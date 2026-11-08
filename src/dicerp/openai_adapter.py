from collections.abc import Sequence
from typing import cast

from openai import OpenAI, OpenAIError
from openai.types.responses import ResponseInputItemParam, ResponseInputParam, ToolParam

from dicerp.agent import INSTRUCTIONS, Call, EvidenceSelection, Output, Turn
from dicerp.errors import ProviderUnavailable
from dicerp.tools import TOOL_SPECS


def create_client(key: str) -> OpenAI:
    return OpenAI(api_key=key, base_url="https://api.openai.com/v1", timeout=10.0, max_retries=0)


class OpenAIEmbedder:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model, input=list(texts), encoding_format="float"
            )
        except OpenAIError as exc:
            raise ProviderUnavailable("Embeddings service unavailable.") from exc
        ordered = sorted(response.data, key=lambda row: row.index)
        if [row.index for row in ordered] != list(range(len(texts))):
            raise ProviderUnavailable("Embeddings service returned invalid indices.")
        return [row.embedding for row in ordered]


class OpenAIModel:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def start(self, question: str) -> "OpenAIConversation":
        return OpenAIConversation(self.client, self.model, question)


class OpenAIConversation:
    def __init__(self, client: OpenAI, model: str, question: str) -> None:
        self.client = client
        self.model = model
        self.transcript: ResponseInputParam = [{"role": "user", "content": question}]

    def step(self, outputs: Sequence[Output]) -> Turn:
        self.transcript.extend(
            {"type": "function_call_output", "call_id": item.call_id, "output": item.payload}
            for item in outputs
        )
        tools: list[ToolParam] = [
            {
                "type": "function",
                "name": name,
                "description": spec.description,
                "strict": True,
                "parameters": spec.schema.model_json_schema(),
            }
            for name, spec in TOOL_SPECS.items()
        ]
        try:
            response = self.client.responses.create(
                model=self.model,
                instructions=INSTRUCTIONS,
                input=self.transcript,
                tools=tools,
                parallel_tool_calls=False,
                store=False,
                include=["reasoning.encrypted_content"],
                max_output_tokens=768,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "evidence_selection",
                        "strict": True,
                        "schema": EvidenceSelection.model_json_schema(),
                    }
                },
            )
        except OpenAIError as exc:
            raise ProviderUnavailable("Responses service unavailable.") from exc
        if response.status != "completed":
            raise ProviderUnavailable("Responses service returned an incomplete response.")
        self.transcript.extend(
            cast(ResponseInputItemParam, item.model_dump(mode="json", exclude_none=True))
            for item in response.output
        )
        return Turn(
            calls=tuple(
                Call(item.call_id, item.name, item.arguments)
                for item in response.output
                if item.type == "function_call"
            ),
            selection_json=response.output_text,
        )

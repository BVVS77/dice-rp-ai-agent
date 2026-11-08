import re
from collections.abc import Sequence

import httpx
from pydantic import ValidationError

from dicerp.agent import INSTRUCTIONS, Call, Output, Turn
from dicerp.errors import ProviderTimeout, ProviderUnavailable
from dicerp.models import StrictModel
from dicerp.tools import TOOL_SPECS

SELECTION_INSTRUCTIONS = """
When finished calling tools, respond with exactly one JSON object containing only
"evidence_ids", for example {"evidence_ids":["E1","E2"]}. If the evidence is insufficient,
respond with {"evidence_ids":[]}. Do not write an answer or add any explanation.
"""
JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def create_ollama_client(base_url: httpx.URL, timeout: float) -> httpx.Client:
    return httpx.Client(
        base_url=base_url,
        timeout=httpx.Timeout(timeout, connect=min(3.0, timeout)),
        trust_env=False,
        follow_redirects=False,
    )


def post(client: httpx.Client, route: str, payload: dict[str, object]) -> bytes:
    try:
        response = client.post(route, json=payload)
        response.raise_for_status()
        return response.content
    except httpx.TimeoutException:
        raise ProviderTimeout(
            "Ollama timed out. Check model loading or increase the timeout."
        ) from None
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            raise ProviderUnavailable(
                "Ollama endpoint or model missing. Check ollama list."
            ) from None
        raise ProviderUnavailable("Ollama rejected the request.") from None
    except httpx.RequestError:
        raise ProviderUnavailable(
            "Could not reach Ollama. Check that the server is running."
        ) from None


class WireModel(StrictModel):
    # Ignore extension fields such as Ollama's reasoning_content and usage.
    model_config = {"extra": "ignore"}


class Embedding(WireModel):
    index: int
    embedding: list[float]


class EmbeddingResponse(WireModel):
    data: list[Embedding]


class Function(WireModel):
    name: str
    arguments: str


class ToolCall(WireModel):
    id: str
    type: str
    function: Function


class Message(WireModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCall] | None = None


class Choice(WireModel):
    finish_reason: str
    message: Message


class Completion(WireModel):
    choices: list[Choice]


class OllamaEmbedder:
    def __init__(self, client: httpx.Client, model: str) -> None:
        self.client = client
        self.model = model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raw = post(self.client, "embeddings", {"model": self.model, "input": list(texts)})
        try:
            response = EmbeddingResponse.model_validate_json(raw)
        except ValidationError:
            raise ProviderUnavailable("Ollama returned an invalid embedding response.") from None
        ordered = sorted(response.data, key=lambda row: row.index)
        if [row.index for row in ordered] != list(range(len(texts))):
            raise ProviderUnavailable("Ollama returned invalid embedding indices.")
        return [row.embedding for row in ordered]


class OllamaModel:
    def __init__(self, client: httpx.Client, model: str) -> None:
        self.client = client
        self.model = model

    def start(self, question: str) -> "OllamaConversation":
        return OllamaConversation(self.client, self.model, question)


class OllamaConversation:
    def __init__(self, client: httpx.Client, model: str, question: str) -> None:
        self.client = client
        self.model = model
        self.messages: list[dict[str, object]] = [
            {"role": "system", "content": INSTRUCTIONS + SELECTION_INSTRUCTIONS},
            {"role": "user", "content": question},
        ]

    def step(self, outputs: Sequence[Output]) -> Turn:
        self.messages.extend(
            {"role": "tool", "tool_call_id": item.call_id, "content": item.payload}
            for item in outputs
        )
        tools = [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": spec.description,
                    "parameters": spec.schema.model_json_schema(),
                },
            }
            for name, spec in TOOL_SPECS.items()
        ]
        raw = post(
            self.client,
            "chat/completions",
            {
                "model": self.model,
                "messages": self.messages,
                "tools": tools,
                "temperature": 0,
                "max_tokens": 768,
                "reasoning_effort": "none",
                "stream": False,
            },
        )
        try:
            (choice,) = Completion.model_validate_json(raw).choices
        except ValueError:
            raise ProviderUnavailable("Ollama returned an invalid chat response.") from None

        message = choice.message
        calls = message.tool_calls or []
        if (
            message.role != "assistant"
            or choice.finish_reason not in {"stop", "tool_calls"}
            or (choice.finish_reason == "tool_calls" and not calls)
            or any(call.type != "function" for call in calls)
        ):
            raise ProviderUnavailable("Ollama returned an incomplete or unsupported chat response.")

        if calls:
            self.messages.append(message.model_dump(exclude_none=True))
            return Turn(
                calls=tuple(
                    Call(call.id, call.function.name, call.function.arguments) for call in calls
                )
            )
        content = (message.content or "").strip()
        fenced = JSON_FENCE.fullmatch(content)
        return Turn(selection_json=fenced[1] if fenced else content)

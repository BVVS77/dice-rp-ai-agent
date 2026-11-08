import json

import httpx
import pytest
from openai import OpenAI

from dicerp.agent import Call, Output
from dicerp.errors import ProviderUnavailable
from dicerp.openai_adapter import OpenAIEmbedder, OpenAIModel


def test_sdk_sends_stateless_tool_outputs() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/responses"
        requests.append(json.loads(request.content))
        output: dict[str, object]
        if len(requests) == 1:
            output = {
                "type": "function_call",
                "id": "fc_test",
                "call_id": "call_test",
                "name": "get_server_status",
                "arguments": "{}",
                "status": "completed",
            }
        else:
            output = {
                "type": "message",
                "id": "msg_test",
                "status": "completed",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": '{"evidence_ids":["E1"]}', "annotations": []}
                ],
            }
        return httpx.Response(200, json={"status": "completed", "output": [output]})

    with OpenAI(
        api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ) as client:
        session = OpenAIModel(client, "test-model").start("Status?")
        assert session.step([]).calls == (Call("call_test", "get_server_status", "{}"),)
        answer = session.step([Output("call_test", '{"evidence":{"E1":"example"}}')])
    assert answer.selection_json == '{"evidence_ids":["E1"]}'
    assert requests[0]["store"] is False
    assert requests[0]["parallel_tool_calls"] is False
    assert isinstance(requests[0]["tools"], list) and len(requests[0]["tools"]) == 3
    inputs = requests[1]["input"]
    assert isinstance(inputs, list)
    assert [item.get("type") for item in inputs] == [None, "function_call", "function_call_output"]
    assert inputs[-1]["call_id"] == "call_test"


def test_embeddings_preserve_input_order() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [0.0, 1.0]},
                    {"object": "embedding", "index": 0, "embedding": [1.0, 0.0]},
                ],
                "model": "test-model",
                "object": "list",
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )

    with OpenAI(
        api_key="test-only", http_client=httpx.Client(transport=httpx.MockTransport(handler))
    ) as client:
        assert OpenAIEmbedder(client, "test-model").embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
        with pytest.raises(ProviderUnavailable, match="indices"):
            OpenAIEmbedder(client, "test-model").embed(["a"])


def test_provider_http_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, json={"error": {"message": "provider internal detail"}})

    with (
        OpenAI(
            api_key="test-only",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        ) as client,
        pytest.raises(ProviderUnavailable, match="Responses service unavailable"),
    ):
        OpenAIModel(client, "test-model").start("Status?").step([])

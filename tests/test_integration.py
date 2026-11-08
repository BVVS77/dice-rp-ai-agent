import json
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI

from dicerp.api import create_app
from dicerp.db import seed_database
from dicerp.openai_adapter import OpenAIEmbedder
from dicerp.retrieval import build_index, handbook


def test_chat_through_configured_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("DICERP_AI_PROVIDER", "openai")
    monkeypatch.setenv("DICERP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DICERP_EMBEDDING_MODEL", "test-embeddings")
    monkeypatch.setenv("DICERP_CHAT_MODEL", "test-model")
    seed_database(tmp_path / "snapshot.sqlite3")
    expected = next(rule for rule in handbook() if rule.title == "Metagaming")
    requested_paths: list[str] = []

    def upstream(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        payload = json.loads(request.content)
        if request.url.path == "/v1/embeddings":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {
                            "index": index,
                            "embedding": [1.0, 0.0]
                            if "metagaming" in value.lower()
                            else [0.0, 1.0],
                        }
                        for index, value in enumerate(payload["input"])
                    ],
                },
            )
        assert request.url.path == "/v1/responses"
        last_input = payload["input"][-1]
        output: dict[str, object]
        if last_input.get("role") == "user":
            output = {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call_1",
                "name": "search_rules",
                "arguments": '{"query":"metagaming"}',
            }
        else:
            assert last_input["call_id"] == "call_1"
            selected = json.loads(last_input["output"])["evidence"]["E1"]
            assert selected == expected.model_dump()
            output = {
                "type": "message",
                "content": [{"type": "output_text", "text": '{"evidence_ids":["E1"]}'}],
            }
        return httpx.Response(200, json={"status": "completed", "output": [output]})

    def sdk_client(key: str) -> OpenAI:
        return OpenAI(
            api_key=key, http_client=httpx.Client(transport=httpx.MockTransport(upstream))
        )

    with sdk_client("test-only") as sdk:
        build_index(
            tmp_path / "rules.json", "test-embeddings", OpenAIEmbedder(sdk, "test-embeddings")
        )
    monkeypatch.setattr("dicerp.providers.create_client", sdk_client)

    with TestClient(create_app(), base_url="http://localhost") as client:
        response = client.post("/v1/chat", json={"question": "What is metagaming?"})

    assert response.status_code == 200
    assert response.json()["sources"] == [expected.model_dump()]
    assert response.json()["tool_calls"] == [{"name": "search_rules", "outcome": "ok"}]
    assert requested_paths == ["/v1/embeddings", "/v1/responses", "/v1/embeddings", "/v1/responses"]

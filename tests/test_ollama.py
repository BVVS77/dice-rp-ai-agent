import json
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from dicerp.agent import Output
from dicerp.api import create_app
from dicerp.config import Settings
from dicerp.db import seed_database
from dicerp.errors import ProviderUnavailable
from dicerp.ollama_adapter import OllamaEmbedder, OllamaModel
from dicerp.retrieval import RuleIndex, build_index

URL = "http://127.0.0.1:11434/v1"


class Vectors:
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0, 1.0] for _ in texts]


def test_index_rejects_provider_switch_even_with_same_model_name(tmp_path: Path) -> None:
    path = tmp_path / "rules.json"
    model = "same-model"
    embedder = Vectors()
    build_index(path, model, embedder)
    legacy = json.loads(path.read_bytes())
    del legacy["provider"]
    path.write_text(json.dumps(legacy), encoding="utf-8")
    RuleIndex.load(path, model, embedder)
    with pytest.raises(ValueError, match="provider"):
        RuleIndex.load(path, model, embedder, provider="ollama")
    build_index(path, model, embedder, provider="ollama")
    RuleIndex.load(path, model, embedder, provider="ollama")
    with pytest.raises(ValueError, match="provider"):
        RuleIndex.load(path, model, embedder)


def test_provider_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DICERP_CHAT_MODEL", raising=False)
    monkeypatch.delenv("DICERP_EMBEDDING_MODEL", raising=False)
    monkeypatch.setenv("DICERP_AI_PROVIDER", "ollama")
    local = Settings(dicerp_chat_model=None, dicerp_embedding_model=None)
    assert (local.chat_model, local.embedding_model) == ("qwen3:8b", "qwen3-embedding:0.6b")
    monkeypatch.setenv("DICERP_AI_PROVIDER", "openai")
    cloud = Settings(dicerp_chat_model=None, dicerp_embedding_model=None)
    assert (cloud.chat_model, cloud.embedding_model) == ("gpt-4.1-mini", "text-embedding-3-small")
    assert Settings(dicerp_chat_model="explicit").chat_model == "explicit"


@pytest.mark.parametrize(
    "url",
    [
        "https://ollama.com/v1",
        "http://127.0.0.1:invalid/v1",
        "http://127.0.0.1:11434/api",
        "http://localhost.evil.test/v1",
        "http://user:pass@127.0.0.1/v1",
        "http://127.0.0.1/v1?redirect=elsewhere",
    ],
)
def test_ollama_rejects_nonlocal_or_ambiguous_urls(url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        Settings(ollama_base_url=url).local_ollama_url()


def test_chat_completion_replays_batched_tool_calls() -> None:
    bodies: list[dict[str, object]] = []
    tools = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "search_rules", "arguments": '{"query":"Metagaming"}'},
        },
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "get_server_status", "arguments": "{}"},
        },
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/chat/completions"
        assert "Authorization" not in request.headers
        bodies.append(json.loads(request.content))
        if len(bodies) == 1:
            message = {"role": "assistant", "tool_calls": tools, "reasoning_content": "ignored"}
        else:
            message = {"role": "assistant", "content": '```json\n{"evidence_ids":["E1","E2"]}\n```'}
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "tool_calls" if len(bodies) == 1 else "stop",
                        "message": message,
                    }
                ]
            },
        )

    with httpx.Client(base_url=URL, transport=httpx.MockTransport(handler)) as client:
        conversation = OllamaModel(client, "qwen3:8b").start("Rules and status?")
        turn = conversation.step([])
        assert [call.call_id for call in turn.calls] == ["call_1", "call_2"]
        finish = conversation.step([Output("call_1", '{"evidence":{}}'), Output("call_2", "{}")])
        assert finish.selection_json == '{"evidence_ids":["E1","E2"]}'

    assert bodies[0]["model"] == "qwen3:8b"
    assert bodies[0]["reasoning_effort"] == "none"
    assert not {"store", "include", "previous_response_id", "response_format"} & bodies[0].keys()
    messages = bodies[1]["messages"]
    assert isinstance(messages, list)
    assert messages[-3] == {"role": "assistant", "tool_calls": tools}
    assert messages[-2] == {"role": "tool", "tool_call_id": "call_1", "content": '{"evidence":{}}'}
    assert messages[-1] == {"role": "tool", "tool_call_id": "call_2", "content": "{}"}


def test_embedder_uses_model_and_reorders_indexed_vectors() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/embeddings"
        assert "Authorization" not in request.headers
        payload = json.loads(request.content)
        assert payload["model"] == "qwen3-embedding:0.6b"
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": 1, "embedding": [0, 1]},
                    {"index": 0, "embedding": [1, 0]},
                ]
            },
        )

    with httpx.Client(base_url=URL, transport=httpx.MockTransport(handler)) as client:
        embedder = OllamaEmbedder(client, "qwen3-embedding:0.6b")
        assert embedder.embed(["first", "second"]) == [[1, 0], [0, 1]]
        with pytest.raises(ProviderUnavailable, match="indices"):
            embedder.embed(["only one"])


@pytest.mark.parametrize(
    ("problem", "expected_status", "detail"),
    [
        ("timeout", 504, "Ollama timed out while loading or running a model."),
        ("connection", 502, "The AI provider could not complete the request."),
        ("missing", 502, "The AI provider could not complete the request."),
    ],
)
def test_local_api_errors_are_sanitized_without_an_openai_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str, expected_status: int, detail: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("DICERP_AI_PROVIDER", "ollama")
    monkeypatch.setenv("DICERP_STATE_DIR", str(tmp_path))
    monkeypatch.setenv("DICERP_EMBEDDING_MODEL", "qwen3-embedding:0.6b")
    monkeypatch.setenv("OLLAMA_BASE_URL", URL)
    seed_database(tmp_path / "snapshot.sqlite3")
    build_index(tmp_path / "rules.json", "qwen3-embedding:0.6b", Vectors(), provider="ollama")

    def handler(request: httpx.Request) -> httpx.Response:
        if problem == "timeout":
            raise httpx.ReadTimeout("private diagnostics", request=request)
        if problem == "connection":
            raise httpx.ConnectError("private diagnostics", request=request)
        return httpx.Response(404, json={"error": "private diagnostics"})

    monkeypatch.setattr(
        "dicerp.providers.create_ollama_client",
        lambda url, timeout: httpx.Client(base_url=url, transport=httpx.MockTransport(handler)),
    )
    with TestClient(create_app(), base_url="http://localhost") as client:
        response = client.post("/v1/chat", json={"question": "What is metagaming?"})
    assert response.status_code == expected_status
    assert response.json() == {"detail": detail}

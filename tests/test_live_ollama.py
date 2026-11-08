import os
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from dicerp.api import create_app
from dicerp.config import Settings
from dicerp.db import seed_database
from dicerp.providers import open_provider
from dicerp.retrieval import build_index, handbook

pytestmark = [
    pytest.mark.ollama_live,
    pytest.mark.skipif(
        os.getenv("DICERP_RUN_OLLAMA_LIVE") != "1", reason="local Ollama models are opt-in"
    ),
]


@pytest.fixture(scope="module")
def local_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    path = tmp_path_factory.mktemp("dicerp-ollama-live")
    settings = Settings(
        dicerp_ai_provider="ollama",
        dicerp_state_dir=path,
        dicerp_chat_model="qwen3:8b",
        dicerp_embedding_model="qwen3-embedding:0.6b",
        ollama_base_url="http://127.0.0.1:11434/v1",
        openai_api_key=None,
    )
    seed_database(path / "snapshot.sqlite3")
    with open_provider(settings) as provider:
        build_index(
            path / "rules.json", settings.embedding_model, provider.embedder, provider="ollama"
        )
    return settings


@pytest.fixture
def local_api(local_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr("dicerp.api.Settings", lambda: local_settings)
    with TestClient(create_app(), base_url="http://localhost") as client:
        yield client


def test_live_ollama_mixed_question(local_api: TestClient) -> None:
    response = local_api.post(
        "/v1/chat",
        json={"question": "What is metagaming? Also show stats for demo-001 and server status."},
    )
    assert response.status_code == 200, response.json()
    result = response.json()
    assert result["status"] == "answered", result
    assert {call["name"] for call in result["tool_calls"] if call["outcome"] == "ok"} == {
        "search_rules",
        "get_player_stats",
        "get_server_status",
    }
    source = next(chunk for chunk in handbook() if chunk.title == "Metagaming")
    assert result["sources"] == [source.model_dump()]
    assert source.excerpt in result["answer"]
    assert f"{source.file}, lines {source.start_line}–{source.end_line}" in result["answer"]
    assert "42 hours played" in result["answer"]
    assert "17/64 players" in result["answer"]
    assert "2026-01-15T18:00:00Z" in result["answer"]


@pytest.mark.parametrize(
    "question",
    [
        "What is the maximum number of items allowed in a vehicle glove compartment?",
        "What are the stats for demo-999?",
    ],
)
def test_live_ollama_unknown_question(local_api: TestClient, question: str) -> None:
    response = local_api.post("/v1/chat", json={"question": question})
    assert response.status_code == 200, response.json()
    result = response.json()
    assert result["status"] == "insufficient_evidence", result
    assert result["sources"] == []
    if "demo-999" in question:
        assert {"name": "get_player_stats", "outcome": "no_match"} in result["tool_calls"]

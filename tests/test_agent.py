import json
from collections.abc import Iterator, Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from dicerp.agent import Agent, Call, Output, Turn
from dicerp.api import create_app
from dicerp.db import SnapshotRepository, seed_database
from dicerp.errors import AgentLimitReached, ProviderUnavailable
from dicerp.models import Citation
from dicerp.retrieval import handbook
from dicerp.tools import Tools


class SampleRules:
    def search(self, query: str) -> list[Citation]:
        return [rule for rule in handbook() if query.lower() in rule.title.lower()]


class ScriptedModel:
    def __init__(self, *turns: Turn) -> None:
        self.turns = iter(turns)
        self.received: list[list[Output]] = []

    def start(self, question: str) -> "ScriptedModel":
        return self

    def step(self, outputs: Sequence[Output]) -> Turn:
        self.received.append(list(outputs))
        return next(self.turns)


@pytest.fixture
def tools(tmp_path: Path) -> Iterator[Tools]:
    path = tmp_path / "snapshot.sqlite3"
    seed_database(path)
    repo = SnapshotRepository.open(path)
    try:
        yield Tools(SampleRules(), repo)
    finally:
        repo.engine.dispose()


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        ("run_sql", "{}"),
        ("search_rules", "not JSON"),
        ("search_rules", '{"query":"   "}'),
        ("search_rules", '{"query":"metagaming","path":"private.txt"}'),
        ("get_server_status", '{"server":"somewhere"}'),
        ("get_player_stats", '{"player_id": 42}'),
        ("get_player_stats", '{"player_id":"real-player"}'),
        ("get_player_stats", '{"player_id":"demo-001"}'),
    ],
)
def test_invalid_calls(name: str, arguments: str, tools: Tools) -> None:
    result = tools.run(name, arguments, "Show stats for demo-002.")
    assert result.outcome == "invalid_call"
    assert not result.evidence


def test_lookup_outcomes(tools: Tools) -> None:
    assert tools.run("get_player_stats", '{"player_id":"demo-999"}', "demo-999").outcome == (
        "no_match"
    )
    assert tools.run("search_rules", '{"query":"moon landing"}', "moon landing").outcome == (
        "no_match"
    )
    assert tools.run("get_player_stats", '{"player_id":"demo-001"}', "DEMO-001").outcome == "ok"


def test_multi_tool_answer_is_rendered_from_evidence(tools: Tools) -> None:
    model = ScriptedModel(
        Turn(calls=(Call("c1", "search_rules", '{"query":"metagaming"}'),)),
        Turn(calls=(Call("c2", "get_player_stats", '{"player_id":"demo-001"}'),)),
        Turn(calls=(Call("c3", "get_server_status", "{}"),)),
        Turn(selection_json='{"evidence_ids":["E1","E1","E2","E3"]}'),
    )
    result = Agent(model, tools).answer("Explain metagaming, show demo-001 and server status.")
    assert result.status == "answered"
    assert len(result.sources) == 1
    source = result.sources[0]
    assert source == next(chunk for chunk in handbook() if chunk.title == "Metagaming")
    assert f"{source.file}, lines {source.start_line}–{source.end_line}" in result.answer
    assert "42 hours played" in result.answer
    assert "17/64 players" in result.answer
    assert [outputs[0].call_id for outputs in model.received[1:]] == ["c1", "c2", "c3"]
    assert json.loads(model.received[1][0].payload)["evidence"]["E1"]["excerpt"] == source.excerpt
    assert [item.name for item in result.tool_calls] == [
        "search_rules",
        "get_player_stats",
        "get_server_status",
    ]


@pytest.mark.parametrize(
    "selection",
    [
        '{"evidence_ids":["E1","invented-rule"]}',
        '{"evidence_ids":["E1"],"answer":"Metagaming is allowed"}',
        '{"evidence_ids":[]}',
        "Trust me. You can metagame.",
    ],
)
def test_unsupported_model_output_abstains(selection: str, tools: Tools) -> None:
    model = ScriptedModel(
        Turn(calls=(Call("c1", "search_rules", '{"query":"metagaming"}'),)),
        Turn(selection_json=selection),
    )
    result = Agent(model, tools).answer("Question")
    assert result.status == "insufficient_evidence"
    assert not result.sources
    assert "allowed" not in result.answer
    assert len(result.tool_calls) == 1


def test_cannot_cite_results_never_fetched(tools: Tools) -> None:
    model = ScriptedModel(Turn(selection_json='{"evidence_ids":["E1"]}'))
    assert Agent(model, tools).answer("Invent E1").status == "insufficient_evidence"


def test_duplicate_call_id_fails_closed(tools: Tools) -> None:
    turn = Turn(calls=(Call("same", "get_server_status", "{}"),))
    with pytest.raises(ProviderUnavailable):
        Agent(ScriptedModel(turn, turn), tools).answer("Status?")


def test_tool_budget_exhaustion(tools: Tools) -> None:
    turns = [Turn(calls=(Call(str(i), "get_server_status", "{}"),)) for i in range(5)]
    with pytest.raises(AgentLimitReached):
        Agent(ScriptedModel(*turns), tools).answer("Status?")


def test_api_validates_host_origin_and_input(tools: Tools) -> None:
    model = ScriptedModel(Turn(selection_json='{"evidence_ids":[]}'))
    with TestClient(create_app(Agent(model, tools)), base_url="http://localhost") as client:
        assert client.get("/healthz").json() == {"status": "ready"}
        assert client.post("/v1/chat", json={"question": " "}).status_code == 422
        assert client.post("/v1/chat", json={"question": "x" * 1001}).status_code == 422
        assert client.get("/healthz", headers={"Host": "other.test"}).status_code == 400
        request = {"question": "What about dinosaurs?"}
        assert (
            client.post(
                "/v1/chat", json=request, headers={"Origin": "https://other.test"}
            ).status_code
            == 403
        )
        response = client.post("/v1/chat", json=request)
        assert response.status_code == 200
        assert response.json()["status"] == "insufficient_evidence"


def test_api_sanitizes_provider_failure(tools: Tools) -> None:
    class FailingModel:
        def start(self, question: str) -> ScriptedModel:
            raise ProviderUnavailable("sensitive backend diagnostics")

    with TestClient(
        create_app(Agent(FailingModel(), tools)), base_url="http://localhost"
    ) as client:
        response = client.post("/v1/chat", json={"question": "Status?"})
    assert response.status_code == 502
    assert "sensitive" not in response.text

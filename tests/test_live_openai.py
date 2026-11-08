"""Opt-in, budgeted smoke tests using only the project's synthetic fixtures."""

import os
from collections import Counter
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import OpenAI
from pydantic import SecretStr, ValidationError

from dicerp.api import create_app
from dicerp.config import Settings
from dicerp.db import seed_database
from dicerp.errors import ProviderUnavailable
from dicerp.models import ChatResponse
from dicerp.openai_adapter import OpenAIEmbedder
from dicerp.retrieval import build_index, handbook

CHAT_MODEL = "gpt-4.1-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
LIMITS = {"/v1/responses": 9, "/v1/embeddings": 5}


@dataclass
class Usage:
    requests: Counter[str] = field(default_factory=Counter)
    statuses: list[tuple[str, int]] = field(default_factory=list)
    tokens: Counter[str] = field(default_factory=Counter)
    errors: list[str] = field(default_factory=list)

    def before(self, request: httpx.Request) -> None:
        path = request.url.path
        if (
            request.url.scheme != "https"
            or request.url.host != "api.openai.com"
            or path not in LIMITS
            or self.requests[path] >= LIMITS[path]
        ):
            raise httpx.RequestError("Live test request exceeded its allowlist or budget.")
        self.requests[path] += 1

    def after(self, response: httpx.Response) -> None:
        path = response.request.url.path
        self.statuses.append((path, response.status_code))
        response.read()
        try:
            payload = response.json()
        except ValueError:
            return
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        if isinstance(usage, dict):
            fields = (
                {"input_tokens": "response_input", "output_tokens": "response_output"}
                if path == "/v1/responses"
                else {"total_tokens": "embedding"}
            )
            for key, label in fields.items():
                if isinstance(value := usage.get(key), int) and value >= 0:
                    self.tokens[label] += value
        error = payload.get("error")
        code = error.get("code") if isinstance(error, dict) else None
        if code in (
            "credit_balance_exhausted",
            "insufficient_quota",
            "invalid_api_key",
            "model_not_found",
            "rate_limit_exceeded",
        ):
            self.errors.append(code)

    def summary(self) -> str:
        return (
            f"requests={dict(self.requests)} statuses={self.statuses} "
            f"tokens={dict(self.tokens)} error_codes={self.errors}"
        )


@dataclass(frozen=True)
class LiveSetup:
    settings: Settings
    sdk: Callable[[str], OpenAI]
    usage: Usage


def prepare_index(path: Path, client: OpenAI) -> None:
    try:
        build_index(path / "rules.json", EMBEDDING_MODEL, OpenAIEmbedder(client, EMBEDDING_MODEL))
        return
    except ProviderUnavailable:
        pass
    # Report outside the exception handler so pytest cannot display the SDK error chain.
    pytest.fail("Live handbook indexing failed. See sanitized provider usage.", pytrace=False)


@pytest.fixture(scope="module")
def live_setup(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LiveSetup]:
    if os.environ.get("DICERP_RUN_LIVE") != "1":
        pytest.skip("live OpenAI is opt-in")
    env_file = Path(os.environ.get("DICERP_LIVE_ENV_FILE", ".env"))
    try:
        credentials = Settings(_env_file=env_file)  # type: ignore[call-arg]
        key = credentials.api_key()
    except (OSError, ValueError):
        key = ""
    if not key:
        pytest.fail("A development API key could not be loaded for live tests.", pytrace=False)
    path = tmp_path_factory.mktemp("dicerp-live")
    settings = Settings.model_construct(
        openai_api_key=SecretStr(key),
        dicerp_chat_model=CHAT_MODEL,
        dicerp_embedding_model=EMBEDDING_MODEL,
        dicerp_state_dir=path,
    )
    usage = Usage()

    def sdk(key: str) -> OpenAI:
        return OpenAI(
            api_key=key,
            base_url="https://api.openai.com/v1",
            timeout=10.0,
            max_retries=0,
            http_client=httpx.Client(
                event_hooks={"request": [usage.before], "response": [usage.after]},
                trust_env=False,
            ),
        )

    try:
        seed_database(path / "snapshot.sqlite3")
        with sdk(key) as client:
            prepare_index(path, client)
        yield LiveSetup(settings, sdk, usage)
    finally:
        print(f"Live OpenAI usage {usage.summary()}")


@pytest.fixture
def live_api(live_setup: LiveSetup, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    monkeypatch.setattr("dicerp.api.Settings", lambda: live_setup.settings)
    monkeypatch.setattr("dicerp.providers.create_client", live_setup.sdk)
    with TestClient(create_app(), base_url="http://localhost") as client:
        yield client


def ask(client: TestClient, question: str) -> ChatResponse:
    response = client.post("/v1/chat", json={"question": question})
    if response.status_code != 200:
        pytest.fail(f"Live chat returned HTTP {response.status_code}.", pytrace=False)
    try:
        return ChatResponse.model_validate_json(response.content)
    except ValidationError:
        pass
    pytest.fail("Live chat returned an invalid response schema.", pytrace=False)


@pytest.mark.live
def test_live_mixed_question(live_api: TestClient) -> None:
    assert live_api.post("/v1/chat", json={"question": " "}).status_code == 422
    result = ask(
        live_api, "What is metagaming? Also show stats for demo-001 and the server status."
    )
    assert result.status == "answered"
    assert {call.name for call in result.tool_calls if call.outcome == "ok"} == {
        "search_rules",
        "get_player_stats",
        "get_server_status",
    }
    source = next(chunk for chunk in handbook() if chunk.title == "Metagaming")
    assert result.sources == [source]
    assert source.excerpt in result.answer
    assert f"{source.file}, lines {source.start_line}–{source.end_line}" in result.answer
    assert "42 hours played" in result.answer
    assert "17/64 players" in result.answer
    assert "2026-01-15T18:00:00Z" in result.answer


@pytest.mark.live
def test_live_missing_player_abstains(live_api: TestClient) -> None:
    result = ask(live_api, "What are the stats for demo-999?")
    assert result.status == "insufficient_evidence"
    assert not result.sources
    assert any(
        call.name == "get_player_stats" and call.outcome == "no_match" for call in result.tool_calls
    )


@pytest.mark.live
def test_live_unknown_rule_abstains(live_api: TestClient) -> None:
    result = ask(
        live_api, "What is the maximum number of items allowed in a vehicle glove compartment?"
    )
    assert result.status == "insufficient_evidence"
    assert not result.sources


@pytest.mark.live
def test_live_provider_failure_is_sanitized(
    live_setup: LiveSetup, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = live_setup.settings.model_copy(
        update={"dicerp_chat_model": "dicerp-deliberately-nonexistent-smoke-test-model"}
    )
    monkeypatch.setattr("dicerp.api.Settings", lambda: settings)
    monkeypatch.setattr("dicerp.providers.create_client", live_setup.sdk)
    with TestClient(create_app(), base_url="http://localhost") as client:
        response = client.post("/v1/chat", json={"question": "What is the server status?"})
    assert live_setup.usage.statuses[-1] in {
        ("/v1/responses", 400),
        ("/v1/responses", 404),
    }
    assert response.status_code == 502
    assert response.json() == {"detail": "The AI provider could not complete the request."}


def test_live_index_setup_redacts_provider_error(tmp_path: Path) -> None:
    usage = Usage()

    def upstream(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            json={
                "error": {
                    "code": "credit_balance_exhausted",
                    "message": "private diagnostic",
                }
            },
        )

    with (
        OpenAI(
            api_key="test-only",
            max_retries=0,
            http_client=httpx.Client(
                transport=httpx.MockTransport(upstream),
                event_hooks={"request": [usage.before], "response": [usage.after]},
            ),
        ) as client,
        pytest.raises(pytest.fail.Exception, match="Live handbook indexing failed") as failed,
    ):
        prepare_index(tmp_path, client)

    assert failed.value.__context__ is None
    assert "private diagnostic" not in usage.summary()
    assert usage.errors == ["credit_balance_exhausted"]
    assert usage.requests == {"/v1/embeddings": 1}
    assert not usage.tokens
    assert not (tmp_path / "rules.json").exists()

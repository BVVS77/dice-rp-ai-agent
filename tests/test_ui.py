from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def app() -> Iterator[AppTest]:
    yield AppTest.from_file(str(APP)).run()


def test_chat_displays_answer_citation_and_trace(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    def post(self: httpx.Client, url: str, *, json: dict[str, str]) -> httpx.Response:
        assert url == "http://127.0.0.1:8000/v1/chat"
        assert json == {"question": "What is metagaming?"}
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "status": "answered",
                "answer": "Metagaming [1]",
                "sources": [
                    {
                        "id": "R-03:10",
                        "title": "Metagaming",
                        "file": "conduct.txt",
                        "start_line": 10,
                        "end_line": 10,
                        "excerpt": "Use in-character information.",
                    }
                ],
                "tool_calls": [{"name": "search_rules", "outcome": "ok"}],
            },
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    app.chat_input[0].set_value("What is metagaming?").run()
    assert not app.exception
    assert len(app.chat_message) == 2
    assert "conduct.txt · lines 10–10" in [caption.value for caption in app.caption]
    assert "search_rules  ·  ok" in [node.value for node in app.text]
    clear_button = next(button for button in app.button if button.label == "Clear chat")
    clear_button.click().run()
    assert len(app.chat_message) == 0


def test_provider_error_is_visible_and_sanitized(
    app: AppTest, monkeypatch: pytest.MonkeyPatch
) -> None:
    def post(self: httpx.Client, url: str, *, json: dict[str, str]) -> httpx.Response:
        return httpx.Response(
            502, request=httpx.Request("POST", url), json={"detail": "backend secret"}
        )

    monkeypatch.setattr(httpx.Client, "post", post)
    app.chat_input[0].set_value("What is the server status?").run()
    assert not app.exception
    assert len(app.error) == 1
    assert "AI provider is unavailable" in app.error[0].value
    assert "backend secret" not in app.error[0].value

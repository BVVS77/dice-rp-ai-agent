import httpx
import streamlit as st
from pydantic import ValidationError

from dicerp.models import ChatResponse

API = "http://127.0.0.1:8000/v1/chat"
EXAMPLES = {
    "Server rules": "What does the handbook say about metagaming?",
    "Player profile": "Show the stats for demo-001.",
    "Server status": "What is the server status and player count?",
}


def ask(question: str) -> ChatResponse:
    with httpx.Client(timeout=httpx.Timeout(1100.0, connect=3.0), trust_env=False) as client:
        response = client.post(API, json={"question": question})
        response.raise_for_status()
        return ChatResponse.model_validate_json(response.content)


def show_reply(reply: ChatResponse) -> None:
    with st.chat_message("assistant"):
        st.text(reply.answer)
        if reply.sources:
            with st.expander(f"Handbook sources · {len(reply.sources)}"):
                for index, source in enumerate(reply.sources, 1):
                    st.text(f"[{index}] {source.title}")
                    st.caption(f"{source.file} · lines {source.start_line}–{source.end_line}")
                    st.text(source.excerpt)
        if reply.tool_calls:
            with st.expander(f"Tool activity · {len(reply.tool_calls)}"):
                for call in reply.tool_calls:
                    st.text(f"{call.name}  ·  {call.outcome.replace('_', ' ')}")


def main() -> None:
    st.set_page_config(page_title="DiceRP AI Agent", page_icon="🎲", layout="centered")
    st.title("DiceRP AI Agent")
    st.caption("FIVEM ROLEPLAY  /  FICTIONAL SERVER SANDBOX")
    st.write("Find a server rule, look up a demo player, or inspect the latest sample snapshot.")
    st.caption("Use sample data only. Each question is sent independently to the AI backend.")

    if "turns" not in st.session_state:
        st.session_state.turns = []
    with st.sidebar:
        st.header("About this sandbox")
        st.write("Rules cite exact handbook lines. Player stats and status are synthetic.")
        st.caption("Demo IDs · demo-001, demo-002, demo-003")
        if st.button("Clear chat"):
            st.session_state.turns = []

    example = None
    for column, (label, example_question) in zip(st.columns(3), EXAMPLES.items(), strict=True):
        if column.button(label):
            example = example_question
    for previous_question, reply in st.session_state.turns:
        with st.chat_message("user"):
            st.text(previous_question)
        show_reply(reply)

    question = st.chat_input("Ask about DiceRP…", max_chars=1000) or example
    if not question or not question.strip():
        return
    with st.chat_message("user"):
        st.text(question)
    try:
        with st.spinner("Checking the handbook and sandbox tools…"):
            reply = ask(question)
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        message = {
            422: "Enter a question of 1–1,000 characters.",
            502: "The AI provider is unavailable. Check the backend configuration and try again.",
            503: "The sandbox could not complete this request. Try a narrower question.",
            504: "Ollama timed out loading or running a model. Check the local server.",
        }.get(status, "The backend returned an unexpected error.")
        st.error(message)
        return
    except httpx.TimeoutException:
        st.error("The backend took too long to respond. Check the provider and try again.")
        return
    except (httpx.RequestError, ValidationError):
        st.error("Could not get a valid response. Check that FastAPI is running on port 8000.")
        return
    st.session_state.turns = [*st.session_state.turns, (question, reply)][-30:]
    show_reply(reply)


if __name__ == "__main__":
    main()

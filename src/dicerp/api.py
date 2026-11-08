from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI, HTTPException, Request
from pydantic import ValidationError
from sqlalchemy.exc import SQLAlchemyError
from starlette.middleware.trustedhost import TrustedHostMiddleware

from dicerp.agent import Agent
from dicerp.config import Settings
from dicerp.db import SnapshotRepository
from dicerp.errors import AgentLimitReached, ProviderTimeout, ProviderUnavailable
from dicerp.models import ChatRequest, ChatResponse
from dicerp.providers import open_provider
from dicerp.retrieval import RuleIndex
from dicerp.tools import Tools


def create_app(agent: Agent | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if agent is not None:
            app.state.agent = agent
            yield
            return

        settings = Settings()
        with open_provider(settings) as provider:
            try:
                rules = RuleIndex.load(
                    settings.dicerp_state_dir / "rules.json",
                    settings.embedding_model,
                    provider.embedder,
                    provider=settings.dicerp_ai_provider,
                )
            except (OSError, ValueError) as exc:
                raise RuntimeError("Rule index unavailable. Run dicerp index.") from exc
            repo = SnapshotRepository.open(settings.dicerp_state_dir / "snapshot.sqlite3")
            try:
                try:
                    if repo.status() is None or repo.player("demo-001") is None:
                        raise ValueError("Incomplete fixture.")
                except (SQLAlchemyError, ValueError) as exc:
                    raise RuntimeError("Snapshot unavailable. Run dicerp seed.") from exc
                app.state.agent = Agent(provider.model, Tools(rules, repo))
                yield
            finally:
                repo.engine.dispose()

    app = FastAPI(title="DiceRP AI Agent", version="0.1.0", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])

    @app.get("/healthz")
    def health() -> dict[str, str]:
        return {"status": "ready"}

    @app.post("/v1/chat")
    def chat(payload: ChatRequest, request: Request) -> ChatResponse:
        origin = request.headers.get("origin")
        if origin is not None and origin != f"{request.url.scheme}://{request.headers['host']}":
            raise HTTPException(403, "Cross-origin requests are disabled.")
        try:
            return cast(Agent, request.app.state.agent).answer(payload.question)
        except ProviderTimeout as exc:
            raise HTTPException(504, "Ollama timed out while loading or running a model.") from exc
        except ProviderUnavailable as exc:
            raise HTTPException(502, "The AI provider could not complete the request.") from exc
        except AgentLimitReached as exc:
            raise HTTPException(503, "Tool budget exceeded. Try a narrower question.") from exc
        except (SQLAlchemyError, ValidationError) as exc:
            raise HTTPException(503, "The synthetic snapshot is unavailable.") from exc

    return app

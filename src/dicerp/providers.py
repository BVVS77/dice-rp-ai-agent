from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from dicerp.agent import Model
from dicerp.config import Settings
from dicerp.ollama_adapter import OllamaEmbedder, OllamaModel, create_ollama_client
from dicerp.openai_adapter import OpenAIEmbedder, OpenAIModel, create_client
from dicerp.retrieval import Embedder


@dataclass(frozen=True)
class Provider:
    model: Model
    embedder: Embedder


@contextmanager
def open_provider(settings: Settings) -> Iterator[Provider]:
    if settings.dicerp_ai_provider == "ollama":
        with create_ollama_client(
            settings.local_ollama_url(), settings.dicerp_ollama_timeout_seconds
        ) as client:
            yield Provider(
                OllamaModel(client, settings.chat_model),
                OllamaEmbedder(client, settings.embedding_model),
            )
    else:
        with create_client(settings.api_key()) as client:
            yield Provider(
                OpenAIModel(client, settings.chat_model),
                OpenAIEmbedder(client, settings.embedding_model),
            )

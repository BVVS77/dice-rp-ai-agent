from pathlib import Path
from typing import Literal

import httpx
from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

type ProviderName = Literal["openai", "ollama"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: SecretStr | None = None
    dicerp_ai_provider: ProviderName = "openai"
    dicerp_chat_model: str | None = None
    dicerp_embedding_model: str | None = None
    dicerp_state_dir: Path = Path(".dicerp")
    ollama_base_url: str = "http://127.0.0.1:11434/v1"
    dicerp_ollama_timeout_seconds: float = Field(default=60.0, gt=0, le=120)

    @property
    def chat_model(self) -> str:
        default = "qwen3:8b" if self.dicerp_ai_provider == "ollama" else "gpt-4.1-mini"
        return self.dicerp_chat_model or default

    @property
    def embedding_model(self) -> str:
        default = (
            "qwen3-embedding:0.6b"
            if self.dicerp_ai_provider == "ollama"
            else "text-embedding-3-small"
        )
        return self.dicerp_embedding_model or default

    def api_key(self) -> str:
        if self.openai_api_key is None or not self.openai_api_key.get_secret_value().strip():
            raise ValueError("Set OPENAI_API_KEY with a dedicated development project key.")
        return self.openai_api_key.get_secret_value().strip()

    def local_ollama_url(self) -> httpx.URL:
        try:
            url = httpx.URL(self.ollama_base_url)
        except httpx.InvalidURL:
            raise ValueError("OLLAMA_BASE_URL must be a loopback HTTP URL ending in /v1.") from None
        if (
            url.scheme != "http"
            or url.host not in {"127.0.0.1", "localhost", "::1"}
            or url.userinfo
            or url.query
            or url.fragment
            or url.path.rstrip("/") != "/v1"
        ):
            raise ValueError("OLLAMA_BASE_URL must be a loopback HTTP URL ending in /v1.")
        # Use a numeric loopback address even when localhost is configured.
        if url.host == "localhost":
            url = url.copy_with(host="127.0.0.1")
        return url

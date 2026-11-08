import hashlib
import json
import math
import re
from collections.abc import Sequence
from importlib.resources import files
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Protocol

from dicerp.config import ProviderName
from dicerp.errors import ProviderUnavailable
from dicerp.models import Citation, StrictModel

HEADING = re.compile(r"## (R-[0-9]{2}) \| (.+)")
MAX_CHARS = 900


class Embedder(Protocol):
    def embed(self, texts: Sequence[str]) -> list[list[float]]: ...


def chunk_document(file: str, text: str) -> list[Citation]:
    lines = text.splitlines()
    sections = [(i, match) for i, line in enumerate(lines) if (match := HEADING.fullmatch(line))]
    chunks = []
    for index, (heading, match) in enumerate(sections):
        stop = sections[index + 1][0] if index + 1 < len(sections) else len(lines)
        start = heading + 1
        while start < stop:
            if not lines[start].strip():
                start += 1
                continue
            end = start
            size = 0
            while end < stop and size + len(lines[end]) + (end > start) <= MAX_CHARS:
                size += len(lines[end]) + (end > start)
                end += 1
            if end == start:
                raise ValueError(f"{file}:{start + 1} exceeds {MAX_CHARS} characters.")
            while not lines[end - 1].strip():
                end -= 1
            chunks.append(
                Citation(
                    id=f"{match[1]}:{start + 1}",
                    file=file,
                    title=match[2],
                    start_line=start + 1,
                    end_line=end,
                    excerpt="\n".join(lines[start:end]),
                )
            )
            start = end
    return chunks


def handbook() -> list[Citation]:
    root = files("dicerp").joinpath("handbook")
    chunks = [
        chunk
        for entry in sorted(root.iterdir(), key=lambda item: item.name)
        if entry.name.endswith(".txt")
        for chunk in chunk_document(entry.name, entry.read_text(encoding="utf-8"))
    ]
    if not chunks or len({chunk.id for chunk in chunks}) != len(chunks):
        raise ValueError("Handbook requires uniquely identified rule chunks.")
    return chunks


class IndexFile(StrictModel):
    fingerprint: str
    provider: ProviderName = "openai"
    vectors: list[list[float]]


def fingerprint(chunks: Sequence[Citation], model: str) -> str:
    payload = [model, [chunk.model_dump() for chunk in chunks]]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def normalized(vector: list[float]) -> list[float]:
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Embedding must be finite and nonzero.")
    return [component / norm for component in vector]


class RuleIndex:
    def __init__(
        self, chunks: Sequence[Citation], vectors: list[list[float]], embedder: Embedder
    ) -> None:
        if not chunks or len(chunks) != len(vectors) or len({len(v) for v in vectors}) != 1:
            raise ValueError("Embedding index has invalid shape.")
        self._chunks = tuple(chunks)
        self._vectors = [normalized(vector) for vector in vectors]
        self._embedder = embedder

    def search(self, query: str) -> list[Citation]:
        try:
            (vector,) = self._embedder.embed([query])
            unit = normalized(vector)
            if len(unit) != len(self._vectors[0]):
                raise ValueError("Embedding dimensions changed.")
        except ValueError as exc:
            raise ProviderUnavailable("Query embedding was invalid.") from exc
        ranked = sorted(
            zip(self._vectors, self._chunks, strict=True),
            key=lambda item: (-sum(a * b for a, b in zip(unit, item[0], strict=True)), item[1].id),
        )
        return [chunk for _, chunk in ranked[:3]]

    @classmethod
    def load(
        cls, path: Path, model: str, embedder: Embedder, *, provider: ProviderName = "openai"
    ) -> "RuleIndex":
        chunks = handbook()
        index = IndexFile.model_validate_json(path.read_bytes())
        if index.provider != provider or index.fingerprint != fingerprint(chunks, model):
            raise ValueError(
                "Handbook, embedding provider or model changed. Run dicerp index again."
            )
        return cls(chunks, index.vectors, embedder)


def build_index(
    path: Path, model: str, embedder: Embedder, *, provider: ProviderName = "openai"
) -> None:
    chunks = handbook()
    vectors = embedder.embed([f"{chunk.title}\n{chunk.excerpt}" for chunk in chunks])
    RuleIndex(chunks, vectors, embedder)
    payload = IndexFile(fingerprint=fingerprint(chunks, model), vectors=vectors, provider=provider)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as temp:
        temporary = Path(temp.name)
    try:
        temporary.write_text(payload.model_dump_json(), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)

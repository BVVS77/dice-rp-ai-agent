from collections.abc import Sequence
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.exc import OperationalError

from dicerp.db import SnapshotRepository, seed_database
from dicerp.errors import ProviderUnavailable
from dicerp.retrieval import RuleIndex, build_index, chunk_document, handbook


class TestVectors:
    __test__ = False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0, 1.0] if "Metagaming" in text else [1.0, 0.0] for text in texts]


def test_seed_and_read_only(tmp_path: Path) -> None:
    path = tmp_path / "test # %.sqlite3"
    seed_database(path)
    with pytest.raises(FileExistsError):
        seed_database(path)
    repo = SnapshotRepository.open(path)
    try:
        player = repo.player("demo-001")
        assert player is not None and player.playtime_hours == 42
        assert repo.player("demo-999") is None
        assert repo.player("' OR 1=1 --") is None
        snapshot = repo.status()
        assert snapshot is not None and snapshot.online_players == 17
        with repo.engine.begin() as connection:
            # Even a client that disables query_only cannot bypass SQLite mode=ro.
            connection.execute(text("PRAGMA query_only = OFF"))
            with pytest.raises(OperationalError, match="readonly"):
                connection.execute(text("DELETE FROM players"))
    finally:
        repo.engine.dispose()


def test_chunking_preserves_line_spans() -> None:
    lines = ["# Test", "", "## R-11 | Demo", "a" * 500, "", "b" * 500]
    chunks = chunk_document("test.txt", "\n".join(lines))
    assert [(chunk.start_line, chunk.end_line) for chunk in chunks] == [(4, 4), (6, 6)]
    for chunk in chunks:
        assert chunk.excerpt == "\n".join(lines[chunk.start_line - 1 : chunk.end_line])
    with pytest.raises(ValueError, match="exceeds"):
        chunk_document("test.txt", "## R-11 | Demo\n" + "a" * 901)


def test_index_ranks_and_validates_model(tmp_path: Path) -> None:
    path = tmp_path / "index.json"
    embedder = TestVectors()
    build_index(path, "unit-test-model", embedder)
    index = RuleIndex.load(path, "unit-test-model", embedder)
    assert [chunk.id.split(":")[0] for chunk in index.search("Metagaming")] == [
        "R-03",
        "R-01",
        "R-02",
    ]
    with pytest.raises(ValueError, match="changed"):
        RuleIndex.load(path, "another-model", embedder)


def test_invalid_vectors_fail_closed() -> None:
    chunk = handbook()[0]
    with pytest.raises(ValueError, match="finite"):
        RuleIndex([chunk], [[float("nan")]], TestVectors())
    index = RuleIndex([chunk], [[1.0]], TestVectors())
    with pytest.raises(ProviderUnavailable):
        index.search("question")

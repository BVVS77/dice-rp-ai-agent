import sqlite3
from pathlib import Path

from sqlalchemy import Column, Integer, MetaData, String, Table, create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.pool import NullPool

from dicerp.models import PlayerStats, ServerStatus

metadata = MetaData()
players = Table(
    "players",
    metadata,
    Column("player_id", String, primary_key=True),
    Column("character_name", String, nullable=False),
    Column("role", String, nullable=False),
    Column("playtime_hours", Integer, nullable=False),
    Column("completed_sessions", Integer, nullable=False),
)
server = Table(
    "server_snapshot",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("server_name", String, nullable=False),
    Column("state", String, nullable=False),
    Column("online_players", Integer, nullable=False),
    Column("capacity", Integer, nullable=False),
    Column("snapshot_utc", String, nullable=False),
)


def seed_database(path: Path) -> None:
    """Create a new fixture database, refusing to overwrite an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb"):
        pass
    engine = create_engine("sqlite+pysqlite://", creator=lambda: sqlite3.connect(path))
    try:
        with engine.begin() as connection:
            metadata.create_all(connection)
            connection.execute(
                players.insert(),
                [
                    {
                        "player_id": "demo-001",
                        "character_name": "Morgan Vale",
                        "role": "Mechanic",
                        "playtime_hours": 42,
                        "completed_sessions": 12,
                    },
                    {
                        "player_id": "demo-002",
                        "character_name": "Riley Quinn",
                        "role": "Paramedic",
                        "playtime_hours": 67,
                        "completed_sessions": 21,
                    },
                    {
                        "player_id": "demo-003",
                        "character_name": "Avery Reed",
                        "role": "Taxi Driver",
                        "playtime_hours": 18,
                        "completed_sessions": 6,
                    },
                ],
            )
            connection.execute(
                server.insert().values(
                    id=1,
                    server_name="DiceRP Sandbox",
                    state="online",
                    online_players=17,
                    capacity=64,
                    snapshot_utc="2026-01-15T18:00:00Z",
                )
            )
    except Exception:
        engine.dispose()
        path.unlink()
        raise
    finally:
        engine.dispose()


class SnapshotRepository:
    def __init__(self, engine: Engine) -> None:
        self.engine = engine

    @classmethod
    def open(cls, path: Path) -> "SnapshotRepository":
        uri = path.resolve().as_uri() + "?mode=ro"

        def connect() -> sqlite3.Connection:
            connection = sqlite3.connect(uri, uri=True, check_same_thread=False)
            connection.execute("PRAGMA query_only = ON")
            return connection

        return cls(create_engine("sqlite+pysqlite://", creator=connect, poolclass=NullPool))

    def player(self, player_id: str) -> PlayerStats | None:
        with self.engine.connect() as connection:
            row = (
                connection.execute(select(players).where(players.c.player_id == player_id))
                .mappings()
                .one_or_none()
            )
        return PlayerStats.model_validate(dict(row)) if row is not None else None

    def status(self) -> ServerStatus | None:
        columns = [column for column in server.c if column.name != "id"]
        with self.engine.connect() as connection:
            row = (
                connection.execute(select(*columns).where(server.c.id == 1))
                .mappings()
                .one_or_none()
            )
        return ServerStatus.model_validate(dict(row)) if row is not None else None

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

Question = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
PlayerID = Annotated[str, StringConstraints(pattern=r"^demo-[0-9]{3}$")]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class Citation(StrictModel):
    id: str
    file: str
    start_line: int
    end_line: int
    title: str
    excerpt: str


class PlayerStats(StrictModel):
    player_id: str
    character_name: str
    role: str
    playtime_hours: int
    completed_sessions: int


class ServerStatus(StrictModel):
    server_name: str
    state: Literal["online", "offline", "maintenance"]
    online_players: int
    capacity: int
    snapshot_utc: str


class ToolTrace(StrictModel):
    name: str
    outcome: Literal["ok", "no_match", "invalid_call"]


class ChatRequest(StrictModel):
    question: Question


class ChatResponse(StrictModel):
    status: Literal["answered", "insufficient_evidence"]
    answer: str
    sources: list[Citation]
    tool_calls: list[ToolTrace]

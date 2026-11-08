import re
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import ValidationError

from dicerp.models import Citation, PlayerID, PlayerStats, Question, ServerStatus, StrictModel

type Evidence = Citation | PlayerStats | ServerStatus

PLAYER_REFERENCE = re.compile(r"(?<![\w-])demo-[0-9]{3}(?![\w-])")


class Rules(Protocol):
    def search(self, query: str) -> list[Citation]: ...


class Snapshot(Protocol):
    def player(self, player_id: str) -> PlayerStats | None: ...
    def status(self) -> ServerStatus | None: ...


class SearchArgs(StrictModel):
    query: Question


class PlayerArgs(StrictModel):
    player_id: PlayerID


class StatusArgs(StrictModel):
    pass


@dataclass(frozen=True)
class ToolSpec:
    description: str
    schema: type[StrictModel]


TOOL_SPECS = {
    "search_rules": ToolSpec("Retrieve verbatim excerpts from the fictional handbook.", SearchArgs),
    "get_player_stats": ToolSpec(
        "Look up one synthetic player by their exact demo-NNN ID. Never guess an ID.", PlayerArgs
    ),
    "get_server_status": ToolSpec("Read the timestamped synthetic server snapshot.", StatusArgs),
}


@dataclass(frozen=True)
class ToolResult:
    outcome: Literal["ok", "no_match", "invalid_call"]
    evidence: tuple[Evidence, ...] = ()


class Tools:
    def __init__(self, rules: Rules, snapshot: Snapshot) -> None:
        self.rules = rules
        self.snapshot = snapshot

    def run(self, name: str, arguments: str, question: str) -> ToolResult:
        if name not in TOOL_SPECS or len(arguments) > 4096:
            return ToolResult("invalid_call")
        try:
            args = TOOL_SPECS[name].schema.model_validate_json(arguments)
        except ValidationError:
            return ToolResult("invalid_call")

        if isinstance(args, SearchArgs):
            hits = self.rules.search(args.query)
            return ToolResult("ok", tuple(hits)) if hits else ToolResult("no_match")
        if isinstance(args, PlayerArgs) and args.player_id not in PLAYER_REFERENCE.findall(
            question.lower()
        ):
            return ToolResult("invalid_call")
        record = (
            self.snapshot.player(args.player_id)
            if isinstance(args, PlayerArgs)
            else (self.snapshot.status())
        )
        return ToolResult("ok", (record,)) if record is not None else ToolResult("no_match")

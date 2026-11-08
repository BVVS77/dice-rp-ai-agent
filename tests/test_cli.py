import os
import subprocess
import sys
from pathlib import Path


def test_seed_command_isolated_and_non_overwriting(tmp_path: Path) -> None:
    env = {**os.environ, "DICERP_STATE_DIR": str(tmp_path / "state"), "OPENAI_API_KEY": ""}
    command = [sys.executable, "-m", "dicerp.cli", "seed"]
    first = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert first.returncode == 0, first.stderr
    path = tmp_path / "state" / "snapshot.sqlite3"
    original = path.read_bytes()
    second = subprocess.run(command, cwd=tmp_path, env=env, capture_output=True, text=True)
    assert second.returncode == 1
    assert "already exists" in second.stderr
    assert path.read_bytes() == original


def test_index_without_key_fails_safely(tmp_path: Path) -> None:
    result = subprocess.run(
        [sys.executable, "-m", "dicerp.cli", "index"],
        cwd=tmp_path,
        env={
            **os.environ,
            "DICERP_AI_PROVIDER": "openai",
            "DICERP_STATE_DIR": str(tmp_path / "state"),
            "OPENAI_API_KEY": "",
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "API key" in result.stderr
    assert not (tmp_path / "state").exists()

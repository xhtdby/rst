from pathlib import Path
import subprocess, sys, os

DATA = Path(__file__).resolve().parents[1] / "data" / "edges.sample.csv"


def run_cmd(*args: str) -> str:
    cmd = [sys.executable, "-m", "rst_trap_finder.cli", *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.check_output(cmd, text=True, env=env)


def test_rank_command():
    out = run_cmd("rank", "--csv", str(DATA), "--top", "3")
    assert "word" in out


def test_next_command():
    out = run_cmd("next", "--word", "color", "--csv", str(DATA))
    assert "Best" in out

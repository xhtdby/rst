"""Tests for the CLI functionality."""
from pathlib import Path
import subprocess
import sys
import os

DATA = Path(__file__).resolve().parents[1] / "data" / "edges.sample.csv"


def run_cmd(*args: str) -> str:
    """Run CLI command and return output."""
    cmd = [sys.executable, "-m", "rst_trap_finder.cli", *args]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    return subprocess.check_output(cmd, text=True, env=env)


def test_analyze_command():
    """Test the analyze command."""
    out = run_cmd("analyze", str(DATA), "--top", "3")
    assert "Top 3 trap words" in out
    assert "Word" in out
    assert "Score" in out


def test_recommend_command():
    """Test the recommend command."""
    out = run_cmd("recommend", "color", str(DATA), "--top", "5")
    assert "Best next word from 'color'" in out
    assert "recommendations" in out


def test_word_command():
    """Test the word command."""
    out = run_cmd("word", "start", str(DATA))
    assert "Analysis for 'start'" in out
    assert "Composite Score" in out

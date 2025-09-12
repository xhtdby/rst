"""RST Trap Finder package."""
from __future__ import annotations

from typing import FrozenSet

TRAP_LETTERS: FrozenSet[str] = frozenset({"r", "s", "t"})

__all__ = ["TRAP_LETTERS"]

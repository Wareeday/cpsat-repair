from __future__ import annotations

from .base import Family
from .binpacking import BinPacking
from .coloring import GraphColoring
from .jobshop import JobShop
from .knapsack import KnapsackConflicts
from .latin import LatinCompletion
from .nqueens import NQueensBlocked
from .shifts import ShiftScheduling
from .tsp import TSP

_ALL = [NQueensBlocked, LatinCompletion, GraphColoring, KnapsackConflicts, BinPacking, TSP, JobShop, ShiftScheduling]
REGISTRY: dict[str, Family] = {cls.name: cls() for cls in _ALL}


def get_family(name: str) -> Family:
    if name not in REGISTRY:
        raise KeyError(f"unknown family '{name}'. Available: {', '.join(REGISTRY)}")
    return REGISTRY[name]


def all_families() -> list[Family]:
    return list(REGISTRY.values())

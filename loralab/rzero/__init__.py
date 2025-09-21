"""R-Zero: Self-Evolving Reasoning LLM from Zero Data

Implementation of the R-Zero framework for training reasoning models
through Challenger-Solver co-evolution.
"""

from .challenger import ChallengerAgent
from .solver import SolverAgent
from .evolution_engine import RZeroEvolutionEngine

__all__ = [
    "ChallengerAgent",
    "SolverAgent",
    "RZeroEvolutionEngine",
]

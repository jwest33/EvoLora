"""Configuration for R-Zero evolution system."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Optional
from pathlib import Path
from .task_goals import TaskGoal, get_task_goal

@dataclass
class EvolutionConfig:
    """Main configuration for evolution system."""

    # Task goal configuration
    task_goal: Optional[TaskGoal] = None
    task_goal_name: Optional[str] = None  # Name of predefined goal or "custom"

    # Server endpoints
    llm_endpoint: str = "http://localhost:8000"
    embedding_endpoint: str = "http://localhost:8002"

    # Evolution parameters
    generations: int = 100
    population_size: int = 20  # Number of instruction variants
    dataset_size_per_gen: int = 50  # Queries generated per generation

    # Genetic algorithm parameters
    mutation_rate: float = 0.2
    crossover_rate: float = 0.5
    elite_size: int = 4  # Top performers to keep

    # Difficulty curriculum
    initial_difficulty: float = 0.3
    difficulty_increment: float = 0.05
    max_difficulty: float = 0.95

    # Reward weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        "accuracy": 0.6,
        "difficulty": 0.3,
        "diversity": 0.1
    })

    # Challenger parameters
    challenger_temperature: float = 0.8
    challenger_max_tokens: int = 150
    ambiguity_target: float = 0.4  # Target ambiguity level

    # Solver parameters
    solver_temperature: float = 0.3
    solver_max_tokens: int = 100
    confidence_threshold: float = 0.7

    # Data generation
    seed_examples_per_node: int = 5
    augmentation_factor: int = 3

    # Evaluation
    test_split: float = 0.2
    min_test_size: int = 20
    success_threshold: float = 0.85

    # Checkpointing
    checkpoint_interval: int = 10
    checkpoint_dir: Path = field(default_factory=lambda: Path("checkpoints"))

    # Logging
    log_level: str = "INFO"
    verbose: bool = True

    def __post_init__(self):
        """Ensure checkpoint directory exists and load task goal if specified."""
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Load task goal if name is specified but goal object is not
        if self.task_goal_name and not self.task_goal:
            self.task_goal = get_task_goal(self.task_goal_name)
            if not self.task_goal:
                # Default to hierarchical routing if invalid name
                self.task_goal = get_task_goal("hierarchical_routing")

        # Default task goal if none specified
        if not self.task_goal:
            self.task_goal = get_task_goal("hierarchical_routing")

@dataclass
class InstructionGene:
    """Represents a single instruction variant."""
    id: str
    content: str
    fitness: float = 0.0
    generation: int = 0
    parent_ids: list[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)

@dataclass
class ChallengerState:
    """State of the Challenger agent."""
    current_difficulty: float = 0.3  # Default initial difficulty
    generated_count: int = 0
    success_rate: float = 0.0
    recent_queries: list[str] = field(default_factory=list)

@dataclass
class SolverState:
    """State of the Solver agent."""
    current_accuracy: float = 0.0
    attempts: int = 0
    successes: int = 0
    failure_patterns: Dict[str, int] = field(default_factory=dict)

@dataclass
class EvolutionState:
    """Complete state of the evolution process."""
    generation: int = 0
    best_fitness: float = 0.0
    best_instruction: Optional[InstructionGene] = None
    population: list[InstructionGene] = field(default_factory=list)
    challenger: ChallengerState = field(default_factory=ChallengerState)
    solver: SolverState = field(default_factory=SolverState)
    history: list[Dict] = field(default_factory=list)

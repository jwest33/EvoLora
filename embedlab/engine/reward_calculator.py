"""Reward calculator for evolution agents."""
from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np

from ..config.evolution_config import EvolutionConfig

class RewardCalculator:
    """
    Calculates rewards for Challenger and Solver agents.
    Implements reward shaping to guide co-evolution.
    """

    def __init__(self, config: EvolutionConfig):
        self.config = config

    def calculate_challenger_reward(
        self,
        solver_accuracy: float,
        query_difficulty: float,
        query_diversity: float
    ) -> float:
        """
        Calculate Challenger's reward.

        Challenger is rewarded for:
        - Generating queries at appropriate difficulty (not too easy, not too hard)
        - Maintaining diversity in generated queries
        - Pushing Solver to optimal learning zone

        Args:
            solver_accuracy: How well Solver performed on generated queries
            query_difficulty: Average difficulty of generated queries
            query_diversity: Diversity score of generated queries

        Returns:
            Reward value between 0 and 1
        """
        # Optimal solver accuracy is around 60-70%
        # Too easy (>80%) or too hard (<40%) gets lower reward
        optimal_accuracy = 0.65
        accuracy_distance = abs(solver_accuracy - optimal_accuracy)
        accuracy_reward = 1.0 - min(1.0, accuracy_distance * 2)

        # Reward for appropriate difficulty progression
        difficulty_reward = min(1.0, query_difficulty)

        # Reward for diversity
        diversity_reward = query_diversity

        # Weighted combination
        weights = self.config.reward_weights
        total_reward = (
            weights["accuracy"] * accuracy_reward +
            weights["difficulty"] * difficulty_reward +
            weights["diversity"] * diversity_reward
        )

        return max(0.0, min(1.0, total_reward))

    def calculate_solver_reward(
        self,
        accuracy: float,
        confidence: float,
        improvement: float
    ) -> float:
        """
        Calculate Solver's reward.

        Solver is rewarded for:
        - High routing accuracy
        - High confidence on correct predictions
        - Continuous improvement

        Args:
            accuracy: Routing accuracy on test queries
            confidence: Average confidence of predictions
            improvement: Improvement from previous generation

        Returns:
            Reward value between 0 and 1
        """
        # Direct reward for accuracy
        accuracy_reward = accuracy

        # Bonus for confident correct predictions
        confidence_bonus = confidence * 0.2 if accuracy > 0.5 else 0

        # Reward for improvement
        improvement_bonus = max(0, improvement) * 0.1

        total_reward = accuracy_reward + confidence_bonus + improvement_bonus

        return max(0.0, min(1.0, total_reward))

    def calculate_instruction_fitness(
        self,
        accuracy: float,
        confidence: float,
        complexity: float,
        coverage: float
    ) -> float:
        """
        Calculate fitness score for an instruction variant.

        Args:
            accuracy: Routing accuracy with this instruction
            confidence: Average confidence when using this instruction
            complexity: Complexity/length penalty (lower is better)
            coverage: How well instruction covers all categories

        Returns:
            Fitness score
        """
        # Main fitness from accuracy
        base_fitness = accuracy

        # Bonus for high confidence
        confidence_bonus = confidence * 0.1

        # Penalty for overly complex instructions
        complexity_penalty = max(0, complexity - 0.5) * 0.1

        # Bonus for good coverage
        coverage_bonus = coverage * 0.05

        fitness = base_fitness + confidence_bonus - complexity_penalty + coverage_bonus

        return max(0.0, min(1.0, fitness))

    def calculate_query_quality(
        self,
        ambiguity: float,
        difficulty: float,
        relevance: float
    ) -> float:
        """
        Calculate quality score for a generated query.

        Args:
            ambiguity: How ambiguous the query is (moderate is good)
            difficulty: Difficulty level of the query
            relevance: How relevant/realistic the query is

        Returns:
            Quality score between 0 and 1
        """
        # Optimal ambiguity is moderate (around 0.4-0.6)
        if 0.4 <= ambiguity <= 0.6:
            ambiguity_score = 1.0
        else:
            distance = min(abs(ambiguity - 0.4), abs(ambiguity - 0.6))
            ambiguity_score = 1.0 - distance

        # Higher difficulty is generally good (with limits)
        difficulty_score = min(1.0, difficulty * 1.2)

        # Relevance is directly scored
        relevance_score = relevance

        # Weighted average
        quality = (ambiguity_score + difficulty_score + relevance_score) / 3

        return max(0.0, min(1.0, quality))

    def calculate_generation_progress(
        self,
        current_fitness: float,
        previous_fitness: List[float],
        target_fitness: float = 0.85
    ) -> Dict[str, float]:
        """
        Calculate progress metrics for current generation.

        Args:
            current_fitness: Current best fitness
            previous_fitness: List of previous generation fitness values
            target_fitness: Target fitness to achieve

        Returns:
            Dictionary of progress metrics
        """
        progress = {
            "current_fitness": current_fitness,
            "target_fitness": target_fitness,
            "progress_to_target": current_fitness / target_fitness
        }

        if previous_fitness:
            # Calculate trend
            recent = previous_fitness[-5:] if len(previous_fitness) >= 5 else previous_fitness
            progress["recent_improvement"] = current_fitness - np.mean(recent)
            progress["overall_improvement"] = current_fitness - previous_fitness[0]

            # Calculate stability
            if len(recent) > 1:
                progress["stability"] = 1.0 - np.std(recent)
            else:
                progress["stability"] = 0.5

            # Estimate generations to target
            if len(previous_fitness) >= 2:
                recent_rate = (current_fitness - previous_fitness[-1])
                if recent_rate > 0:
                    progress["estimated_gens_to_target"] = int(
                        (target_fitness - current_fitness) / recent_rate
                    )
                else:
                    progress["estimated_gens_to_target"] = -1  # No progress
            else:
                progress["estimated_gens_to_target"] = -1

        return progress

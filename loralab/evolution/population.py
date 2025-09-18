"""Population manager for evolutionary LoRA optimization

Manages the lifecycle of LoRA variant populations during evolution.
"""

import logging
from typing import List, Optional
import numpy as np

from ..core.lora_factory import LoRAVariant

logger = logging.getLogger(__name__)


class PopulationManager:
    """Manages populations of LoRA variants"""

    def __init__(self):
        """Initialize population manager"""
        self.generation_count = 0
        self.total_variants_created = 0
        self.best_fitness_history = []

    def select_survivors(self,
                        population: List[LoRAVariant],
                        keep_top: int) -> List[LoRAVariant]:
        """Select top performing variants as survivors

        Args:
            population: Current population
            keep_top: Number of variants to keep

        Returns:
            List of surviving variants
        """
        # Sort by fitness score (higher is better)
        sorted_pop = sorted(population,
                           key=lambda v: v.fitness_score(),
                           reverse=True)

        survivors = sorted_pop[:keep_top]

        # Track best fitness
        if sorted_pop:
            best_fitness = sorted_pop[0].fitness_score()
            self.best_fitness_history.append(best_fitness)

        logger.info(f"Selected {len(survivors)} survivors from population of {len(population)}")

        return survivors

    def tournament_selection(self,
                            population: List[LoRAVariant],
                            num_select: int,
                            tournament_size: int = 3) -> List[LoRAVariant]:
        """Select variants using tournament selection

        Args:
            population: Current population
            num_select: Number of variants to select
            tournament_size: Size of each tournament

        Returns:
            Selected variants
        """
        selected = []

        for _ in range(num_select):
            # Random tournament participants
            tournament = np.random.choice(population, size=tournament_size, replace=False)

            # Winner is the one with best fitness
            winner = max(tournament, key=lambda v: v.fitness_score())
            selected.append(winner)

        return selected

    def diversity_selection(self,
                           population: List[LoRAVariant],
                           keep_top: int) -> List[LoRAVariant]:
        """Select diverse variants to maintain population diversity

        Args:
            population: Current population
            keep_top: Number to select

        Returns:
            Diverse selection of variants
        """
        if len(population) <= keep_top:
            return population

        selected = []
        remaining = population.copy()

        # Always keep the best
        best = max(remaining, key=lambda v: v.fitness_score())
        selected.append(best)
        remaining.remove(best)

        # Select diverse variants based on configuration distance
        while len(selected) < keep_top and remaining:
            # Find variant most different from current selection
            max_distance = -1
            most_diverse = None

            for variant in remaining:
                # Calculate minimum distance to selected variants
                min_dist = min(self._config_distance(variant, s) for s in selected)

                if min_dist > max_distance:
                    max_distance = min_dist
                    most_diverse = variant

            if most_diverse:
                selected.append(most_diverse)
                remaining.remove(most_diverse)

        return selected

    def _config_distance(self, v1: LoRAVariant, v2: LoRAVariant) -> float:
        """Calculate configuration distance between two variants

        Args:
            v1: First variant
            v2: Second variant

        Returns:
            Distance measure
        """
        # Normalized differences
        rank_diff = abs(v1.rank - v2.rank) / 256.0
        lr_diff = abs(np.log10(v1.learning_rate) - np.log10(v2.learning_rate)) / 4.0
        dropout_diff = abs(v1.dropout - v2.dropout) / 0.2

        # Weighted sum
        distance = (rank_diff * 0.5 + lr_diff * 0.3 + dropout_diff * 0.2)

        return distance

    def get_population_stats(self, population: List[LoRAVariant]) -> dict:
        """Get statistics about the current population

        Args:
            population: Current population

        Returns:
            Dictionary of statistics
        """
        if not population:
            return {}

        accuracies = [v.eval_accuracy for v in population]
        perplexities = [v.eval_perplexity for v in population]
        ranks = [v.rank for v in population]
        lrs = [v.learning_rate for v in population]

        stats = {
            'size': len(population),
            'accuracy': {
                'best': max(accuracies),
                'worst': min(accuracies),
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            },
            'perplexity': {
                'best': min(perplexities),
                'worst': max(perplexities),
                'mean': np.mean(perplexities),
                'std': np.std(perplexities)
            },
            'rank_distribution': {
                'min': min(ranks),
                'max': max(ranks),
                'mean': np.mean(ranks),
                'unique': len(set(ranks))
            },
            'learning_rate_distribution': {
                'min': min(lrs),
                'max': max(lrs),
                'geometric_mean': np.exp(np.mean(np.log(lrs)))
            }
        }

        return stats

    def prune_population(self,
                        population: List[LoRAVariant],
                        target_size: int,
                        strategy: str = 'fitness') -> List[LoRAVariant]:
        """Prune population to target size

        Args:
            population: Current population
            target_size: Target population size
            strategy: Pruning strategy ('fitness', 'diversity', 'hybrid')

        Returns:
            Pruned population
        """
        if len(population) <= target_size:
            return population

        if strategy == 'fitness':
            # Keep top performers
            return self.select_survivors(population, target_size)
        elif strategy == 'diversity':
            # Keep diverse variants
            return self.diversity_selection(population, target_size)
        elif strategy == 'hybrid':
            # Mix of top performers and diverse variants
            num_elite = target_size // 2
            elite = self.select_survivors(population, num_elite)

            # Remove elite from population for diversity selection
            remaining = [v for v in population if v not in elite]
            diverse = self.diversity_selection(remaining, target_size - num_elite)

            return elite + diverse
        else:
            raise ValueError(f"Unknown pruning strategy: {strategy}")

    def get_convergence_metrics(self) -> dict:
        """Get metrics about population convergence

        Returns:
            Dictionary of convergence metrics
        """
        if len(self.best_fitness_history) < 2:
            return {'converged': False, 'generations': len(self.best_fitness_history)}

        # Check if fitness has plateaued
        recent_history = self.best_fitness_history[-5:]
        if len(recent_history) >= 3:
            # Calculate improvement over last few generations
            improvement = recent_history[-1] - recent_history[0]
            relative_improvement = improvement / (recent_history[0] + 1e-10)

            converged = relative_improvement < 0.01  # Less than 1% improvement

            return {
                'converged': converged,
                'generations': len(self.best_fitness_history),
                'best_fitness': self.best_fitness_history[-1],
                'improvement_rate': relative_improvement,
                'fitness_history': self.best_fitness_history
            }

        return {'converged': False, 'generations': len(self.best_fitness_history)}

"""LoRA factory for creating and mutating adapter configurations

Handles the creation of LoRA variants with different hyperparameters
for evolutionary optimization.
"""

import random
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json

logger = logging.getLogger(__name__)


@dataclass
class LoRAVariant:
    """Represents a LoRA adapter configuration variant"""

    # Core configuration
    rank: int
    alpha: int
    dropout: float
    learning_rate: float
    target_modules: List[str]

    # Metadata
    variant_id: str = ""
    generation: int = 0
    parent_id: Optional[str] = None

    # Performance metrics
    train_loss: float = float('inf')
    eval_accuracy: float = 0.0
    eval_perplexity: float = float('inf')
    training_time: float = 0.0

    # Model reference (not serialized)
    model: Optional[Any] = field(default=None, repr=False)

    def __post_init__(self):
        """Generate variant ID if not provided"""
        if not self.variant_id:
            # Convert learning_rate to float if it's a string
            lr = float(self.learning_rate) if isinstance(self.learning_rate, str) else self.learning_rate
            self.variant_id = f"gen{self.generation}_r{self.rank}_lr{lr:.0e}"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'variant_id': self.variant_id,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'rank': self.rank,
            'alpha': self.alpha,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'target_modules': self.target_modules,
            'train_loss': self.train_loss,
            'eval_accuracy': self.eval_accuracy,
            'eval_perplexity': self.eval_perplexity,
            'training_time': self.training_time
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LoRAVariant':
        """Create from dictionary"""
        return cls(
            variant_id=data.get('variant_id', ''),
            generation=data.get('generation', 0),
            parent_id=data.get('parent_id'),
            rank=data['rank'],
            alpha=data['alpha'],
            dropout=data['dropout'],
            learning_rate=data['learning_rate'],
            target_modules=data['target_modules'],
            train_loss=data.get('train_loss', float('inf')),
            eval_accuracy=data.get('eval_accuracy', 0.0),
            eval_perplexity=data.get('eval_perplexity', float('inf')),
            training_time=data.get('training_time', 0.0)
        )

    def fitness_score(self) -> float:
        """Calculate overall fitness score for evolution"""
        # Combine accuracy (higher is better) and perplexity (lower is better)
        # Normalize perplexity to 0-1 range (inverse)
        perplexity_score = 1.0 / (1.0 + self.eval_perplexity)

        # Weight accuracy more heavily
        score = (self.eval_accuracy * 0.7) + (perplexity_score * 0.3)

        # Slight penalty for very large ranks (efficiency consideration)
        if self.rank > 128:
            score *= 0.95

        return score


class LoRAFactory:
    """Factory for creating and mutating LoRA configurations"""

    def __init__(self, search_space: Dict[str, Any]):
        """Initialize with hyperparameter search space

        Args:
            search_space: Dictionary defining the search space:
                - rank: List of possible ranks
                - alpha_multiplier: List of multipliers for alpha
                - dropout: List of dropout rates
                - learning_rate: List of learning rates
                - target_modules: List of target module names
        """
        self.search_space = search_space
        self.generation_counter = 0

    def create_random_variant(self) -> LoRAVariant:
        """Create a random LoRA variant from the search space"""
        rank = random.choice(self.search_space['rank'])
        alpha_mult = random.choice(self.search_space['alpha_multiplier'])

        # Ensure learning rate is a float
        lr = random.choice(self.search_space['learning_rate'])
        lr = float(lr) if isinstance(lr, str) else lr

        variant = LoRAVariant(
            rank=rank,
            alpha=rank * alpha_mult,
            dropout=random.choice(self.search_space['dropout']),
            learning_rate=lr,
            target_modules=self.search_space['target_modules'],
            generation=self.generation_counter
        )

        logger.info(f"Created random variant: {variant.variant_id}")
        return variant

    def mutate_variant(self, parent: LoRAVariant, mutation_rate: float = 0.3) -> LoRAVariant:
        """Create a mutated version of a parent variant

        Args:
            parent: Parent variant to mutate
            mutation_rate: Probability of mutating each parameter

        Returns:
            New mutated variant
        """
        # Start with parent's configuration
        new_rank = parent.rank
        new_alpha_mult = parent.alpha // parent.rank if parent.rank > 0 else 2
        new_dropout = parent.dropout
        new_lr = parent.learning_rate

        # Mutate rank
        if random.random() < mutation_rate:
            # Either slightly adjust or pick new value
            if random.random() < 0.5:
                # Slight adjustment
                rank_options = self.search_space['rank']
                current_idx = rank_options.index(parent.rank) if parent.rank in rank_options else 0
                new_idx = max(0, min(len(rank_options) - 1,
                                    current_idx + random.choice([-1, 1])))
                new_rank = rank_options[new_idx]
            else:
                # Random new value
                new_rank = random.choice(self.search_space['rank'])

        # Mutate alpha multiplier
        if random.random() < mutation_rate:
            new_alpha_mult = random.choice(self.search_space['alpha_multiplier'])

        # Mutate dropout
        if random.random() < mutation_rate:
            new_dropout = random.choice(self.search_space['dropout'])

        # Mutate learning rate
        if random.random() < mutation_rate:
            if random.random() < 0.5:
                # Scale existing LR
                parent_lr = float(parent.learning_rate) if isinstance(parent.learning_rate, str) else parent.learning_rate
                new_lr = parent_lr * random.choice([0.5, 0.8, 1.25, 2.0])
                # Clamp to reasonable range
                new_lr = max(1e-6, min(1e-2, new_lr))
            else:
                # Pick new value
                new_lr = random.choice(self.search_space['learning_rate'])
                new_lr = float(new_lr) if isinstance(new_lr, str) else new_lr

        variant = LoRAVariant(
            rank=new_rank,
            alpha=new_rank * new_alpha_mult,
            dropout=new_dropout,
            learning_rate=new_lr,
            target_modules=parent.target_modules,
            generation=self.generation_counter,
            parent_id=parent.variant_id
        )

        logger.info(f"Created mutated variant: {variant.variant_id} (parent: {parent.variant_id})")
        return variant

    def crossover_variants(self, parent1: LoRAVariant, parent2: LoRAVariant) -> LoRAVariant:
        """Create offspring by combining two parent variants

        Args:
            parent1: First parent variant
            parent2: Second parent variant

        Returns:
            New variant combining traits from both parents
        """
        # Randomly choose parameters from each parent
        rank = random.choice([parent1.rank, parent2.rank])

        # For alpha, maintain the ratio from the chosen rank's parent
        if rank == parent1.rank:
            alpha = parent1.alpha
        else:
            alpha = parent2.alpha

        dropout = random.choice([parent1.dropout, parent2.dropout])
        lr1 = float(parent1.learning_rate) if isinstance(parent1.learning_rate, str) else parent1.learning_rate
        lr2 = float(parent2.learning_rate) if isinstance(parent2.learning_rate, str) else parent2.learning_rate
        learning_rate = random.choice([lr1, lr2])

        variant = LoRAVariant(
            rank=rank,
            alpha=alpha,
            dropout=dropout,
            learning_rate=learning_rate,
            target_modules=parent1.target_modules,
            generation=self.generation_counter,
            parent_id=f"{parent1.variant_id}+{parent2.variant_id}"
        )

        logger.info(f"Created crossover variant: {variant.variant_id}")
        return variant

    def create_population(self, size: int) -> List[LoRAVariant]:
        """Create an initial random population

        Args:
            size: Population size

        Returns:
            List of LoRA variants
        """
        population = []
        for _ in range(size):
            population.append(self.create_random_variant())
        return population

    def evolve_population(self,
                         survivors: List[LoRAVariant],
                         population_size: int,
                         mutation_rate: float = 0.3,
                         crossover_rate: float = 0.2) -> List[LoRAVariant]:
        """Create next generation from survivors

        Args:
            survivors: Top performing variants from previous generation
            population_size: Target population size
            mutation_rate: Probability of mutation
            crossover_rate: Proportion of population from crossover

        Returns:
            New population for next generation
        """
        self.generation_counter += 1
        new_population = []

        # Keep the survivors (elitism)
        new_population.extend(survivors)

        # Calculate how many new variants we need
        needed = population_size - len(survivors)
        num_crossover = int(needed * crossover_rate)
        num_mutation = needed - num_crossover

        # Create crossover variants if we have at least 2 survivors
        if len(survivors) >= 2:
            for _ in range(num_crossover):
                parents = random.sample(survivors, 2)
                child = self.crossover_variants(parents[0], parents[1])
                new_population.append(child)
        else:
            # If not enough for crossover, do more mutations
            num_mutation = needed

        # Create mutated variants
        for _ in range(num_mutation):
            parent = random.choice(survivors)
            child = self.mutate_variant(parent, mutation_rate)
            new_population.append(child)

        # If still need more (shouldn't happen), add random
        while len(new_population) < population_size:
            new_population.append(self.create_random_variant())

        logger.info(f"Evolved to generation {self.generation_counter}: "
                   f"{len(survivors)} survivors, {num_mutation} mutations, "
                   f"{num_crossover} crossovers")

        return new_population[:population_size]

    def save_variant(self, variant: LoRAVariant, filepath: str):
        """Save variant configuration to file"""
        with open(filepath, 'w') as f:
            json.dump(variant.to_dict(), f, indent=2)
        logger.info(f"Saved variant to {filepath}")

    def load_variant(self, filepath: str) -> LoRAVariant:
        """Load variant configuration from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        variant = LoRAVariant.from_dict(data)
        logger.info(f"Loaded variant from {filepath}")
        return variant
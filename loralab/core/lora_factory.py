"""LoRA factory for creating and mutating adapter configurations

Handles the creation of LoRA variants with different hyperparameters
for evolutionary optimization.
"""

import random
import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
import json
import hashlib

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

    # Additional training configuration
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # LoRA-specific configuration
    use_rslora: bool = False
    target_modules_preset: str = "standard"  # "minimal", "standard", or "extended"

    # Metadata
    variant_id: str = ""
    generation: int = 0
    parent_id: Optional[str] = None

    # Performance metrics
    train_loss: float = float('inf')
    eval_accuracy: float = 0.0
    eval_perplexity: float = float('inf')
    training_time: float = 0.0

    # Additional metrics for GRPO
    rewards: float = 0.0

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
            'weight_decay': self.weight_decay,
            'warmup_ratio': self.warmup_ratio,
            'max_grad_norm': self.max_grad_norm,
            'use_rslora': self.use_rslora,
            'target_modules_preset': self.target_modules_preset,
            'train_loss': self.train_loss,
            'eval_accuracy': self.eval_accuracy,
            'eval_perplexity': self.eval_perplexity,
            'training_time': self.training_time,
            'rewards': self.rewards
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
            weight_decay=data.get('weight_decay', 0.01),
            warmup_ratio=data.get('warmup_ratio', 0.1),
            max_grad_norm=data.get('max_grad_norm', 1.0),
            use_rslora=data.get('use_rslora', False),
            target_modules_preset=data.get('target_modules_preset', 'standard'),
            train_loss=data.get('train_loss', float('inf')),
            eval_accuracy=data.get('eval_accuracy', 0.0),
            eval_perplexity=data.get('eval_perplexity', float('inf')),
            training_time=data.get('training_time', 0.0),
            rewards=data.get('rewards', 0.0)
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

    def get_configuration_hash(self) -> str:
        """Generate a unique hash for this configuration

        Used for duplicate detection. Only includes parameters that affect
        the model training, not metadata or performance metrics.
        """
        config_str = f"{self.rank}_{self.alpha}_{self.dropout:.3f}_{self.learning_rate:.2e}"
        config_str += f"_{self.weight_decay:.3f}_{self.warmup_ratio:.2f}_{self.max_grad_norm:.1f}"
        config_str += f"_{self.use_rslora}_{self.target_modules_preset}"
        config_str += f"_{'_'.join(sorted(self.target_modules))}"

        return hashlib.md5(config_str.encode()).hexdigest()[:16]


class LoRAFactory:
    """Factory for creating and mutating LoRA configurations"""

    # Target module presets
    TARGET_MODULE_PRESETS = {
        'minimal': ['q_proj', 'v_proj'],
        'standard': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        'extended': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
    }

    def __init__(self, search_space: Dict[str, Any]):
        """Initialize with hyperparameter search space

        Args:
            search_space: Dictionary defining the search space:
                - rank: List of possible ranks
                - alpha_multiplier: List of multipliers for alpha
                - dropout: List of dropout rates
                - learning_rate: List of learning rates
                - target_modules: List of target module names
                - weight_decay: List of weight decay values
                - warmup_ratio: List of warmup ratios
                - max_grad_norm: List of gradient clipping values
                - use_rslora: List of boolean values
                - target_modules_preset: List of preset names
        """
        self.search_space = search_space
        self.generation_counter = 0
        self.seen_configurations: Set[str] = set()  # Track unique configurations

    def create_random_variant(self, max_attempts: int = 10) -> LoRAVariant:
        """Create a random LoRA variant from the search space

        Args:
            max_attempts: Maximum attempts to create a unique variant

        Returns:
            A unique LoRA variant
        """
        for attempt in range(max_attempts):
            rank = random.choice(self.search_space['rank'])
            alpha_mult = random.choice(self.search_space.get('alpha_multiplier', [2]))

            # Ensure learning rate is a float
            lr = random.choice(self.search_space['learning_rate'])
            lr = float(lr) if isinstance(lr, str) else lr

            # Get target modules preset and corresponding modules
            preset = random.choice(self.search_space.get('target_modules_preset', ['standard']))
            target_modules = self.TARGET_MODULE_PRESETS.get(preset,
                                                           self.search_space.get('target_modules',
                                                           self.TARGET_MODULE_PRESETS['standard']))

            variant = LoRAVariant(
                rank=rank,
                alpha=rank * alpha_mult,
                dropout=random.choice(self.search_space.get('dropout', [0.0])),
                learning_rate=lr,
                target_modules=target_modules,
                weight_decay=random.choice(self.search_space.get('weight_decay', [0.01])),
                warmup_ratio=random.choice(self.search_space.get('warmup_ratio', [0.1])),
                max_grad_norm=random.choice(self.search_space.get('max_grad_norm', [1.0])),
                use_rslora=random.choice(self.search_space.get('use_rslora', [False])),
                target_modules_preset=preset,
                generation=self.generation_counter
            )

            # Check for duplicate
            config_hash = variant.get_configuration_hash()
            if config_hash not in self.seen_configurations:
                self.seen_configurations.add(config_hash)
                logger.info(f"Created unique variant: {variant.variant_id} (hash: {config_hash[:8]})")
                return variant
            else:
                logger.debug(f"Duplicate configuration detected (attempt {attempt + 1}), retrying...")

        # If we couldn't create unique after max_attempts, force slight variation
        logger.warning(f"Could not create unique variant after {max_attempts} attempts, forcing variation")
        variant.learning_rate *= random.uniform(0.95, 1.05)  # Slight LR variation
        config_hash = variant.get_configuration_hash()
        self.seen_configurations.add(config_hash)
        return variant

    def mutate_variant(self, parent: LoRAVariant, mutation_rate: float = 0.3, max_attempts: int = 10) -> LoRAVariant:
        """Create a mutated version of a parent variant

        Args:
            parent: Parent variant to mutate
            mutation_rate: Probability of mutating each parameter
            max_attempts: Maximum attempts to create unique variant

        Returns:
            New mutated variant
        """
        for attempt in range(max_attempts):
            # Start with parent's configuration
            new_rank = parent.rank
            new_alpha_mult = parent.alpha // parent.rank if parent.rank > 0 else 2
            new_dropout = parent.dropout
            new_lr = parent.learning_rate
            new_weight_decay = parent.weight_decay
            new_warmup_ratio = parent.warmup_ratio
            new_max_grad_norm = parent.max_grad_norm
            new_use_rslora = parent.use_rslora
            new_preset = parent.target_modules_preset

            # Mutate rank
            if random.random() < mutation_rate:
                # Either slightly adjust or pick new value
                if random.random() < 0.5 and 'rank' in self.search_space:
                    # Slight adjustment
                    rank_options = self.search_space['rank']
                    current_idx = rank_options.index(parent.rank) if parent.rank in rank_options else 0
                    new_idx = max(0, min(len(rank_options) - 1,
                                        current_idx + random.choice([-1, 1])))
                    new_rank = rank_options[new_idx]
                else:
                    # Random new value
                    new_rank = random.choice(self.search_space.get('rank', [parent.rank]))

            # Mutate alpha multiplier
            if random.random() < mutation_rate:
                new_alpha_mult = random.choice(self.search_space.get('alpha_multiplier', [new_alpha_mult]))

            # Mutate dropout
            if random.random() < mutation_rate:
                new_dropout = random.choice(self.search_space.get('dropout', [0.0]))

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
                    new_lr = random.choice(self.search_space.get('learning_rate', [parent.learning_rate]))
                    new_lr = float(new_lr) if isinstance(new_lr, str) else new_lr

            # Mutate new properties
            if random.random() < mutation_rate:
                new_weight_decay = random.choice(self.search_space.get('weight_decay', [parent.weight_decay]))

            if random.random() < mutation_rate:
                new_warmup_ratio = random.choice(self.search_space.get('warmup_ratio', [parent.warmup_ratio]))

            if random.random() < mutation_rate:
                new_max_grad_norm = random.choice(self.search_space.get('max_grad_norm', [parent.max_grad_norm]))

            if random.random() < mutation_rate:
                new_use_rslora = random.choice(self.search_space.get('use_rslora', [parent.use_rslora]))

            if random.random() < mutation_rate:
                new_preset = random.choice(self.search_space.get('target_modules_preset', [parent.target_modules_preset]))

            # Get target modules from preset
            target_modules = self.TARGET_MODULE_PRESETS.get(new_preset, parent.target_modules)

            variant = LoRAVariant(
                rank=new_rank,
                alpha=new_rank * new_alpha_mult,
                dropout=new_dropout,
                learning_rate=new_lr,
                target_modules=target_modules,
                weight_decay=new_weight_decay,
                warmup_ratio=new_warmup_ratio,
                max_grad_norm=new_max_grad_norm,
                use_rslora=new_use_rslora,
                target_modules_preset=new_preset,
                generation=self.generation_counter,
                parent_id=parent.variant_id
            )

            # Check for duplicate
            config_hash = variant.get_configuration_hash()
            if config_hash not in self.seen_configurations:
                self.seen_configurations.add(config_hash)
                logger.info(f"Created unique mutated variant: {variant.variant_id} (parent: {parent.variant_id}, hash: {config_hash[:8]})")
                return variant
            else:
                logger.debug(f"Mutation created duplicate (attempt {attempt + 1}), retrying...")

        # If we couldn't create unique, force variation
        logger.warning(f"Could not create unique mutation after {max_attempts} attempts, forcing variation")
        variant.learning_rate *= random.uniform(0.9, 1.1)
        config_hash = variant.get_configuration_hash()
        self.seen_configurations.add(config_hash)
        return variant

    def crossover_variants(self, parent1: LoRAVariant, parent2: LoRAVariant, max_attempts: int = 10) -> LoRAVariant:
        """Create offspring by combining two parent variants

        Args:
            parent1: First parent variant
            parent2: Second parent variant
            max_attempts: Maximum attempts to create unique variant

        Returns:
            New variant combining traits from both parents
        """
        for attempt in range(max_attempts):
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

            # Crossover new properties
            weight_decay = random.choice([parent1.weight_decay, parent2.weight_decay])
            warmup_ratio = random.choice([parent1.warmup_ratio, parent2.warmup_ratio])
            max_grad_norm = random.choice([parent1.max_grad_norm, parent2.max_grad_norm])
            use_rslora = random.choice([parent1.use_rslora, parent2.use_rslora])
            target_modules_preset = random.choice([parent1.target_modules_preset, parent2.target_modules_preset])

            # Get target modules from chosen preset
            target_modules = self.TARGET_MODULE_PRESETS.get(target_modules_preset,
                                                           random.choice([parent1.target_modules, parent2.target_modules]))

            variant = LoRAVariant(
                rank=rank,
                alpha=alpha,
                dropout=dropout,
                learning_rate=learning_rate,
                target_modules=target_modules,
                weight_decay=weight_decay,
                warmup_ratio=warmup_ratio,
                max_grad_norm=max_grad_norm,
                use_rslora=use_rslora,
                target_modules_preset=target_modules_preset,
                generation=self.generation_counter,
                parent_id=f"{parent1.variant_id}+{parent2.variant_id}"
            )

            # Check for duplicate
            config_hash = variant.get_configuration_hash()
            if config_hash not in self.seen_configurations:
                self.seen_configurations.add(config_hash)
                logger.info(f"Created unique crossover variant: {variant.variant_id} (hash: {config_hash[:8]})")
                return variant
            else:
                logger.debug(f"Crossover created duplicate (attempt {attempt + 1}), retrying...")

        # Force variation if couldn't create unique
        logger.warning(f"Could not create unique crossover after {max_attempts} attempts, forcing variation")
        variant.learning_rate *= random.uniform(0.95, 1.05)
        config_hash = variant.get_configuration_hash()
        self.seen_configurations.add(config_hash)
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

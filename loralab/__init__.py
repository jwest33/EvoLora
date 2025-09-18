"""LoRALab Self-Supervised Evolution System

Evolutionary optimization for finding optimal LoRA adapters without teacher models.
"""

__version__ = "2.0.0"

from .core.model_manager import ModelManager
from .core.lora_factory import LoRAFactory, LoRAVariant
from .evolution.evolutionary_trainer import EvolutionaryTrainer
from .evolution.population import PopulationManager
from .evolution.fitness_evaluator import FitnessEvaluator
from .training.self_supervised import SelfSupervisedTrainer
from .datasets.dataset_loader import DatasetLoader
from .config.config_loader import ConfigLoader

__all__ = [
    'ModelManager',
    'LoRAFactory',
    'LoRAVariant',
    'EvolutionaryTrainer',
    'PopulationManager',
    'FitnessEvaluator',
    'SelfSupervisedTrainer',
    'DatasetLoader',
    'ConfigLoader'
]

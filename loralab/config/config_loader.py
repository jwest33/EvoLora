"""
Configuration loader and validator for LoRA evolution system.
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Loads and validates configuration files."""

    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML or JSON file.

        Args:
            config_path: Path to configuration file

        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        # Load based on extension
        if config_path.suffix in ['.yaml', '.yml']:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        elif config_path.suffix == '.json':
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")

        # Validate and fill defaults
        config = ConfigLoader._validate_config(config)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration and fill in defaults.

        Args:
            config: Raw configuration dictionary

        Returns:
            Validated configuration with defaults
        """
        # Challenger defaults
        if 'challenger' not in config:
            raise ValueError("Challenger configuration is required")

        challenger = config['challenger']
        challenger.setdefault('port', 8080)
        # Don't set defaults for optional parameters - let them be undefined
        # challenger.setdefault('context_size', 8192)
        # challenger.setdefault('gpu_layers', 20)

        # Solver defaults
        if 'solver' not in config:
            raise ValueError("Solver configuration is required")

        solver = config['solver']
        solver.setdefault('device', 'cuda')

        if 'lora_config' not in solver:
            solver['lora_config'] = {}

        lora = solver['lora_config']
        lora.setdefault('rank', 16)
        lora.setdefault('alpha', 32)
        lora.setdefault('dropout', 0.1)
        lora.setdefault('target_modules', ["q_proj", "v_proj", "k_proj", "o_proj"])

        # Task configuration
        if 'task' not in config:
            config['task'] = {}

        task = config['task']
        task.setdefault('type', 'code_documentation')
        task.setdefault('difficulty_progression', 'adaptive')

        if 'curriculum' not in task:
            task['curriculum'] = {}

        curriculum = task['curriculum']
        curriculum.setdefault('initial_difficulty', 0.3)
        curriculum.setdefault('target_success_rate', 0.7)

        # Training configuration
        if 'training' not in config:
            config['training'] = {}

        training = config['training']
        training.setdefault('learning_rate', 1e-5)
        training.setdefault('batch_size', 8)
        training.setdefault('num_rollouts', 4)
        training.setdefault('kl_penalty', 0.1)
        training.setdefault('clip_ratio', 0.2)
        training.setdefault('max_grad_norm', 1.0)

        # Evolution configuration
        if 'evolution' not in config:
            config['evolution'] = {}

        evolution = config['evolution']
        evolution.setdefault('generations', 50)
        evolution.setdefault('population_size', 10)
        evolution.setdefault('dataset_size_per_gen', 100)
        evolution.setdefault('eval_ratio', 0.2)

        # Output directory
        if 'output_dir' not in config:
            config['output_dir'] = 'experiments/default'

        return config

    @staticmethod
    def save_config(config: Dict[str, Any], save_path: str):
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            save_path: Path to save configuration
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        if save_path.suffix in ['.yaml', '.yml']:
            with open(save_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        elif save_path.suffix == '.json':
            with open(save_path, 'w') as f:
                json.dump(config, f, indent=2)
        else:
            raise ValueError(f"Unsupported config file format: {save_path.suffix}")

        logger.info(f"Saved configuration to {save_path}")

    @staticmethod
    def merge_configs(base_config: Dict[str, Any],
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configurations, with override taking precedence.

        Args:
            base_config: Base configuration
            override_config: Override configuration

        Returns:
            Merged configuration
        """
        import copy

        def deep_merge(base, override):
            """Recursively merge dictionaries."""
            result = copy.deepcopy(base)
            for key, value in override.items():
                if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = deep_merge(result[key], value)
                else:
                    result[key] = value
            return result

        return deep_merge(base_config, override_config)
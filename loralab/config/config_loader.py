"""Configuration loader for LoRALab self-supervised evolution

Provides utilities for loading and managing evolution configurations.
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
        # Check mode
        mode = config.get('mode', 'self_supervised')

        if mode != 'self_supervised':
            logger.warning(f"Mode '{mode}' not supported, using 'self_supervised'")
            config['mode'] = 'self_supervised'

        # Model configuration
        if 'model' not in config:
            config['model'] = {}

        model = config['model']
        model.setdefault('path', 'Qwen/Qwen3-4B-Instruct-2507')
        model.setdefault('type', 'transformers')
        model.setdefault('torch_dtype', 'float16')
        model.setdefault('device_map', 'auto')
        model.setdefault('trust_remote_code', True)
        model.setdefault('low_cpu_mem_usage', True)

        # Evolution configuration
        if 'evolution' not in config:
            config['evolution'] = {}

        evolution = config['evolution']
        evolution.setdefault('population_size', 6)
        evolution.setdefault('generations', 10)
        evolution.setdefault('keep_top', 2)
        evolution.setdefault('mutation_rate', 0.3)
        evolution.setdefault('crossover_rate', 0.2)

        # LoRA search space
        if 'lora_search_space' not in config:
            config['lora_search_space'] = {}

        lora = config['lora_search_space']
        lora.setdefault('rank', [4, 8, 16, 32, 64])
        lora.setdefault('alpha_multiplier', [1, 2, 3])
        lora.setdefault('dropout', [0.05, 0.1, 0.15])
        lora.setdefault('learning_rate', [1e-5, 2e-5, 5e-5, 1e-4, 2e-4])
        lora.setdefault('target_modules', ["q_proj", "k_proj", "v_proj", "o_proj"])

        # Training configuration
        if 'training' not in config:
            config['training'] = {}

        training = config['training']
        training.setdefault('batch_size', 4)
        training.setdefault('gradient_accumulation_steps', 16)
        training.setdefault('epochs_per_variant', 1)
        training.setdefault('max_grad_norm', 1.0)
        training.setdefault('weight_decay', 0.01)
        training.setdefault('warmup_ratio', 0.1)
        training.setdefault('fp16', True)
        training.setdefault('gradient_checkpointing', False)

        # Dataset configuration
        if 'dataset' not in config:
            config['dataset'] = {}

        dataset = config['dataset']
        dataset.setdefault('sources', ['mmlu-pro'])
        dataset.setdefault('train_size', 10000)
        dataset.setdefault('eval_size', 1000)
        dataset.setdefault('seed', 42)

        # Output directory
        config.setdefault('output_dir', 'evolved_adapters')

        # Logging configuration
        if 'logging' not in config:
            config['logging'] = {}

        log_config = config['logging']
        log_config.setdefault('level', 'INFO')
        log_config.setdefault('save_history', True)
        log_config.setdefault('track_metrics', ['accuracy', 'perplexity', 'fitness_score'])

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

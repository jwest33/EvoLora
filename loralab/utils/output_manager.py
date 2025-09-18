"""Centralized output directory management for LoRALab

Provides consistent output structure across all modules.
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class OutputManager:
    """Manages all output directories and paths for LoRALab"""

    def __init__(self, base_dir: str = "lora_runs", run_name: Optional[str] = None):
        """Initialize output manager with base directory

        Args:
            base_dir: Base directory for all outputs
            run_name: Optional run name, defaults to timestamp
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_name

        # Create directory structure
        self._create_directories()

        # Store paths for easy access
        self.paths = {
            'base': self.run_dir,
            'checkpoints': self.run_dir / 'checkpoints',
            'models': self.run_dir / 'models',
            'best_model': self.run_dir / 'models' / 'best',
            'reports': self.run_dir / 'reports',
            'analysis': self.run_dir / 'analysis',
            'logs': self.run_dir / 'logs',
            'config': self.run_dir / 'config',
            'history': self.run_dir / 'history'
        }

        logger.info(f"Output manager initialized at: {self.run_dir}")

    def _create_directories(self):
        """Create all required directories"""
        directories = [
            self.run_dir,
            self.run_dir / 'checkpoints',        # Evolution checkpoints
            self.run_dir / 'checkpoints' / 'generations',  # Per-generation checkpoints
            self.run_dir / 'models',              # Saved models
            self.run_dir / 'models' / 'best',     # Best variant
            self.run_dir / 'models' / 'variants', # All variant models
            self.run_dir / 'reports',             # Evaluation reports
            self.run_dir / 'reports' / 'comparisons',  # Model comparison reports
            self.run_dir / 'reports' / 'evaluations',  # Individual evaluation reports
            self.run_dir / 'analysis',            # Analysis outputs
            self.run_dir / 'analysis' / 'visualizations',  # Graphs and charts
            self.run_dir / 'logs',                # Training and system logs
            self.run_dir / 'config',              # Configuration backups
            self.run_dir / 'history'              # Evolution history
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    def get_path(self, key: str) -> Path:
        """Get a specific path by key

        Args:
            key: Path key (e.g., 'checkpoints', 'best_model')

        Returns:
            Path object
        """
        if key not in self.paths:
            raise KeyError(f"Unknown path key: {key}. Available keys: {list(self.paths.keys())}")
        return self.paths[key]

    def get_generation_checkpoint_dir(self, generation: int) -> Path:
        """Get checkpoint directory for a specific generation

        Args:
            generation: Generation number

        Returns:
            Path to generation checkpoint directory
        """
        gen_dir = self.paths['checkpoints'] / 'generations' / f'gen{generation}'
        gen_dir.mkdir(parents=True, exist_ok=True)
        return gen_dir

    def get_variant_model_dir(self, variant_id: str) -> Path:
        """Get model directory for a specific variant

        Args:
            variant_id: Variant identifier

        Returns:
            Path to variant model directory
        """
        variant_dir = self.paths['models'] / 'variants' / variant_id
        variant_dir.mkdir(parents=True, exist_ok=True)
        return variant_dir

    def save_config(self, config: Dict[str, Any], filename: str = "config.yaml"):
        """Save configuration to config directory

        Args:
            config: Configuration dictionary
            filename: Config filename
        """
        config_path = self.paths['config'] / filename

        if filename.endswith('.yaml'):
            import yaml
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
        else:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)

        logger.info(f"Config saved to: {config_path}")

    def save_history(self, history: Any, filename: str = "evolution_history.json"):
        """Save history to history directory

        Args:
            history: History data
            filename: History filename
        """
        history_path = self.paths['history'] / filename

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

        return history_path

    def get_report_path(self, report_type: str, identifier: str) -> Path:
        """Get path for a specific report

        Args:
            report_type: Type of report ('comparison', 'evaluation')
            identifier: Report identifier

        Returns:
            Path to report file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        if report_type == 'comparison':
            return self.paths['reports'] / 'comparisons' / f"comparison_{identifier}_{timestamp}.md"
        elif report_type == 'evaluation':
            return self.paths['reports'] / 'evaluations' / f"evaluation_{identifier}_{timestamp}.json"
        else:
            return self.paths['reports'] / f"{report_type}_{identifier}_{timestamp}.md"

    def get_visualization_path(self, viz_name: str, extension: str = "png") -> Path:
        """Get path for a visualization file

        Args:
            viz_name: Visualization name
            extension: File extension

        Returns:
            Path to visualization file
        """
        return self.paths['analysis'] / 'visualizations' / f"{viz_name}.{extension}"

    def get_log_path(self, log_name: str = "training") -> Path:
        """Get path for a log file

        Args:
            log_name: Log file name

        Returns:
            Path to log file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return self.paths['logs'] / f"{log_name}_{timestamp}.log"

    def list_runs(self) -> list:
        """List all available runs

        Returns:
            List of run directories
        """
        if not self.base_dir.exists():
            return []

        runs = [d.name for d in self.base_dir.iterdir() if d.is_dir()]
        return sorted(runs, reverse=True)  # Most recent first

    def get_run_info(self) -> Dict[str, Any]:
        """Get information about current run

        Returns:
            Dictionary with run information
        """
        info = {
            'run_name': self.run_dir.name,
            'base_path': str(self.run_dir),
            'created': datetime.now().isoformat(),
            'structure': {}
        }

        # Count files in each directory
        for key, path in self.paths.items():
            if path.exists():
                file_count = len(list(path.rglob('*')))
                info['structure'][key] = {
                    'path': str(path),
                    'files': file_count
                }

        return info

    def cleanup_old_runs(self, keep_last: int = 5):
        """Clean up old runs, keeping only the most recent ones

        Args:
            keep_last: Number of runs to keep
        """
        runs = self.list_runs()

        if len(runs) <= keep_last:
            logger.info(f"Only {len(runs)} runs exist, no cleanup needed")
            return

        runs_to_delete = runs[keep_last:]

        for run in runs_to_delete:
            run_path = self.base_dir / run
            if run_path.exists():
                import shutil
                shutil.rmtree(run_path)
                logger.info(f"Deleted old run: {run}")

        logger.info(f"Cleanup complete. Kept {keep_last} most recent runs")


# Singleton instance
_output_manager = None


def get_output_manager(base_dir: str = "lora_runs", run_name: Optional[str] = None) -> OutputManager:
    """Get or create the global output manager instance

    Args:
        base_dir: Base directory for outputs
        run_name: Optional run name

    Returns:
        OutputManager instance
    """
    global _output_manager

    if _output_manager is None or run_name is not None:
        _output_manager = OutputManager(base_dir, run_name)

    return _output_manager
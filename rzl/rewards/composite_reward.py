"""Composite reward combining multiple reward components"""
from typing import List, Dict, Optional
import numpy as np


class CompositeReward:
    """Combine multiple reward components into a composite score

    Following R-Zero paper's approach to combine:
    - Uncertainty reward (maximize learning potential)
    - Repetition penalty (encourage diversity)
    - Format reward (ensure quality)
    """

    def __init__(
        self,
        uncertainty_weight: float = 1.0,
        repetition_weight: float = 1.0,
        format_weight: float = 0.5
    ):
        """Initialize composite reward calculator

        Args:
            uncertainty_weight: Weight for uncertainty reward
            repetition_weight: Weight for repetition penalty
            format_weight: Weight for format reward
        """
        self.uncertainty_weight = uncertainty_weight
        self.repetition_weight = repetition_weight
        self.format_weight = format_weight

    def compute(
        self,
        uncertainty_rewards: List[float],
        repetition_penalties: List[float],
        format_rewards: List[float],
        clamp_negative: bool = True
    ) -> List[float]:
        """Compute composite rewards

        Following R-Zero paper: r = max(0, r_uncertainty - r_repetition + r_format)

        Args:
            uncertainty_rewards: Uncertainty rewards (higher is better)
            repetition_penalties: Repetition penalties (higher is worse)
            format_rewards: Format rewards (higher is better)
            clamp_negative: Whether to clamp negative rewards to 0

        Returns:
            List of composite rewards
        """
        batch_size = len(uncertainty_rewards)

        # Ensure all components have same length
        assert len(repetition_penalties) == batch_size
        assert len(format_rewards) == batch_size

        composite_rewards = []

        for i in range(batch_size):
            # Weighted combination
            reward = (
                self.uncertainty_weight * uncertainty_rewards[i]
                - self.repetition_weight * repetition_penalties[i]
                + self.format_weight * format_rewards[i]
            )

            # Clamp negative rewards if requested
            if clamp_negative:
                reward = max(0.0, reward)

            composite_rewards.append(reward)

        return composite_rewards

    def compute_normalized(
        self,
        uncertainty_rewards: List[float],
        repetition_penalties: List[float],
        format_rewards: List[float]
    ) -> List[float]:
        """Compute normalized composite rewards

        Normalizes each component before combining for balanced contribution

        Args:
            uncertainty_rewards: Uncertainty rewards
            repetition_penalties: Repetition penalties
            format_rewards: Format rewards

        Returns:
            List of normalized composite rewards
        """
        # Normalize each component
        u_norm = self._normalize(uncertainty_rewards)
        r_norm = self._normalize(repetition_penalties)
        f_norm = self._normalize(format_rewards)

        # Compute composite with normalized components
        return self.compute(u_norm, r_norm, f_norm, clamp_negative=True)

    def _normalize(self, values: List[float]) -> List[float]:
        """Normalize values to [0, 1] range

        Args:
            values: List of values to normalize

        Returns:
            Normalized values
        """
        if not values:
            return []

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            # All values are the same
            return [0.5] * len(values)

        # Min-max normalization
        normalized = [
            (v - min_val) / (max_val - min_val)
            for v in values
        ]

        return normalized

    def get_component_statistics(
        self,
        uncertainty_rewards: List[float],
        repetition_penalties: List[float],
        format_rewards: List[float]
    ) -> Dict:
        """Get statistics for each reward component

        Args:
            uncertainty_rewards: Uncertainty rewards
            repetition_penalties: Repetition penalties
            format_rewards: Format rewards

        Returns:
            Dictionary with component statistics
        """
        composite = self.compute(
            uncertainty_rewards,
            repetition_penalties,
            format_rewards
        )

        stats = {
            "uncertainty": {
                "mean": np.mean(uncertainty_rewards),
                "std": np.std(uncertainty_rewards),
                "min": np.min(uncertainty_rewards),
                "max": np.max(uncertainty_rewards)
            },
            "repetition": {
                "mean": np.mean(repetition_penalties),
                "std": np.std(repetition_penalties),
                "min": np.min(repetition_penalties),
                "max": np.max(repetition_penalties)
            },
            "format": {
                "mean": np.mean(format_rewards),
                "std": np.std(format_rewards),
                "min": np.min(format_rewards),
                "max": np.max(format_rewards)
            },
            "composite": {
                "mean": np.mean(composite),
                "std": np.std(composite),
                "min": np.min(composite),
                "max": np.max(composite),
                "positive_ratio": sum(1 for r in composite if r > 0) / len(composite)
            }
        }

        return stats
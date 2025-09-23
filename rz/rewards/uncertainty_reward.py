"""Uncertainty reward for Challenger training in R-Zero"""
import re
from typing import List, Dict
from collections import Counter


class UncertaintyReward:
    """Compute uncertainty reward based on Solver's self-consistency

    Following R-Zero paper: r_uncertainty = 1 - 2|p̂(x; Sφ) - 0.5|

    This reward is maximized when the Solver's empirical accuracy is 0.5,
    meaning the problem is at the edge of the Solver's capability.
    """

    def __init__(self):
        """Initialize uncertainty reward calculator"""
        self.name = "uncertainty"

    def compute(
        self,
        problems: List[Dict],
        solver_responses: List[List[str]]
    ) -> List[float]:
        """Compute uncertainty rewards for generated problems

        Args:
            problems: List of generated problems with questions and answers
            solver_responses: For each problem, list of m solver responses

        Returns:
            List of uncertainty rewards (one per problem)
        """
        rewards = []

        for problem_idx, responses in enumerate(solver_responses):
            if not responses:
                rewards.append(0.0)
                continue

            # Extract numerical answers from solver responses
            extracted_answers = []
            for response in responses:
                answer = self._extract_answer(response)
                if answer:
                    extracted_answers.append(answer)

            if not extracted_answers:
                # No valid answers extracted - maximum uncertainty
                rewards.append(0.0)
                continue

            # Compute empirical accuracy (self-consistency)
            p_hat = self._compute_self_consistency(extracted_answers)

            # Uncertainty reward: maximized when p_hat = 0.5
            r_uncertainty = 1.0 - 2.0 * abs(p_hat - 0.5)
            rewards.append(r_uncertainty)

        return rewards

    def _extract_answer(self, response: str) -> str:
        """Extract numerical answer from solver response

        Args:
            response: Solver's generated response

        Returns:
            Extracted numerical answer or empty string
        """
        # Look for answer in various formats
        patterns = [
            r'<SOLUTION>(.*?)</SOLUTION>',  # Solution tags
            r'\\boxed\{([\d\.\-]+)\}',      # LaTeX boxed
            r'answer is ([\d\.\-]+)',        # Natural language
            r'= ([\d\.\-]+)$',               # Final calculation
        ]

        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                return match.group(1).strip()

        # Fallback: extract last number
        numbers = re.findall(r'[\d\.\-]+', response)
        return numbers[-1] if numbers else ""

    def _compute_self_consistency(self, answers: List[str]) -> float:
        """Compute self-consistency (majority vote accuracy)

        Args:
            answers: List of extracted answers

        Returns:
            Empirical accuracy (fraction agreeing with majority)
        """
        if not answers:
            return 0.5  # Default to maximum uncertainty

        # Count answer frequencies
        answer_counts = Counter(answers)
        most_common = answer_counts.most_common(1)[0]

        # Empirical accuracy is fraction agreeing with majority
        p_hat = most_common[1] / len(answers)
        return p_hat

    def get_optimal_range(self) -> tuple:
        """Get the optimal uncertainty range for curriculum

        Returns:
            Tuple of (min_accuracy, max_accuracy) for optimal learning
        """
        # Optimal when solver accuracy is between 30-70%
        return (0.3, 0.7)

"""Format reward for verifying problem structure"""
import re
from typing import List, Dict


class FormatReward:
    """Check if generated problems have proper format and structure

    Ensures problems are well-formed with:
    - Valid question text
    - Numerical answer
    - Appropriate length
    """

    def __init__(
        self,
        min_question_length: int = 20,
        max_question_length: int = 500,
        require_tags: bool = False
    ):
        """Initialize format reward calculator

        Args:
            min_question_length: Minimum acceptable question length
            max_question_length: Maximum acceptable question length
            require_tags: Whether to require XML-style tags
        """
        self.min_question_length = min_question_length
        self.max_question_length = max_question_length
        self.require_tags = require_tags
        self.name = "format"

    def compute(self, problems: List[Dict]) -> List[float]:
        """Compute format rewards for problems

        Args:
            problems: List of generated problems

        Returns:
            List of format rewards (1.0 = good, 0.0 = bad format)
        """
        rewards = []

        for problem in problems:
            reward = self._check_format(problem)
            rewards.append(reward)

        return rewards

    def _check_format(self, problem: Dict) -> float:
        """Check format of a single problem

        Args:
            problem: Problem dictionary

        Returns:
            Format reward (0.0 to 1.0)
        """
        score = 1.0

        # Check question exists and has reasonable length
        question = problem.get("question", "")
        if not question:
            return 0.0

        q_len = len(question)
        if q_len < self.min_question_length:
            score *= 0.5  # Too short
        elif q_len > self.max_question_length:
            score *= 0.7  # Too long

        # Check answer exists and is numerical
        answer = problem.get("answer", "")
        if not answer:
            return 0.0

        # Verify answer contains a number
        if not re.search(r'\d+', answer):
            score *= 0.3

        # Check for XML tags if required
        if self.require_tags:
            raw_response = problem.get("raw_response", "")
            if "<question>" not in raw_response or "</question>" not in raw_response:
                score *= 0.5

        # Check for common quality indicators
        quality_score = self._check_quality(question)
        score *= quality_score

        return max(0.0, min(1.0, score))

    def _check_quality(self, question: str) -> float:
        """Check quality indicators in question text

        Args:
            question: Question text

        Returns:
            Quality score (0.0 to 1.0)
        """
        score = 1.0

        # Should be a complete sentence
        if not question.strip().endswith(('?', '.')):
            score *= 0.8

        # Should contain some context (not just numbers)
        words = question.split()
        num_words = len(words)
        if num_words < 5:
            score *= 0.5

        # Check for word problem indicators
        context_words = [
            'has', 'have', 'buys', 'sells', 'gives', 'receives',
            'total', 'altogether', 'remaining', 'left', 'more', 'less'
        ]
        has_context = any(word in question.lower() for word in context_words)
        if not has_context:
            score *= 0.7

        return score

    def get_format_statistics(self, problems: List[Dict]) -> Dict:
        """Get detailed format statistics for a batch

        Args:
            problems: List of problems to analyze

        Returns:
            Dictionary with format statistics
        """
        rewards = self.compute(problems)

        stats = {
            "total": len(problems),
            "perfect_format": sum(1 for r in rewards if r >= 0.9),
            "good_format": sum(1 for r in rewards if 0.7 <= r < 0.9),
            "poor_format": sum(1 for r in rewards if 0.3 <= r < 0.7),
            "bad_format": sum(1 for r in rewards if r < 0.3),
            "average_score": sum(rewards) / len(rewards) if rewards else 0
        }

        return stats

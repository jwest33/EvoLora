"""GSM8K Dataset Handler for R-Zero

Handles loading and processing of the GSM8K dataset for training and evaluation.
"""

import logging
import re
import random
from typing import List, Dict, Any, Optional, Tuple
from datasets import load_dataset

logger = logging.getLogger(__name__)


class GSM8KHandler:
    """Handler for GSM8K dataset operations"""

    def __init__(self, config, cache_dir: str = None):
        """Initialize GSM8K handler

        Args:
            config: GSM8KConfig with dataset settings
            cache_dir: Directory for caching dataset
        """
        self.config = config
        self.cache_dir = cache_dir
        self.dataset = None
        self.train_set = None
        self.test_set = None

        # Load dataset
        self._load_dataset()

    def _load_dataset(self):
        """Load GSM8K dataset from HuggingFace"""
        logger.info("Loading GSM8K dataset...")

        try:
            # Load GSM8K from HuggingFace
            self.dataset = load_dataset(
                self.config.dataset_name,
                self.config.subset,
                cache_dir=self.cache_dir
            )

            # Process train and test splits
            self._process_splits()

            logger.info(f"GSM8K loaded: {len(self.train_set)} train, {len(self.test_set)} test examples")

        except Exception as e:
            logger.error(f"Failed to load GSM8K: {e}")
            logger.info("Creating synthetic dataset as fallback...")
            self._create_synthetic_dataset()

    def _process_splits(self):
        """Process and split the dataset"""
        # GSM8K has train and test splits
        train_data = self.dataset["train"]
        test_data = self.dataset["test"]

        # Process train set
        self.train_set = []
        for item in train_data:
            processed = self._process_example(item)
            if processed:
                self.train_set.append(processed)

        # Process test set (use a subset for faster evaluation)
        self.test_set = []
        test_subset_size = int(len(test_data) * self.config.test_split_ratio)
        test_subset = random.sample(range(len(test_data)), min(test_subset_size, len(test_data)))

        for idx in test_subset:
            processed = self._process_example(test_data[idx])
            if processed:
                self.test_set.append(processed)

    def _process_example(self, example: Dict) -> Optional[Dict]:
        """Process a single GSM8K example

        Args:
            example: Raw example from dataset

        Returns:
            Processed example with question and numeric answer
        """
        try:
            # Extract question and answer
            question = example.get("question", "")
            answer_text = example.get("answer", "")

            if not question or not answer_text:
                return None

            # Extract numeric answer from GSM8K format
            # GSM8K answers have format: "... #### numeric_answer"
            numeric_answer = self._extract_numeric_answer(answer_text)

            if numeric_answer is None:
                return None

            return {
                "question": question.strip(),
                "answer": str(numeric_answer),
                "full_answer": answer_text,
            }

        except Exception as e:
            logger.debug(f"Failed to process example: {e}")
            return None

    def _extract_numeric_answer(self, answer_text: str) -> Optional[str]:
        """Extract numeric answer from GSM8K answer format

        Args:
            answer_text: Full answer text with reasoning

        Returns:
            Numeric answer as string
        """
        # GSM8K format: "reasoning... #### numeric_answer"
        if "####" in answer_text:
            parts = answer_text.split("####")
            if len(parts) >= 2:
                answer = parts[-1].strip()
                # Clean the answer
                answer = answer.replace(",", "")  # Remove commas
                answer = answer.replace("$", "")  # Remove dollar signs

                # Verify it's numeric
                try:
                    float(answer)
                    return answer
                except:
                    # Try to extract number from the answer
                    numbers = re.findall(r"[-]?\d+\.?\d*", answer)
                    if numbers:
                        return numbers[0]

        # Fallback: try to find last number in the text
        numbers = re.findall(r"[-]?\d+\.?\d*", answer_text)
        if numbers:
            return numbers[-1]

        return None

    def _create_synthetic_dataset(self):
        """Create a synthetic dataset as fallback"""
        logger.info("Creating synthetic math problems...")

        self.train_set = []
        self.test_set = []

        # Create some synthetic problems
        templates = [
            ("John has {a} apples. He buys {b} more apples. How many apples does he have now?", lambda a, b: a + b),
            ("A store has {a} items. They sell {b} items. How many items are left?", lambda a, b: a - b),
            ("Each box contains {a} items. If there are {b} boxes, how many items are there in total?", lambda a, b: a * b),
            ("A pizza is cut into {a} slices. {b} friends share it equally. How many slices does each friend get?", lambda a, b: a // b),
            ("A train travels {a} miles in {b} hours. What is its average speed in miles per hour?", lambda a, b: a / b),
        ]

        for i in range(100):
            template, func = random.choice(templates)
            a = random.randint(10, 100)
            b = random.randint(2, 20)

            # Ensure valid division
            if "equally" in template or "per hour" in template:
                a = b * random.randint(2, 10)  # Make it divisible

            question = template.format(a=a, b=b)
            answer = func(a, b)

            # Handle float answers
            if isinstance(answer, float):
                answer = round(answer, 2)

            example = {
                "question": question,
                "answer": str(answer),
                "full_answer": f"The answer is {answer}. #### {answer}",
            }

            if i < 80:
                self.train_set.append(example)
            else:
                self.test_set.append(example)

        logger.info(f"Created {len(self.train_set)} train and {len(self.test_set)} test examples")

    def get_train_set(self, limit: Optional[int] = None) -> List[Dict]:
        """Get training examples

        Args:
            limit: Maximum number of examples to return

        Returns:
            List of training examples
        """
        if limit:
            return self.train_set[:limit]
        return self.train_set

    def get_test_set(self, limit: Optional[int] = None) -> List[Dict]:
        """Get test examples

        Args:
            limit: Maximum number of examples to return

        Returns:
            List of test examples
        """
        if limit:
            return self.test_set[:limit]
        return self.test_set

    def get_random_examples(self, n: int = 10, split: str = "train") -> List[Dict]:
        """Get random examples from dataset

        Args:
            n: Number of examples to get
            split: Which split to sample from ("train" or "test")

        Returns:
            List of random examples
        """
        dataset = self.train_set if split == "train" else self.test_set
        n = min(n, len(dataset))
        return random.sample(dataset, n)

    def format_for_prompt(self, example: Dict) -> str:
        """Format an example for use in a prompt

        Args:
            example: Dataset example

        Returns:
            Formatted prompt string
        """
        question = example["question"]
        return f"Problem: {question}\n\nSolve this step-by-step and provide your final answer."

    def validate_answer(self, predicted: str, target: str, tolerance: float = None) -> bool:
        """Validate if predicted answer matches target

        Args:
            predicted: Predicted answer
            target: Target answer
            tolerance: Numeric tolerance for comparison

        Returns:
            True if answers match
        """
        tolerance = tolerance or self.config.tolerance

        # Clean answers
        predicted = str(predicted).strip().replace(",", "").replace("$", "")
        target = str(target).strip().replace(",", "").replace("$", "")

        # Try exact match first
        if predicted == target:
            return True

        # Try numeric comparison
        try:
            pred_num = float(predicted)
            target_num = float(target)
            return abs(pred_num - target_num) <= tolerance
        except:
            return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics

        Returns:
            Dictionary of statistics
        """
        stats = {
            "train_size": len(self.train_set),
            "test_size": len(self.test_set),
            "total_size": len(self.train_set) + len(self.test_set),
        }

        # Calculate question length statistics
        if self.train_set:
            train_lengths = [len(ex["question"].split()) for ex in self.train_set]
            stats["avg_question_length"] = sum(train_lengths) / len(train_lengths)
            stats["min_question_length"] = min(train_lengths)
            stats["max_question_length"] = max(train_lengths)

        # Calculate answer value statistics
        if self.train_set:
            answer_values = []
            for ex in self.train_set:
                try:
                    val = float(ex["answer"])
                    answer_values.append(val)
                except:
                    continue

            if answer_values:
                stats["avg_answer_value"] = sum(answer_values) / len(answer_values)
                stats["min_answer_value"] = min(answer_values)
                stats["max_answer_value"] = max(answer_values)

        return stats

    def create_bootstrap_examples(self, n: int = 10) -> List[Dict]:
        """Create bootstrap examples for initial Solver training

        These are simple, high-quality examples to help the Solver
        learn the basic format before R-Zero evolution begins.

        Args:
            n: Number of bootstrap examples

        Returns:
            List of bootstrap examples with reasoning
        """
        bootstrap_examples = []

        simple_problems = [
            {
                "question": "Sarah has 5 apples. She buys 3 more apples. How many apples does she have in total?",
                "reasoning": "Sarah starts with 5 apples. She buys 3 more apples. Total apples = 5 + 3 = 8.",
                "answer": "8"
            },
            {
                "question": "A store has 20 books. They sell 7 books. How many books are left?",
                "reasoning": "The store starts with 20 books. They sell 7 books. Books remaining = 20 - 7 = 13.",
                "answer": "13"
            },
            {
                "question": "Each box contains 6 cookies. If there are 4 boxes, how many cookies are there in total?",
                "reasoning": "Each box has 6 cookies. There are 4 boxes. Total cookies = 6 × 4 = 24.",
                "answer": "24"
            },
            {
                "question": "A pizza is cut into 12 slices. 3 friends share it equally. How many slices does each friend get?",
                "reasoning": "The pizza has 12 slices. It's shared by 3 friends equally. Slices per friend = 12 ÷ 3 = 4.",
                "answer": "4"
            },
            {
                "question": "John walks 2 miles every day. How many miles does he walk in a week?",
                "reasoning": "John walks 2 miles per day. A week has 7 days. Total miles = 2 × 7 = 14.",
                "answer": "14"
            },
            {
                "question": "A car travels 60 miles in 2 hours. What is its average speed?",
                "reasoning": "The car travels 60 miles in 2 hours. Average speed = distance ÷ time = 60 ÷ 2 = 30 miles per hour.",
                "answer": "30"
            },
            {
                "question": "Lisa has $50. She spends $18 on a book. How much money does she have left?",
                "reasoning": "Lisa starts with $50. She spends $18. Money remaining = 50 - 18 = 32 dollars.",
                "answer": "32"
            },
            {
                "question": "A recipe needs 3 cups of flour. How much flour is needed for 4 batches?",
                "reasoning": "One batch needs 3 cups of flour. For 4 batches: 3 × 4 = 12 cups of flour.",
                "answer": "12"
            },
            {
                "question": "There are 24 students in a class. They form groups of 4. How many groups are there?",
                "reasoning": "Total students = 24. Each group has 4 students. Number of groups = 24 ÷ 4 = 6.",
                "answer": "6"
            },
            {
                "question": "Tom saves $5 each week. How much will he save in 8 weeks?",
                "reasoning": "Tom saves $5 per week. In 8 weeks, he saves: 5 × 8 = 40 dollars.",
                "answer": "40"
            }
        ]

        return simple_problems[:n]

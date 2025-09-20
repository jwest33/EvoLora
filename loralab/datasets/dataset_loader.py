"""Unified dataset loader for self-supervised LoRA training

Handles loading datasets from HuggingFace and other sources.
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path
import json
import random

logger = logging.getLogger(__name__)

# Try to import datasets library
try:
    from datasets import load_dataset
    import datasets

    # Apply Windows compatibility settings
    from ..utils.windows_compat import disable_multiprocessing, is_windows
    if is_windows():
        disable_multiprocessing()
        datasets.config.NUM_PROC = 1

    DATASETS_AVAILABLE = True
except ImportError:
    DATASETS_AVAILABLE = False
    logger.warning("datasets library not installed. Install with: pip install datasets")


class DatasetLoader:
    """Unified loader for training datasets"""

    # Mapping of dataset names to HuggingFace configurations
    DATASET_CONFIGS = {
        'mmlu-pro': {
            'name': 'TIGER-Lab/MMLU-Pro',
            'split': 'test',
            'question_field': 'question',
            'answer_field': 'answer',
            'subset': None,
            'process_answer': lambda x: x if isinstance(x, str) else str(x)
        },
        'gsm8k': {
            'name': 'gsm8k',
            'split': 'train',
            'question_field': 'question',
            'answer_field': 'answer',
            'subset': 'main'
        },
        'squad': {
            'name': 'squad',
            'split': 'train',
            'question_field': 'question',
            'answer_field': 'answers',
            'subset': None,
            'process_answer': lambda x: x['text'][0] if x and 'text' in x else ''
        },
        'alpaca': {
            'name': 'tatsu-lab/alpaca',
            'split': 'train',
            'question_field': 'instruction',
            'answer_field': 'output',
            'subset': None
        },
        'dolly': {
            'name': 'databricks/databricks-dolly-15k',
            'split': 'train',
            'question_field': 'instruction',
            'answer_field': 'response',
            'subset': None
        },
        'openassistant': {
            'name': 'OpenAssistant/oasst1',
            'split': 'train',
            'question_field': 'text',
            'answer_field': 'text',
            'subset': None,
            'process_answer': lambda x: x if isinstance(x, str) else ''
        },
        'wizardlm': {
            'name': 'WizardLM/WizardLM_evol_instruct_V2_196k',
            'split': 'train',
            'question_field': 'instruction',
            'answer_field': 'output',
            'subset': None
        }
    }

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize dataset loader

        Args:
            cache_dir: Directory to cache datasets
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path('cache')
        # Don't create directory immediately - let it be created when needed

    def load_dataset(self,
                    dataset_name: str,
                    train_size: Optional[int] = None,
                    eval_size: Optional[int] = None,
                    seed: int = 42) -> Dict[str, List[Dict]]:
        """Load a dataset for training and evaluation

        Args:
            dataset_name: Name of dataset to load
            train_size: Number of training examples (None for all)
            eval_size: Number of evaluation examples (None for 10% of train)
            seed: Random seed for splitting

        Returns:
            Dictionary with 'train' and 'eval' splits
        """
        logger.info(f"Loading dataset: {dataset_name}")

        # Check if it's a known dataset
        if dataset_name in self.DATASET_CONFIGS:
            return self._load_huggingface_dataset(
                dataset_name, train_size, eval_size, seed
            )
        else:
            # Try to load as a local file
            return self._load_local_dataset(
                dataset_name, train_size, eval_size, seed
            )

    def _load_huggingface_dataset(self,
                                 dataset_name: str,
                                 train_size: Optional[int],
                                 eval_size: Optional[int],
                                 seed: int) -> Dict[str, List[Dict]]:
        """Load dataset from HuggingFace

        Args:
            dataset_name: Name of dataset
            train_size: Number of training examples
            eval_size: Number of evaluation examples
            seed: Random seed

        Returns:
            Dictionary with train and eval splits
        """
        if not DATASETS_AVAILABLE:
            logger.error("datasets library not available")
            return self._create_dummy_dataset(train_size, eval_size)

        config = self.DATASET_CONFIGS[dataset_name]

        try:
            # Load dataset from HuggingFace
            logger.info(f"Fetching {dataset_name} from HuggingFace...")

            try:
                if config['subset']:
                    dataset = load_dataset(config['name'], config['subset'], trust_remote_code=True)
                else:
                    dataset = load_dataset(config['name'], trust_remote_code=True)
            except Exception as e:
                logger.warning(f"Failed with trust_remote_code=True: {e}")
                # Try without trust_remote_code
                if config['subset']:
                    dataset = load_dataset(config['name'], config['subset'])
                else:
                    dataset = load_dataset(config['name'])

            # Get the appropriate split
            if config['split'] in dataset:
                data = dataset[config['split']]
            else:
                # If specified split doesn't exist, try to find any available split
                available_splits = list(dataset.keys())
                logger.warning(f"Split '{config['split']}' not found. Using '{available_splits[0]}'")
                data = dataset[available_splits[0]]

            # Convert to our format
            formatted_data = []
            for item in data:
                question = item.get(config['question_field'], '')
                answer = item.get(config['answer_field'], '')

                # Process answer if needed
                if 'process_answer' in config:
                    answer = config['process_answer'](answer)

                formatted_data.append({
                    'question': str(question),
                    'answer': str(answer)
                })

            # Shuffle data
            random.seed(seed)
            random.shuffle(formatted_data)

            # Split into train and eval
            if train_size:
                formatted_data = formatted_data[:train_size + (eval_size or 0)]

            if eval_size is None:
                eval_size = max(100, len(formatted_data) // 10)  # 10% for eval

            train_data = formatted_data[eval_size:]
            eval_data = formatted_data[:eval_size]

            logger.info(f"Loaded {len(train_data)} training and {len(eval_data)} evaluation examples")

            return {
                'train': train_data,
                'eval': eval_data
            }

        except Exception as e:
            logger.error(f"Failed to load dataset from HuggingFace: {e}")
            return self._create_dummy_dataset(train_size, eval_size)

    def _load_local_dataset(self,
                           file_path: str,
                           train_size: Optional[int],
                           eval_size: Optional[int],
                           seed: int) -> Dict[str, List[Dict]]:
        """Load dataset from local file

        Args:
            file_path: Path to dataset file (JSON or JSONL)
            train_size: Number of training examples
            eval_size: Number of evaluation examples
            seed: Random seed

        Returns:
            Dictionary with train and eval splits
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"Dataset file not found: {file_path}")
            return self._create_dummy_dataset(train_size, eval_size)

        try:
            # Load data based on file extension
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
            elif file_path.suffix == '.jsonl':
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        data.append(json.loads(line))
            else:
                logger.error(f"Unsupported file format: {file_path.suffix}")
                return self._create_dummy_dataset(train_size, eval_size)

            # Ensure data is in correct format
            formatted_data = []
            for item in data:
                if isinstance(item, dict):
                    question = item.get('question', item.get('text', item.get('instruction', '')))
                    answer = item.get('answer', item.get('label', item.get('response', '')))
                    formatted_data.append({
                        'question': str(question),
                        'answer': str(answer)
                    })

            # Shuffle and split
            random.seed(seed)
            random.shuffle(formatted_data)

            if train_size:
                formatted_data = formatted_data[:train_size + (eval_size or 0)]

            if eval_size is None:
                eval_size = max(100, len(formatted_data) // 10)

            train_data = formatted_data[eval_size:]
            eval_data = formatted_data[:eval_size]

            logger.info(f"Loaded {len(train_data)} training and {len(eval_data)} evaluation examples from {file_path}")

            return {
                'train': train_data,
                'eval': eval_data
            }

        except Exception as e:
            logger.error(f"Failed to load local dataset: {e}")
            return self._create_dummy_dataset(train_size, eval_size)

    def _create_dummy_dataset(self,
                             train_size: Optional[int],
                             eval_size: Optional[int]) -> Dict[str, List[Dict]]:
        """Create a dummy dataset for testing

        Args:
            train_size: Number of training examples
            eval_size: Number of evaluation examples

        Returns:
            Dictionary with dummy train and eval data
        """
        logger.warning("Creating dummy dataset for testing")

        train_size = train_size or 100
        eval_size = eval_size or 20

        # Create simple Q&A pairs
        questions = [
            "What is the capital of France?",
            "How many days are in a week?",
            "What is 2 + 2?",
            "Who wrote Romeo and Juliet?",
            "What is the largest planet?",
            "How many continents are there?",
            "What is the speed of light?",
            "Who painted the Mona Lisa?",
            "What is the smallest prime number?",
            "How many sides does a triangle have?"
        ]

        answers = [
            "Paris",
            "7",
            "4",
            "William Shakespeare",
            "Jupiter",
            "7",
            "299,792,458 meters per second",
            "Leonardo da Vinci",
            "2",
            "3"
        ]

        # Generate dataset by repeating and varying questions
        train_data = []
        for i in range(train_size):
            idx = i % len(questions)
            train_data.append({
                'question': questions[idx],
                'answer': answers[idx]
            })

        eval_data = []
        for i in range(eval_size):
            idx = i % len(questions)
            eval_data.append({
                'question': questions[idx],
                'answer': answers[idx]
            })

        return {
            'train': train_data,
            'eval': eval_data
        }

    def list_available_datasets(self) -> List[str]:
        """List all available dataset names

        Returns:
            List of dataset names
        """
        return list(self.DATASET_CONFIGS.keys())

    def save_dataset(self, data: Dict[str, List[Dict]], output_path: str):
        """Save dataset to file

        Args:
            data: Dataset dictionary with train/eval splits
            output_path: Path to save dataset
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Dataset saved to {output_path}")

"""Main entry point for self-supervised LoRA evolution

Run evolutionary optimization to find optimal LoRA adapters.
"""

import sys
import os
import logging
import warnings
from pathlib import Path
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Disable ALL multiprocessing on Windows before any imports
import platform
import multiprocessing

if platform.system() == 'Windows':
    # Check if we're in a child process
    if multiprocessing.current_process().name != 'MainProcess':
        # In child process - completely skip Unsloth imports
        os.environ["SKIP_UNSLOTH"] = "1"
        # Exit early from child processes that shouldn't import Unsloth
        sys.modules['unsloth'] = None  # Prevent import

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["DATASETS_PARALLEL_DOWNLOADS"] = "false"
    os.environ["UNSLOTH_NUM_PROC"] = "1"
    os.environ["DATASETS_NUM_PROC"] = "1"
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Enable logits for evaluation

    # Suppress Windows multiprocessing redirect warnings
    warnings.filterwarnings("ignore", message=".*Redirects are currently not supported in Windows.*")
    # Suppress Unsloth TRL patching warnings
    warnings.filterwarnings("ignore", message=".*Unsloth TRL patching issue.*")

    # Windows multiprocessing fix
    if __name__ == '__main__':
        multiprocessing.freeze_support()
        # Force spawn method (required for Windows)
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except:
            pass

    # Force single process for datasets
    try:
        import datasets
        datasets.disable_progress_bars()
        datasets.config.NUM_PROC = 1
        # Also set through environment variable to be sure
        os.environ["HF_DATASETS_NUM_PROC"] = "1"
    except ImportError:
        pass

# Force disable multiprocessing in datasets globally before any imports
os.environ["HF_DATASETS_NUM_PROC"] = "1"
os.environ["DATASETS_NUM_PROC"] = "1"

# Monkey patch datasets to never use multiprocessing
try:
    import datasets
    # Override default num_proc for all dataset operations
    _original_map = datasets.Dataset.map
    def _patched_map(self, *args, **kwargs):
        if 'num_proc' not in kwargs or kwargs['num_proc'] is None:
            kwargs['num_proc'] = 1
        return _original_map(self, *args, **kwargs)
    datasets.Dataset.map = _patched_map
except ImportError:
    pass

# Configure and initialize Unsloth before anything else
from loralab.utils.unsloth_config import init_unsloth
UNSLOTH_AVAILABLE = init_unsloth()

from loralab.cli_evolution import main

if __name__ == '__main__':
    # Configure root logger - only show critical errors from external libraries
    logging.basicConfig(
        level=logging.WARNING,  # Reduce default verbosity
        format='%(message)s',  # Simplified format for console
        handlers=[
            logging.StreamHandler()
        ]
    )

    # File logging with full details
    file_handler = logging.FileHandler('evolution.log')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    ))
    logging.getLogger().addHandler(file_handler)

    # Set loralab loggers to INFO
    logging.getLogger('loralab').setLevel(logging.INFO)

    # Suppress external library messages
    logging.getLogger('transformers').setLevel(logging.ERROR)
    logging.getLogger('datasets').setLevel(logging.ERROR)
    logging.getLogger('trl').setLevel(logging.ERROR)
    logging.getLogger('unsloth').setLevel(logging.ERROR)

    # Suppress redirect warnings at runtime too
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    # Run main CLI
    main()

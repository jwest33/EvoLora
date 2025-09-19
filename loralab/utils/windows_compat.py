"""Windows compatibility utilities for Unsloth and multiprocessing

Provides wrappers and utilities to handle Windows-specific issues with
Unsloth's dynamic patching and multiprocessing.
"""

import os
import platform
import functools
import logging
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

def is_windows():
    """Check if running on Windows"""
    return platform.system() == 'Windows'

def disable_multiprocessing():
    """Disable all multiprocessing for Windows compatibility"""
    if is_windows():
        # Disable all forms of multiprocessing
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["DATASETS_PARALLEL_DOWNLOADS"] = "false"
        os.environ["DATASETS_NUM_PROC"] = "1"
        os.environ["UNSLOTH_NUM_PROC"] = "1"
        os.environ["RAYON_RS_NUM_CPUS"] = "1"

        # Disable HuggingFace datasets multiprocessing
        try:
            import datasets
            datasets.config.NUM_PROC = 1
            datasets.disable_progress_bars()
        except ImportError:
            pass

def sequential_map(func: Callable, items: List[Any], desc: str = None) -> List[Any]:
    """Map a function over items sequentially (no multiprocessing)

    Args:
        func: Function to apply
        items: Items to process
        desc: Description for progress bar

    Returns:
        List of results
    """
    results = []

    try:
        from tqdm import tqdm
        iterator = tqdm(items, desc=desc) if desc else items
    except ImportError:
        iterator = items

    for item in iterator:
        results.append(func(item))

    return results

def windows_safe_dataset_map(dataset, function, **kwargs):
    """Wrapper for dataset.map() that forces single process on Windows

    Args:
        dataset: HuggingFace dataset
        function: Mapping function
        **kwargs: Additional arguments for map()

    Returns:
        Mapped dataset
    """
    if is_windows():
        # Force single process
        kwargs['num_proc'] = 1
        # Disable caching issues
        kwargs['load_from_cache_file'] = False

    return dataset.map(function, **kwargs)

def windows_safe_trainer(trainer_class):
    """Decorator to make a trainer class Windows-compatible

    Wraps trainer methods to avoid multiprocessing issues.
    """
    if not is_windows():
        return trainer_class

    original_init = trainer_class.__init__

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        # Disable multiprocessing in training args if present
        if hasattr(self, 'args'):
            self.args.dataloader_num_workers = 0
            self.args.dataloader_pin_memory = False

        # Call original init
        original_init(self, *args, **kwargs)

        # Patch dataset processing methods
        if hasattr(self, '_prepare_dataset'):
            original_prepare = self._prepare_dataset

            @functools.wraps(original_prepare)
            def wrapped_prepare(*args, **kwargs):
                # Ensure single process
                if 'num_proc' in kwargs:
                    kwargs['num_proc'] = 1
                return original_prepare(*args, **kwargs)

            self._prepare_dataset = wrapped_prepare

    trainer_class.__init__ = wrapped_init
    return trainer_class

class WindowsSafeUnslothManager:
    """Windows-safe wrapper for Unsloth operations

    Ensures all Unsloth operations happen in the main process only.
    """

    def __init__(self):
        self._unsloth_available = False
        self._initialized = False

    def initialize(self):
        """Initialize Unsloth if not already done"""
        if self._initialized:
            return self._unsloth_available

        if not is_windows():
            # Normal initialization on non-Windows
            try:
                import unsloth
                self._unsloth_available = True
            except ImportError:
                self._unsloth_available = False
        else:
            # Windows-specific initialization
            import multiprocessing
            if multiprocessing.current_process().name != 'MainProcess':
                # Never use Unsloth in child processes on Windows
                self._unsloth_available = False
            else:
                try:
                    # Ensure environment is configured
                    disable_multiprocessing()

                    # Import Unsloth
                    import unsloth
                    self._unsloth_available = True
                    logger.info("Unsloth initialized in main process (Windows mode)")
                except ImportError:
                    self._unsloth_available = False
                except Exception as e:
                    logger.warning(f"Unsloth initialization failed: {e}")
                    self._unsloth_available = False

        self._initialized = True
        return self._unsloth_available

    def is_available(self):
        """Check if Unsloth is available"""
        if not self._initialized:
            self.initialize()
        return self._unsloth_available

# Global instance
_windows_safe_manager = WindowsSafeUnslothManager()

def get_unsloth_manager():
    """Get the Windows-safe Unsloth manager"""
    return _windows_safe_manager
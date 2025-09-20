"""Unsloth configuration and initialization utilities

Handles Windows multiprocessing compatibility and TRL patching issues.
"""

import os
import sys
import io
import platform
import warnings
import logging
import multiprocessing

logger = logging.getLogger(__name__)

# Global state to prevent re-initialization
_UNSLOTH_INITIALIZED = False
_UNSLOTH_AVAILABLE = None

def configure_unsloth():
    """Configure Unsloth settings for optimal performance"""

    # Windows-specific settings - MUST be set before any imports
    if platform.system() == 'Windows':
        # Completely disable multiprocessing on Windows
        os.environ["UNSLOTH_NUM_PROC"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        os.environ["DATASETS_PARALLEL_DOWNLOADS"] = "false"
        os.environ["DATASETS_NUM_PROC"] = "1"

        # Try to set spawn method if not already set
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass  # Already set

    # Suppress warnings
    warnings.filterwarnings("ignore", message=".*Unsloth should be imported before.*")
    warnings.filterwarnings("ignore", message=".*Xformers.*")
    warnings.filterwarnings("ignore", message=".*Redirects are currently not supported.*")
    warnings.filterwarnings("ignore", message=".*AttributeError.*Unsloth.*")

    # Set environment flags
    os.environ["UNSLOTH_IS_PRESENT"] = "1"
    os.environ["SUPPRESS_UNSLOTH_WARNINGS"] = "1"
    os.environ["UNSLOTH_DISABLE_MULTIPROCESSING"] = "1"  # Force disable multiprocessing
    os.environ["UNSLOTH_RETURN_LOGITS"] = "1"  # Force return logits for evaluation

def init_unsloth():
    """Initialize Unsloth with proper configuration

    Returns:
        bool: Whether Unsloth was successfully initialized
    """
    global _UNSLOTH_INITIALIZED, _UNSLOTH_AVAILABLE

    # Return cached result if already initialized
    if _UNSLOTH_INITIALIZED:
        return _UNSLOTH_AVAILABLE

    # Skip completely if SKIP_UNSLOTH is set
    if os.environ.get("SKIP_UNSLOTH") == "1":
        logger.debug("Skipping Unsloth (SKIP_UNSLOTH=1)")
        _UNSLOTH_INITIALIZED = True
        _UNSLOTH_AVAILABLE = False
        return False

    # Only initialize in main process
    if multiprocessing.current_process().name != 'MainProcess':
        logger.debug("Skipping Unsloth init in child process")
        _UNSLOTH_INITIALIZED = True
        _UNSLOTH_AVAILABLE = False
        return False

    configure_unsloth()

    try:
        # Only import if not already imported
        if 'unsloth' not in sys.modules:
            # Pre-import TRL to prevent patching issues
            try:
                # Import TRL components that Unsloth will patch
                from trl import SFTTrainer, DPOTrainer
                from trl import SFTConfig
            except ImportError:
                logger.debug("TRL not installed, skipping pre-import")

            # Temporarily suppress stdout to hide Unsloth patching messages
            old_stdout = sys.stdout
            sys.stdout = io.StringIO()

            try:
                # Now import Unsloth - this will patch TRL if it's installed
                import unsloth

                # Verify core components are available
                from unsloth import FastLanguageModel
                from unsloth.chat_templates import get_chat_template
            finally:
                # Restore stdout
                sys.stdout = old_stdout

            logger.debug("Unsloth initialized successfully")
            _UNSLOTH_INITIALIZED = True
            _UNSLOTH_AVAILABLE = True
            return True

        # Already imported
        _UNSLOTH_INITIALIZED = True
        _UNSLOTH_AVAILABLE = True
        return True

    except ImportError as e:
        logger.warning(f"Unsloth not installed: {e}")
        _UNSLOTH_INITIALIZED = True
        _UNSLOTH_AVAILABLE = False
        return False
    except NotImplementedError as e:
        # GPU not detected
        logger.warning(f"GPU not detected by Unsloth: {e}")
        _UNSLOTH_INITIALIZED = True
        _UNSLOTH_AVAILABLE = False
        return False
    except AttributeError as e:
        # TRL patching issue - try to continue
        if "Unsloth" in str(e) or "TRL" in str(e):
            logger.warning(f"Unsloth TRL patching issue, attempting to continue: {e}")
            # Mark as available since the core functionality might still work
            _UNSLOTH_INITIALIZED = True
            _UNSLOTH_AVAILABLE = True
            return True
        logger.error(f"Unsloth initialization failed: {e}")
        _UNSLOTH_INITIALIZED = True
        _UNSLOTH_AVAILABLE = False
        return False
    except Exception as e:
        logger.error(f"Unsloth initialization error: {e}")
        _UNSLOTH_INITIALIZED = True
        _UNSLOTH_AVAILABLE = False
        return False

def is_unsloth_available():
    """Check if Unsloth is available

    Returns:
        bool: Whether Unsloth can be used
    """
    global _UNSLOTH_AVAILABLE

    if _UNSLOTH_AVAILABLE is None:
        init_unsloth()

    return _UNSLOTH_AVAILABLE or False
"""Initialize Unsloth before other imports

This module should be imported first to ensure Unsloth optimizations are applied.
"""

import sys
import os
import logging

# Check if already initialized
if 'unsloth' in sys.modules:
    UNSLOTH_AVAILABLE = True
else:
    # Suppress repeated warnings
    import warnings
    warnings.filterwarnings("ignore", message=".*Xformers.*")

    # Try to import and initialize Unsloth first
    try:
        import unsloth
        UNSLOTH_AVAILABLE = True
        # Mark as initialized to prevent repeated imports
        os.environ["UNSLOTH_INITIALIZED"] = "1"
    except ImportError:
        UNSLOTH_AVAILABLE = False
    except NotImplementedError:
        # Unsloth couldn't detect GPU
        UNSLOTH_AVAILABLE = False

# Export flag for other modules to check
__all__ = ['UNSLOTH_AVAILABLE']
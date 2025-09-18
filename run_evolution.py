"""Main entry point for self-supervised LoRA evolution

Run evolutionary optimization to find optimal LoRA adapters.
"""

import sys
import logging
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from loralab.cli_evolution import main

if __name__ == '__main__':
    # Configure root logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evolution.log')
        ]
    )

    # Run main CLI
    main()

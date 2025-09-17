#!/usr/bin/env python
"""
Wrapper for quality comparison demos.
"""

import sys
from loralab.demos.quality_comparison import QualityComparison


def main():
    """Run comparison based on command line argument."""
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        print("Running full quality analysis...")
        comparison = QualityComparison()
        comparison.full_analysis()
    else:
        print("Running quick visual comparison...")
        comparison = QualityComparison()
        comparison.quick_comparison()


if __name__ == "__main__":
    main()
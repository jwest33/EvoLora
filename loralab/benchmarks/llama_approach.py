#!/usr/bin/env python
"""
Benchmark script to determine whether server or direct llama.cpp is faster.
This will help you decide which approach to use for your specific hardware.
"""

import time
import yaml
import sys
from pathlib import Path


def main():
    print("="*70)
    print("LLAMA.CPP APPROACH BENCHMARK")
    print("="*70)
    print("\nThis will test which approach is faster for your system:")
    print("1. Server-based (current approach)")
    print("2. Direct library calls (new approach)")
    print("\nNote: Direct approach requires llama-cpp-python to be installed.")

    # Check if config exists
    config_path = Path("loralab/configs/documentation.yaml")
    if not config_path.exists():
        print(f"\nError: Config not found at {config_path}")
        sys.exit(1)

    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    challenger_config = config['challenger']

    # Ask user for test size
    print("\nHow many test generations to run?")
    print("1. Quick (5 generations)")
    print("2. Standard (20 generations)")
    print("3. Thorough (50 generations)")

    choice = input("\nSelect (1-3): ").strip()
    num_tests = {'1': 5, '2': 20, '3': 50}.get(choice, 10)

    print(f"\nRunning benchmark with {num_tests} generations...")

    # First, try the direct approach to see if it's even available
    try:
        from loralab.core.llama_direct import LlamaDirectBenchmark, LLAMA_CPP_AVAILABLE

        if not LLAMA_CPP_AVAILABLE:
            print("\n⚠️  llama-cpp-python is not installed.")
            print("To test direct approach, install it with:")
            print("pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
            print("\nUsing SERVER approach by default.")
            return

        # Run comparison
        print("\nStarting benchmark...")
        server_results, direct_results = LlamaDirectBenchmark.compare(
            challenger_config,
            num_tests=num_tests
        )

        # Additional analysis for your use case
        print("\n" + "="*70)
        print("ANALYSIS FOR YOUR USE CASE")
        print("="*70)

        # Calculate for a typical evolution run
        tasks_per_generation = 20
        generations = 5
        total_calls = tasks_per_generation * generations * 2  # Tasks + labels

        server_total = (server_results['startup_time'] +
                       server_results['avg_generation_time'] * total_calls)
        direct_total = (direct_results['startup_time'] +
                       direct_results['avg_generation_time'] * total_calls)

        print(f"\nFor a typical quick evolution run ({generations} generations, {tasks_per_generation} tasks):")
        print(f"  Server approach: {server_total/60:.1f} minutes")
        print(f"  Direct approach: {direct_total/60:.1f} minutes")
        print(f"  Time saved: {abs(server_total - direct_total)/60:.1f} minutes")

        # Memory considerations
        print("\n" + "="*70)
        print("MEMORY CONSIDERATIONS")
        print("="*70)
        print("\nServer approach:")
        print("  ✓ Runs in separate process")
        print("  ✓ Can be killed if OOM")
        print("  ✓ Doesn't interfere with Solver model")

        print("\nDirect approach:")
        print("  ✓ Single process (simpler)")
        print("  ✓ Faster model swapping")
        print("  ⚠️  Shares memory with Solver")

        # Final recommendation
        print("\n" + "="*70)
        print("RECOMMENDATION FOR YOUR SETUP")
        print("="*70)

        if direct_total < server_total * 0.7:
            print("\n✅ USE DIRECT APPROACH")
            print(f"   Saves {(server_total - direct_total)/60:.1f} minutes per run")
            print("   Significantly faster for your hardware")
        elif server_total < direct_total * 0.9:
            print("\n✅ USE SERVER APPROACH")
            print("   More stable and isolated")
            print("   Better for long-running experiments")
        else:
            print("\n✅ EITHER APPROACH IS FINE")
            print("   Performance difference is minimal")
            print("   Choose based on your preference")

            if config.get('evolution', {}).get('batched_mode'):
                print("\n   Since you're using batched mode:")
                print("   → DIRECT approach is recommended (simpler)")

    except ImportError as e:
        print(f"\nError importing benchmark module: {e}")
        print("Make sure the loralab package is properly installed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
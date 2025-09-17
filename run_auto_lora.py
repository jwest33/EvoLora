#!/usr/bin/env python
"""
AutoLoRA - Main entry point for automatic LoRA adapter generation.
Uses direct llama-cpp-python approach (no server required).
"""

import subprocess
import sys
import yaml
from pathlib import Path
import datetime

def main():
    print(" LoRA Adapter Generator Demo (Direct Mode)")
    print("=" * 60)

    print("\n[OK] Using direct llama-cpp-python (no server required)")
    print("  Models will alternate to save memory")

    # Check if config exists
    config_path = "loralab/config/config.yaml"
    if not Path(config_path).exists():
        print(f"\nConfiguration file not found: {config_path}")
        sys.exit(1)

    # Update config to use direct mode
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Ensure we're using direct mode
    config['challenger']['use_direct'] = True
    print(f"  Challenger model: {config['challenger'].get('model_path', 'Not specified')}")
    print(f"  Solver model: {config['solver'].get('model_name', 'Not specified')}")

    # Demo options
    print("\nDemo Options:")
    print("1. Quick test (5 generations)")
    print("2. Standard run (20 generations)")
    print("3. Full run (50 generations)")
    print("4. Custom run (specify generations)")
    print("5. Test existing adapter")

    choice = input("\nSelect option (1-5): ")

    if choice == '1':
        # Quick test
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("experiments") / f"job_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print("\n" + "="*60)
        print("Starting Quick Test (5 generations) - Direct Mode")
        print("="*60)
        subprocess.run([
            sys.executable, "-m", "loralab.cli", "evolve",
            "--config", str(config),
            "--output", str(output_dir),
            "--generations", "5",
            "--use-direct"  # Force direct mode
        ])

    elif choice == '2':
        # Standard run
        print("\n" + "="*60)
        print("Starting Standard Evolution (20 generations) - Direct Mode")
        print("="*60)
        subprocess.run([
            sys.executable, "-m", "loralab.cli", "evolve",
            "--config", str(config),
            "--output", "experiments/standard_demo",
            "--generations", "20",
            "--use-direct"  # Force direct mode
        ])

    elif choice == '3':
        # Full run
        print("\n" + "="*60)
        print("Starting Full Evolution (50 generations) - Direct Mode")
        print("="*60)
        subprocess.run([
            sys.executable, "-m", "loralab.cli", "evolve",
            "--config", str(config),
            "--output", "experiments/full_demo",
            "--generations", "50",
            "--use-direct"  # Force direct mode
        ])

    elif choice == '4':
        # Custom run
        while True:
            num_gen = input("\nEnter number of generations (1-100): ").strip()
            try:
                generations = int(num_gen)
                if 1 <= generations <= 100:
                    break
                else:
                    print("Please enter a number between 1 and 100")
            except ValueError:
                print("Please enter a valid number")

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path("experiments") / f"custom_{generations}gen_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        print("\n" + "="*60)
        print(f"Starting Custom Evolution ({generations} generations) - Direct Mode")
        print("="*60)
        subprocess.run([
            sys.executable, "-m", "loralab.cli", "evolve",
            "--config", str(config),
            "--output", str(output_dir),
            "--generations", str(generations),
            "--use-direct"  # Force direct mode
        ])

    elif choice == '5':
        # Test adapter
        adapter_path = input("\nEnter adapter path (or press Enter for default): ").strip()
        if not adapter_path:
            # Look for most recent adapter
            experiments = Path("experiments")
            if experiments.exists():
                adapters = list(experiments.glob("*/best_checkpoint/adapter"))
                if adapters:
                    adapter_path = str(adapters[-1])
                    print(f"Using adapter: {adapter_path}")
                else:
                    print("No adapters found. Run training first.")
                    sys.exit(1)

        # Create test code
        test_code = '''def calculate_fibonacci(n, cache=None):
    if cache is None:
        cache = {}

    if n in cache:
        return cache[n]

    if n <= 1:
        return n

    result = calculate_fibonacci(n-1, cache) + calculate_fibonacci(n-2, cache)
    cache[n] = result
    return result'''

        # Save test code
        with open("test_code.py", "w") as f:
            f.write(test_code)

        print("\nTesting adapter with sample code...")
        subprocess.run([
            sys.executable, "-m", "loralab.cli", "test",
            "--adapter", adapter_path,
            "--input", "test_code.py",
            "--task-type", "code_documentation"
        ])

    else:
        print(f"\nInvalid option: {choice}")
        print("Please select a valid option (1-5)")
        sys.exit(1)

    print("\nDemo completed!")
    print("\nNext steps:")
    print("1. Check results in experiments/ directory")
    print("2. Monitor progress: python -m loralab.cli monitor --experiment experiments/[your_experiment]")
    print("3. Test adapter: python -m loralab.cli test --adapter experiments/[your_experiment]/best_checkpoint/adapter")

    # Clean up temp config
    if config.exists():
        config.unlink()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nX Error: {e}")
        sys.exit(1)

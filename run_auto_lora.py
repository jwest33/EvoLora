#!/usr/bin/env python
"""
AutoLoRA - Main entry point for automatic LoRA adapter generation.
Uses direct llama-cpp-python approach (no server required).
"""

import subprocess
import sys
import yaml
from pathlib import Path

def check_llama_cpp_python():
    """Check if llama-cpp-python is installed."""
    try:
        import llama_cpp
        return True
    except ImportError:
        return False

def install_llama_cpp_python():
    """Offer to install llama-cpp-python."""
    print("\n⚠️  llama-cpp-python is not installed.")
    print("This is required for the direct approach (no server needed).")
    print("\nInstall options:")
    print("1. CPU only: pip install llama-cpp-python")
    print("2. CUDA 12.1: pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121")
    print("3. Skip installation (exit)")

    choice = input("\nSelect option (1-3): ")

    if choice == '1':
        print("\nInstalling llama-cpp-python (CPU only)...")
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"], check=True)
        return True
    elif choice == '2':
        print("\nInstalling llama-cpp-python (CUDA 12.1)...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "llama-cpp-python",
            "--extra-index-url", "https://abetlen.github.io/llama-cpp-python/whl/cu121"
        ], check=True)
        return True
    else:
        return False

def main():
    print(" LoRA Adapter Generator Demo (Direct Mode)")
    print("=" * 60)

    # Check for llama-cpp-python
    if not check_llama_cpp_python():
        if not install_llama_cpp_python():
            print("\nCannot proceed without llama-cpp-python. Exiting.")
            sys.exit(1)
        # Verify installation
        if not check_llama_cpp_python():
            print("\nInstallation failed. Please install manually.")
            sys.exit(1)

    print("\n✓ Using direct llama-cpp-python (no server required)")
    print("  Models will alternate to save memory")

    # Check if config exists
    config_path = "loralab/configs/documentation.yaml"
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

    # Save updated config temporarily
    temp_config = Path("loralab/configs/temp_direct_config.yaml")
    temp_config.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)

    # Demo options
    print("\nDemo Options:")
    print("1. Quick test (5 generations)")
    print("2. Standard run (20 generations)")
    print("3. Full run (50 generations)")
    print("4. Test existing adapter")

    choice = input("\nSelect option (1-4): ")

    if choice == '1':
        # Quick test
        print("\n" + "="*60)
        print("Starting Quick Test (5 generations) - Direct Mode")
        print("This will take approximately 5-10 minutes")
        print("="*60)
        subprocess.run([
            sys.executable, "-m", "loralab.cli", "evolve",
            "--config", str(temp_config),
            "--output", "experiments/quick_demo",
            "--generations", "5",
            "--use-direct"  # Force direct mode
        ])

    elif choice == '2':
        # Standard run
        print("\n" + "="*60)
        print("Starting Standard Evolution (20 generations) - Direct Mode")
        print("This will take approximately 20-40 minutes")
        print("="*60)
        subprocess.run([
            sys.executable, "-m", "loralab.cli", "evolve",
            "--config", str(temp_config),
            "--output", "experiments/standard_demo",
            "--generations", "20",
            "--use-direct"  # Force direct mode
        ])

    elif choice == '3':
        # Full run
        print("\n" + "="*60)
        print("Starting Full Evolution (50 generations) - Direct Mode")
        print("This will take approximately 1-2 hours")
        print("="*60)
        subprocess.run([
            sys.executable, "-m", "loralab.cli", "evolve",
            "--config", str(temp_config),
            "--output", "experiments/full_demo",
            "--use-direct"  # Force direct mode
        ])

    elif choice == '4':
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

    print("\nDemo completed!")
    print("\nNext steps:")
    print("1. Check results in experiments/ directory")
    print("2. Monitor progress: python -m loralab.cli monitor --experiment experiments/[your_experiment]")
    print("3. Test adapter: python -m loralab.cli test --adapter experiments/[your_experiment]/best_checkpoint/adapter")

    # Clean up temp config
    if temp_config.exists():
        temp_config.unlink()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nX Error: {e}")
        sys.exit(1)
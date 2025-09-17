#!/usr/bin/env python
"""
LLM Configuration Tuning Script
Helps find optimal settings for both Challenger and Solver models
"""

import time
import json
import yaml
import subprocess
import sys
import torch
import psutil
import GPUtil
from pathlib import Path
from typing import Dict, List, Any, Tuple
import requests
from datetime import datetime


class LLMTuner:
    """Tune LLM configurations for optimal performance."""

    def __init__(self):
        self.results = []
        self.config_path = Path("loralab/config/config.yaml")
        with open(self.config_path, 'r') as f:
            self.base_config = yaml.safe_load(f)

    def test_challenger_config(self, gpu_layers: int = None, context_size: int = None,
                               batch_size: int = None, parallel: int = None) -> Dict[str, Any]:
        """Test a specific Challenger configuration."""
        print("\n" + "="*60)
        print("Testing Challenger Configuration")
        print("="*60)

        config = self.base_config['challenger'].copy()

        # Override with test parameters
        if gpu_layers is not None:
            config['gpu_layers'] = gpu_layers
        if context_size is not None:
            config['context_size'] = context_size
        if batch_size is not None:
            config['batch_size'] = batch_size
        if parallel is not None:
            config['parallel'] = parallel

        print(f"Config: GPU Layers={config.get('gpu_layers', 'default')}, "
              f"Context={config.get('context_size', 'default')}, "
              f"Batch={config.get('batch_size', 'default')}")

        # Start server with config
        server_process = self._start_challenger_server(config)

        if not server_process:
            return {'error': 'Failed to start server'}

        try:
            # Warm up
            print("Warming up...")
            self._test_generation("Write a simple hello world function", max_tokens=100)

            # Test different workloads
            results = {
                'config': config,
                'tests': {}
            }

            # Test 1: Short generation
            print("\nTest 1: Short code generation (100 tokens)")
            start = time.time()
            output = self._test_generation(
                "Generate a Python function that calculates factorial",
                max_tokens=100
            )
            results['tests']['short_gen'] = {
                'time': time.time() - start,
                'tokens': len(output.split()) if output else 0,
                'success': bool(output)
            }
            print(f"  Time: {results['tests']['short_gen']['time']:.2f}s")
            print(f"  Tokens: {results['tests']['short_gen']['tokens']}")

            # Test 2: Medium generation
            print("\nTest 2: Medium code generation (500 tokens)")
            start = time.time()
            output = self._test_generation(
                "Generate a complex Python class with multiple methods",
                max_tokens=500
            )
            results['tests']['medium_gen'] = {
                'time': time.time() - start,
                'tokens': len(output.split()) if output else 0,
                'success': bool(output)
            }
            print(f"  Time: {results['tests']['medium_gen']['time']:.2f}s")
            print(f"  Tokens: {results['tests']['medium_gen']['tokens']}")

            # Test 3: Documentation generation
            print("\nTest 3: Documentation generation (300 tokens)")
            code_sample = """def process_data(data, threshold=0.5):
    filtered = [x for x in data if x > threshold]
    return sum(filtered) / len(filtered) if filtered else 0"""

            start = time.time()
            output = self._test_generation(
                f"Generate detailed documentation for this code:\n{code_sample}",
                max_tokens=300
            )
            results['tests']['doc_gen'] = {
                'time': time.time() - start,
                'tokens': len(output.split()) if output else 0,
                'success': bool(output)
            }
            print(f"  Time: {results['tests']['doc_gen']['time']:.2f}s")
            print(f"  Tokens: {results['tests']['doc_gen']['tokens']}")

            # Monitor resources
            results['resources'] = self._get_resource_usage()
            print(f"\nResource Usage:")
            print(f"  CPU: {results['resources']['cpu_percent']:.1f}%")
            print(f"  RAM: {results['resources']['ram_gb']:.2f} GB")
            print(f"  VRAM: {results['resources']['vram_gb']:.2f} GB")

            # Calculate throughput
            total_time = sum(test['time'] for test in results['tests'].values())
            total_tokens = sum(test['tokens'] for test in results['tests'].values())
            results['throughput'] = {
                'tokens_per_second': total_tokens / total_time if total_time > 0 else 0,
                'avg_time_per_100_tokens': (total_time / total_tokens * 100) if total_tokens > 0 else 0
            }
            print(f"\nThroughput:")
            print(f"  Tokens/sec: {results['throughput']['tokens_per_second']:.1f}")
            print(f"  Time per 100 tokens: {results['throughput']['avg_time_per_100_tokens']:.1f}s")

            return results

        finally:
            # Stop server
            print("\nStopping server...")
            server_process.terminate()
            server_process.wait()

    def test_solver_config(self, rank: int = None, batch_size: int = None,
                          load_in_4bit: bool = True) -> Dict[str, Any]:
        """Test a specific Solver configuration."""
        print("\n" + "="*60)
        print("Testing Solver Configuration")
        print("="*60)

        config = self.base_config['solver'].copy()

        # Override with test parameters
        if rank is not None:
            config['lora_config']['rank'] = rank
        if batch_size is not None:
            self.base_config['training']['batch_size'] = batch_size

        # Ensure quantization config exists
        if 'quantization' not in config:
            config['quantization'] = {}
        config['quantization']['load_in_4bit'] = load_in_4bit

        print(f"Config: LoRA Rank={config['lora_config']['rank']}, "
              f"Batch={batch_size or self.base_config.get('training', {}).get('batch_size', 8)}, "
              f"4bit={load_in_4bit}")

        try:
            # Load model with config
            from loralab.core.solver_client import SolverModel

            print("Loading model...")
            start = time.time()
            solver = SolverModel(config)
            load_time = time.time() - start
            print(f"  Load time: {load_time:.2f}s")

            results = {
                'config': config,
                'load_time': load_time,
                'tests': {}
            }

            # Test inference speed
            test_prompts = [
                "Generate documentation for this function:\ndef add(a, b): return a + b\n\nDocumentation:",
                "Explain this code:\nfor i in range(10): print(i**2)\n\nExplanation:",
                "Complete this function:\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    "
            ]

            print("\nTesting inference speed...")
            for i, prompt in enumerate(test_prompts, 1):
                print(f"  Test {i}/3...", end="")
                start = time.time()
                output = solver.generate(prompt, max_tokens=100, temperature=0.7, num_return=1)
                gen_time = time.time() - start
                results['tests'][f'inference_{i}'] = {
                    'time': gen_time,
                    'tokens': len(output[0].split()) if output else 0
                }
                print(f" {gen_time:.2f}s")

            # Test training speed (mock)
            print("\nTesting LoRA training speed...")
            from loralab.adaptation.lora_solver import LoRASolverTrainer

            # Ensure training config exists with defaults
            training_config = self.base_config.get('training', {})
            if not training_config:
                training_config = {
                    'learning_rate': 1e-5,
                    'batch_size': batch_size or 8,
                    'num_rollouts': 1,  # Use 1 for speed test
                    'kl_penalty': 0.0,
                    'clip_ratio': 0.2,
                    'max_grad_norm': 1.0
                }

            trainer = LoRASolverTrainer(solver, training_config)

            # Create mock data
            from loralab.generation.task_challenger import Task
            mock_tasks = [
                Task(f"task_{i}", f"def func_{i}(): pass", 0.5, {'task_type': 'code_documentation'})
                for i in range(3)  # Reduced for faster test
            ]
            mock_labels = ["Mock documentation"] * 3
            mock_scores = [0.5] * 3

            start = time.time()
            # Note: This will do actual training, but on tiny data
            try:
                metrics = trainer.train_iteration(mock_tasks, mock_labels, mock_scores)
            except Exception as e:
                print(f"  Training test failed: {e}")
                metrics = {'loss': 0, 'error': str(e)}
            train_time = time.time() - start

            results['tests']['training'] = {
                'time': train_time,
                'loss': metrics.get('loss', 0)
            }
            print(f"  Training iteration: {train_time:.2f}s")

            # Get resource usage
            results['resources'] = self._get_resource_usage()
            print(f"\nResource Usage:")
            print(f"  CPU: {results['resources']['cpu_percent']:.1f}%")
            print(f"  RAM: {results['resources']['ram_gb']:.2f} GB")
            print(f"  VRAM: {results['resources']['vram_gb']:.2f} GB")

            return results

        except Exception as e:
            print(f"Error testing solver: {e}")
            return {'error': str(e)}

    def _start_challenger_server(self, config: Dict) -> subprocess.Popen:
        """Start the Challenger server with given config."""
        cmd = [
            config['executable'],
            "-m", config['model_path'],
            "--port", str(config.get('port', 8080))
        ]

        # Add optional parameters
        if 'gpu_layers' in config:
            cmd.extend(["--n-gpu-layers", str(config['gpu_layers'])])
        if 'context_size' in config:
            cmd.extend(["--ctx-size", str(config['context_size'])])
        if 'batch_size' in config:
            cmd.extend(["--n-batch", str(config['batch_size'])])
        if 'parallel' in config:
            cmd.extend(["--parallel", str(config['parallel'])])
        # Add CPU MoE layers if specified
        if 'cpu_moe_layers' in config:
            cmd.extend(["--n-cpu-moe", str(config['cpu_moe_layers'])])
        if config.get('cont_batching'):
            cmd.append("--cont-batching")

        print(f"Starting server: {' '.join(cmd[-6:])}")  # Show last 6 args

        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Wait for server to be ready
            port = config.get('port', 8080)
            for _ in range(30):
                try:
                    response = requests.get(f"http://localhost:{port}/health", timeout=1)
                    if response.status_code == 200:
                        print("Server ready!")
                        return process
                except:
                    time.sleep(1)

            print("Server failed to start")
            process.terminate()
            return None

        except Exception as e:
            print(f"Error starting server: {e}")
            return None

    def _test_generation(self, prompt: str, max_tokens: int) -> str:
        """Test generation with the current server."""
        try:
            response = requests.post(
                "http://localhost:8080/completion",
                json={
                    "prompt": prompt,
                    "n_predict": max_tokens,
                    "temperature": 0.7,
                    "stream": False
                },
                timeout=60
            )
            response.raise_for_status()
            return response.json().get("content", "")
        except Exception as e:
            print(f"Generation error: {e}")
            return ""

    def _get_resource_usage(self) -> Dict[str, float]:
        """Get current resource usage."""
        # CPU and RAM
        cpu_percent = psutil.cpu_percent(interval=1)
        ram = psutil.virtual_memory()

        # GPU
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                vram_used = gpu.memoryUsed / 1024  # Convert to GB
                gpu_util = gpu.load * 100
            else:
                vram_used = 0
                gpu_util = 0
        except:
            vram_used = 0
            gpu_util = 0

        return {
            'cpu_percent': cpu_percent,
            'ram_gb': ram.used / (1024**3),
            'ram_percent': ram.percent,
            'vram_gb': vram_used,
            'gpu_util': gpu_util
        }

    def run_challenger_sweep(self):
        """Run a sweep of Challenger configurations."""
        print("\n" + "="*80)
        print("CHALLENGER CONFIGURATION SWEEP")
        print("="*80)

        # Test different GPU layer counts with CPU MoE offloading
        gpu_layers_options = [12, 20, 30]
        cpu_moe_options = [0, 10, 20, 30]
        context_options = [2048, 8192]

        best_config = None
        best_score = float('inf')

        for gpu_layers in gpu_layers_options:
            for cpu_moe in cpu_moe_options:
                print(f"\n\nTesting GPU Layers={gpu_layers}, CPU MoE={cpu_moe}")
                print("-" * 40)

                # Set cpu_moe_layers in config for test
                test_config = self.base_config['challenger'].copy()
                test_config['gpu_layers'] = gpu_layers
                test_config['cpu_moe_layers'] = cpu_moe

                result = self.test_challenger_config(
                    gpu_layers=gpu_layers,
                    context_size=8192,
                    batch_size=512
                )

                if 'error' not in result:
                    # Score based on speed and resource usage
                    avg_time = sum(test['time'] for test in result['tests'].values()) / len(result['tests'])
                    vram = result['resources']['vram_gb']

                    # Weighted score (lower is better)
                    score = avg_time + (vram * 0.1)  # Penalize VRAM usage slightly

                    if score < best_score:
                        best_score = score
                        best_config = result['config']
                        best_config['cpu_moe_layers'] = cpu_moe  # Save the CPU MoE setting

                    self.results.append(result)

        return best_config

    def run_solver_sweep(self):
        """Run a sweep of Solver configurations."""
        print("\n" + "="*80)
        print("SOLVER CONFIGURATION SWEEP")
        print("="*80)

        rank_options = [4, 8, 16, 32]
        batch_options = [4, 8, 16]

        best_config = None
        best_score = float('inf')

        for rank in rank_options:
            for batch_size in batch_options:
                print(f"\n\nTesting LoRA Rank={rank}, Batch={batch_size}")
                print("-" * 40)

                result = self.test_solver_config(
                    rank=rank,
                    batch_size=batch_size,
                    load_in_4bit=True
                )

                if 'error' not in result:
                    # Score based on speed and quality trade-off
                    train_time = result['tests']['training']['time']
                    vram = result['resources']['vram_gb']

                    # Weighted score
                    score = train_time + (vram * 0.2)

                    if score < best_score:
                        best_score = score
                        best_config = result['config']

                    self.results.append(result)

        return best_config

    def save_results(self):
        """Save tuning results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"tuning_results_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        print(f"\nResults saved to {output_file}")

    def generate_optimized_config(self, challenger_config: Dict, solver_config: Dict):
        """Generate an optimized configuration file."""
        optimized = self.base_config.copy()

        # Update with best configs
        if challenger_config:
            optimized['challenger'].update(challenger_config)
        if solver_config:
            optimized['solver'].update(solver_config)

        # Save optimized config
        with open("loralab/config/optimized.yaml", 'w') as f:
            yaml.dump(optimized, f, default_flow_style=False, sort_keys=False)

        print("\nOptimized configuration saved to loralab/config/optimized.yaml")


def main():
    """Main tuning process."""
    print("LLM Configuration Tuner")
    print("=" * 80)
    print("\nThis will test various configurations to find optimal settings.")
    print("Expected time: 15-30 minutes")

    tuner = LLMTuner()

    # Menu
    print("\nWhat would you like to tune?")
    print("1. Challenger (30B model) only")
    print("2. Solver (4B model) only")
    print("3. Both models")
    print("4. Quick test (single config)")

    choice = input("\nSelect option (1-4): ").strip()

    if choice == '1':
        best_challenger = tuner.run_challenger_sweep()
        tuner.save_results()
        if best_challenger:
            print(f"\nBest Challenger config:")
            print(json.dumps(best_challenger, indent=2))
            tuner.generate_optimized_config(best_challenger, None)

    elif choice == '2':
        best_solver = tuner.run_solver_sweep()
        tuner.save_results()
        if best_solver:
            print(f"\nBest Solver config:")
            print(json.dumps(best_solver, indent=2))
            tuner.generate_optimized_config(None, best_solver)

    elif choice == '3':
        best_challenger = tuner.run_challenger_sweep()
        best_solver = tuner.run_solver_sweep()
        tuner.save_results()

        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)

        if best_challenger:
            print(f"\nBest Challenger config:")
            print(json.dumps(best_challenger, indent=2))

        if best_solver:
            print(f"\nBest Solver config:")
            print(json.dumps(best_solver, indent=2))

        if best_challenger or best_solver:
            tuner.generate_optimized_config(best_challenger, best_solver)

    elif choice == '4':
        # Quick test with current config
        print("\nRunning quick test with current configuration...")
        result = tuner.test_challenger_config()
        print("\nChallenger results:")
        print(json.dumps(result, indent=2))

        result = tuner.test_solver_config()
        print("\nSolver results:")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

"""
Direct llama-cpp-python client for the Challenger model.
Uses the library directly instead of running a separate server process.
"""

import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Check if llama-cpp-python is available
try:
    from llama_cpp import Llama
    LLAMA_CPP_AVAILABLE = True
except ImportError:
    LLAMA_CPP_AVAILABLE = False
    logger.warning("llama-cpp-python not installed. Install with: pip install llama-cpp-python")


class LlamaDirectClient:
    """Direct client using llama-cpp-python library."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize direct llama.cpp client.

        Args:
            config: Configuration dictionary containing:
                - model_path: Path to GGUF model file
                - gpu_layers: Number of layers to offload to GPU
                - context_size: Context window size
                - cpu_moe_layers: Number of MoE layers for CPU
                - n_threads: Number of CPU threads to use
        """
        if not LLAMA_CPP_AVAILABLE:
            raise ImportError(
                "llama-cpp-python is not installed. Install it with:\n"
                "pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121"
            )

        self.model_path = Path(config['model_path'])
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.gpu_layers = config.get('gpu_layers', 20)
        self.context_size = config.get('context_size', 8192)
        self.n_threads = config.get('n_threads', 8)
        self.cpu_moe_layers = config.get('cpu_moe_layers', 20)

        self.model = None
        self._load_model()

    def _load_model(self):
        """Load the model into memory."""
        logger.info(f"Loading model from {self.model_path}")
        start_time = time.time()

        try:
            self.model = Llama(
                model_path=str(self.model_path),
                n_gpu_layers=self.gpu_layers,
                n_ctx=self.context_size,
                n_threads=self.n_threads,
                n_batch=512,
                verbose=False,
                # Note: cpu_moe_layers might not be directly supported
                # in llama-cpp-python, depends on version
            )

            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def generate(self,
                 prompt: str,
                 max_tokens: int = 2048,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 stop: Optional[List[str]] = None) -> str:
        """
        Generate text using direct llama.cpp.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated text
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            # Direct generation using llama-cpp-python
            response = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop or [],
                echo=False  # Don't include prompt in output
            )

            # Extract generated text
            if isinstance(response, dict) and 'choices' in response:
                return response['choices'][0]['text']
            else:
                return str(response)

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return ""

    def evaluate_responses(self,
                          responses: List[str],
                          criteria: str) -> List[float]:
        """
        Evaluate multiple responses based on criteria.

        Args:
            responses: List of responses to evaluate
            criteria: Evaluation criteria description

        Returns:
            List of scores (0.0 to 1.0)
        """
        scores = []
        for response in responses:
            prompt = f"""Evaluate the following response based on: {criteria}

Response:
{response}

Provide a score from 0.0 to 1.0 where 1.0 is perfect.
Output only the numerical score.

Score: """

            score_text = self.generate(prompt, max_tokens=5, temperature=0.1)
            try:
                import re
                match = re.search(r'([0-9]*\.?[0-9]+)', score_text.strip())
                if match:
                    score = float(match.group(1))
                    score = max(0.0, min(1.0, score))
                else:
                    score = 0.5
            except (ValueError, AttributeError):
                logger.warning(f"Failed to parse score: {score_text}")
                score = 0.5
            scores.append(score)

        return scores

    def unload(self):
        """Unload the model from memory."""
        if self.model:
            del self.model
            self.model = None
            logger.info("Model unloaded from memory")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.unload()


class LlamaDirectBenchmark:
    """Benchmark tool to compare server vs direct approaches."""

    @staticmethod
    def benchmark_server(config: Dict[str, Any], test_prompts: List[str]) -> Dict[str, float]:
        """Benchmark the server approach."""
        from loralab.core.llama_client import LlamaCppClient

        results = {}
        client = LlamaCppClient(config)

        # Start server
        start_time = time.time()
        client.start_server()
        results['startup_time'] = time.time() - start_time

        # Test generations
        gen_times = []
        for prompt in test_prompts:
            start = time.time()
            output = client.generate(prompt, max_tokens=200)
            gen_times.append(time.time() - start)

        results['avg_generation_time'] = sum(gen_times) / len(gen_times)
        results['total_time'] = results['startup_time'] + sum(gen_times)

        # Cleanup
        client.stop_server()

        return results

    @staticmethod
    def benchmark_direct(config: Dict[str, Any], test_prompts: List[str]) -> Dict[str, float]:
        """Benchmark the direct approach."""
        results = {}

        # Load model
        start_time = time.time()
        client = LlamaDirectClient(config)
        results['startup_time'] = time.time() - start_time

        # Test generations
        gen_times = []
        for prompt in test_prompts:
            start = time.time()
            output = client.generate(prompt, max_tokens=200)
            gen_times.append(time.time() - start)

        results['avg_generation_time'] = sum(gen_times) / len(gen_times)
        results['total_time'] = results['startup_time'] + sum(gen_times)

        # Cleanup
        client.unload()

        return results

    @staticmethod
    def compare(config: Dict[str, Any], num_tests: int = 10):
        """Run comparative benchmark."""
        print("\n" + "="*60)
        print("LLAMA.CPP SERVER vs DIRECT COMPARISON")
        print("="*60)

        # Create test prompts
        test_prompts = [
            f"Write a Python function that calculates the {i}th Fibonacci number"
            for i in range(num_tests)
        ]

        # Test server approach
        print("\nTesting SERVER approach...")
        server_results = LlamaDirectBenchmark.benchmark_server(config, test_prompts)

        # Test direct approach
        print("\nTesting DIRECT approach...")
        direct_results = LlamaDirectBenchmark.benchmark_direct(config, test_prompts)

        # Compare results
        print("\n" + "-"*60)
        print("RESULTS:")
        print("-"*60)
        print(f"{'Metric':<30} {'Server':<15} {'Direct':<15} {'Difference'}")
        print("-"*60)

        for metric in ['startup_time', 'avg_generation_time', 'total_time']:
            server_val = server_results[metric]
            direct_val = direct_results[metric]
            diff = ((direct_val - server_val) / server_val) * 100

            print(f"{metric:<30} {server_val:<15.3f} {direct_val:<15.3f} {diff:+.1f}%")

        print("-"*60)

        # Recommendation
        if direct_results['total_time'] < server_results['total_time'] * 0.8:
            print("\n✓ RECOMMENDATION: Use DIRECT approach (>20% faster)")
        elif server_results['total_time'] < direct_results['total_time'] * 0.8:
            print("\n✓ RECOMMENDATION: Use SERVER approach (>20% faster)")
        else:
            print("\n✓ RECOMMENDATION: Either approach is fine (similar performance)")

        return server_results, direct_results


def main():
    """Run benchmark comparison."""
    import yaml

    # Load config
    with open("loralab/config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    challenger_config = config['challenger']

    # Run comparison
    server_results, direct_results = LlamaDirectBenchmark.compare(
        challenger_config,
        num_tests=5
    )


if __name__ == "__main__":
    main()

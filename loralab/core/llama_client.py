"""
LLaMA.cpp client for the Challenger model (Qwen3-30B).
Handles communication with llama-server for task generation and evaluation.
"""

import json
import time
import subprocess
import requests
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class LlamaCppClient:
    """Client for interacting with llama.cpp server."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LlamaCppClient.

        Args:
            config: Configuration dictionary containing:
                - executable: Path to llama-server.exe
                - model_path: Path to GGUF model file
                - gpu_layers: Number of layers to offload to GPU
                - port: Server port (default 8080)
                - context_size: Context window size (default 8192)
        """
        self.config = config  # Store config for later use
        self.executable = Path(config['executable'])
        self.model_path = Path(config['model_path'])
        self.gpu_layers = config.get('gpu_layers', 12)  # Default reduced for VRAM
        self.port = config.get('port', 8080)
        self.context_size = config.get('context_size', 4096)  # Default reduced
        self.parallel = config.get('parallel', 2)  # Default reduced for memory
        self.batch_size = config.get('batch_size', 512)  # Batch size optimization
        self.cpu_moe_layers = config.get('cpu_moe_layers', None)  # Number of MoE layers for CPU
        self.cont_batching = config.get('cont_batching', True)  # Continuous batching

        self.base_url = f"http://localhost:{self.port}"
        self.server_process = None

    def start_server(self) -> None:
        """Start the llama.cpp server."""
        if self.server_process is not None:
            logger.warning("Server already running")
            return

        cmd = [
            str(self.executable),
            "-m", str(self.model_path),
            "--port", str(self.port)
        ]

        # Add optional parameters only if they were provided in config
        if 'gpu_layers' in self.config:
            cmd.extend(["--n-gpu-layers", str(self.gpu_layers)])
        if 'context_size' in self.config:
            cmd.extend(["--ctx-size", str(self.context_size)])
        if 'parallel' in self.config:
            cmd.extend(["--parallel", str(self.parallel)])
        if 'batch_size' in self.config:
            cmd.extend(["--n-batch", str(self.batch_size)])

        # Add CPU MoE layers if specified
        if 'cpu_moe_layers' in self.config and self.config['cpu_moe_layers']:
            cmd.extend(["--n-cpu-moe", str(self.config['cpu_moe_layers'])])

        # Add continuous batching if enabled
        if self.config.get('cont_batching', False):
            cmd.append("--cont-batching")

        logger.info(f"Starting llama.cpp server with command: {' '.join(cmd)}")
        self.server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Wait for server to start
        time.sleep(5)
        self._wait_for_server()

    def _wait_for_server(self, timeout: int = 30) -> None:
        """Wait for server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}/health")
                if response.status_code == 200:
                    logger.info("LlamaCpp server is ready")
                    return
            except requests.exceptions.ConnectionError:
                time.sleep(1)
        raise TimeoutError(f"Server failed to start within {timeout} seconds")

    def stop_server(self) -> None:
        """Stop the llama.cpp server."""
        if self.server_process is not None:
            self.server_process.terminate()
            self.server_process.wait()
            self.server_process = None
            logger.info("LlamaCpp server stopped")

    def generate(self,
                 prompt: str,
                 max_tokens: int = 2048,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 stop: Optional[List[str]] = None) -> str:
        """
        Generate text using the llama.cpp server.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            stop: Stop sequences

        Returns:
            Generated text
        """
        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stop": stop or [],
            "stream": False
        }

        print(f"               [Challenger] Processing request (max {max_tokens} tokens)...")
        response = requests.post(
            f"{self.base_url}/completion",
            json=payload,
            timeout=120
        )
        response.raise_for_status()

        result = response.json()
        generated = result.get("content", "")
        print(f"               [Challenger] Generated {len(generated.split())} words")
        return generated

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

            score_text = self.generate(prompt, max_tokens=10, temperature=0.1)
            try:
                score = float(score_text.strip())
                score = max(0.0, min(1.0, score))
            except ValueError:
                logger.warning(f"Failed to parse score: {score_text}")
                score = 0.5
            scores.append(score)

        return scores

    def __enter__(self):
        """Context manager entry."""
        self.start_server()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_server()
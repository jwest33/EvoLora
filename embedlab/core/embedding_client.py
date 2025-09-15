"""Embedding Client for interacting with local embedding server."""
from __future__ import annotations
import logging
from typing import List, Optional, Union
import numpy as np
import requests
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class EmbeddingConfig:
    """Configuration for embedding client."""
    base_url: str = "http://localhost:8002"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    batch_size: int = 32
    normalize: bool = True
    timeout: int = 30

class EmbeddingClient:
    """Client for interacting with local embedding server."""

    def __init__(self, config: Optional[EmbeddingConfig] = None):
        self.config = config or EmbeddingConfig()
        self.session = requests.Session()

    def embed(
        self,
        texts: Union[str, List[str]],
        instruction: Optional[str] = None,
        is_query: bool = True
    ) -> np.ndarray:
        """
        Generate embeddings for texts.

        Args:
            texts: Single text or list of texts
            instruction: Optional instruction for query embedding
            is_query: Whether these are queries (True) or documents (False)

        Returns:
            Numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Prepare texts with instruction if provided
        if instruction and is_query:
            formatted_texts = [
                f"Instruct: {instruction}\nQuery: {text}"
                for text in texts
            ]
        else:
            formatted_texts = texts

        # Process in batches if needed
        all_embeddings = []
        for i in range(0, len(formatted_texts), self.config.batch_size):
            batch = formatted_texts[i:i + self.config.batch_size]
            embeddings = self._embed_batch(batch)
            all_embeddings.extend(embeddings)

        result = np.array(all_embeddings)

        if self.config.normalize:
            # Normalize to unit vectors
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / (norms + 1e-12)

        return result

    def _embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of texts."""
        payload = {
            "input": texts,
            "model": self.config.model,
            "encoding_format": "float"
        }

        try:
            response = self.session.post(
                f"{self.config.base_url}/v1/embeddings",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()

            # Extract embeddings from response
            embeddings = [item["embedding"] for item in data["data"]]
            return embeddings

        except requests.RequestException as e:
            logger.error(f"Embedding request failed: {e}")
            # Return zero vectors as fallback
            return [[0.0] * 768 for _ in texts]  # Assuming 768-dim embeddings

    def embed_query(
        self,
        texts: Union[str, List[str]],
        instruction: Optional[str] = None
    ) -> np.ndarray:
        """Embed query texts with optional instruction."""
        return self.embed(texts, instruction, is_query=True)

    def embed_document(
        self,
        texts: Union[str, List[str]]
    ) -> np.ndarray:
        """Embed document texts without instruction."""
        return self.embed(texts, instruction=None, is_query=False)

    def similarity(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray
    ) -> np.ndarray:
        """
        Compute cosine similarity between embeddings.

        Args:
            embeddings1: First set of embeddings (n, d)
            embeddings2: Second set of embeddings (m, d)

        Returns:
            Similarity matrix (n, m)
        """
        # Ensure 2D arrays
        if embeddings1.ndim == 1:
            embeddings1 = embeddings1.reshape(1, -1)
        if embeddings2.ndim == 1:
            embeddings2 = embeddings2.reshape(1, -1)

        # Compute cosine similarity
        return embeddings1 @ embeddings2.T

    def health_check(self) -> bool:
        """Check if embedding server is healthy."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            # Try alternative health endpoint
            try:
                response = self.session.post(
                    f"{self.config.base_url}/v1/embeddings",
                    json={"input": ["test"], "model": self.config.model},
                    timeout=5
                )
                return response.status_code == 200
            except:
                return False

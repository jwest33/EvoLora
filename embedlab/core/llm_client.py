"""LLM Client for interacting with local Qwen3-4B server."""
from __future__ import annotations
import json
import logging
from typing import Dict, List, Optional, Any
import requests
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class LLMConfig:
    """Configuration for LLM client."""
    base_url: str = "http://localhost:8000"
    model: str = "qwen3-4b-instruct"
    temperature: float = 0.7
    max_tokens: int = 2048
    top_p: float = 0.9
    timeout: int = 60

@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    model: str = ""

class LLMClient:
    """Client for interacting with local LLM server."""

    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.session = requests.Session()

    def complete(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_mode: bool = False
    ) -> LLMResponse:
        """
        Generate completion from LLM.

        Args:
            prompt: User prompt
            system: System message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            json_mode: Whether to enforce JSON output

        Returns:
            LLMResponse with generated content
        """
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens,
            "top_p": self.config.top_p,
            "stream": False
        }

        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response = self.session.post(
                f"{self.config.base_url}/v1/chat/completions",
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            data = response.json()

            return LLMResponse(
                content=data["choices"][0]["message"]["content"],
                usage=data.get("usage", {}),
                model=data.get("model", self.config.model)
            )
        except requests.RequestException as e:
            logger.error(f"LLM request failed: {e}")
            raise

    def batch_complete(
        self,
        prompts: List[str],
        system: Optional[str] = None,
        **kwargs
    ) -> List[LLMResponse]:
        """Generate completions for multiple prompts."""
        responses = []
        for prompt in prompts:
            try:
                response = self.complete(prompt, system, **kwargs)
                responses.append(response)
            except Exception as e:
                logger.error(f"Failed to process prompt: {e}")
                responses.append(LLMResponse(content="", usage={}))
        return responses

    def generate_json(
        self,
        prompt: str,
        system: Optional[str] = None,
        schema: Optional[Dict] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate JSON output from LLM.

        Args:
            prompt: User prompt
            system: System message
            schema: Optional JSON schema to include in prompt
            **kwargs: Additional arguments for complete()

        Returns:
            Parsed JSON dictionary
        """
        if schema:
            prompt = f"{prompt}\n\nPlease respond with valid JSON matching this schema:\n{json.dumps(schema, indent=2)}"

        response = self.complete(prompt, system, json_mode=True, **kwargs)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response content: {response.content}")
            return {}

    def health_check(self) -> bool:
        """Check if LLM server is healthy."""
        try:
            response = self.session.get(
                f"{self.config.base_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

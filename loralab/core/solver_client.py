"""
Solver model client with LoRA adapter support.
Manages the Qwen3-4B model with dynamic LoRA adapters.
Supports both HuggingFace models and GGUF files via llama.cpp.
"""

import torch
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging
import subprocess
import requests
import time
import json
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)

logger = logging.getLogger(__name__)


class SolverModel:
    """Solver model with LoRA adapter support."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize SolverModel.

        Args:
            config: Configuration dictionary containing:
                - model_name: Name or path of base model
                - device: Device to use (cuda/cpu)
                - lora_config: LoRA configuration parameters
                - quantization: Optional quantization config
        """
        self.model_name = config['model_name']
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.lora_config = config['lora_config']

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Quantization config if specified
        quantization_config = None
        if config.get('quantization', {}).get('enabled', False):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=config['quantization'].get('load_in_8bit', False),
                load_in_4bit=config['quantization'].get('load_in_4bit', True),
                bnb_4bit_compute_dtype=torch.float16
            )

        # Load base model
        logger.info(f"Loading base model: {self.model_name}")
        self.base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32,
            device_map='auto' if self.device == 'cuda' else None,
            quantization_config=quantization_config,
            trust_remote_code=True
        )

        # Initialize with LoRA
        self.model = self._initialize_lora()
        self.current_adapter = "default"

    def _initialize_lora(self) -> PeftModel:
        """Initialize model with LoRA configuration."""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=self.lora_config['rank'],
            lora_alpha=self.lora_config['alpha'],
            lora_dropout=self.lora_config.get('dropout', 0.1),
            target_modules=self.lora_config.get('target_modules',
                ["q_proj", "v_proj", "k_proj", "o_proj"])
        )

        logger.info("Initializing LoRA adapter")
        model_with_lora = get_peft_model(self.base_model, lora_config)
        model_with_lora.print_trainable_parameters()
        return model_with_lora

    def generate(self,
                 prompt: str,
                 max_tokens: int = 512,
                 temperature: float = 0.7,
                 top_p: float = 0.9,
                 num_return: int = 1) -> List[str]:
        """
        Generate responses using the solver model.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            num_return: Number of responses to generate

        Returns:
            List of generated responses
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                num_return_sequences=num_return,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        responses = []
        for output in outputs:
            response = self.tokenizer.decode(
                output[inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            )
            responses.append(response)

        return responses

    def get_logits(self, prompt: str) -> torch.Tensor:
        """
        Get model logits for a prompt.

        Args:
            prompt: Input prompt

        Returns:
            Logits tensor
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        return outputs.logits

    def compute_perplexity(self, text: str) -> float:
        """
        Compute perplexity for given text.

        Args:
            text: Input text

        Returns:
            Perplexity score
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs.input_ids)

        return torch.exp(outputs.loss).item()

    def save_adapter(self, save_path: str, adapter_name: Optional[str] = None):
        """
        Save LoRA adapter to disk.

        Args:
            save_path: Path to save adapter
            adapter_name: Name of adapter to save (default: current)
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        if adapter_name:
            self.model.set_adapter(adapter_name)

        logger.info(f"Saving adapter to {save_path}")
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def load_adapter(self, load_path: str, adapter_name: str = "loaded"):
        """
        Load LoRA adapter from disk.

        Args:
            load_path: Path to load adapter from
            adapter_name: Name to give loaded adapter
        """
        logger.info(f"Loading adapter from {load_path}")
        self.model = PeftModel.from_pretrained(
            self.base_model,
            load_path,
            adapter_name=adapter_name
        )
        self.current_adapter = adapter_name

    def add_adapter(self, adapter_name: str, lora_config: Optional[Dict] = None):
        """
        Add a new LoRA adapter.

        Args:
            adapter_name: Name for the new adapter
            lora_config: Optional LoRA configuration (uses default if None)
        """
        if lora_config is None:
            lora_config = self.lora_config

        config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config['rank'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config.get('dropout', 0.1),
            target_modules=lora_config.get('target_modules',
                ["q_proj", "v_proj", "k_proj", "o_proj"])
        )

        logger.info(f"Adding adapter: {adapter_name}")
        self.model.add_adapter(adapter_name, config)
        self.current_adapter = adapter_name

    def set_adapter(self, adapter_name: str):
        """
        Switch to a specific adapter.

        Args:
            adapter_name: Name of adapter to activate
        """
        logger.info(f"Switching to adapter: {adapter_name}")
        self.model.set_adapter(adapter_name)
        self.current_adapter = adapter_name

    def merge_and_unload(self) -> AutoModelForCausalLM:
        """
        Merge LoRA weights with base model and unload LoRA.

        Returns:
            Base model with merged weights
        """
        logger.info("Merging LoRA weights with base model")
        return self.model.merge_and_unload()

    def get_trainable_parameters(self) -> Tuple[int, int]:
        """
        Get number of trainable parameters.

        Returns:
            Tuple of (trainable_params, all_params)
        """
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        return trainable_params, all_param
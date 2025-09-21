"""R-Zero Configuration for Self-Evolving Reasoning Models

Configuration for the R-Zero framework implementing Challenger-Solver co-evolution
based on the paper "R-Zero: Self-Evolving Reasoning LLM from Zero Data"
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class ChallengerConfig:
    """Configuration for the Challenger agent that generates difficult questions"""

    # Model configuration
    model_name: str = "unsloth/Qwen3-4B-unsloth-bnb-4bit"  # Pre-quantized Qwen3 4B model
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "load_in_4bit": True,
        "use_gradient_checkpointing": "unsloth",
        "max_seq_length": 256,  # Extremely reduced to save memory
    })

    # GRPO training parameters
    learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    max_steps: int = 5
    batch_size: int = 1  # Reduced for memory
    gradient_accumulation_steps: int = 128  # Effective batch = 1

    # Generation parameters
    temperature: float = 1.0
    top_p: float = 0.99
    max_prompt_length: int = 64  # Extremely reduced for memory
    max_completion_length: int = 64  # Extremely reduced for memory
    num_generations_per_prompt: int = 2  # GRPO minimum requirement

    # Reward configuration
    uncertainty_target: float = 0.5  # Target solver accuracy for maximum reward
    repetition_penalty_weight: float = 1.0
    bleu_threshold: float = 0.5  # For clustering similar questions
    format_reward_weight: float = 0.5

    # Sampling configuration
    num_questions_per_iteration: int = 1000  # Reduced for memory (was 8000)
    num_solver_responses: int = 5  # Reduced for memory (was 10)


@dataclass
class SolverConfig:
    """Configuration for the Solver agent that learns to solve problems"""

    # Model configuration
    model_name: str = "unsloth/gemma-3-270m-it-unsloth-bnb-4bit"  # Pre-quantized Gemma3 270M model
    model_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        "load_in_4bit": True,
        "use_gradient_checkpointing": "unsloth",
        "max_seq_length": 256,  # Extremely reduced to save memory
    })

    # GRPO training parameters
    learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    warmup_ratio: float = 0.1
    max_steps: int = 15
    batch_size: int = 1  # Reduced for memory
    gradient_accumulation_steps: int = 128  # Effective batch = 1

    # Generation parameters for training
    temperature: float = 0.1  # Lower for more focused generation
    top_p: float = 0.9
    max_prompt_length: int = 64  # Extremely reduced for memory
    max_completion_length: int = 48  # Extremely reduced for memory
    num_generations_per_prompt: int = 2  # GRPO minimum requirement

    # Dataset filtering
    difficulty_filter_delta: float = 0.25  # δ in paper, keeps 25%-75% accuracy
    min_accuracy: float = 0.25  # Minimum solver accuracy to include
    max_accuracy: float = 0.75  # Maximum solver accuracy to include

    # Pseudo-label generation
    num_responses_for_voting: int = 10  # m for majority voting
    min_consensus_ratio: float = 0.3  # Minimum agreement for pseudo-label


@dataclass
class EvolutionConfig:
    """Configuration for the co-evolution loop"""

    # Evolution parameters
    num_iterations: int = 3  # Number of co-evolution cycles
    initial_difficulty: float = 0.3  # Starting difficulty level
    difficulty_increment: float = 0.1  # How much to increase per iteration

    # Checkpointing
    save_checkpoints: bool = True
    checkpoint_dir: str = "checkpoints/rzero"
    save_intermediate: bool = True  # Save after each agent training

    # Monitoring
    log_samples: bool = True
    samples_per_iteration: int = 10  # Number of examples to log
    track_pseudo_label_accuracy: bool = True

    # Early stopping
    enable_early_stopping: bool = True
    min_improvement: float = 0.01  # Minimum improvement to continue
    patience: int = 2  # Iterations without improvement before stopping


@dataclass
class GSM8KConfig:
    """Configuration specific to GSM8K dataset"""

    # Dataset parameters
    dataset_name: str = "gsm8k"
    subset: str = "main"  # or "socratic" for step-by-step version
    test_split_ratio: float = 0.1  # Fraction to use for evaluation

    # Prompt formatting
    system_prompt: str = """You are a mathematical reasoning assistant. Solve problems step-by-step.

Show your work clearly and end with:
Final Answer: [numeric value]"""

    # Answer extraction
    answer_pattern: str = r"Final Answer:\s*([-\d,.]+)"
    extract_numeric_only: bool = True
    tolerance: float = 0.01  # For numeric comparison


@dataclass
class CUDAConfig:
    """CUDA optimization settings for RTX 5060 Ti 16GB"""

    # Memory optimization
    use_flash_attention: bool = True
    gradient_checkpointing: bool = True
    mixed_precision: str = "bf16"  # or "fp16"

    # Batch size tuning
    max_batch_size_challenger: int = 1  # Minimal for memory
    max_batch_size_solver: int = 1  # Minimal for memory

    # VRAM management
    empty_cache_frequency: int = 10  # More frequent clearing
    max_memory_reserved: float = 0.85  # Leave more headroom

    # Compilation
    use_torch_compile: bool = False  # Disable for GRPO stability
    compile_mode: str = "default"  # "default", "reduce-overhead", "max-autotune"


@dataclass
class RZeroConfig:
    """Main configuration combining all components"""

    challenger: ChallengerConfig = field(default_factory=ChallengerConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    gsm8k: GSM8KConfig = field(default_factory=GSM8KConfig)
    cuda: CUDAConfig = field(default_factory=CUDAConfig)

    # Global settings
    seed: int = 42
    experiment_name: str = "rzero_gsm8k"
    output_dir: str = "outputs/rzero"
    use_wandb: bool = False
    wandb_project: str = "rzero-evolution"

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "challenger": self.challenger.__dict__,
            "solver": self.solver.__dict__,
            "evolution": self.evolution.__dict__,
            "gsm8k": self.gsm8k.__dict__,
            "cuda": self.cuda.__dict__,
            "seed": self.seed,
            "experiment_name": self.experiment_name,
            "output_dir": self.output_dir,
            "use_wandb": self.use_wandb,
            "wandb_project": self.wandb_project,
        }

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RZeroConfig":
        """Create config from dictionary"""
        config = cls()

        # Update challenger config
        if "challenger" in config_dict:
            for key, value in config_dict["challenger"].items():
                setattr(config.challenger, key, value)

        # Update solver config
        if "solver" in config_dict:
            for key, value in config_dict["solver"].items():
                setattr(config.solver, key, value)

        # Update evolution config
        if "evolution" in config_dict:
            for key, value in config_dict["evolution"].items():
                setattr(config.evolution, key, value)

        # Update GSM8K config
        if "gsm8k" in config_dict:
            for key, value in config_dict["gsm8k"].items():
                setattr(config.gsm8k, key, value)

        # Update CUDA config
        if "cuda" in config_dict:
            for key, value in config_dict["cuda"].items():
                setattr(config.cuda, key, value)

        # Update global settings
        for key in ["seed", "experiment_name", "output_dir", "use_wandb", "wandb_project"]:
            if key in config_dict:
                setattr(config, key, config_dict[key])

        return config

    def update_for_hardware(self, vram_gb: int = 16):
        """Adjust settings based on available VRAM"""
        if vram_gb < 12:
            # Small VRAM adjustments
            self.cuda.max_batch_size_challenger = 1
            self.cuda.max_batch_size_solver = 1
            self.challenger.gradient_accumulation_steps = 128
            self.solver.gradient_accumulation_steps = 128
            self.cuda.gradient_checkpointing = True
        elif vram_gb < 24:
            # Medium VRAM (RTX 5060 Ti 16GB)
            self.cuda.max_batch_size_challenger = 4
            self.cuda.max_batch_size_solver = 4
            self.challenger.gradient_accumulation_steps = 32
            self.solver.gradient_accumulation_steps = 32
        else:
            # Large VRAM
            self.cuda.max_batch_size_challenger = 8
            self.cuda.max_batch_size_solver = 8
            self.challenger.gradient_accumulation_steps = 16
            self.solver.gradient_accumulation_steps = 16


def create_default_config() -> RZeroConfig:
    """Create default R-Zero configuration"""
    config = RZeroConfig()

    # Override with specific models requested by user
    config.challenger.model_name = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    config.solver.model_name = "unsloth/gemma-3-270m-it-unsloth-bnb-4bit"

    # Adjust for 16GB VRAM
    config.update_for_hardware(vram_gb=16)

    return config


def load_config(config_path: Optional[str] = None) -> RZeroConfig:
    """Load configuration from file or create default"""
    if config_path:
        import json
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return RZeroConfig.from_dict(config_dict)
    return create_default_config()


def save_config(config: RZeroConfig, config_path: str):
    """Save configuration to file"""
    import json
    import os

    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)

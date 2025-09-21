"""R-Zero implementation using Gemma models for both Challenger and Solver"""
from unsloth import FastLanguageModel
import os
import sys
import torch
import gc
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, Trainer
import json
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loralab.utils.cli_formatter import CLIFormatter, SpinnerProgress

# Disable multiprocessing to avoid Windows issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def clear_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

class ChallengerAgent:
    """Qwen3-4B Challenger that generates difficult problems"""

    def __init__(self):
        CLIFormatter.print_info("Initializing Challenger...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
            max_seq_length=1024,
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False
        )

        # Apply LoRA for fine-tuning
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None  # And LoftQ
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        CLIFormatter.print_success(f"Challenger loaded: {trainable:,} trainable params")

    def generate_problems(self, num_problems: int, difficulty: float = 0.5) -> List[Dict]:
        """Generate math problems based on difficulty level"""
        problems = []

        # Create prompts for different difficulty levels
        if difficulty < 0.3:
            prompt_template = "Generate a simple arithmetic problem (addition or subtraction) with single digits. Problem:"
        elif difficulty < 0.7:
            prompt_template = "Generate a moderately complex math problem involving multiplication or two-step operations. Problem:"
        else:
            prompt_template = "Generate a challenging math problem involving fractions, percentages, or multi-step word problems. Problem:"

        self.model.eval()
        for _ in tqdm(range(num_problems), desc="Generating problems"):
            inputs = self.tokenizer(prompt_template, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                )

            problem_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract just the problem part (after "Problem:")
            if "Problem:" in problem_text:
                problem_text = problem_text.split("Problem:")[-1].strip()

            problems.append({
                "question": problem_text,
                "difficulty": difficulty
            })

        return problems

    def train_on_feedback(self, dataset: Dataset, solver_accuracy: float):
        """Train Challenger based on Solver's performance"""
        CLIFormatter.print_status("Training Challenger", f"Solver accuracy: {solver_accuracy:.2%}")

        # Adjust reward based on solver accuracy
        # Optimal when Solver achieves 60-70% accuracy
        if 0.6 <= solver_accuracy <= 0.7:
            reward_multiplier = 1.0  # Optimal
        elif solver_accuracy < 0.6:
            reward_multiplier = 0.5  # Problems too hard
        else:
            reward_multiplier = 0.5  # Problems too easy

        # Manual tokenization to avoid multiprocessing issues
        tokenized_data = []
        for example in dataset:
            text = f"Generate a problem that is {'easy' if solver_accuracy > 0.7 else 'harder'}: {example['question']}"
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors=None,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            tokenized_data.append(tokens)

        train_dataset = Dataset.from_list(tokenized_data)

        training_args = TrainingArguments(
            output_dir="outputs/challenger",
            per_device_train_batch_size=1,
            max_steps=10,  # Quick training
            learning_rate=2e-5 * reward_multiplier,
            optim="adamw_8bit",
            logging_steps=1,
            report_to="none",
            dataloader_num_workers=0,
            fp16=True,
            bf16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        CLIFormatter.print_success("Challenger training completed")

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        clear_memory()


class SolverAgent:
    """Gemma-270m Solver that solves problems"""

    def __init__(self):
        CLIFormatter.print_info("Initializing Solver (Gemma-270m)...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3-270m-it",
            max_seq_length=512,
            load_in_4bit=False,  # Use float32 to avoid dtype issues
            dtype=torch.float32,
        )

        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,  # Disable to avoid dtype issues
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        CLIFormatter.print_success(f"Solver loaded: {trainable:,} trainable params")

    def solve_problems(self, problems: List[Dict]) -> List[Dict]:
        """Generate solutions for problems"""
        results = []
        self.model.eval()

        for problem in tqdm(problems, desc="Solving problems"):
            prompt = f"Question: {problem['question']}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Generate multiple solutions for uncertainty estimation
                solutions = []
                for _ in range(2):  # Generate 2 solutions
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=50,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )
                    answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                    # Extract answer part
                    if "Answer:" in answer:
                        answer = answer.split("Answer:")[-1].strip()
                    solutions.append(answer)

            # Calculate uncertainty (simplified - based on answer consistency)
            uncertainty = 0.0 if solutions[0] == solutions[1] else 1.0

            results.append({
                "question": problem["question"],
                "answer": solutions[0],
                "all_answers": solutions,
                "uncertainty": uncertainty,
                "difficulty": problem.get("difficulty", 0.5)
            })

        return results

    def train_on_problems(self, dataset: Dataset):
        """Train Solver on problem-solution pairs"""
        CLIFormatter.print_info("Training Solver...")

        # Manual tokenization
        tokenized_data = []
        for example in dataset:
            text = f"Question: {example['question']}\nAnswer: {example.get('answer', 'Unknown')}"
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors=None,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            tokenized_data.append(tokens)

        train_dataset = Dataset.from_list(tokenized_data)

        training_args = TrainingArguments(
            output_dir="outputs/solver",
            per_device_train_batch_size=1,
            max_steps=10,
            learning_rate=2e-5,
            optim="adamw_torch",  # Regular optimizer for float32
            logging_steps=1,
            report_to="none",
            dataloader_num_workers=0,
            fp16=False,  # No mixed precision
            bf16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        CLIFormatter.print_success("Solver training completed")

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        clear_memory()


def run_rzero_evolution(num_iterations: int = 5):
    """Main R-Zero evolution loop"""
    CLIFormatter.print_header("R-Zero Training with Gemma Models")

    # Check CUDA
    if torch.cuda.is_available():
        CLIFormatter.print_status("CUDA Device", torch.cuda.get_device_name(0))
        CLIFormatter.print_status("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        CLIFormatter.print_warning("CUDA not available - training will be slow")

    # Initialize with GSM8K for bootstrapping
    CLIFormatter.print_info("Loading GSM8K dataset for bootstrapping...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train[:100]")  # Small subset

    # Prepare initial dataset
    initial_data = []
    for example in gsm8k:
        initial_data.append({
            "question": example["question"],
            "answer": example["answer"].split("####")[-1].strip() if "####" in example["answer"] else example["answer"]
        })

    current_dataset = Dataset.from_list(initial_data)
    difficulty = 0.3  # Start with easy problems

    for iteration in range(num_iterations):
        CLIFormatter.print_generation_header(iteration + 1, num_iterations)
        CLIFormatter.print_metric("Current Difficulty", difficulty, good_threshold=0.7, bad_threshold=0.3)

        # Phase 1: Solver training
        CLIFormatter.print_subheader("Phase 1: Solver Training")
        solver = SolverAgent()
        solver.train_on_problems(current_dataset)

        # Evaluate Solver
        test_problems = current_dataset.select(range(min(10, len(current_dataset))))
        results = solver.solve_problems(test_problems.to_list())

        # Calculate accuracy (simplified - would need ground truth in real scenario)
        avg_uncertainty = sum(r["uncertainty"] for r in results) / len(results)
        solver_accuracy = 1.0 - avg_uncertainty  # Simplified metric
        CLIFormatter.print_metric("Solver Accuracy", solver_accuracy * 100, "%",
                                  good_threshold=70, bad_threshold=40)

        # Clean up Solver
        solver.cleanup()

        # Phase 2: Challenger training and problem generation
        CLIFormatter.print_subheader("Phase 2: Challenger Training")
        challenger = ChallengerAgent()
        challenger.train_on_feedback(current_dataset, solver_accuracy)

        # Generate new problems with adjusted difficulty
        if solver_accuracy > 0.7:
            difficulty = min(1.0, difficulty + 0.1)  # Increase difficulty
        elif solver_accuracy < 0.6:
            difficulty = max(0.1, difficulty - 0.1)  # Decrease difficulty

        CLIFormatter.print_info(f"Generating New Problems (difficulty: {difficulty:.2f})")
        new_problems = challenger.generate_problems(20, difficulty)

        # Clean up Challenger
        challenger.cleanup()

        # Create new dataset for next iteration
        current_dataset = Dataset.from_list(new_problems)

        # Save checkpoint
        checkpoint = {
            "iteration": iteration + 1,
            "difficulty": difficulty,
            "solver_accuracy": solver_accuracy,
            "dataset_size": len(current_dataset)
        }

        os.makedirs("outputs/checkpoints", exist_ok=True)
        with open(f"outputs/checkpoints/iteration_{iteration + 1}.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

        CLIFormatter.print_success(f"Checkpoint saved: iteration {iteration + 1}")

    CLIFormatter.print_summary_box(
        "R-Zero Evolution Complete",
        {
            "Total Iterations": num_iterations,
            "Final Difficulty": f"{difficulty:.2f}",
            "Final Dataset Size": len(current_dataset),
            "Checkpoints Saved": f"outputs/checkpoints/"
        }
    )


if __name__ == "__main__":
    # Test with just 2 iterations to verify it works
    run_rzero_evolution(num_iterations=2)
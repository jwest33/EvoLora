"""Simplified R-Zero implementation using llama.cpp Teacher and trainable Solver"""
from llama_cpp import Llama
from unsloth import FastModel
import os
import sys
import torch
import gc
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, Trainer
from trl import GRPOConfig, GRPOTrainer
import json
import re

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loralab.utils.cli_formatter import CLIFormatter, SpinnerProgress

# Disable multiprocessing to avoid Windows issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# Define reasoning format tokens (following Unsloth notebook)
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"

def clear_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()


class TeacherAgent:
    """Qwen3-30B Teacher that generates problems and evolves its prompts"""

    def __init__(self):
        CLIFormatter.print_info("Initializing Teacher with Qwen3-30B GGUF model...")

        model_path = r"C:\models\Qwen3-30B-A3B-Instruct-2507\Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"

        if not os.path.exists(model_path):
            CLIFormatter.print_error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        # Load the Qwen3 chat template
        chat_template_path = os.path.join(os.path.dirname(__file__), "qwen3_nonthinking.jinja")
        with open(chat_template_path, 'r') as f:
            chat_template = f.read()

        # Initialize llama.cpp with the GGUF model and chat template
        self.model = Llama(
            model_path=model_path,
            n_ctx=8192,  # Increased context window for GSM8K format
            n_threads=8,  # CPU threads
            n_gpu_layers=-1,  # Use all GPU layers
            offload_kqv=True,  # Offload KQV to GPU for MoE models
            split_mode=1,  # Split mode for MoE (1 = split by layer)
            tensor_split=None,  # Auto-split across GPUs if multiple
            chat_format="chatml",  # Use ChatML format for Qwen
            verbose=False
        )

        # Initialize the generation prompt (will evolve over time)
        self.generation_prompt = self._get_initial_prompt()
        self.prompt_history = [self.generation_prompt]
        self.iteration = 0

        CLIFormatter.print_success("Teacher model loaded successfully")

    def _get_initial_prompt(self) -> str:
        """Get the initial problem generation prompt"""
        return f"""Generate a math WORD PROBLEM with a real-world context.

The problem must:
1. Tell a story or describe a real situation
2. Involve people, objects, or scenarios
3. Require mathematical reasoning to solve

Format EXACTLY as follows:
Question: [A complete word problem with context]
Answer: {REASONING_START}
[Show your step-by-step reasoning here with calculations in <<>> brackets]
{REASONING_END}
{SOLUTION_START}[Final numerical answer only]{SOLUTION_END}

Example:
Question: Sarah has 15 apples. She gives 3 to her friend and buys 7 more. How many apples does she have?
Answer: {REASONING_START}
Sarah starts with 15 apples.
She gives away 3, so she has 15 - 3 = <<15-3=12>>12 apples.
She buys 7 more, so she has 12 + 7 = <<12+7=19>>19 apples.
{REASONING_END}
{SOLUTION_START}19{SOLUTION_END}

Generate ONE word problem:"""

    def _parse_problem_response(self, response: str) -> tuple:
        """Parse the model response to extract question, answer, and reasoning"""
        question = ""
        answer = ""
        full_solution = ""
        reasoning = ""

        if "Answer:" in response and "Question:" in response:
            # Extract question
            q_start = response.find("Question:")
            a_start = response.find("Answer:")
            if q_start != -1 and a_start != -1:
                question = response[q_start+9:a_start].strip()
                answer_part = response[a_start+7:].strip()
                full_solution = answer_part

                # Extract reasoning section
                if REASONING_START in answer_part and REASONING_END in answer_part:
                    r_start = answer_part.find(REASONING_START) + len(REASONING_START)
                    r_end = answer_part.find(REASONING_END)
                    reasoning = answer_part[r_start:r_end].strip()

                # Extract final answer from SOLUTION tags
                if SOLUTION_START in answer_part and SOLUTION_END in answer_part:
                    s_start = answer_part.find(SOLUTION_START) + len(SOLUTION_START)
                    s_end = answer_part.find(SOLUTION_END)
                    answer = answer_part[s_start:s_end].strip()
                # Fallback to extracting numbers
                else:
                    import re
                    numbers = re.findall(r'\d+(?:\.\d+)?', answer_part)
                    answer = numbers[-1] if numbers else ""

        return question, answer, full_solution, reasoning

    def generate_problems(self, num_problems: int, difficulty: float = 0.5, dataset_examples: List[Dict] = None) -> List[Dict]:
        """Generate math problems using the current prompt"""
        problems = []

        # Adjust prompt based on difficulty
        difficulty_prompt = self._adjust_prompt_for_difficulty(difficulty, dataset_examples)

        with SpinnerProgress(f"Generating {num_problems} problems at difficulty {difficulty:.1f}") as spinner:
            for i in range(num_problems):
                spinner.update(f"Generating problem {i+1}/{num_problems}")

                # Build the full prompt with example if this is the first problem and generation is failing
                if i == 0:
                    example_prompt = "\n\nExample:\nQuestion: Sarah has 8 apples and buys 5 more. How many apples does she have now?\nAnswer: Sarah has 8 apples. She buys 5 more, so 8 + 5 = <<8+5=13>>13. Therefore, she has \\boxed{13} apples.\n\n"
                    full_prompt = f"{self.generation_prompt}{example_prompt}{difficulty_prompt}\n\nNow generate a new problem:\n\nQuestion: "
                else:
                    full_prompt = f"{self.generation_prompt}\n\n{difficulty_prompt}\n\nQuestion: "

                # Generate with llama.cpp using chat format
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that creates math word problems in GSM8K format. Always include real-world context and step-by-step solutions."},
                    {"role": "user", "content": full_prompt}
                ]

                response = self.model.create_chat_completion(
                    messages=messages,
                    max_tokens=512,  # Increased for complete solutions
                    temperature=0.7 + (i * 0.01),  # Slightly lower temperature for more focused generation
                    top_p=0.9,
                    stop=["\n\nQuestion:", "\n\nNow", "\n\n**", "Example:", "\nGenerate"]  # Better stop tokens
                )

                # Extract the generated text from chat completion
                generated_text = response['choices'][0]['message']['content'].strip()

                # Debug: Print what the model actually generated (comment out after debugging)
                # if i == 0:  # Only print for first problem to avoid spam
                #     print(f"\n[DEBUG] Full prompt:\n{full_prompt[:200]}...")
                #     print(f"\n[DEBUG] Generated text:\n{generated_text}\n")

                # Parse the response using the GSM8K format parser
                question, answer, full_solution, reasoning = self._parse_problem_response(generated_text)

                # If parsing failed, try simpler extraction
                if not question and not answer:
                    lines = generated_text.split('\n')
                    if lines:
                        # Try to extract from first few lines
                        for line in lines:
                            if line.strip():
                                if not question and not line.startswith("Answer"):
                                    question = line.strip()
                                elif "####" in line:
                                    answer = line.split("####")[1].strip()
                                    break

                # Fallback if generation fails - with variety
                if not question or not answer:
                    import random
                    num1 = random.randint(5, 15) + int(difficulty * 10)
                    num2 = random.randint(3, 12) + int(difficulty * 8)
                    operations = ['+', '-'] if difficulty < 0.5 else ['+', '-', '*']
                    op = random.choice(operations)

                    if op == '+':
                        question = f"What is {num1} + {num2}?"
                        answer = str(num1 + num2)
                    elif op == '-':
                        question = f"What is {max(num1, num2)} - {min(num1, num2)}?"
                        answer = str(abs(num1 - num2))
                    else:  # multiplication
                        num1 = random.randint(2, 9)
                        num2 = random.randint(2, 9)
                        question = f"What is {num1} × {num2}?"
                        answer = str(num1 * num2)

                problems.append({
                    "question": question,
                    "answer": answer,
                    "full_solution": full_solution if full_solution else f"Direct calculation: {answer}",
                    "reasoning": reasoning if reasoning else "",
                    "difficulty": difficulty
                })

        return problems

    def _adjust_prompt_for_difficulty(self, difficulty: float, dataset_examples: List[Dict] = None) -> str:
        """Adjust the prompt based on difficulty level"""
        if difficulty <= 0.3:
            return """Create a SIMPLE word problem about everyday situations (like buying items, sharing toys, counting objects).
Use small numbers (under 20) and only one calculation step.
Include a real-world context with people or objects.
Show the calculation with <<>> brackets and put final answer in \\boxed{}."""
        elif difficulty <= 0.6:
            return """Create a MODERATELY COMPLEX word problem involving real scenarios (like planning events, managing money, measuring things).
The problem should require 2-3 calculation steps.
Include realistic context with names, places, or activities.
Show each calculation step with <<>> brackets and put final answer in \\boxed{}."""
        else:
            return """Create a CHALLENGING word problem with a rich story context (like running a business, planning a trip, solving a puzzle).
The problem should require multiple steps and logical reasoning.
Include detailed scenario with multiple constraints or conditions.
Show all calculation steps with <<>> brackets and put final answer in \\boxed{}."""

    def evolve_prompt(self, solver_accuracy: float, problem_samples: List[Dict] = None):
        """Evolve the generation prompt based on solver performance"""
        self.iteration += 1

        CLIFormatter.print_info(f"Evolving Teacher prompt (Iteration {self.iteration}, Solver accuracy: {solver_accuracy:.2%})")

        # Build evolution prompt asking the model to revise itself
        evolution_prompt = f"""You are a math problem generator. Your current prompt is:

{self.generation_prompt}

The solver achieved {solver_accuracy:.1%} accuracy on the problems you generated.

Based on this performance:
- If accuracy is below 60%, problems are too hard. Make them simpler.
- If accuracy is 60-70%, problems are at the right difficulty level.
- If accuracy is above 70%, problems are too easy. Make them harder.

Please revise your problem generation prompt to better challenge the solver.
Write only the new prompt, nothing else.

New prompt:"""

        # Get the model's revised prompt using chat format
        messages = [
            {"role": "system", "content": "You are an expert at improving prompts for generating math word problems."},
            {"role": "user", "content": evolution_prompt}
        ]

        response = self.model.create_chat_completion(
            messages=messages,
            max_tokens=300,
            temperature=0.8,
            top_p=0.95
        )

        # Extract the new prompt from chat completion
        new_prompt = response['choices'][0]['message']['content'].strip()

        if new_prompt and len(new_prompt) > 50:  # Ensure we got a meaningful response
            self.generation_prompt = new_prompt
            self.prompt_history.append(new_prompt)
            CLIFormatter.print_success("Teacher prompt evolved successfully")
        else:
            CLIFormatter.print_warning("Prompt evolution failed, keeping current prompt")

    def save_state(self, path: str):
        """Save the current state including prompts"""
        state = {
            "generation_prompt": self.generation_prompt,
            "prompt_history": self.prompt_history,
            "iteration": self.iteration
        }
        # Only create directory if path contains a directory
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, path: str):
        """Load a saved state"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            self.generation_prompt = state["generation_prompt"]
            self.prompt_history = state["prompt_history"]
            self.iteration = state["iteration"]
            CLIFormatter.print_info(f"Loaded Teacher state from iteration {self.iteration}")

    def cleanup(self):
        """Clean up model from memory"""
        # llama.cpp models are cleaned up automatically
        pass


class SolverAgent:
    """Gemma-3-1B Solver that learns to reason step-by-step using GRPO"""

    def __init__(self, checkpoint_path=None):
        CLIFormatter.print_info("Initializing Solver (Gemma-3-1B with GRPO)...")

        self.max_seq_length = 1024  # Increased for reasoning steps

        # Load from checkpoint if provided, otherwise from base model
        if checkpoint_path and os.path.exists(checkpoint_path):
            CLIFormatter.print_info(f"Loading Solver from checkpoint: {checkpoint_path}")
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name=checkpoint_path,
                max_seq_length=self.max_seq_length,
                load_in_4bit=False,
                load_in_8bit=False,
            )
        else:
            CLIFormatter.print_info("Loading base Gemma-3-1B model from Hugging Face")
            self.model, self.tokenizer = FastModel.from_pretrained(
                model_name="unsloth/gemma-3-1b-it",
                max_seq_length=self.max_seq_length,
                load_in_4bit=False,
                load_in_8bit=False,
                full_finetuning=False,
            )

        # Apply LoRA for GRPO (only if loading base model)
        if not (checkpoint_path and os.path.exists(checkpoint_path)):
            self.model = FastModel.get_peft_model(
                self.model,
                finetune_vision_layers=False,
                finetune_language_layers=True,
                finetune_attention_modules=True,  # Important for reasoning
                finetune_mlp_modules=True,
                r=8,  # Following Unsloth notebook
                lora_alpha=8,
                lora_dropout=0,
                bias="none",
                random_state=3407,
            )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Create system prompt for reasoning
        self.system_prompt = f"""You are given a problem.
Think about the problem and provide your working out.
Place it between {REASONING_START} and {REASONING_END}.
Then, provide your solution between {SOLUTION_START}{SOLUTION_END}"""

        # Compile regex patterns for extracting answers
        self.match_format = re.compile(
            rf"^[\s]{{0,}}"
            rf"{REASONING_START}.+?{REASONING_END}.*?"
            rf"{SOLUTION_START}(.+?){SOLUTION_END}"
            rf"[\s]{{0,}}$",
            flags=re.MULTILINE | re.DOTALL
        )

        # Pattern for extracting numbers from solution
        self.match_numbers = re.compile(
            rf"{SOLUTION_START}.*?([\d\.]{{1,}})",
            flags=re.MULTILINE | re.DOTALL
        )

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        CLIFormatter.print_success(f"Solver loaded: {trainable:,} trainable params")

    def match_format_exactly(self, completions, **kwargs):
        """Reward function: Check if format matches exactly"""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Match if format is seen exactly!
            if self.match_format.search(response) is not None:
                score += 3.0
            scores.append(score)
        return scores

    def match_format_approximately(self, completions, **kwargs):
        """Reward function: Partial credit for format elements"""
        scores = []
        for completion in completions:
            score = 0
            response = completion[0]["content"]
            # Count how many keywords are seen - penalize if too many!
            score += 0.5 if response.count(REASONING_START) == 1 else -0.5
            score += 0.5 if response.count(REASONING_END) == 1 else -0.5
            score += 0.5 if response.count(SOLUTION_START) == 1 else -0.5
            score += 0.5 if response.count(SOLUTION_END) == 1 else -0.5
            scores.append(score)
        return scores

    def check_answer(self, prompts, completions, answer, **kwargs):
        """Reward function: Check answer correctness"""
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_format.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            score = 0
            if guess is None:
                scores.append(0)
                continue
            # Correct answer gets 3 points!
            if guess == true_answer:
                score += 3.0
            # Match if spaces are seen
            elif guess.strip() == true_answer.strip():
                score += 1.5
            else:
                # Reward if answer is close via ratios
                try:
                    ratio = float(guess) / float(true_answer)
                    if ratio >= 0.9 and ratio <= 1.1:
                        score += 0.5
                    elif ratio >= 0.8 and ratio <= 1.2:
                        score += 0.25
                    else:
                        score -= 1.0  # Penalize wrong answers
                except:
                    score -= 0.5  # Penalize non-numeric
            scores.append(score)
        return scores

    def check_numbers(self, prompts, completions, answer, **kwargs):
        """Reward function: Extract and check numerical answers"""
        responses = [completion[0]["content"] for completion in completions]

        extracted_responses = [
            guess.group(1)
            if (guess := self.match_numbers.search(r)) is not None else None
            for r in responses
        ]

        scores = []
        for guess, true_answer in zip(extracted_responses, answer):
            if guess is None:
                scores.append(0)
                continue
            # Convert to numbers
            try:
                true_answer = float(true_answer.strip())
                guess = float(guess.strip())
                scores.append(1.5 if guess == true_answer else 0.0)
            except:
                scores.append(0)
                continue
        return scores

    def solve_problems(self, problems: List[Dict], return_accuracy: bool = True) -> List[Dict]:
        """Generate solutions for problems"""
        results = []
        correct_count = 0

        CLIFormatter.print_info(f"Solving {len(problems)} problems...")

        with SpinnerProgress(f"Solving {len(problems)} problems") as spinner:
            for idx, problem in enumerate(problems):
                spinner.update(f"Solving problem {idx+1}/{len(problems)}")
                prompt = f"Question: {problem['question']}\nAnswer:"

                inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.15,
                    )

                # Decode only the generated part
                generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                # Extract numerical answer
                import re
                numbers = re.findall(r'[-]?\d+(?:\.\d+)?', answer)
                solver_answer = numbers[0] if numbers else ""

                # Check correctness if ground truth available
                is_correct = False
                if "answer" in problem and problem["answer"]:
                    try:
                        is_correct = abs(float(solver_answer) - float(problem["answer"])) < 0.01
                        if is_correct:
                            correct_count += 1
                    except:
                        pass

                result = {
                    "question": problem["question"],
                    "solver_answer": solver_answer,
                    "full_answer": answer,
                    "ground_truth": problem.get("answer", ""),
                    "is_correct": is_correct,
                    "difficulty": problem.get("difficulty", 0.5)
                }
                results.append(result)

        # Calculate accuracy
        if return_accuracy and problems:
            accuracy = correct_count / len(problems)
            for r in results:
                r["accuracy"] = accuracy

        return results

    def train_with_grpo(self, problems: List[Dict], max_steps: int = 50):
        """Train Solver using GRPO to learn reasoning"""
        CLIFormatter.print_info("Training Solver with GRPO...")

        # Format dataset for GRPO
        dataset_items = []
        for problem in problems:
            dataset_items.append({
                "prompt": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": problem["question"]},
                ],
                "answer": problem["answer"],
            })

        dataset = Dataset.from_list(dataset_items)

        # GRPO configuration
        max_prompt_length = 256
        training_args = GRPOConfig(
            learning_rate=5e-6,
            adam_beta1=0.9,
            adam_beta2=0.99,
            weight_decay=0.1,
            warmup_ratio=0.1,
            lr_scheduler_type="cosine",
            optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
            logging_steps=1,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,  # For smoother training
            num_generations=4,  # Number of completions per prompt
            max_prompt_length=max_prompt_length,
            max_completion_length=self.max_seq_length - max_prompt_length,
            max_steps=max_steps,
            save_steps=max_steps,
            max_grad_norm=0.1,
            report_to="none",
            output_dir="outputs/grpo_solver",
        )

        # Initialize GRPO trainer with reward functions
        trainer = GRPOTrainer(
            model=self.model,
            processing_class=self.tokenizer,
            reward_funcs=[
                self.match_format_exactly,
                self.match_format_approximately,
                self.check_answer,
                self.check_numbers,
            ],
            args=training_args,
            train_dataset=dataset,
        )

        # Monkey-patch the log method to format output nicely
        original_log = trainer.log

        def formatted_log(logs, start_time=None):
            # Call original log method with both arguments
            if start_time is not None:
                original_log(logs, start_time)
            else:
                original_log(logs)

            # Format and display key metrics
            if logs and isinstance(logs, dict):
                step = trainer.state.global_step

                # Only print formatted output when we have complete training metrics
                # (reward data indicates a full training step, not just eval)
                has_rewards = 'reward' in logs and 'epoch' in logs
                has_components = any(k.startswith('rewards/') for k in logs.keys())

                if step > 0 and has_rewards and has_components:
                    print("\n")  # Clear line after default output
                    # Use the new synthwave-styled GRPO formatter
                    CLIFormatter.format_grpo_stats(logs, step, max_steps)

        trainer.log = formatted_log

        # Train with GRPO
        trainer.train()
        CLIFormatter.print_success("GRPO training completed")

    def save_checkpoint(self, iteration: int) -> str:
        """Save model checkpoint and return the path"""
        checkpoint_dir = f"outputs/solver/iteration_{iteration}"
        os.makedirs(checkpoint_dir, exist_ok=True)

        CLIFormatter.print_info(f"Saving Solver checkpoint to {checkpoint_dir}")
        self.model.save_pretrained(checkpoint_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)

        return checkpoint_dir

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        clear_memory()


def run_simplified_rzero(num_iterations: int = 5, grpo_steps_per_iteration: int = 100):
    """Main simplified R-Zero evolution loop with GRPO

    Args:
        num_iterations: Number of evolution cycles
        grpo_steps_per_iteration: GRPO training steps per iteration (50-200 recommended)
    """
    CLIFormatter.print_header("R-Zero Training with GRPO")

    # Check CUDA
    if torch.cuda.is_available():
        CLIFormatter.print_status("CUDA Device", torch.cuda.get_device_name(0))
        CLIFormatter.print_status("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        CLIFormatter.print_warning("CUDA not available - training will be slow")

    # Training configuration
    CLIFormatter.print_subheader("Configuration")
    CLIFormatter.print_status("Evolution iterations", str(num_iterations))
    CLIFormatter.print_status("GRPO steps per iteration", str(grpo_steps_per_iteration))
    CLIFormatter.print_status("Model", "Gemma-3-1B with LoRA")
    CLIFormatter.print_status("Teacher", "Qwen3-30B (GGUF)")

    # Initialize with GSM8K for bootstrapping
    CLIFormatter.print_subheader("Loading GSM8K Dataset")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train[:100]")

    # Prepare initial dataset
    initial_data = []
    for example in gsm8k:
        initial_data.append({
            "question": example["question"],
            "answer": example["answer"].split("####")[-1].strip() if "####" in example["answer"] else example["answer"]
        })

    current_dataset = Dataset.from_list(initial_data)
    difficulty = 0.3  # Start with easy problems

    # Initialize Teacher (no training needed)
    CLIFormatter.print_subheader("Initializing Agents")
    teacher = TeacherAgent()
    teacher_state_path = "outputs/teacher/state.json"

    # Load teacher state if exists
    if os.path.exists(teacher_state_path):
        CLIFormatter.print_info("Loading existing Teacher state...")
        teacher.load_state(teacher_state_path)

    # Pre-train Solver on GSM8K examples using GRPO
    CLIFormatter.print_subheader("Pre-training Solver with GRPO")
    CLIFormatter.print_info("This teaches the model the basic reasoning format...")
    solver = SolverAgent()

    # Pre-train with GRPO on initial dataset
    initial_problems = []
    for item in current_dataset.select(range(min(30, len(current_dataset)))):
        initial_problems.append({
            "question": item["question"],
            "answer": item["answer"],
        })

    # Use more steps for pre-training to establish format
    solver.train_with_grpo(initial_problems, max_steps=50)

    # Save pre-trained solver
    solver_checkpoint = solver.save_checkpoint(0)
    solver.cleanup()

    # Evolution loop
    CLIFormatter.print_header("Starting Evolution Loop")

    for iteration in range(num_iterations):
        CLIFormatter.print_header(f"Evolution Iteration {iteration + 1}/{num_iterations}")
        CLIFormatter.print_status("Current Difficulty", f"{difficulty:.2f}")

        # Phase 1: Teacher generates problems
        CLIFormatter.print_subheader("Phase 1: Teacher Generating Problems")
        num_problems = 20 + (iteration * 5)  # Increase dataset size over time
        CLIFormatter.print_info(f"Generating {num_problems} problems at difficulty {difficulty:.2f}...")

        new_problems = teacher.generate_problems(num_problems, difficulty, initial_data[:10])

        # Show sample problem
        if new_problems:
            CLIFormatter.print_info("Sample generated problem:")
            CLIFormatter.print_status("Question", new_problems[0]['question'][:100] + "..." if len(new_problems[0]['question']) > 100 else new_problems[0]['question'])
            CLIFormatter.print_status("Answer", new_problems[0]['answer'])
            if new_problems[0].get('reasoning'):
                CLIFormatter.print_status("Has reasoning", "Yes")

        # Save teacher samples
        os.makedirs("outputs/samples", exist_ok=True)
        with open(f"outputs/samples/teacher_iter_{iteration + 1}.json", "w") as f:
            json.dump({
                "iteration": iteration + 1,
                "difficulty": difficulty,
                "prompt": teacher.generation_prompt[:500] + "..." if len(teacher.generation_prompt) > 500 else teacher.generation_prompt,
                "samples": new_problems[:5]
            }, f, indent=2)

        # Update dataset
        current_dataset = Dataset.from_list(new_problems)

        # Phase 2: GRPO Training
        CLIFormatter.print_subheader("Phase 2: Solver GRPO Training")
        CLIFormatter.print_info(f"Training for {grpo_steps_per_iteration} steps to learn reasoning...")
        CLIFormatter.print_info("Watch the reward metrics to see learning progress!")

        solver = SolverAgent(checkpoint_path=solver_checkpoint)

        # Train with GRPO on new problems - use configured step count
        solver.train_with_grpo(new_problems, max_steps=grpo_steps_per_iteration)

        # Phase 3: Evaluate solver
        CLIFormatter.print_subheader("Phase 3: Evaluating Solver Performance")
        # Test on subset for speed
        test_problems = new_problems[:min(10, len(new_problems))]
        results = solver.solve_problems(test_problems)
        solver_accuracy = results[0]["accuracy"] if results else 0

        CLIFormatter.print_metric("Solver Accuracy", solver_accuracy * 100, "%", good_threshold=70, bad_threshold=40)

        # Show example solution
        if results:
            CLIFormatter.print_info("Example solver output:")
            CLIFormatter.print_status("Question", results[0]['question'][:80] + "..." if len(results[0]['question']) > 80 else results[0]['question'])
            CLIFormatter.print_status("Solver answer", results[0]['solver_answer'])
            CLIFormatter.print_status("Correct answer", results[0]['ground_truth'])
            CLIFormatter.print_status("Result", "✓ Correct" if results[0]['is_correct'] else "✗ Incorrect")

        # Save solver samples
        with open(f"outputs/samples/solver_iter_{iteration + 1}.json", "w") as f:
            json.dump({
                "iteration": iteration + 1,
                "accuracy": solver_accuracy,
                "samples": results[:5]
            }, f, indent=2)

        # Phase 4: Teacher evolves its prompt
        CLIFormatter.print_subheader("Phase 4: Teacher Prompt Evolution")
        old_prompt_len = len(teacher.generation_prompt)
        teacher.evolve_prompt(solver_accuracy, new_problems[:5])
        new_prompt_len = len(teacher.generation_prompt)

        CLIFormatter.print_status("Prompt evolution", f"{old_prompt_len} → {new_prompt_len} chars")
        teacher.save_state(teacher_state_path)

        # Adjust difficulty for next iteration
        old_difficulty = difficulty
        if solver_accuracy > 0.7:
            difficulty = min(1.0, difficulty + 0.1)
            CLIFormatter.print_success(f"Performance good! Increasing difficulty: {old_difficulty:.2f} → {difficulty:.2f}")
        elif solver_accuracy < 0.4:
            difficulty = max(0.1, difficulty - 0.1)
            CLIFormatter.print_warning(f"Performance low. Decreasing difficulty: {old_difficulty:.2f} → {difficulty:.2f}")
        else:
            CLIFormatter.print_info(f"Maintaining difficulty at {difficulty:.2f}")

        # Save solver checkpoint
        solver_checkpoint = solver.save_checkpoint(iteration + 1)
        solver.cleanup()

        # Iteration summary
        CLIFormatter.print_success(f"✓ Iteration {iteration + 1} completed!")
        CLIFormatter.print_status("Final accuracy", f"{solver_accuracy:.2%}")
        CLIFormatter.print_status("Next difficulty", f"{difficulty:.2f}")
        CLIFormatter.print_status("Prompt iterations", str(len(teacher.prompt_history)))
        print()  # Add spacing between iterations

    # Final summary
    CLIFormatter.print_header("R-Zero GRPO Training Complete!")
    CLIFormatter.print_success(f"Completed {num_iterations} evolution iterations")
    CLIFormatter.print_status("Final difficulty level", f"{difficulty:.2f}")
    CLIFormatter.print_status("Total prompt evolutions", str(len(teacher.prompt_history)))

    # Save final summary
    with open("outputs/evolution_summary.json", "w") as f:
        json.dump({
            "num_iterations": num_iterations,
            "final_difficulty": difficulty,
            "grpo_steps_per_iteration": grpo_steps_per_iteration,
            "teacher_prompts": teacher.prompt_history
        }, f, indent=2)

    CLIFormatter.print_info("Results saved:")
    CLIFormatter.print_status("Samples", "outputs/samples/")
    CLIFormatter.print_status("Model checkpoints", "outputs/solver/")
    CLIFormatter.print_status("Teacher state", "outputs/teacher/")
    CLIFormatter.print_status("Evolution summary", "outputs/evolution_summary.json")

    # Final cleanup
    teacher.cleanup()


if __name__ == "__main__":
    # Run with recommended settings based on test results
    # 100 GRPO steps per iteration is a good balance between training time and learning
    run_simplified_rzero(num_iterations=3, grpo_steps_per_iteration=100)

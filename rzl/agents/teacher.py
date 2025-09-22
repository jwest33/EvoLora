"""Teacher Agent for R-Zero implementation - Generates problems and evolves prompts"""
from llama_cpp import Llama
import os
import json
import re
from typing import List, Dict
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cli_formatter import CLIFormatter, SpinnerProgress

# Define reasoning format tokens
REASONING_START = "<start_working_out>"
REASONING_END = "<end_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


class TeacherAgent:
    """Qwen3-30B Teacher that generates problems and evolves its prompts"""

    def __init__(self):
        CLIFormatter.print_info("Initializing Teacher with Qwen3-30B GGUF model...")

        model_path = r"C:\models\Qwen3-30B-A3B-Instruct-2507\Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"

        if not os.path.exists(model_path):
            CLIFormatter.print_error(f"Model not found at {model_path}")
            raise FileNotFoundError(f"GGUF model not found: {model_path}")

        # Load the Qwen3 chat template
        chat_template_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "qwen3_nonthinking.jinja")
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
        """Load saved state"""
        if os.path.exists(path):
            with open(path, 'r') as f:
                state = json.load(f)
            self.generation_prompt = state.get("generation_prompt", self._get_initial_prompt())
            self.prompt_history = state.get("prompt_history", [self.generation_prompt])
            self.iteration = state.get("iteration", 0)
            CLIFormatter.print_success(f"Loaded Teacher state from iteration {self.iteration}")

    def cleanup(self):
        """Clean up model from memory"""
        # llama.cpp models are cleaned up automatically
        pass

"""Test script for GRPO-based R-Zero implementation"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_rzl import TeacherAgent, SolverAgent, REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
from loralab.utils.cli_formatter import CLIFormatter, SpinnerProgress
from datasets import Dataset
import torch

def test_grpo_training():
    """Test the GRPO training with a small dataset"""

    CLIFormatter.print_header("GRPO R-Zero Test")

    # Check CUDA
    if torch.cuda.is_available():
        CLIFormatter.print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        CLIFormatter.print_status("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        CLIFormatter.print_warning("CUDA not available - training will be slow")

    CLIFormatter.print_subheader("Step 1: Initialize Teacher and generate problems")

    teacher = TeacherAgent()

    # Generate a small set of problems at different difficulties
    problems = []
    for difficulty in [0.2, 0.5]:
        CLIFormatter.print_status("Generating problems", f"Difficulty {difficulty}")
        batch = teacher.generate_problems(num_problems=3, difficulty=difficulty)
        problems.extend(batch)

    CLIFormatter.print_success(f"Generated {len(problems)} problems")

    # Display sample problem
    if problems:
        CLIFormatter.print_info("Sample problem generated:")
        CLIFormatter.print_item("Question", problems[0]['question'])
        CLIFormatter.print_item("Answer", problems[0]['answer'])
        if problems[0].get('reasoning'):
            CLIFormatter.print_item("Has reasoning", "Yes")

    CLIFormatter.print_subheader("Step 2: Initialize Solver with GRPO")

    solver = SolverAgent()

    CLIFormatter.print_subheader("Step 3: Train Solver with GRPO")
    CLIFormatter.print_info("This will train for 10 steps to test the setup...")
    CLIFormatter.print_info("Watch for the 'reward' column to see if the model is learning!")

    # Train with GRPO for just a few steps to test
    solver.train_with_grpo(problems, max_steps=10)

    CLIFormatter.print_subheader("Step 4: Test Solver's reasoning")

    # Test on a simple problem
    test_problem = {
        "question": "Tommy has 5 apples. He gets 3 more from his friend. How many apples does Tommy have now?"
    }

    messages = [
        {"role": "system", "content": solver.system_prompt},
        {"role": "user", "content": test_problem["question"]},
    ]

    text = solver.tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=False,
    )

    CLIFormatter.print_info("Generating solution with reasoning...")

    from transformers import TextStreamer
    streamer = TextStreamer(solver.tokenizer, skip_prompt=True)

    outputs = solver.model.generate(
        **solver.tokenizer(text, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=256,
        temperature=1.0,
        top_p=0.95,
        top_k=64,
        streamer=streamer,
    )

    # Check if the output contains reasoning format
    generated_text = solver.tokenizer.decode(outputs[0], skip_special_tokens=True)

    if REASONING_START in generated_text and REASONING_END in generated_text:
        CLIFormatter.print_success("Model is using reasoning format!")
    else:
        CLIFormatter.print_warning("Model is not using reasoning format yet (needs more training)")

    if SOLUTION_START in generated_text and SOLUTION_END in generated_text:
        CLIFormatter.print_success("Model is using solution format!")
    else:
        CLIFormatter.print_warning("Model is not using solution format yet (needs more training)")

    CLIFormatter.print_subheader("Step 5: Test Teacher prompt evolution")

    CLIFormatter.print_status("Original prompt length", f"{len(teacher.generation_prompt)} chars")

    # Simulate different solver accuracies
    for accuracy in [0.3, 0.7]:
        CLIFormatter.print_info(f"Evolving prompt with solver accuracy: {accuracy:.0%}")
        teacher.evolve_prompt(solver_accuracy=accuracy, problem_samples=problems[:2])

    CLIFormatter.print_status("Evolved prompt length", f"{len(teacher.generation_prompt)} chars")
    CLIFormatter.print_status("Prompt history", f"{len(teacher.prompt_history)} versions")

    # Cleanup
    solver.cleanup()
    teacher.cleanup()

    CLIFormatter.print_header("GRPO R-Zero Test Completed")
    CLIFormatter.print_success("Test completed successfully!")

    CLIFormatter.print_info("Notes:")
    CLIFormatter.print_item("", "The reward should increase over training steps")
    CLIFormatter.print_item("", "The model needs more steps (50-200) to learn the format properly")
    CLIFormatter.print_item("", "With full training, it will generate step-by-step reasoning")

if __name__ == "__main__":
    test_grpo_training()

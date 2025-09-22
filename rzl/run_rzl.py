"""Main entry - orchestrates training pipeline"""
import os
import sys
import json
import torch
from datetime import datetime
from datasets import Dataset, load_dataset

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.cli_formatter import CLIFormatter
from agents import TeacherAgent, SolverAgent

# Disable multiprocessing to avoid Windows issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'


def run_simplified_rzero(num_iterations: int = 8, grpo_steps_per_iteration: int = 75):
    """Main simplified R-Zero evolution loop with GRPO

    Args:
        num_iterations: Number of evolution cycles (default: 8)
        grpo_steps_per_iteration: GRPO training steps per iteration (default: 75)
    """
    CLIFormatter.print_header("RZL Training with GRPO")

    # Create unique run ID for this training session
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = f"outputs/{run_id}"
    CLIFormatter.print_status("Run ID", run_id)
    CLIFormatter.print_status("Output Directory", run_dir)

    # Create the run directory
    os.makedirs(run_dir, exist_ok=True)

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
    teacher_state_path = f"{run_dir}/teacher/state.json"

    # Pre-train Solver on GSM8K examples using GRPO
    CLIFormatter.print_subheader("Pre-training Solver with GRPO")
    CLIFormatter.print_info("This teaches the model the basic reasoning format...")
    solver = SolverAgent()

    # Pre-train with GRPO on initial dataset
    initial_problems = []
    for item in current_dataset.select(range(min(30, len(current_dataset)))):  # Use 30 examples for better diversity
        initial_problems.append({
            "question": item["question"],
            "answer": item["answer"],
        })

    # Use moderate steps for pre-training to establish format well
    solver.train_with_grpo(initial_problems, max_steps=10)  # 10 steps to learn format properly

    # Save pre-trained solver
    solver_checkpoint = solver.save_checkpoint(0, run_id)
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
        samples_dir = f"{run_dir}/samples"
        os.makedirs(samples_dir, exist_ok=True)
        with open(f"{samples_dir}/teacher_iter_{iteration + 1}.json", "w") as f:
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

        CLIFormatter.print_status("Solver Accuracy", f"{solver_accuracy:.2%}")

        # Show example solver output
        if results:
            CLIFormatter.print_info("Example solver output:")
            print(f"Question: {results[0]['question'][:100]}...")
            print(f"Solver answer: {results[0]['solver_answer']}")
            print(f"Correct answer: {results[0]['ground_truth']}")
            print(f"Result: {'✓ Correct' if results[0]['is_correct'] else '✗ Incorrect'}")

        # Phase 4: Teacher evolves
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
        solver_checkpoint = solver.save_checkpoint(iteration + 1, run_id)
        solver.cleanup()

        # Iteration summary
        CLIFormatter.print_success(f"✓ Iteration {iteration + 1} completed!")
        CLIFormatter.print_status("Final accuracy", f"{solver_accuracy:.2%}")
        CLIFormatter.print_status("Next difficulty", f"{difficulty:.2f}")
        CLIFormatter.print_status("Prompt iterations", str(len(teacher.prompt_history)))
        print()  # Add spacing between iterations

    # Final summary
    CLIFormatter.print_header("GRPO Training Complete!")
    CLIFormatter.print_success(f"Completed {num_iterations} evolution iterations")
    CLIFormatter.print_status("Final difficulty level", f"{difficulty:.2f}")
    CLIFormatter.print_status("Total prompt evolutions", str(len(teacher.prompt_history)))

    # Save final summary
    summary_path = f"{run_dir}/evolution_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "run_id": run_id,
            "num_iterations": num_iterations,
            "final_difficulty": difficulty,
            "grpo_steps_per_iteration": grpo_steps_per_iteration,
            "teacher_prompts": teacher.prompt_history
        }, f, indent=2)

    CLIFormatter.print_info("Results saved in:")
    CLIFormatter.print_status("Run directory", run_dir)
    CLIFormatter.print_status("Model checkpoints", f"{run_dir}/solver/")
    CLIFormatter.print_status("Teacher state", f"{run_dir}/teacher/")
    CLIFormatter.print_status("Training samples", f"{run_dir}/samples/")
    CLIFormatter.print_status("Summary", summary_path)

    # Final cleanup
    teacher.cleanup()

    print("\n" + "="*50)
    CLIFormatter.print_info("To export the best checkpoint to GGUF:")
    CLIFormatter.print_info(f"  python helpers/export_to_gguf.py --checkpoint {run_dir}/solver/iteration_{num_iterations}")
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RZL training with GRPO")
    parser.add_argument(
        "--iterations",
        type=int,
        default=8,
        help="Number of evolution iterations (default: 8)"
    )
    parser.add_argument(
        "--grpo-steps",
        type=int,
        default=75,
        help="GRPO training steps per iteration (default: 75)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run in test mode with minimal iterations"
    )

    args = parser.parse_args()

    # Test mode for quick validation
    if args.test:
        run_simplified_rzero(num_iterations=2, grpo_steps_per_iteration=5)
    else:
        run_simplified_rzero(
            num_iterations=args.iterations,
            grpo_steps_per_iteration=args.grpo_steps
        )

"""Minimal test for the complete R-Zero pipeline"""
import os
import sys
import torch
import gc
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from utils.cli_formatter import CLIFormatter
from agents import ChallengerAgent, SolverAgent
from utils.dataset_filter import DatasetFilter
from rewards import UncertaintyReward, RepetitionPenalty, FormatReward, CompositeReward


def clear_memory():
    """Clear GPU memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()


def test_rzero_minimal():
    """Minimal test of the R-Zero pipeline

    Tests:
    1. Model initialization
    2. Problem generation
    3. Dataset filtering
    4. Solver training
    5. Challenger training with frozen solver
    """
    CLIFormatter.print_header("R-Zero Pipeline Minimal Test")

    try:
        # Test 1: Initialize models
        CLIFormatter.print_subheader("Test 1: Model Initialization")

        challenger = ChallengerAgent()
        CLIFormatter.print_success("✓ Challenger initialized")

        solver = SolverAgent()
        CLIFormatter.print_success("✓ Solver initialized")

        # Test 2: Generate problems
        CLIFormatter.print_subheader("Test 2: Problem Generation")

        problems = challenger.generate_candidate_problems(
            num_problems=10,  # Very small for testing
            temperature=0.8
        )
        CLIFormatter.print_success(f"✓ Generated {len(problems)} problems")

        if problems:
            sample = problems[0]
            CLIFormatter.print_info(f"Sample question: {sample['question'][:50]}...")
            CLIFormatter.print_info(f"Sample answer: {sample['answer']}")

        # Test 3: Dataset filtering
        CLIFormatter.print_subheader("Test 3: Dataset Filtering")

        dataset_filter = DatasetFilter(delta=0.25)
        filtered_problems, stats = dataset_filter.filter_dataset(
            problems,
            solver,
            m_samples=3  # Reduced for speed
        )
        CLIFormatter.print_success(f"✓ Filtered {len(filtered_problems)} problems")
        CLIFormatter.print_info(f"  Kept: {stats['kept']}, Too easy: {stats['too_easy']}, Too hard: {stats['too_hard']}")

        # Test 4: Solver self-consistency
        CLIFormatter.print_subheader("Test 4: Solver Self-Consistency")

        if problems:
            test_question = problems[0]["question"]
            pseudo_label, empirical_acc, solutions = solver.solve_with_self_consistency(
                test_question,
                m_samples=3
            )
            CLIFormatter.print_success(f"✓ Self-consistency: answer={pseudo_label}, accuracy={empirical_acc:.2f}")

        # Test 5: Reward computation
        CLIFormatter.print_subheader("Test 5: Reward System")

        # Test uncertainty reward
        uncertainty_reward = UncertaintyReward()
        solver_responses = []
        for p in problems[:3]:  # Test on subset
            responses = []
            for _ in range(3):
                resp = solver.generate_solution(p["question"])
                responses.append(resp)
            solver_responses.append(responses)

        u_rewards = uncertainty_reward.compute(problems[:3], solver_responses)
        CLIFormatter.print_success(f"✓ Uncertainty rewards: {[f'{r:.2f}' for r in u_rewards]}")

        # Test repetition penalty
        repetition_penalty = RepetitionPenalty()
        r_penalties = repetition_penalty.compute(problems)
        CLIFormatter.print_success(f"✓ Repetition penalties computed")

        # Test format reward
        format_reward = FormatReward()
        f_rewards = format_reward.compute(problems)
        CLIFormatter.print_success(f"✓ Format rewards computed")

        # Test composite reward
        composite_reward = CompositeReward()
        if len(u_rewards) == len(problems):
            c_rewards = composite_reward.compute(
                u_rewards[:len(problems)],
                r_penalties[:len(problems)],
                f_rewards[:len(problems)]
            )
            CLIFormatter.print_success(f"✓ Composite rewards computed")

        # Test 6: Mini training loop (1 step each)
        CLIFormatter.print_subheader("Test 6: Mini Training Loop")

        if filtered_problems:
            # Train solver for 1 step
            solver.train_with_grpo(filtered_problems[:5], max_steps=1)
            CLIFormatter.print_success("✓ Solver GRPO training (1 step)")

            # Train challenger for 1 step
            challenger.train_with_grpo(
                frozen_solver=solver,
                num_problems=5,
                max_steps=1,
                m_solver_samples=2
            )
            CLIFormatter.print_success("✓ Challenger GRPO training (1 step)")
        else:
            CLIFormatter.print_warning("Skipping training - no filtered problems")

        # Cleanup
        CLIFormatter.print_subheader("Cleanup")
        challenger.cleanup()
        solver.cleanup()
        clear_memory()
        CLIFormatter.print_success("✓ Models cleaned up")

        CLIFormatter.print_header("✅ All Tests Passed!")
        return True

    except Exception as e:
        CLIFormatter.print_error(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_rzero_integration():
    """Integration test with slightly more iterations

    Tests a mini version of the full co-evolution loop
    """
    CLIFormatter.print_header("R-Zero Integration Test")

    try:
        # Initialize
        challenger = ChallengerAgent()
        solver = SolverAgent()
        dataset_filter = DatasetFilter()

        # Bootstrap solver with simple problems
        bootstrap_problems = [
            {"question": "What is 2 + 3?", "answer": "5"},
            {"question": "What is 4 + 6?", "answer": "10"},
            {"question": "What is 7 + 8?", "answer": "15"},
        ]

        CLIFormatter.print_info("Bootstrapping solver...")
        solver.train_with_grpo(bootstrap_problems, max_steps=2)

        # Run 2 co-evolution iterations
        for iteration in range(1, 3):
            CLIFormatter.print_subheader(f"Iteration {iteration}")

            # Generate problems
            problems = challenger.generate_candidate_problems(
                num_problems=20,
                temperature=0.8
            )
            CLIFormatter.print_info(f"Generated {len(problems)} problems")

            # Filter
            filtered, stats = dataset_filter.filter_dataset(
                problems,
                solver,
                m_samples=3
            )
            CLIFormatter.print_info(f"Filtered to {len(filtered)} problems")

            if filtered:
                # Train solver
                solver.train_with_grpo(filtered[:10], max_steps=2)
                CLIFormatter.print_success("✓ Solver trained")

                # Train challenger
                challenger.train_with_grpo(
                    frozen_solver=solver,
                    num_problems=10,
                    max_steps=2,
                    m_solver_samples=3
                )
                CLIFormatter.print_success("✓ Challenger trained")

            # Evaluate
            if filtered:
                test_results = solver.solve_problems(filtered[:3])
                accuracy = test_results[0]["accuracy"] if test_results else 0
                CLIFormatter.print_status("Accuracy", f"{accuracy:.2%}")

        # Cleanup
        challenger.cleanup()
        solver.cleanup()
        clear_memory()

        CLIFormatter.print_header("✅ Integration Test Passed!")
        return True

    except Exception as e:
        CLIFormatter.print_error(f"❌ Integration test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test R-Zero pipeline")
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run integration test with more iterations"
    )

    args = parser.parse_args()

    if args.integration:
        success = test_rzero_integration()
    else:
        success = test_rzero_minimal()

    exit(0 if success else 1)
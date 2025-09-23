"""Simple test script for TeacherAgent problem generation using run_rz functions"""
import sys
import os
# Add parent directory to path to import from rz module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import TeacherAgent
import json

def test_teacher_generation():
    """Test the TeacherAgent's problem generation capabilities"""

    print("Teacher Problem Generation Test")
    print("="*80)

    # Initialize the actual TeacherAgent from run_rz
    teacher = TeacherAgent()

    # Test different difficulty levels
    difficulty_levels = [
        (0.2, "Very Easy"),
        (0.5, "Medium"),
        (0.8, "Hard")
    ]

    all_results = {}

    # Example dataset (simulating GSM8K examples)
    example_dataset = [
        {
            "question": "Tommy has 3 toy cars. His friend gives him 5 more toy cars. How many toy cars does Tommy have now?",
            "answer": "8"
        },
        {
            "question": "A bakery sold 24 cookies in the morning and 18 cookies in the afternoon. How many cookies did they sell in total?",
            "answer": "42"
        },
        {
            "question": "Lisa has 15 stickers. She gives 3 to her friend and buys 7 more. How many stickers does Lisa have now?",
            "answer": "19"
        }
    ]

    for difficulty, diff_name in difficulty_levels:
        print(f"\n\n{'='*80}")
        print(f"TESTING DIFFICULTY: {difficulty} ({diff_name})")
        print(f"{'='*80}")

        # Show what prompt adjustment will be used
        difficulty_prompt = teacher._adjust_prompt_for_difficulty(difficulty, example_dataset)
        print(f"\nDifficulty instruction: {difficulty_prompt}")

        # Generate problems using the actual method
        print(f"\nGenerating 3 problems at difficulty {difficulty}...")
        problems = teacher.generate_problems(
            num_problems=3,
            difficulty=difficulty,
            dataset_examples=example_dataset
        )

        all_results[f"difficulty_{difficulty}"] = problems

        # Display results
        print(f"\n\nResults for difficulty {difficulty} ({diff_name}):")
        print("-" * 40)

        success_count = sum(1 for p in problems if p.get('question') and p.get('answer'))
        print(f"Successfully generated: {success_count}/{len(problems)}")
        print()

        for i, p in enumerate(problems, 1):
            has_question = bool(p.get('question'))
            has_answer = bool(p.get('answer'))
            status = "✓" if has_question and has_answer else "✗"

            print(f"{status} Problem {i}:")
            print(f"   Question: {p.get('question', '[NO QUESTION]')}")
            print(f"   Answer: {p.get('answer', '[NO ANSWER]')}")
            print()

    # Save results
    output_file = "test_teacher_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to {output_file}")

    # Test prompt evolution
    print("\n" + "="*80)
    print("TESTING PROMPT EVOLUTION")
    print("="*80)

    print("\nInitial generation prompt:")
    print("-" * 40)
    print(teacher.generation_prompt)
    print("-" * 40)

    # Simulate different solver accuracies
    test_accuracies = [0.3, 0.65, 0.85]

    for accuracy in test_accuracies:
        print(f"\n\nSimulating solver accuracy: {accuracy:.0%}")
        print("Evolving prompt...")

        # Save the prompt before evolution
        old_prompt = teacher.generation_prompt

        # Evolve the prompt
        teacher.evolve_prompt(solver_accuracy=accuracy, problem_samples=problems[:2])

        # Check if prompt changed
        if teacher.generation_prompt != old_prompt:
            print("✓ Prompt evolved successfully!")
            print("\nNew prompt:")
            print("-" * 40)
            print(teacher.generation_prompt)
            print("-" * 40)
        else:
            print("✗ Prompt did not change (evolution may have failed)")

    # Show prompt history
    print("\n" + "="*80)
    print("PROMPT EVOLUTION HISTORY")
    print("="*80)

    for i, prompt in enumerate(teacher.prompt_history):
        print(f"\nIteration {i}:")
        print("-" * 40)
        print(prompt[:200] + "..." if len(prompt) > 200 else prompt)

    # Save teacher state
    teacher.save_state("test_teacher_state.json")
    print(f"\n\nTeacher state saved to test_teacher_state.json")

    # Final statistics
    print("\n" + "="*80)
    print("FINAL STATISTICS:")
    print("-" * 40)

    total_problems = 0
    successful_problems = 0

    for diff_key, problems in all_results.items():
        diff_success = sum(1 for p in problems if p.get('question') and p.get('answer'))
        total_problems += len(problems)
        successful_problems += diff_success
        print(f"{diff_key}: {diff_success}/{len(problems)} successful")

    if total_problems > 0:
        print(f"\nOverall success rate: {successful_problems}/{total_problems} ({100*successful_problems/total_problems:.1f}%)")

    # Cleanup
    teacher.cleanup()
    print("\nTest completed!")


if __name__ == "__main__":
    test_teacher_generation()

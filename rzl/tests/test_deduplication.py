"""Test script for verifying Teacher problem deduplication"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import TeacherAgent
from utils.cli_formatter import CLIFormatter

def test_deduplication():
    CLIFormatter.print_header("Testing Problem Deduplication")

    # Create teacher
    teacher = TeacherAgent()

    # Test duplicate detection
    CLIFormatter.print_subheader("Testing Duplicate Detection")

    # Add some test problems to history
    test_questions = [
        "What is 5 + 3?",
        "what is 5 + 3?",  # Same but lowercase
        "What is 5 + 3??",  # Extra punctuation
        "What   is   5   +   3?",  # Extra spaces
        "Sarah has 5 apples and gets 3 more. How many does she have?",
        "SARAH has 5 apples and gets 3 more! How many does she have??",  # Case/punct variation
    ]

    for q in test_questions:
        if teacher._is_duplicate(q):
            CLIFormatter.print_warning(f"Duplicate detected: {q}")
        else:
            CLIFormatter.print_success(f"New problem: {q}")
            teacher._add_to_history(q)

    CLIFormatter.print_status("Unique problems in history", str(len(teacher.problem_history)))

    # Test generation with deduplication
    CLIFormatter.print_subheader("Testing Generation with Deduplication")

    # Generate problems multiple times to test for duplicates
    for i in range(3):
        CLIFormatter.print_info(f"\nGeneration round {i+1}")
        problems = teacher.generate_problems(5, difficulty=0.3)

        for j, p in enumerate(problems):
            CLIFormatter.print_status(f"Problem {j+1}", p['question'][:50] + "..." if len(p['question']) > 50 else p['question'])

        CLIFormatter.print_status("Total unique problems", str(len(teacher.problem_history)))
        CLIFormatter.print_status("Total duplicates avoided", str(teacher.duplicate_count))

    # Test state persistence
    CLIFormatter.print_subheader("Testing State Persistence")

    # Save state
    test_state_path = "test_teacher_state.json"
    teacher.save_state(test_state_path)
    CLIFormatter.print_success("State saved")

    # Create new teacher and load state
    teacher2 = TeacherAgent()
    teacher2.load_state(test_state_path)

    # Verify history was loaded
    if len(teacher2.problem_history) == len(teacher.problem_history):
        CLIFormatter.print_success(f"History loaded correctly: {len(teacher2.problem_history)} problems")
    else:
        CLIFormatter.print_error(f"History mismatch: {len(teacher2.problem_history)} vs {len(teacher.problem_history)}")

    # Clean up
    if os.path.exists(test_state_path):
        os.remove(test_state_path)

    teacher.cleanup()
    teacher2.cleanup()

    CLIFormatter.print_header("Deduplication Test Complete")

if __name__ == "__main__":
    test_deduplication()

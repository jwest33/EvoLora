"""Test script for fixed evaluation set

This script demonstrates using the fixed evaluation questions
to compare base and trained models.
"""

from loralab.evaluation.fixed_eval_set import (
    EVALUATION_QUESTIONS,
    get_evaluation_set,
    format_for_model,
    check_answer,
    get_question_by_category
)

def main():
    print("="*60)
    print("FIXED EVALUATION SET FOR MODEL COMPARISON")
    print("="*60)

    # Get all questions
    all_questions = get_evaluation_set()
    print(f"\nTotal questions: {len(all_questions)}")

    # Show questions by category
    categories = [
        ('Basic Arithmetic', 'basic_arithmetic'),
        ('Percentage Calculations', 'percentage'),
        ('Ratio & Proportion', 'ratio'),
        ('Multi-Step Problems', 'multi_step'),
        ('Time/Rate/Distance', 'time_rate'),
        ('Money/Tax/Discounts', 'money_tax'),
        ('Division with Remainders', 'division_remainder')
    ]

    print("\nQuestions by category:")
    print("-"*60)

    for name, key in categories:
        questions = get_question_by_category(key)
        print(f"\n{name}: {len(questions)} questions")

        # Show first question as example
        if questions:
            q = questions[0]
            print(f"  Example: {q['question'][:80]}...")
            print(f"  Answer: {q['answer']}")

    # Demonstrate formatting for model input
    print("\n" + "="*60)
    print("EXAMPLE MODEL INPUT FORMAT")
    print("="*60)

    sample = all_questions[0]
    formatted = format_for_model(sample)
    print(formatted)

    # Demonstrate answer checking
    print("\n" + "="*60)
    print("ANSWER CHECKING EXAMPLES")
    print("="*60)

    test_cases = [
        ("The answer is 90", "90", True),
        ("Therefore: 15.12", "15.12", True),
        ("6 stickers each, 13 left over", "6 stickers each, 13 left over", True),
        ("The result is 100", "90", False),
    ]

    for output, answer, expected in test_cases:
        result = check_answer(output, answer)
        status = "✓" if result == expected else "✗"
        print(f"{status} Output: '{output}' | Answer: '{answer}' | Match: {result}")

    print("\n" + "="*60)
    print("Fixed evaluation set loaded successfully!")
    print("These questions will be used to compare base vs trained models.")
    print("="*60)

if __name__ == "__main__":
    main()

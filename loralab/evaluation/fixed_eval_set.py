"""Fixed evaluation set for comparing base and trained models

These questions are designed to test various mathematical reasoning capabilities
and are intentionally kept separate from the training data.
"""

# Fixed evaluation questions for model comparison
EVALUATION_QUESTIONS = [
    # Basic arithmetic word problems
    {
        "question": "Sarah has 47 marbles in her collection. Her brother gives her 28 more marbles for her birthday, and her friend gives her 15 marbles. How many marbles does Sarah have now?",
        "answer": "90"
    },
    {
        "question": "A bakery had 234 cupcakes in the morning. They sold 89 cupcakes by noon and 67 cupcakes in the afternoon. How many cupcakes are left?",
        "answer": "78"
    },
    {
        "question": "Tom collected 156 baseball cards. He traded away 45 cards, bought 23 new ones, then gave 12 to his sister. How many cards does Tom have now?",
        "answer": "122"
    },

    # Percentage calculations
    {
        "question": "At a restaurant, the bill comes to $84. If you want to leave an 18% tip, how much should the tip be?",
        "answer": "15.12"
    },
    {
        "question": "Maria scored 42 points out of 60 on her math test. What percentage did she get?",
        "answer": "70"
    },
    {
        "question": "A laptop originally costs $1,200. During a sale, it's offered at 35% off. What is the sale price?",
        "answer": "780"
    },

    # Ratio and proportion problems
    {
        "question": "A recipe calls for flour and sugar in the ratio 5:2. If you use 15 cups of flour, how many cups of sugar do you need?",
        "answer": "6"
    },
    {
        "question": "In an animal shelter, the ratio of dogs to cats is 3:4. If there are 36 cats, how many dogs are there?",
        "answer": "27"
    },
    {
        "question": "On a map, 1 inch represents 25 miles. If two cities are 4.5 inches apart on the map, what is the actual distance between them in miles?",
        "answer": "112.5"
    },

    # Multi-step problems
    {
        "question": "John works 8 hours a day at $15 per hour. He works 5 days a week. After receiving his weekly pay, he spends $180 on groceries and $95 on gas. How much money does he have left from his weekly earnings?",
        "answer": "325"
    },
    {
        "question": "A pizza costs $12 and feeds 3 people. Drinks cost $2.50 each. If you're ordering for a party of 18 people and everyone gets a drink, what's the total cost?",
        "answer": "117"
    },
    {
        "question": "Lisa buys 3 shirts at $24 each, 2 pairs of pants at $38 each, and a jacket for $85. If tax is 8%, what is her total cost?",
        "answer": "253.80"
    },
    {
        "question": "A factory produces 450 units on Monday, 380 units on Tuesday, and 420 units on Wednesday. They need to produce 2,000 units by Friday. If they produce the same amount on Thursday as they did on Wednesday, how many units must they produce on Friday?",
        "answer": "330"
    },

    # Time/rate/distance problems
    {
        "question": "A train travels at 75 miles per hour. How far will it travel in 3 hours and 20 minutes?",
        "answer": "250"
    },
    {
        "question": "Emma can type 65 words per minute. How many words can she type in 45 minutes?",
        "answer": "2925"
    },
    {
        "question": "Two cars start from the same point and drive in opposite directions. One travels at 55 mph and the other at 65 mph. How far apart will they be after 2.5 hours?",
        "answer": "300"
    },

    # Money/shopping with tax or discounts
    {
        "question": "You buy groceries totaling $156.50. If the sales tax is 6.5%, what is your final total?",
        "answer": "166.67"
    },
    {
        "question": "A jacket marked at $180 is first discounted by 20%, then an additional 15% off the reduced price. What is the final price?",
        "answer": "122.40"
    },
    {
        "question": "A gym membership costs $89 per month. Members get a 30% discount on all classes. If a non-member pays $25 for a yoga class, how much does a member pay?",
        "answer": "17.50"
    },

    # Division with remainders in context
    {
        "question": "Mrs. Chen has 157 stickers to distribute equally among her 24 students. How many stickers will each student get, and how many stickers will be left over?",
        "answer": "6 stickers each, 13 left over"
    },
    {
        "question": "A bakery packs cookies in boxes of 12. If they have 146 cookies, how many full boxes can they make, and how many cookies will remain unboxed?",
        "answer": "12 boxes, 2 cookies remain"
    }
]

def get_evaluation_set():
    """Return the fixed evaluation questions"""
    return EVALUATION_QUESTIONS

def get_question_by_category(category):
    """Get questions by category for targeted evaluation

    Categories:
    - basic_arithmetic: indices 0-2
    - percentage: indices 3-5
    - ratio: indices 6-8
    - multi_step: indices 9-12
    - time_rate: indices 13-15
    - money_tax: indices 16-18
    - division_remainder: indices 19-20
    """
    categories = {
        'basic_arithmetic': EVALUATION_QUESTIONS[0:3],
        'percentage': EVALUATION_QUESTIONS[3:6],
        'ratio': EVALUATION_QUESTIONS[6:9],
        'multi_step': EVALUATION_QUESTIONS[9:13],
        'time_rate': EVALUATION_QUESTIONS[13:16],
        'money_tax': EVALUATION_QUESTIONS[16:19],
        'division_remainder': EVALUATION_QUESTIONS[19:21]
    }
    return categories.get(category, [])

def format_for_model(question_dict):
    """Format a question for model input"""
    return f"Question: {question_dict['question']}\n\nLet's solve this step-by-step:"

def check_answer(model_output, correct_answer):
    """Check if model output contains the correct answer

    Args:
        model_output: String output from the model
        correct_answer: The correct answer string

    Returns:
        Boolean indicating if answer is correct
    """
    # Handle answers with "left over" or "remain" for division problems
    if "left over" in correct_answer or "remain" in correct_answer:
        # For division with remainder problems, check both parts
        parts = correct_answer.lower().replace(",", "").split()
        return all(part in model_output.lower().replace(",", "") for part in parts if part.isdigit())

    # For numeric answers, check if the number appears in the output
    answer_num = correct_answer.replace(".", "").replace(",", "")
    output_clean = model_output.replace(".", "").replace(",", "")

    # Check for exact match or with formatting variations
    return (answer_num in output_clean or
            correct_answer in model_output or
            f"= {correct_answer}" in model_output or
            f"is {correct_answer}" in model_output or
            f": {correct_answer}" in model_output)
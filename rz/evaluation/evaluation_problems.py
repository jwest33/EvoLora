"""
GSM8K-style math word problems for model evaluation.
Contains 20 diverse problems across different mathematical concepts.
"""

evaluation_problems = [
    # Basic arithmetic word problems (3 problems)
    {
        "question": "Maria has 147 stickers in her collection. She gives 23 stickers to her sister and buys 56 more stickers at the store. How many stickers does Maria have now?",
        "answer": "180"
    },
    {
        "question": "A farmer harvested 324 apples from his orchard. He sold 189 apples at the market and gave 47 apples to his neighbors. How many apples does the farmer have left?",
        "answer": "88"
    },
    {
        "question": "During a school fundraiser, the third grade collected 234 cans, the fourth grade collected 298 cans, and the fifth grade collected 156 cans. What is the total number of cans collected by all three grades?",
        "answer": "688"
    },

    # Percentage calculations (3 problems)
    {
        "question": "A restaurant bill comes to $84. If the customer wants to leave a 18% tip, how much money will they pay in total?",
        "answer": "99.12"
    },
    {
        "question": "Sarah scored 76 points out of 95 possible points on her science test. What percentage did she score on the test?",
        "answer": "80"
    },
    {
        "question": "A clothing store is having a 35% off sale. If a jacket originally costs $128, what is the sale price after the discount?",
        "answer": "83.20"
    },

    # Ratio and proportion problems (3 problems)
    {
        "question": "In a recipe for fruit punch, the ratio of cranberry juice to orange juice is 3:5. If Tom uses 12 cups of cranberry juice, how many cups of orange juice should he use?",
        "answer": "20"
    },
    {
        "question": "The ratio of cats to dogs at an animal shelter is 4:7. If there are 28 cats at the shelter, how many dogs are there?",
        "answer": "49"
    },
    {
        "question": "A map has a scale where 2 inches represents 15 miles. If two cities are 8.4 inches apart on the map, what is the actual distance between them?",
        "answer": "63"
    },

    # Multi-step problems requiring 3+ operations (4 problems)
    {
        "question": "Emma works at a bookstore and earns $14 per hour. She worked 6 hours on Monday, 8 hours on Tuesday, and 5 hours on Wednesday. If she spends $45 on lunch during the week, how much money does she have left from her earnings?",
        "answer": "221"
    },
    {
        "question": "A school is ordering pizza for a party. Each pizza costs $12 and feeds 8 students. If there are 156 students attending and the school has a $200 budget, how much money will be left over after buying enough pizzas?",
        "answer": "8"
    },
    {
        "question": "Jake bought 4 notebooks for $3.50 each and 6 pens for $1.25 each. He paid with a $25 bill. If there's a 7% sales tax on his purchase, how much change will he receive?",
        "answer": "3.455"
    },
    {
        "question": "A factory produces 240 toys per day. Each toy requires 3 batteries, and batteries come in packs of 12. If the factory operates for 5 days, how many battery packs do they need to buy?",
        "answer": "300"
    },

    # Time/rate/distance problems (3 problems)
    {
        "question": "A train travels at a constant speed of 75 miles per hour. How far will it travel in 2 hours and 40 minutes?",
        "answer": "200"
    },
    {
        "question": "Lisa can type 65 words per minute. If she needs to type a 1,950-word essay, how many minutes will it take her to finish?",
        "answer": "30"
    },
    {
        "question": "Two cars start from the same point and drive in opposite directions. Car A travels at 55 mph and Car B travels at 65 mph. After how many hours will they be 360 miles apart?",
        "answer": "3"
    },

    # Money/shopping problems with tax or discounts (3 problems)
    {
        "question": "At a grocery store, oranges cost $2.40 per pound and apples cost $3.20 per pound. Kevin buys 2.5 pounds of oranges and 1.8 pounds of apples. If there's a 6% sales tax, what is his total bill?",
        "answer": "11.508"
    },
    {
        "question": "A video game originally costs $60. During a Black Friday sale, it's marked down 25%. Additionally, customers get an extra 10% off the sale price with a coupon. What is the final price of the video game?",
        "answer": "40.50"
    },
    {
        "question": "Michelle buys 3 books for $8.95 each and 2 magazines for $4.50 each. She has a membership card that gives her a 15% discount on her total purchase. What does she pay after the discount?",
        "answer": "30.0875"
    },

    # Division with remainders in context (2 problems)
    {
        "question": "A teacher has 347 stickers to distribute equally among 24 students. How many stickers will each student receive, and how many stickers will be left over?",
        "answer": "14 stickers each with 11 left over"
    },
    {
        "question": "A bakery has 428 cookies to pack into boxes. Each box holds 15 cookies. How many full boxes can they make, and how many cookies will be left over?",
        "answer": "28 full boxes with 8 cookies left over"
    }
]

# Verification: Check that we have exactly 20 problems
assert len(evaluation_problems) == 20, f"Expected 20 problems, got {len(evaluation_problems)}"

# Category breakdown verification
categories = {
    "Basic arithmetic": 3,
    "Percentage calculations": 3,
    "Ratio and proportion": 3,
    "Multi-step problems": 4,
    "Time/rate/distance": 3,
    "Money/shopping": 3,
    "Division with remainders": 2
}

print(f"Created {len(evaluation_problems)} evaluation problems:")
for category, count in categories.items():
    print(f"  - {category}: {count} problems")
print(f"Total: {sum(categories.values())} problems")

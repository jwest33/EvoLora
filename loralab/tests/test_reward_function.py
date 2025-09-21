"""Test the reward function to ensure it properly extracts and rewards answers"""

import re

def calculate_reward(text, answer):
    """Simplified version of the reward function for testing"""
    reward = 0.0

    # Extract the final numeric answer from GSM8K format
    correct_answer_full = str(answer).strip()
    if "####" in correct_answer_full:
        correct_answer = correct_answer_full.split("####")[-1].strip()
    else:
        correct_answer = correct_answer_full

    print(f"Extracted answer: '{correct_answer}'")

    # Try to find the answer in various formats
    answer_found = False

    # First, simple string check - if the exact answer appears anywhere
    if correct_answer in text:
        answer_found = True
        print(f"  [OK] Found exact match: '{correct_answer}' in text")
        # Check position and quality
        if text.strip().endswith(correct_answer):
            reward += 10.0  # Perfect - answer at the end
            print(f"  [OK] Answer at the end: +10.0")
        elif "answer" in text.lower() and correct_answer in text.split("answer")[-1]:
            reward += 8.0  # Good - answer after "answer" keyword
            print(f"  [OK] Answer after 'answer' keyword: +8.0")
        else:
            reward += 6.0  # OK - answer somewhere in text
            print(f"  [OK] Answer somewhere in text: +6.0")

    # If not found, try numeric matching
    if not answer_found:
        print(f"  [X] No exact match, trying numeric matching...")
        try:
            # Extract all numbers from the generated text
            # Better regex that handles commas and doesn't break on periods
            gen_numbers = re.findall(r'-?\d+(?:,\d{3})*(?:\.\d+)?', text)
            print(f"  Numbers found in text: {gen_numbers}")

            # Try to parse the correct answer as a number
            correct_num = float(correct_answer.replace(',', ''))
            print(f"  Correct number: {correct_num}")

            # Check if any generated number matches
            for gen_num_str in gen_numbers:
                try:
                    gen_num = float(gen_num_str.replace(',', ''))
                    diff = abs(gen_num - correct_num)
                    if diff < 0.1:  # Generous tolerance
                        answer_found = True
                        print(f"  [OK] Numeric match found: {gen_num}")
                        # Position-based reward
                        if gen_numbers.index(gen_num_str) == len(gen_numbers) - 1:
                            reward += 8.0  # Last number (likely final answer)
                            print(f"  [OK] Last number in text: +8.0")
                        else:
                            reward += 5.0  # Number appears somewhere
                            print(f"  [OK] Number appears somewhere: +5.0")
                        break
                    elif diff < 1.0:  # Within 1.0 - partial credit
                        answer_found = True
                        print(f"  [OK] Close match found: {gen_num} (diff: {diff:.2f})")
                        reward += 3.0  # Partial credit for being close
                        print(f"  [OK] Close to answer: +3.0")
                        break
                except:
                    continue
        except (ValueError, AttributeError) as e:
            print(f"  [X] Error in numeric parsing: {e}")

    # Heavy penalty for wrong or missing answer
    if not answer_found:
        print(f"  [X] Answer not found")
        # Check if text was truncated while potentially getting to answer
        if text.endswith("...") or len(text) > 250:
            reward -= 1.0
            print(f"  - Text appears truncated: -1.0")
        # Check if there's any attempt at an answer
        elif re.search(r'answer|total|result|therefore|=\s*\d+', text.lower()):
            reward -= 3.0  # Wrong answer but tried
            print(f"  - Wrong answer but attempted: -3.0")
        else:
            reward -= 8.0  # No answer attempt
            print(f"  - No answer attempt: -8.0")

    return reward


# Test cases
test_cases = [
    {
        "name": "Perfect match at end",
        "text": "Let me calculate: 25 * 2 = 50, then 50 + 25 = 75. In 10 hours: 75 * 10 = 750",
        "answer": "#### 750"
    },
    {
        "name": "Answer after keyword",
        "text": "First, Nathan writes 25 letters. Jacob writes twice as fast, so 50. The answer is 750 letters.",
        "answer": "Since Nathan writes 25 letters in an hour, Jacob writes 25*2 = <<25*2=50>>50 letters in an hour.\nTogether, they write 50+25 = <<50+25=75>>75 letters in an hour,\nIn 10 hours, they'll write a total of 75*10 = <<10*75=750>>750 letters\n#### 750"
    },
    {
        "name": "Truncated text",
        "text": "Here's the breakdown:\n1. Nathan's speed is 6 letters per hour.\n2. Calculate the total letters: 6 + 2 = 8 letters.\n3. Calculate for 10 hours: So, the tota...",
        "answer": "#### 750"
    },
    {
        "name": "Wrong answer but tried",
        "text": "Nathan writes 25 letters per hour. Jacob writes faster. Together they write many letters. The total is 500.",
        "answer": "#### 750"
    },
    {
        "name": "Correct answer in middle",
        "text": "Nathan: 25 letters/hour. Jacob: 50 letters/hour. Together: 75. For 10 hours: 750. But wait, let me recalculate...",
        "answer": "#### 750"
    },
    {
        "name": "No answer attempt",
        "text": "This is a complex problem involving letter writing speeds.",
        "answer": "#### 750"
    },
    {
        "name": "Answer with dollar sign",
        "text": "The total cost is $42",
        "answer": "#### 42"
    },
    {
        "name": "Answer with commas",
        "text": "The population is 1,250 people",
        "answer": "#### 1250"
    },
    {
        "name": "Close but not exact",
        "text": "After calculating everything, I get 749.5",
        "answer": "#### 750"
    },
    {
        "name": "Multiple numbers, answer at end",
        "text": "Step 1: 25 letters. Step 2: 50 letters. Step 3: 75 letters per hour. Final: 750",
        "answer": "#### 750"
    }
]

print("="*60)
print("TESTING REWARD FUNCTION")
print("="*60)

for test in test_cases:
    print(f"\nTest: {test['name']}")
    print(f"Text: {test['text'][:100]}...")
    print(f"Answer: {test['answer'][:50]}...")

    reward = calculate_reward(test['text'], test['answer'])

    print(f"Final Reward: {reward:.1f}")
    print("-"*40)

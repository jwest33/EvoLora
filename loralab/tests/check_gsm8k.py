from datasets import load_dataset

# Load a few GSM8K examples
ds = load_dataset('gsm8k', 'main', split='train')

print("GSM8K Answer Format Examples:")
print("="*60)

for i in range(5):
    example = ds[i]
    print(f"\nExample {i+1}:")
    print(f"Question: {example['question'][:100]}...")
    print(f"Full Answer: {example['answer'][:200]}...")

    # Extract just the number
    if "####" in example['answer']:
        final_answer = example['answer'].split("####")[-1].strip()
        print(f"Final Answer (after ####): '{final_answer}'")

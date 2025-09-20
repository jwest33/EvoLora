"""Simple inference test for Gemma3 GRPO model
Quick script to test your trained model
"""

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
import torch

# Path to your trained adapter (update this to your latest run)
ADAPTER_PATH = "single_grpo_runs/gemma3_20250919_110648/adapter"

# Load the base model
print("Loading Gemma3-270M...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/gemma-3-270m-it",
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=False,
)

# Load your trained LoRA adapter
print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = FastLanguageModel.from_pretrained(
    model_name=ADAPTER_PATH,
    max_seq_length=2048,
    dtype=torch.float16,
    load_in_4bit=False,
)

# Set up for inference
FastLanguageModel.for_inference(model)

# Apply Gemma3 chat template
tokenizer = get_chat_template(tokenizer, chat_template="gemma3")

# Test with a math problem
print("\n" + "="*60)
print("Testing with a math problem:")
print("="*60)

question = "John has 12 cookies. He gives 3 to his sister and 2 to his friend. How many cookies does John have left?"

# Format as conversation
messages = [
    {"role": "user", "content": question}
]

# Apply chat template and generate
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt"
).to("cuda")

# Generate response with streaming
print(f"\nQuestion: {question}")
print("\nAnswer: ", end="")

from transformers import TextStreamer
streamer = TextStreamer(tokenizer, skip_prompt=True)

outputs = model.generate(
    input_ids=inputs,
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    streamer=streamer,
    use_cache=True,
)

print("\n" + "="*60)

# Interactive mode
print("\nEntering interactive mode (type 'quit' to exit)...")
print("-"*60)

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['quit', 'exit']:
        break

    messages = [{"role": "user", "content": user_input}]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    print("Model: ", end="")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=256,
        temperature=0.7,
        top_p=0.9,
        streamer=streamer,
        use_cache=True,
    )
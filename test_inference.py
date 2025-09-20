"""Test inference with trained GRPO model
Use this script to test your trained LoRA adapter with sample prompts.
"""

import torch
import sys
import os
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Disable multiprocessing on Windows
import platform
if platform.system() == 'Windows':
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Unsloth
from loralab.utils.unsloth_config import init_unsloth
UNSLOTH_AVAILABLE = init_unsloth()

if UNSLOTH_AVAILABLE:
    from unsloth import FastLanguageModel
else:
    raise ImportError("Unsloth is required for this script")


def load_trained_model(adapter_path: str, base_model: str = "unsloth/gemma-3-270m-it"):
    """Load the base model with trained LoRA adapter

    Args:
        adapter_path: Path to the saved LoRA adapter
        base_model: Base model name (default: gemma3-270m)

    Returns:
        model, tokenizer tuple
    """
    print(f"Loading base model: {base_model}")

    # Load base model and tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=False,  # Gemma3-270M is small, no need for quantization
    )

    # Load the LoRA adapter
    print(f"Loading LoRA adapter from: {adapter_path}")
    model = FastLanguageModel.get_peft_model(
        model,
        adapter_name=adapter_path,
    )

    # Alternatively, if the adapter was saved with save_pretrained:
    from peft import PeftModel
    try:
        model = PeftModel.from_pretrained(model, adapter_path)
        print("Loaded adapter successfully!")
    except:
        print("Note: Adapter loading may require the exact saving format used")

    # Enable inference mode
    FastLanguageModel.for_inference(model)

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 256):
    """Generate a response for a given prompt

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        prompt: Input prompt
        max_new_tokens: Maximum tokens to generate

    Returns:
        Generated text
    """
    # Format the prompt for Gemma3
    messages = [
        {"role": "user", "content": prompt}
    ]

    # Apply chat template
    from unsloth.chat_templates import get_chat_template
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma3",
    )

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # Tokenize
    inputs = tokenizer(text, return_tensors="pt").to("cuda")

    # Generate
    from transformers import TextStreamer
    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        top_k=40,
        streamer=text_streamer,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Decode the full response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the assistant's response
    if "model\n" in response:
        response = response.split("model\n")[-1].strip()

    return response


def test_math_problems(model, tokenizer):
    """Test the model with some math problems"""

    test_problems = [
        "John has 5 apples. He gives 2 to Mary. How many apples does John have left?",
        "A store sells pencils for $2 each. If I buy 7 pencils, how much will I pay?",
        "Sarah walks 3 miles every day. How many miles does she walk in a week?",
        "If a pizza is cut into 8 slices and Tom eats 3 slices, what fraction of the pizza is left?",
        "A car travels 60 miles per hour. How far will it travel in 2.5 hours?",
    ]

    print("\n" + "="*80)
    print("TESTING MATH PROBLEMS")
    print("="*80)

    for i, problem in enumerate(test_problems, 1):
        print(f"\n--- Problem {i} ---")
        print(f"Question: {problem}")
        print(f"Answer: ", end="")

        response = generate_response(model, tokenizer, problem, max_new_tokens=256)
        print("\n")


def interactive_chat(model, tokenizer):
    """Interactive chat mode"""
    print("\n" + "="*80)
    print("INTERACTIVE CHAT MODE")
    print("="*80)
    print("Type 'quit' or 'exit' to stop")
    print("Type 'clear' to clear the screen")
    print("-"*80)

    while True:
        # Get user input
        user_input = input("\nYou: ").strip()

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if user_input.lower() == 'clear':
            os.system('cls' if os.name == 'nt' else 'clear')
            continue

        if not user_input:
            continue

        # Generate response
        print("\nModel: ", end="")
        response = generate_response(model, tokenizer, user_input, max_new_tokens=256)
        print("\n" + "-"*80)


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Test inference with trained GRPO model")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="single_grpo_runs/gemma3_20250919_110648/adapter",
        help="Path to the trained LoRA adapter"
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="unsloth/gemma-3-270m-it",
        help="Base model name"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["test", "chat", "both"],
        default="both",
        help="Mode: test (run test problems), chat (interactive), or both"
    )

    args = parser.parse_args()

    # Check if adapter path exists
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        print(f"Error: Adapter path does not exist: {adapter_path}")
        print("\nAvailable adapters:")

        # List available adapters
        for run_dir in Path("single_grpo_runs").glob("*/adapter"):
            print(f"  - {run_dir}")

        return

    # Load model
    print("Loading model and adapter...")
    model, tokenizer = load_trained_model(str(adapter_path), args.base_model)
    print("Model loaded successfully!")

    # Run based on mode
    if args.mode in ["test", "both"]:
        test_math_problems(model, tokenizer)

    if args.mode in ["chat", "both"]:
        interactive_chat(model, tokenizer)


if __name__ == "__main__":
    main()
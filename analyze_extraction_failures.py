"""Analyze why ExtractUserInfo has low tool selection accuracy"""

import json
from pathlib import Path
from llama_cpp import Llama
from loralab.benchmarks.dataset_utils import load_dataset, get_latest_dataset
from loralab.benchmarks.qwen_tool_evaluator import QwenToolEvaluator

def analyze_failures(model_path: str):
    """Test all ExtractUserInfo examples to see what tools are being selected"""

    # Load dataset
    dataset_path = get_latest_dataset()
    data = load_dataset(dataset_path)

    # Initialize evaluator
    evaluator = QwenToolEvaluator(model_path)

    # Filter for ExtractUserInfo examples
    extraction_examples = [
        (i, ex) for i, ex in enumerate(data["examples"])
        if ex["expected_tool"] == "ExtractUserInfo"
    ]

    print("=" * 60)
    print("ANALYZING EXTRACTUSERINFO TOOL SELECTION")
    print("=" * 60)
    print(f"Total ExtractUserInfo examples: {len(extraction_examples)}")

    # Track what tools are being selected instead
    tool_selections = {}
    failures = []

    for idx, example in extraction_examples:
        result = evaluator.evaluate_example(example, data["tools"])

        predicted = result.predicted_tool or "(no tool)"
        if predicted not in tool_selections:
            tool_selections[predicted] = []
        tool_selections[predicted].append(idx)

        if not result.correct_tool:
            failures.append((idx, example, result))

    # Show results
    print(f"\nTool Selection Distribution:")
    for tool, indices in tool_selections.items():
        print(f"  {tool}: {len(indices)} times")

    # Show specific failures
    print(f"\n" + "=" * 60)
    print(f"FAILED EXAMPLES (selected wrong tool):")
    print("=" * 60)

    for idx, example, result in failures[:5]:  # Show first 5 failures
        print(f"\nExample {idx}:")
        print(f"  Input: {example['user_input']}")
        print(f"  Expected: ExtractUserInfo")
        print(f"  Selected: {result.predicted_tool or '(no tool)'}")
        if result.predicted_params:
            print(f"  Params: {json.dumps(result.predicted_params, indent=4)}")


def test_with_only_extract_tool(model_path: str):
    """Test if model performs better with only ExtractUserInfo available"""

    # Load dataset
    dataset_path = get_latest_dataset()
    data = load_dataset(dataset_path)

    # Get only ExtractUserInfo tool
    extract_tool = None
    for tool in data["tools"]:
        if tool["function"]["name"] == "ExtractUserInfo":
            extract_tool = tool
            break

    # Initialize evaluator
    evaluator = QwenToolEvaluator(model_path)

    # Test a few examples with ONLY the extract tool
    extraction_examples = [
        ex for ex in data["examples"][:10]
        if ex["expected_tool"] == "ExtractUserInfo"
    ]

    print("\n" + "=" * 60)
    print("TESTING WITH ONLY EXTRACTUSERINFO TOOL")
    print("=" * 60)

    for i, example in enumerate(extraction_examples[:3]):
        print(f"\nExample {i+1}:")
        print(f"  Input: {example['user_input']}")

        # Test with all tools
        result_all = evaluator.evaluate_example(example, data["tools"])

        # Test with only extract tool
        result_single = evaluator.evaluate_example(example, [extract_tool])

        print(f"  With all tools: {result_all.predicted_tool or '(no tool)'}")
        print(f"  With only ExtractUserInfo: {result_single.predicted_tool or '(no tool)'}")


def check_instruction_clarity(model_path: str):
    """Test if clearer instructions help"""

    dataset_path = get_latest_dataset()
    data = load_dataset(dataset_path)

    llm = Llama(
        model_path=model_path,
        n_ctx=8192,
        n_gpu_layers=-1,
        verbose=False
    )

    test_input = "My name is Bob and I'm 42 years old"

    print("\n" + "=" * 60)
    print("TESTING DIFFERENT PROMPTS")
    print("=" * 60)

    # Test 1: Current format
    prompt1 = f"""You are a helpful assistant with access to the following functions:

ExtractUserInfo: Extract user information from text
  - name (string): User's name (required)
  - age (integer): User's age (optional)
  - email (string): User's email address (optional)
  - location (string): User's location (optional)

CalculateMath: Perform mathematical calculations
  - operation (string): Operation to perform (required)
  - operands (array): Numbers to operate on (required)

When you need to use a function, respond with a JSON object in this exact format:
{{"function": "function_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}

User: {test_input}
Assistant:"""

    response1 = llm(prompt1, max_tokens=100, temperature=0.1, stop=["User:"])
    print(f"\nStandard prompt response: {response1['choices'][0]['text'].strip()}")

    # Test 2: More explicit instruction
    prompt2 = f"""You are a helpful assistant. Your task is to extract information from the user's message.

Available function:
ExtractUserInfo - extracts name, age, email, and location from text

The user said: "{test_input}"

This message contains personal information that should be extracted.

Respond with: {{"function": "ExtractUserInfo", "arguments": {{"name": "...", "age": ...}}}}
Assistant:"""

    response2 = llm(prompt2, max_tokens=100, temperature=0.1, stop=["User:"])
    print(f"\nExplicit prompt response: {response2['choices'][0]['text'].strip()}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Analyze extraction failures")
    parser.add_argument("--model", required=True, help="Model path")
    parser.add_argument("--mode", choices=["failures", "single", "prompts"], default="failures")

    args = parser.parse_args()

    if args.mode == "failures":
        analyze_failures(args.model)
    elif args.mode == "single":
        test_with_only_extract_tool(args.model)
    elif args.mode == "prompts":
        check_instruction_clarity(args.model)


if __name__ == "__main__":
    main()

"""Show the exact prompt being sent to the model"""

from loralab.benchmarks.dataset_utils import load_dataset, get_latest_dataset

def show_full_prompt():
    """Display the complete prompt the model receives"""

    # Load dataset
    dataset_path = get_latest_dataset()
    data = load_dataset(dataset_path)

    # Get an example
    example = data["examples"][0]

    # Format tools exactly as the evaluator does
    def format_tools_for_qwen(tools):
        tool_descriptions = []
        for tool in tools:
            func = tool["function"]
            name = func["name"]
            desc = func["description"]
            params = func.get("parameters", {})

            tool_str = f"{name}: {desc}"

            if "properties" in params:
                param_list = []
                for param_name, param_spec in params["properties"].items():
                    param_type = param_spec.get("type", "string")
                    param_desc = param_spec.get("description", "")
                    required = param_name in params.get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    param_list.append(f"  - {param_name} ({param_type}): {param_desc}{req_str}")

                if param_list:
                    tool_str += "\n" + "\n".join(param_list)

            tool_descriptions.append(tool_str)

        return "\n\n".join(tool_descriptions)

    # Create the full prompt
    tools_desc = format_tools_for_qwen(data["tools"])

    prompt = f"""You are a helpful assistant with access to the following functions:

{tools_desc}

When you need to use a function, respond with a JSON object in this exact format:
{{"function": "function_name", "arguments": {{"param1": "value1", "param2": "value2"}}}}

Important rules:
1. Output ONLY ONE function call - choose the MOST appropriate function
2. Output ONLY the JSON object, nothing else
3. Do NOT output multiple function calls
4. Do NOT add explanatory text
5. Use the exact function names provided
6. Include all required parameters for your chosen function

User: {example["user_input"]}
Assistant:"""

    print("=" * 60)
    print("COMPLETE PROMPT SENT TO MODEL")
    print("=" * 60)
    print(prompt)
    print("=" * 60)
    print(f"\nPrompt length: {len(prompt)} characters")
    print(f"Example was: {example['expected_tool']} tool")

if __name__ == "__main__":
    show_full_prompt()

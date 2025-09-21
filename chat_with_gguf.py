#!/usr/bin/env python
"""Interactive chat interface for GGUF models using llama.cpp"""

import os
import sys
from pathlib import Path
from typing import Optional, List, Dict
from llama_cpp import Llama
from loralab.utils.cli_formatter import CLIFormatter, Style

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Reasoning format markers
REASONING_START = "<start_working_out>"
REASONING_END = "</start_working_out>"
SOLUTION_START = "<SOLUTION>"
SOLUTION_END = "</SOLUTION>"


class GGUFChat:
    """Interactive chat interface for GGUF models"""

    def __init__(
        self,
        model_path: str,
        system_prompt: Optional[str] = None,
        n_ctx: int = 4096,
        n_threads: int = 8,
        n_gpu_layers: int = -1,
        chat_format: str = "chatml",
        verbose: bool = False
    ):
        """Initialize chat interface

        Args:
            model_path: Path to GGUF model file
            system_prompt: System prompt for the model
            n_ctx: Context window size
            n_threads: Number of CPU threads
            n_gpu_layers: Number of layers to offload to GPU (-1 for all)
            chat_format: Chat template format
            verbose: Enable verbose llama.cpp output
        """
        self.model_path = Path(model_path)
        self.system_prompt = system_prompt or self.get_default_system_prompt()
        self.messages: List[Dict[str, str]] = []

        # Validate model exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        # Initialize llama.cpp model
        CLIFormatter.print_info("Loading model...")

        try:
            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                chat_format=chat_format,
                verbose=verbose
            )
            CLIFormatter.print_success("Model loaded successfully!")

            # Display model info
            model_size = self.model_path.stat().st_size / (1024 ** 3)
            CLIFormatter.print_status("Model", self.model_path.name)
            CLIFormatter.print_status("Size", f"{model_size:.2f} GB")
            CLIFormatter.print_status("Context", f"{n_ctx} tokens")
            CLIFormatter.print_status("GPU layers", "All" if n_gpu_layers == -1 else str(n_gpu_layers))

        except Exception as e:
            raise RuntimeError(f"Failed to load model: {e}")

        # Initialize conversation
        self.reset_conversation()

    @staticmethod
    def get_default_system_prompt() -> str:
        """Get the default R-Zero solver system prompt"""
        return """You are a helpful math tutor who solves problems step by step.

For each math problem:
1. First, write out your reasoning process inside <start_working_out> tags
2. Show each calculation step clearly
3. Explain your thinking at each step
4. Then provide the final answer inside <SOLUTION> tags

Example format:
<start_working_out>
Let me break down this problem:
- First, I need to identify what we're looking for
- Then I'll work through the calculations:
  * Step 1: ...
  * Step 2: ...
- Therefore, the answer is...
</start_working_out>
<SOLUTION>The answer is [your answer]</SOLUTION>

Remember to be clear and thorough in your explanations."""

    def reset_conversation(self):
        """Reset the conversation history"""
        self.messages = [{"role": "system", "content": self.system_prompt}]
        CLIFormatter.print_success("Conversation reset")

    def check_reasoning_format(self, response: str) -> bool:
        """Check if response uses correct reasoning format

        Args:
            response: Model's response

        Returns:
            True if format is correct
        """
        has_reasoning = REASONING_START in response and REASONING_END in response
        has_solution = SOLUTION_START in response and SOLUTION_END in response
        return has_reasoning and has_solution

    def generate_response(
        self,
        user_input: str,
        max_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
        stream: bool = True
    ) -> str:
        """Generate a response from the model

        Args:
            user_input: User's message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stream: Whether to stream the response

        Returns:
            Model's complete response
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_input})

        try:
            if stream:
                # Stream the response
                response_generator = self.llm.create_chat_completion(
                    messages=self.messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=True
                )

                full_response = ""
                for chunk in response_generator:
                    if chunk['choices'][0]['delta'].get('content'):
                        text = chunk['choices'][0]['delta']['content']
                        print(f"{CLIFormatter.ELECTRIC_BLUE}{text}{Style.RESET_ALL}", end="", flush=True)
                        full_response += text

            else:
                # Generate without streaming
                response = self.llm.create_chat_completion(
                    messages=self.messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stream=False
                )
                full_response = response['choices'][0]['message']['content']
                print(f"{CLIFormatter.ELECTRIC_BLUE}{full_response}{Style.RESET_ALL}")

            # Add assistant response to history
            self.messages.append({"role": "assistant", "content": full_response})

            return full_response

        except Exception as e:
            # Remove the failed user message
            self.messages.pop()
            raise RuntimeError(f"Generation failed: {e}")

    def run_interactive_chat(self):
        """Run the interactive chat loop"""
        CLIFormatter.print_header("Interactive Chat with R-Zero Solver")

        CLIFormatter.print_info("Commands:")
        CLIFormatter.print_list_item("'exit' or 'quit' - Exit the chat")
        CLIFormatter.print_list_item("'clear' or 'reset' - Clear conversation history")
        CLIFormatter.print_list_item("'help' - Show this help message")
        CLIFormatter.print_list_item("'system <prompt>' - Change system prompt")

        print()
        CLIFormatter.print_info("Example questions:")
        CLIFormatter.print_list_item("Tommy has 5 apples and gets 3 more. How many does he have?")
        CLIFormatter.print_list_item("A bakery sold 24 cookies in the morning and 18 in the afternoon. How many total?")
        print()

        while True:
            try:
                # Get user input
                print(f"{CLIFormatter.NEON_CYAN}You: {Style.RESET_ALL}", end="")
                user_input = input().strip()

                # Handle commands
                if user_input.lower() in ['exit', 'quit']:
                    CLIFormatter.print_info("Goodbye!")
                    break

                elif user_input.lower() in ['clear', 'reset']:
                    self.reset_conversation()
                    continue

                elif user_input.lower() == 'help':
                    self.show_help()
                    continue

                elif user_input.lower().startswith('system '):
                    new_prompt = user_input[7:].strip()
                    if new_prompt:
                        self.system_prompt = new_prompt
                        self.reset_conversation()
                        CLIFormatter.print_success("System prompt updated")
                    continue

                elif not user_input:
                    continue

                # Generate response
                print(f"{CLIFormatter.NEON_PINK}Solver: {Style.RESET_ALL}", end="")

                response = self.generate_response(user_input)
                print()  # New line after response

                # Check format if it's a math problem
                keywords = ['how many', 'calculate', 'solve', 'what is', 'find', '+', '-', '*', '/', '=']
                is_math = any(kw in user_input.lower() for kw in keywords)

                if is_math:
                    if self.check_reasoning_format(response):
                        print(f"{CLIFormatter.NEON_GREEN}✓ Model used correct reasoning format!{Style.RESET_ALL}")
                    else:
                        print(f"{CLIFormatter.SUNSET_ORANGE}⚠ Model didn't use full reasoning format{Style.RESET_ALL}")

                print()  # Extra line for readability

            except KeyboardInterrupt:
                print()
                CLIFormatter.print_info("Use 'exit' to quit")
                continue

            except Exception as e:
                CLIFormatter.print_error(f"Error: {e}")
                continue

    def show_help(self):
        """Display help information"""
        CLIFormatter.print_subheader("Chat Commands")
        CLIFormatter.print_list_item("'exit' or 'quit' - Exit the chat")
        CLIFormatter.print_list_item("'clear' or 'reset' - Clear conversation history")
        CLIFormatter.print_list_item("'help' - Show this help message")
        CLIFormatter.print_list_item("'system <prompt>' - Change system prompt")

        CLIFormatter.print_subheader("Tips")
        CLIFormatter.print_list_item("Ask math problems to see step-by-step solutions")
        CLIFormatter.print_list_item("The model will show its reasoning process")
        CLIFormatter.print_list_item("Clear the conversation if responses become inconsistent")


def find_gguf_models(directory: str = "outputs/gguf") -> List[Path]:
    """Find all GGUF models in a directory

    Args:
        directory: Directory to search

    Returns:
        List of GGUF file paths
    """
    gguf_dir = Path(directory)
    if not gguf_dir.exists():
        return []

    return sorted(gguf_dir.glob("*.gguf"))


def select_model(models: List[Path]) -> Path:
    """Interactive model selection

    Args:
        models: List of available models

    Returns:
        Selected model path
    """
    CLIFormatter.print_subheader("Available Models")

    for i, model in enumerate(models, 1):
        size_gb = model.stat().st_size / (1024 ** 3)
        CLIFormatter.print_list_item(f"{i}. {model.name} ({size_gb:.2f} GB)")

    print()
    while True:
        try:
            choice = input("Select model (number): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                return models[idx]
            else:
                CLIFormatter.print_warning("Invalid selection")
        except ValueError:
            CLIFormatter.print_warning("Please enter a number")


def main():
    """Main entry point for chat script"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Chat with R-Zero GGUF models using llama.cpp",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chat_with_gguf.py                     # Auto-find and select model
  python chat_with_gguf.py --model model.gguf  # Use specific model
  python chat_with_gguf.py --gpu-layers 20     # Offload 20 layers to GPU
  python chat_with_gguf.py --context 8192      # Use 8K context window
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        help="Path to GGUF model file"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="outputs/gguf",
        help="Directory to search for models (default: outputs/gguf)"
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt"
    )
    parser.add_argument(
        "--context",
        type=int,
        default=4096,
        help="Context window size (default: 4096)"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=8,
        help="Number of CPU threads (default: 8)"
    )
    parser.add_argument(
        "--gpu-layers",
        type=int,
        default=-1,
        help="Number of layers to offload to GPU (default: -1 for all)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p sampling (default: 0.95)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per response (default: 512)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose llama.cpp output"
    )

    args = parser.parse_args()

    # Find or select model
    if args.model:
        model_path = Path(args.model)
        if not model_path.exists():
            CLIFormatter.print_error(f"Model not found: {args.model}")
            sys.exit(1)
    else:
        # Search for models
        CLIFormatter.print_info(f"Searching for models in {args.model_dir}...")
        models = find_gguf_models(args.model_dir)

        if not models:
            CLIFormatter.print_error(f"No GGUF models found in {args.model_dir}")
            CLIFormatter.print_info("Run 'python export_to_gguf.py' to export a model first")
            sys.exit(1)

        if len(models) == 1:
            model_path = models[0]
            CLIFormatter.print_info(f"Using model: {model_path.name}")
        else:
            model_path = select_model(models)

    # Initialize chat
    try:
        chat = GGUFChat(
            model_path=str(model_path),
            system_prompt=args.system_prompt,
            n_ctx=args.context,
            n_threads=args.threads,
            n_gpu_layers=args.gpu_layers,
            verbose=args.verbose
        )

        # Set generation parameters
        chat.temperature = args.temperature
        chat.top_p = args.top_p
        chat.max_tokens = args.max_tokens

        # Run interactive chat
        chat.run_interactive_chat()

    except Exception as e:
        CLIFormatter.print_error(f"Failed to initialize chat: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

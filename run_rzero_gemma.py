"""R-Zero implementation using Gemma models for both Challenger and Solver"""
from unsloth import FastLanguageModel
import os
import sys
import torch
import gc
from typing import List, Dict, Any
from datasets import Dataset, load_dataset
from transformers import TrainingArguments, Trainer
import json
from tqdm import tqdm

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from loralab.utils.cli_formatter import CLIFormatter, SpinnerProgress

# Disable multiprocessing to avoid Windows issues
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

def clear_memory():
    """Clear CUDA memory"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()

class ChallengerAgent:
    """Qwen3-4B Challenger that generates difficult problems"""

    def __init__(self):
        CLIFormatter.print_info("Initializing Challenger...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3-4b-it-unsloth-bnb-4bit", # "unsloth/gemma-3-1b-it-unsloth-bnb-4bit" is stable
            max_seq_length=1024,
            load_in_4bit = True,  # 4 bit quantization to reduce memory
            load_in_8bit = False
        )

        # Apply LoRA for fine-tuning
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=8,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=16,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
            use_rslora=False,  # We support rank stabilized LoRA
            loftq_config=None  # And LoftQ
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        CLIFormatter.print_success(f"Challenger loaded: {trainable:,} trainable params")

    def extract_answer(self, question: str, response: str) -> str:
        """Use LLM to extract the final numerical answer from a response

        Args:
            question: The math problem
            response: The full response containing the answer

        Returns:
            str: The extracted numerical answer
        """
        self.model.eval()

        prompt = f"""Question: {question}
Response: {response}

Extract ONLY the final numerical answer from the response above.
If the answer is a fraction, convert it to decimal.
Reply with just the number, nothing else.
Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,  # Low temperature for consistent extraction
                do_sample=False,  # Greedy for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        extracted = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Handle multi-line responses - get the LAST non-empty line
        lines = extracted.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        if lines:
            extracted = lines[-1]  # Get the last line

        # If "Final Answer:" appears, extract after the LAST occurrence
        if "final answer" in extracted.lower():
            parts = extracted.lower().split("final answer")
            if len(parts) > 1:
                extracted = parts[-1].strip()
                # Remove colon if present
                if extracted.startswith(':'):
                    extracted = extracted[1:].strip()

        # Clean up common patterns
        extracted = extracted.replace("$", "").replace(",", "")
        extracted = extracted.replace("\\boxed{", "").replace("}", "")

        # Remove any trailing periods or non-numeric characters
        import re
        # Try to extract just the number
        number_match = re.search(r'[-]?\d+(?:\.\d+)?', extracted)
        if number_match:
            extracted = number_match.group()

        return extracted

    def generate_problems(self, num_problems: int, difficulty: float = 0.5) -> List[Dict]:
        """Generate math problems with ground truth answers"""
        problems = []

        # Create prompts for different difficulty levels with explicit answer request
        if difficulty < 0.3:
            prompt_template = """Generate a simple math problem with its answer.
Example format:
Question: What is 5 + 3?
Answer: 8

Now generate a new problem:
Question:"""
        elif difficulty < 0.7:
            prompt_template = """Generate a moderately complex math problem with its answer.
Example format:
Question: If John has 12 apples and gives away 4, how many does he have?
Answer: 8

Now generate a new problem:
Question:"""
        else:
            prompt_template = """Generate a challenging math word problem with its answer.
Example format:
Question: A store sells pens for $3 each. If you buy 5 pens and get a 10% discount, what's the total?
Answer: 13.50

Now generate a new problem:
Question:"""

        self.model.eval()
        for _ in tqdm(range(num_problems), desc="Generating problems"):
            inputs = self.tokenizer(prompt_template, return_tensors="pt", truncation=True, max_length=200)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=250,  # Enough for problem + answer
                    min_new_tokens=30,   # Ensure we get both question and answer
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Parse question and answer
            question = ""
            answer = ""

            if "Answer:" in generated_text:
                parts = generated_text.split("Answer:")
                question = parts[0].strip()
                answer = parts[1].strip() if len(parts) > 1 else ""
            else:
                # If no answer provided, generate a simple problem with known answer
                question = generated_text
                answer = ""

            # Fallback to simple problems with known answers if generation fails
            if not question or not answer:
                num1 = int(5 + difficulty * 20)
                num2 = int(3 + difficulty * 15)
                question = f"What is {num1} + {num2}?"
                answer = str(num1 + num2)

            problems.append({
                "question": question,
                "answer": answer,  # Ground truth
                "difficulty": difficulty
            })

        return problems

    def train_on_feedback(self, dataset: Dataset, solver_accuracy: float):
        """Train Challenger based on Solver's performance"""
        CLIFormatter.print_status("Training Challenger", f"Solver accuracy: {solver_accuracy:.2%}")

        # Adjust reward based on solver accuracy
        # Optimal when Solver achieves 60-70% accuracy
        if 0.6 <= solver_accuracy <= 0.7:
            reward_multiplier = 1.0  # Optimal
        elif solver_accuracy < 0.6:
            reward_multiplier = 0.5  # Problems too hard
        else:
            reward_multiplier = 0.5  # Problems too easy

        # Manual tokenization to avoid multiprocessing issues
        tokenized_data = []
        for example in dataset:
            # Handle both pre-training examples (with 'text' key) and feedback examples (with 'question' key)
            if 'text' in example:
                text = example['text']  # Use text directly for pre-training
            else:
                text = f"Generate a problem that is {'easy' if solver_accuracy > 0.7 else 'harder'}: {example['question']}"
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            # Convert to dict format for Dataset
            token_dict = {
                "input_ids": tokens["input_ids"].squeeze(0),  # Remove batch dimension
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": tokens["input_ids"].squeeze(0).clone()
            }
            tokenized_data.append(token_dict)

        train_dataset = Dataset.from_list(tokenized_data)

        training_args = TrainingArguments(
            output_dir="outputs/challenger",
            per_device_train_batch_size=1,
            max_steps=10,  # Quick training
            learning_rate=2e-5 * reward_multiplier,
            optim="adamw_8bit",
            logging_steps=1,
            report_to="none",
            dataloader_num_workers=0,
            fp16=True,
            bf16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        CLIFormatter.print_success("Challenger training completed")

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        clear_memory()


class SolverAgent:
    """Gemma-270m Solver that solves problems"""

    def __init__(self):
        CLIFormatter.print_info("Initializing Solver (Gemma-270m)...")
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name="unsloth/gemma-3-270m-it",
            max_seq_length=512,
            load_in_4bit=False,  # Use float32 to avoid dtype issues
            dtype=torch.float32,
        )

        # Apply LoRA
        self.model = FastLanguageModel.get_peft_model(
            self.model,
            r=16,
            target_modules=["q_proj", "v_proj"],
            lora_alpha=32,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing=False,  # Disable to avoid dtype issues
        )

        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        CLIFormatter.print_success(f"Solver loaded: {trainable:,} trainable params")

    def extract_answer(self, question: str, response: str) -> str:
        """Use LLM to extract the final numerical answer from a response

        Args:
            question: The problem question
            response: The full response containing the answer

        Returns:
            str: The extracted numerical answer
        """
        self.model.eval()

        prompt = f"""Question: {question}
Response: {response}

Extract ONLY the final numerical answer from the response above.
If the answer is a fraction, convert it to decimal.
Reply with just the number, nothing else.
Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=20,
                temperature=0.1,  # Low temperature for consistent extraction
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        extracted = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Handle multi-line responses - get the LAST non-empty line
        lines = extracted.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        if lines:
            extracted = lines[-1]  # Get the last line

        # If "Final Answer:" appears, extract after the LAST occurrence
        if "final answer" in extracted.lower():
            parts = extracted.lower().split("final answer")
            if len(parts) > 1:
                extracted = parts[-1].strip()
                # Remove colon if present
                if extracted.startswith(':'):
                    extracted = extracted[1:].strip()

        # Clean up common patterns
        extracted = extracted.replace("$", "").replace(",", "")
        extracted = extracted.replace("\\boxed{", "").replace("}", "")

        # Remove any trailing periods or non-numeric characters
        import re
        # Try to extract just the number
        number_match = re.search(r'[-]?\d+(?:\.\d+)?', extracted)
        if number_match:
            extracted = number_match.group()

        return extracted

    def solve_problems(self, problems: List[Dict], judge_agent=None, return_accuracy: bool = True) -> List[Dict]:
        """Generate solutions for problems

        Args:
            problems: List of problem dicts with 'question' and optionally 'answer' (ground truth)
            judge_agent: Optional ChallengerAgent to use for judging correctness
            return_accuracy: If True and ground truth available, calculate actual accuracy
        """
        results = []
        correct_count = 0
        has_ground_truth = False

        self.model.eval()

        # Phase 1: Generate all solutions first
        CLIFormatter.print_info("Generating solutions...")
        for problem in tqdm(problems, desc="Solving problems"):
            prompt = f"Question: {problem['question']}\nAnswer:"
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                # Generate multiple solutions for uncertainty estimation
                solutions = []
                for _ in range(2):  # Generate 2 solutions
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,  # Enough space for reasoning steps
                        min_new_tokens=1,   # Allow short answers when appropriate
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id,
                        repetition_penalty=1.3,  # Moderate repetition penalty
                        no_repeat_ngram_size=4,  # Prevent repeating 4-grams like "Final Answer: X"
                    )
                    # Decode only the generated part (skip the input prompt)
                    generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
                    answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

                    # Clean up the answer
                    if not answer:
                        answer = "Unable to solve"
                    solutions.append(answer)

            # Calculate uncertainty based on answer consistency
            uncertainty = 0.0 if solutions[0] == solutions[1] else 1.0

            result = {
                "question": problem["question"],
                "answer": solutions[0],
                "all_answers": solutions,
                "ground_truth": problem.get("answer", ""),
                "uncertainty": uncertainty,
                "difficulty": problem.get("difficulty", 0.5),
                "is_correct": False  # Will be updated in Phase 2
            }
            results.append(result)

        # Phase 2: Batch evaluate all answers (if ground truth available)
        has_ground_truth = any("answer" in p and p["answer"] for p in problems)

        if has_ground_truth:
            CLIFormatter.print_info("Extracting and evaluating answers...")

            # Use provided judge or self for extraction
            extractor = judge_agent if judge_agent else self

            def fuzzy_numeric_compare(num1_str: str, num2_str: str) -> bool:
                """Compare two numeric strings with fuzzy matching"""
                import re

                def clean_and_extract(text):
                    """Clean and extract numeric value from text"""
                    text = str(text).strip()
                    # Remove common formatting
                    text = text.replace("$", "").replace(",", "").replace("\\boxed{", "").replace("}", "")

                    # Try to extract number with various patterns
                    patterns = [
                        r'^[-]?\d+(?:\.\d+)?$',  # Just a number
                        r'[-]?\d+(?:\.\d+)?(?:\s*(?:dollars?|cents?|%|percent))?$',  # Number with units
                        r'(?:answer|is|equals?|=)\s*[-]?\d+(?:\.\d+)?',  # After keywords
                        r'[-]?\d+(?:\.\d+)?',  # Any number (last resort)
                    ]

                    for pattern in patterns:
                        match = re.search(pattern, text.lower())
                        if match:
                            num_str = re.findall(r'[-]?\d+(?:\.\d+)?', match.group())[0]
                            return float(num_str)

                    # Try fraction conversion (e.g., 1/2 -> 0.5)
                    fraction_match = re.search(r'(\d+)/(\d+)', text)
                    if fraction_match:
                        return float(fraction_match.group(1)) / float(fraction_match.group(2))

                    return None

                try:
                    num1 = clean_and_extract(num1_str)
                    num2 = clean_and_extract(num2_str)

                    if num1 is not None and num2 is not None:
                        # Allow small tolerance for floating point comparison
                        relative_error = abs(num1 - num2) / max(abs(num2), 1e-10)
                        return relative_error < 0.01  # 1% tolerance

                    # Fallback to exact string comparison if can't extract numbers
                    return num1_str.strip().lower() == num2_str.strip().lower()

                except:
                    return False

            for i, (problem, result) in enumerate(tqdm(zip(problems, results), desc="Evaluating answers", total=len(problems))):
                if "answer" in problem and problem["answer"]:
                    ground_truth = str(problem["answer"]).strip()

                    try:
                        # Extract answer using LLM
                        extracted_answer = extractor.extract_answer(
                            question=problem["question"],
                            response=result["answer"]
                        )

                        # Store extracted answer for debugging
                        result["extracted_answer"] = extracted_answer

                        # Use fuzzy deterministic comparison
                        is_correct = fuzzy_numeric_compare(extracted_answer, ground_truth)
                        result["is_correct"] = is_correct
                        if is_correct:
                            correct_count += 1

                        # Debug log for edge cases
                        if not is_correct and len(extracted_answer) > 20:
                            CLIFormatter.print_warning(f"Long extraction: {extracted_answer[:50]}...")

                    except Exception as e:
                        # If extraction fails, try direct comparison as fallback
                        is_correct = fuzzy_numeric_compare(result["answer"], ground_truth)
                        result["is_correct"] = is_correct
                        if is_correct:
                            correct_count += 1

        # Add accuracy to results if we have ground truth
        if has_ground_truth and return_accuracy:
            accuracy = correct_count / len(problems) if problems else 0
            for r in results:
                r["accuracy"] = accuracy

        return results

    def train_on_problems(self, dataset: Dataset, format_training: bool = True):
        """Train Solver on problem-solution pairs

        Args:
            dataset: Dataset with questions and answers
            format_training: If True, includes format examples to reduce repetition
        """
        CLIFormatter.print_info("Training Solver...")

        # Manual tokenization
        tokenized_data = []

        # Add format training examples if enabled
        if format_training:
            format_examples = [
                {"question": "What is 5 + 3?", "answer": "5 + 3 = 8\n\nThe answer is 8"},
                {"question": "If John has 10 apples and gives away 4, how many does he have left?",
                 "answer": "John starts with 10 apples.\nHe gives away 4 apples.\n10 - 4 = 6\n\nJohn has 6 apples left."},
                {"question": "Calculate 12 × 5", "answer": "12 × 5 = 60\n\nThe answer is 60"},
                {"question": "A store sells pens for $2 each. How much do 5 pens cost?",
                 "answer": "Each pen costs $2.\nNumber of pens: 5\nTotal cost: $2 × 5 = $10\n\nThe answer is $10"},
                {"question": "What is 100 divided by 4?", "answer": "100 ÷ 4 = 25\n\nThe answer is 25"},
            ]

            for ex in format_examples:
                # Show the model clean, concise answers without repetition
                text = f"Question: {ex['question']}\nAnswer: {ex['answer']}"
                tokens = self.tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=256,
                    return_tensors=None,
                )
                tokens["labels"] = tokens["input_ids"].copy()
                tokenized_data.append(tokens)

        # Add the actual dataset
        for example in dataset:
            # Clean the answer to avoid training on repetitions
            answer = example.get('answer', 'Unknown')
            if isinstance(answer, str):
                # Remove repetitive "Final Answer" patterns
                if "Final Answer" in answer:
                    # Extract just the first answer
                    answer = answer.split("Final Answer")[0].strip()
                    if not answer and "Final Answer:" in example.get('answer', ''):
                        answer = example['answer'].split("Final Answer:")[1].split('\n')[0].strip()

            text = f"Question: {example['question']}\nAnswer: {answer}"
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors="pt",
            )
            # Convert to dict format for Dataset
            token_dict = {
                "input_ids": tokens["input_ids"].squeeze(0),  # Remove batch dimension
                "attention_mask": tokens["attention_mask"].squeeze(0),
                "labels": tokens["input_ids"].squeeze(0).clone()
            }
            tokenized_data.append(token_dict)

        train_dataset = Dataset.from_list(tokenized_data)

        # Use fewer steps if this is format pre-training (small dataset)
        max_steps = 5 if len(tokenized_data) < 20 else 10

        training_args = TrainingArguments(
            output_dir="outputs/solver",
            per_device_train_batch_size=1,
            max_steps=max_steps,
            learning_rate=2e-5,
            optim="adamw_torch",  # Regular optimizer for float32
            logging_steps=1,
            report_to="none",
            dataloader_num_workers=0,
            fp16=False,  # No mixed precision
            bf16=False,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )

        trainer.train()
        CLIFormatter.print_success("Solver training completed")

    def cleanup(self):
        """Clean up model from memory"""
        del self.model
        del self.tokenizer
        clear_memory()


def run_rzero_evolution(num_iterations: int = 5):
    """Main R-Zero evolution loop"""
    CLIFormatter.print_header("R-Zero Training with Gemma Models")

    # Check CUDA
    if torch.cuda.is_available():
        CLIFormatter.print_status("CUDA Device", torch.cuda.get_device_name(0))
        CLIFormatter.print_status("GPU Memory", f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        CLIFormatter.print_warning("CUDA not available - training will be slow")

    # Initialize with GSM8K for bootstrapping
    CLIFormatter.print_info("Loading GSM8K dataset for bootstrapping...")
    gsm8k = load_dataset("openai/gsm8k", "main", split="train[:100]")  # Small subset

    # Prepare initial dataset
    initial_data = []
    for example in gsm8k:
        initial_data.append({
            "question": example["question"],
            "answer": example["answer"].split("####")[-1].strip() if "####" in example["answer"] else example["answer"]
        })

    current_dataset = Dataset.from_list(initial_data)
    difficulty = 0.3  # Start with easy problems

    # Pre-train Solver on format examples before baseline
    CLIFormatter.print_info("Pre-training Solver on Q&A format...")
    solver = SolverAgent()

    # Create format training examples
    format_examples = [
        {"question": "What is 5 + 3?", "answer": "5 + 3 = 8\n\nThe answer is 8"},
        {"question": "If John has 10 apples and gives away 4, how many does he have left?",
         "answer": "John starts with 10 apples.\nHe gives away 4 apples.\n10 - 4 = 6\n\nJohn has 6 apples left."},
        {"question": "Calculate 12 × 5", "answer": "12 × 5 = 60\n\nThe answer is 60"},
        {"question": "A store sells pens for $2 each. How much do 5 pens cost?",
         "answer": "Each pen costs $2.\nNumber of pens: 5\nTotal cost: $2 × 5 = $10\n\nThe answer is $10"},
        {"question": "What is 100 divided by 4?", "answer": "100 ÷ 4 = 25\n\nThe answer is 25"},
        {"question": "Sarah had 15 cookies and ate 3. How many are left?",
         "answer": "Sarah started with 15 cookies.\nShe ate 3 cookies.\n15 - 3 = 12\n\nSarah has 12 cookies left."},
        {"question": "What is 7 × 8?", "answer": "7 × 8 = 56\n\nThe answer is 56"},
        {"question": "If a book costs $15 and you buy 3 books, what's the total?",
         "answer": "Cost per book: $15\nNumber of books: 3\nTotal: $15 × 3 = $45\n\nThe total cost is $45"},
    ]

    format_dataset = Dataset.from_list(format_examples)

    # Quick format training (just a few steps to learn the pattern)
    solver.train_on_problems(format_dataset, format_training=False)  # Don't add more format examples
    CLIFormatter.print_success("Solver pre-trained on Q&A format")

    # Now calculate baseline accuracy
    CLIFormatter.print_info("Calculating baseline performance on GSM8K data...")
    test_problems = current_dataset.select(range(min(10, len(current_dataset))))
    baseline_results = solver.solve_problems(test_problems.to_list(), judge_agent=None)  # Use numeric extraction only

    # Calculate baseline accuracy using ground truth
    if baseline_results and "accuracy" in baseline_results[0]:
        solver_accuracy = baseline_results[0]["accuracy"]
    else:
        # Fallback to uncertainty-based estimate if no ground truth
        avg_uncertainty = sum(r["uncertainty"] for r in baseline_results) / len(baseline_results)
        solver_accuracy = 1.0 - avg_uncertainty

    CLIFormatter.print_metric("Baseline Solver Accuracy (vs ground truth)", solver_accuracy * 100, "%",
                              good_threshold=30, bad_threshold=10)  # Lower thresholds for untrained model

    # Save baseline samples
    baseline_samples = {
        "iteration": 0,
        "type": "baseline",
        "solver_accuracy": solver_accuracy,
        "samples": baseline_results[:5]
    }
    os.makedirs("outputs/samples", exist_ok=True)
    with open("outputs/samples/baseline.json", "w") as f:
        json.dump(baseline_samples, f, indent=2)
    CLIFormatter.print_info("Saved baseline samples to outputs/samples/baseline.json")

    solver.cleanup()

    for iteration in range(num_iterations):
        CLIFormatter.print_generation_header(iteration + 1, num_iterations)
        CLIFormatter.print_metric("Current Difficulty", difficulty, good_threshold=0.7, bad_threshold=0.3)

        # Phase 1: Challenger generates problems
        CLIFormatter.print_subheader("Phase 1: Challenger Generating Problems")
        challenger = ChallengerAgent()

        # Pre-train Challenger on first iteration to learn format
        if iteration == 0:
            CLIFormatter.print_info("Pre-training Challenger on problem generation format...")
            challenger_examples = [
                {"text": "Generate a simple math problem with its answer.\nQuestion: What is 7 + 5?\nAnswer: 12"},
                {"text": "Generate a simple math problem with its answer.\nQuestion: What is 15 - 8?\nAnswer: 7"},
                {"text": "Generate a moderately complex math problem with its answer.\nQuestion: A box has 6 rows with 4 items each. How many items total?\nAnswer: 24"},
            ]
            challenger_dataset = Dataset.from_list(challenger_examples)
            challenger.train_on_feedback(challenger_dataset, solver_accuracy)

        # Train Challenger if we have feedback from previous iteration
        elif iteration > 0:
            challenger.train_on_feedback(current_dataset, solver_accuracy)

        CLIFormatter.print_info(f"Generating New Problems (difficulty: {difficulty:.2f})")
        new_problems = challenger.generate_problems(20, difficulty)

        # Save challenger samples
        challenger_samples = {
            "iteration": iteration + 1,
            "difficulty": difficulty,
            "solver_accuracy": solver_accuracy if iteration > 0 else 0,
            "samples": new_problems[:5]  # Save first 5 generated problems
        }

        os.makedirs("outputs/samples", exist_ok=True)
        with open(f"outputs/samples/challenger_iter_{iteration + 1}.json", "w") as f:
            json.dump(challenger_samples, f, indent=2)
        CLIFormatter.print_info(f"Saved challenger samples to outputs/samples/challenger_iter_{iteration + 1}.json")

        # Clean up Challenger
        challenger.cleanup()

        # Update dataset with new problems
        current_dataset = Dataset.from_list(new_problems)

        # Phase 2: Solver attempts to solve the problems
        CLIFormatter.print_subheader("Phase 2: Solver Solving Problems")
        solver = SolverAgent()

        # Train Solver on the new problems
        solver.train_on_problems(current_dataset)

        # Evaluate Solver on test subset
        test_problems = current_dataset.select(range(min(10, len(current_dataset))))
        results = solver.solve_problems(test_problems.to_list())

        # Calculate accuracy
        avg_uncertainty = sum(r["uncertainty"] for r in results) / len(results)
        solver_accuracy = 1.0 - avg_uncertainty  # Simplified metric
        CLIFormatter.print_metric("Solver Accuracy", solver_accuracy * 100, "%",
                                  good_threshold=70, bad_threshold=40)

        # Save solver samples
        solver_samples = {
            "iteration": iteration + 1,
            "solver_accuracy": solver_accuracy,
            "samples": results[:5]  # Save first 5 samples
        }

        with open(f"outputs/samples/solver_iter_{iteration + 1}.json", "w") as f:
            json.dump(solver_samples, f, indent=2)
        CLIFormatter.print_info(f"Saved solver samples to outputs/samples/solver_iter_{iteration + 1}.json")

        # Adjust difficulty for next iteration based on solver performance
        if solver_accuracy > 0.7:
            difficulty = min(1.0, difficulty + 0.1)  # Increase difficulty
        elif solver_accuracy < 0.6:
            difficulty = max(0.1, difficulty - 0.1)  # Decrease difficulty

        # Clean up Solver
        solver.cleanup()

        # Save checkpoint
        checkpoint = {
            "iteration": iteration + 1,
            "difficulty": difficulty,
            "solver_accuracy": solver_accuracy,
            "dataset_size": len(current_dataset)
        }

        os.makedirs("outputs/checkpoints", exist_ok=True)
        with open(f"outputs/checkpoints/iteration_{iteration + 1}.json", "w") as f:
            json.dump(checkpoint, f, indent=2)

        CLIFormatter.print_success(f"Checkpoint saved: iteration {iteration + 1}")

    CLIFormatter.print_summary_box(
        "R-Zero Evolution Complete",
        {
            "Total Iterations": num_iterations,
            "Final Difficulty": f"{difficulty:.2f}",
            "Final Dataset Size": len(current_dataset),
            "Checkpoints Saved": f"outputs/checkpoints/",
            "Samples Saved": f"outputs/samples/"
        }
    )

    # Create evolution summary
    evolution_summary = {
        "total_iterations": num_iterations,
        "final_difficulty": difficulty,
        "final_solver_accuracy": solver_accuracy,
        "dataset_size": len(current_dataset),
        "samples_location": "outputs/samples/",
        "checkpoints_location": "outputs/checkpoints/"
    }

    with open("outputs/evolution_summary.json", "w") as f:
        json.dump(evolution_summary, f, indent=2)
    CLIFormatter.print_success("Evolution summary saved to outputs/evolution_summary.json")


if __name__ == "__main__":
    # Test with just 2 iterations to verify it works
    run_rzero_evolution(num_iterations=2)

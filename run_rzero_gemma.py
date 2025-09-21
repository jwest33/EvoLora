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

    def judge_answer(self, question: str, student_answer: str, ground_truth: str = None) -> bool:
        """Use LLM to judge if an answer is correct

        Args:
            question: The math problem
            student_answer: The answer to evaluate
            ground_truth: Optional known correct answer

        Returns:
            bool: True if answer is correct, False otherwise
        """
        self.model.eval()

        if ground_truth:
            prompt = f"""Question: {question}
Student's Answer: {student_answer}
Correct Answer: {ground_truth}

Compare ONLY the final numeric values. Is the student's final answer equal to {ground_truth}?
- Student claims the answer is: {student_answer}
- The correct answer is: {ground_truth}
- If these numbers are equal (ignoring units/formatting), reply YES
- If these numbers are different, reply NO

Answer:"""
        else:
            prompt = f"""Question: {question}
Student's Answer: {student_answer}

Solve this math problem step by step, then check if the student's answer is correct.
Reply with only YES if correct or NO if incorrect.

Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Low temperature for consistent judgment
                do_sample=False,  # Greedy for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()

        # Check if response contains YES
        if "YES" in response and "NO" not in response:
            return True
        elif "NO" in response:
            return False
        else:
            # Fallback to numeric comparison if LLM response is unclear
            import re
            def extract_num(text):
                text = text.replace("\\boxed{", "").replace("}", "")
                numbers = re.findall(r'[-]?\d+(?:\.\d+)?', text)
                return float(numbers[-1]) if numbers else None

            student_num = extract_num(student_answer)
            ground_num = extract_num(ground_truth)

            if student_num is not None and ground_num is not None:
                return abs(student_num - ground_num) < 0.01
            return False

    def generate_problems(self, num_problems: int, difficulty: float = 0.5) -> List[Dict]:
        """Generate math problems based on difficulty level"""
        problems = []

        # Create prompts for different difficulty levels
        if difficulty < 0.3:
            prompt_template = "Generate a simple arithmetic problem (addition or subtraction) with single digits. Problem:"
        elif difficulty < 0.7:
            prompt_template = "Generate a moderately complex math problem involving multiplication or two-step operations. Problem:"
        else:
            prompt_template = "Generate a challenging math problem involving fractions, percentages, or multi-step word problems. Problem:"

        self.model.eval()
        for _ in tqdm(range(num_problems), desc="Generating problems"):
            inputs = self.tokenizer(prompt_template, return_tensors="pt", truncation=True, max_length=200)
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=200,  # Increased for complete problems
                    min_new_tokens=20,   # Ensure minimum problem length
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                )

            # Decode only the generated part
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            problem_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

            # Ensure we have a valid problem
            if not problem_text or len(problem_text) < 10:
                problem_text = f"Calculate {2+difficulty*10:.0f} + {3+difficulty*10:.0f}."

            problems.append({
                "question": problem_text,
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
            text = f"Generate a problem that is {'easy' if solver_accuracy > 0.7 else 'harder'}: {example['question']}"
            tokens = self.tokenizer(
                text,
                truncation=True,
                padding="max_length",
                max_length=256,
                return_tensors=None,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            tokenized_data.append(tokens)

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

    def judge_answer(self, question: str, student_answer: str, ground_truth: str = None) -> bool:
        """Use LLM to judge if an answer is correct

        Args:
            question: The problem question
            student_answer: The student's answer to evaluate
            ground_truth: Optional ground truth answer

        Returns:
            True if answer is correct, False otherwise
        """
        self.model.eval()

        if ground_truth:
            prompt = f"""Question: {question}
Student Answer: {student_answer}
Correct Answer: {ground_truth}

Task: Check if the student's final numeric answer matches the correct answer.
- Extract the final number from student's answer
- The correct final answer is: {ground_truth}
- Reply YES only if the numbers match exactly
- Reply NO if the numbers are different

Answer:"""
        else:
            prompt = f"""Question: {question}
Student Answer: {student_answer}

Solve this math problem step by step, then check if the student's answer is correct.
Reply with only 'YES' if the answer is correct, or 'NO' if incorrect.
Answer:"""

        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=500)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,  # Low temperature for consistent judgment
                do_sample=False,  # Greedy decoding for consistency
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part
        generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
        judgment = self.tokenizer.decode(generated_tokens, skip_special_tokens=True).strip().upper()

        # Look for YES/NO in the response
        if "YES" in judgment and "NO" not in judgment:
            return True
        elif "NO" in judgment:
            return False
        else:
            # Fallback to numeric comparison if LLM response is unclear
            import re
            def extract_num(text):
                text = text.replace("\\boxed{", "").replace("}", "")
                # Try to find the last number in the text
                numbers = re.findall(r'[-]?\d+(?:\.\d+)?', text)
                return float(numbers[-1]) if numbers else None

            student_num = extract_num(student_answer)
            ground_num = extract_num(ground_truth) if ground_truth else None

            if student_num is not None and ground_num is not None:
                return abs(student_num - ground_num) < 0.01
            return False

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

        # Phase 2: Batch evaluate all answers with judge (if ground truth available)
        has_ground_truth = any("answer" in p and p["answer"] for p in problems)

        if has_ground_truth:
            CLIFormatter.print_info("Evaluating answers with LLM judge...")

            # Use provided judge or self for evaluation
            judge = judge_agent if judge_agent else self

            for i, (problem, result) in enumerate(tqdm(zip(problems, results), desc="Judging answers", total=len(problems))):
                if "answer" in problem and problem["answer"]:
                    ground_truth = str(problem["answer"]).strip()

                    try:
                        # Use LLM judge to evaluate
                        is_correct = judge.judge_answer(
                            question=problem["question"],
                            student_answer=result["answer"],
                            ground_truth=ground_truth
                        )
                        result["is_correct"] = is_correct
                        if is_correct:
                            correct_count += 1
                    except Exception as e:
                        # Fallback to numeric extraction if judge fails
                        import re

                        def extract_number(text):
                            patterns = [
                                r'(?:=|is|equals?|answer:?)\s*([-]?\d+(?:\.\d+)?)',
                                r'([-]?\d+(?:\.\d+)?)\s*$',
                                r'\$?([-]?\d+(?:\.\d+)?)',
                            ]
                            for pattern in patterns:
                                matches = re.findall(pattern, text.lower())
                                if matches:
                                    return matches[-1]
                            return None

                        gen_num = extract_number(result["answer"])
                        gt_num = extract_number(ground_truth)

                        if gen_num and gt_num:
                            try:
                                is_correct = abs(float(gen_num) - float(gt_num)) < 0.01
                                result["is_correct"] = is_correct
                                if is_correct:
                                    correct_count += 1
                            except:
                                pass

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
                return_tensors=None,
            )
            tokens["labels"] = tokens["input_ids"].copy()
            tokenized_data.append(tokens)

        train_dataset = Dataset.from_list(tokenized_data)

        training_args = TrainingArguments(
            output_dir="outputs/solver",
            per_device_train_batch_size=1,
            max_steps=10,
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

    # Calculate baseline accuracy on initial GSM8K data
    CLIFormatter.print_info("Calculating baseline performance on GSM8K data...")

    # Initialize judge for evaluation
    judge = ChallengerAgent()
    CLIFormatter.print_info("Using Challenger as judge for answer evaluation")

    solver = SolverAgent()
    test_problems = current_dataset.select(range(min(10, len(current_dataset))))
    baseline_results = solver.solve_problems(test_problems.to_list(), judge_agent=judge)

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
    judge.cleanup()  # Clean up judge after baseline

    for iteration in range(num_iterations):
        CLIFormatter.print_generation_header(iteration + 1, num_iterations)
        CLIFormatter.print_metric("Current Difficulty", difficulty, good_threshold=0.7, bad_threshold=0.3)

        # Phase 1: Challenger generates problems
        CLIFormatter.print_subheader("Phase 1: Challenger Generating Problems")
        challenger = ChallengerAgent()

        # Train Challenger if we have feedback from previous iteration
        if iteration > 0:
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

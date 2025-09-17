"""
Challenger agent for generating progressively difficult tasks.
Uses the larger model to create training data for the solver.
"""

import json
import random
import time
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Task:
    """Represents a single training task."""
    id: str
    input_text: str
    expected_output: Optional[str] = None
    difficulty: float = 0.5
    metadata: Dict[str, Any] = None


class TaskChallenger:
    """Generates challenging tasks for the solver."""

    def __init__(self, llama_client, task_config: Dict[str, Any]):
        """
        Initialize TaskChallenger.

        Args:
            llama_client: LlamaCppClient instance
            task_config: Task generation configuration
        """
        self.client = llama_client
        self.task_type = task_config['type']
        self.difficulty = task_config['curriculum']['initial_difficulty']
        self.target_success_rate = task_config['curriculum']['target_success_rate']
        self.task_templates = self._load_task_templates(task_config)

    def _load_task_templates(self, config: Dict) -> Dict[str, str]:
        """Load task-specific generation templates."""
        templates = {
            'code_documentation': """Generate a complex Python function that would be challenging to document.
The function should have difficulty level {difficulty:.1f} (0=easy, 1=very hard).
Include edge cases, complex logic, and non-obvious behavior.

Requirements:
- Use advanced Python features
- Include type hints
- Make the purpose non-obvious from the function name
- Difficulty indicators: {difficulty_hints}

Output the function code only, no documentation:

```python
""",
            'summarization': """Create a technical article excerpt that would be challenging to summarize.
Difficulty level: {difficulty:.1f}
The text should be dense with information and nuanced arguments.

Text:
""",
            'translation': """Generate a sentence with complex grammar and idiomatic expressions.
Difficulty: {difficulty:.1f}
Include cultural references and wordplay where appropriate.

Sentence:
"""
        }

        # Allow custom templates from config
        if 'templates' in config:
            templates.update(config['templates'])

        return templates

    def generate_task_batch(self,
                           batch_size: int,
                           current_solver_performance: float = 0.5) -> List[Task]:
        """
        Generate a batch of tasks based on current solver performance.

        Args:
            batch_size: Number of tasks to generate
            current_solver_performance: Current success rate of solver

        Returns:
            List of generated tasks
        """
        # Adjust difficulty based on solver performance
        self._adjust_difficulty(current_solver_performance)

        tasks = []
        print(f"           Starting to generate {batch_size} tasks...")
        start_time = time.time()

        for i in range(batch_size):
            task_start = time.time()

            # Show progress for first task and every 5th task
            if i == 0:
                print(f"           Generating task 1/{batch_size} (difficulty: {self.difficulty:.2f})...")
            elif (i+1) % 5 == 0:
                # Calculate progress and time estimates
                elapsed = time.time() - start_time
                avg_time_per_task = elapsed / i
                remaining_tasks = batch_size - i
                eta_seconds = remaining_tasks * avg_time_per_task
                eta_minutes = eta_seconds / 60

                print(f"           Task {i+1}/{batch_size} - Avg: {avg_time_per_task:.1f}s/task - ETA: {eta_minutes:.1f} min")

            task = self._generate_single_task(f"task_{i}")
            tasks.append(task)

            # Occasionally show a snippet of what was generated (every 3rd task)
            if (i+1) % 3 == 0 and task.input_text:
                # Show first 150 chars of the generated content
                snippet = task.input_text[:150].replace('\n', ' ')
                if len(task.input_text) > 150:
                    snippet += "..."
                print(f"             → Sample: {snippet}")

        # Final summary
        total_time = time.time() - start_time
        print(f"           Completed! Generated {batch_size} tasks in {total_time/60:.1f} minutes")
        print(f"           Average: {total_time/batch_size:.1f} seconds per task")

        return tasks

    def _generate_single_task(self, task_id: str) -> Task:
        """Generate a single task based on current difficulty."""

        # Get difficulty-specific hints
        difficulty_hints = self._get_difficulty_hints()

        # Generate task input
        if self.task_type == 'code_documentation':
            task_input = self._generate_code_task(difficulty_hints)
        elif self.task_type == 'summarization':
            task_input = self._generate_summarization_task(difficulty_hints)
        else:
            task_input = self._generate_generic_task(difficulty_hints)

        return Task(
            id=task_id,
            input_text=task_input,
            difficulty=self.difficulty,
            metadata={'task_type': self.task_type}
        )

    def _generate_code_task(self, difficulty_hints: List[str]) -> str:
        """Generate a code documentation task."""
        prompt = self.task_templates['code_documentation'].format(
            difficulty=self.difficulty,
            difficulty_hints=", ".join(difficulty_hints)
        )

        code = self.client.generate(
            prompt,
            max_tokens=1024,
            temperature=0.8,
            stop=["```"]
        )

        # Clean up the generated code
        code = code.strip()
        if not code:
            # Fallback to a simple function
            code = self._get_fallback_code()

        return code

    def _generate_summarization_task(self, difficulty_hints: List[str]) -> str:
        """Generate a summarization task."""
        prompt = self.task_templates['summarization'].format(
            difficulty=self.difficulty
        )

        text = self.client.generate(
            prompt,
            max_tokens=512,
            temperature=0.9
        )

        return text.strip()

    def _generate_generic_task(self, difficulty_hints: List[str]) -> str:
        """Generate a generic task based on task type."""
        template = self.task_templates.get(
            self.task_type,
            "Generate a task of type {task_type} with difficulty {difficulty:.1f}"
        )

        prompt = template.format(
            task_type=self.task_type,
            difficulty=self.difficulty,
            difficulty_hints=", ".join(difficulty_hints)
        )

        return self.client.generate(prompt, max_tokens=512)

    def _get_difficulty_hints(self) -> List[str]:
        """Get hints for current difficulty level."""
        if self.difficulty < 0.3:
            return ["simple logic", "clear purpose", "basic operations"]
        elif self.difficulty < 0.6:
            return ["moderate complexity", "some edge cases", "multiple concepts"]
        elif self.difficulty < 0.8:
            return ["complex logic", "non-obvious behavior", "advanced features"]
        else:
            return ["very complex", "multiple abstractions", "subtle bugs", "performance critical"]

    def _adjust_difficulty(self, solver_performance: float):
        """
        Adjust difficulty based on solver performance.

        Args:
            solver_performance: Current success rate (0 to 1)
        """
        # If solver is doing too well, increase difficulty
        if solver_performance > self.target_success_rate + 0.1:
            self.difficulty = min(1.0, self.difficulty + 0.05)
            logger.info(f"Increasing difficulty to {self.difficulty:.2f}")
        # If solver is struggling, decrease difficulty
        elif solver_performance < self.target_success_rate - 0.1:
            self.difficulty = max(0.1, self.difficulty - 0.05)
            logger.info(f"Decreasing difficulty to {self.difficulty:.2f}")

    def _get_fallback_code(self) -> str:
        """Get a fallback code sample if generation fails."""
        return """def process_data(items, config=None):
    if config is None:
        config = {}

    results = []
    for item in items:
        if isinstance(item, dict):
            processed = {k: v * 2 if isinstance(v, (int, float)) else v
                        for k, v in item.items()}
        else:
            processed = item
        results.append(processed)

    return results"""

    def evaluate_solver_outputs(self,
                               tasks: List[Task],
                               solver_outputs: List[str]) -> List[float]:
        """
        Evaluate solver outputs for given tasks.

        Args:
            tasks: Original tasks
            solver_outputs: Solver's responses

        Returns:
            List of scores (0 to 1)
        """
        scores = []

        for task, output in zip(tasks, solver_outputs):
            if self.task_type == 'code_documentation':
                score = self._evaluate_documentation(task.input_text, output)
            else:
                score = self._evaluate_generic(task, output)
            scores.append(score)

        return scores

    def _evaluate_documentation(self, code: str, documentation: str) -> float:
        """Evaluate documentation quality for code."""
        prompt = f"""Evaluate this documentation for the given code.

Code:
```python
{code}
```

Documentation:
{documentation}

Rate the documentation quality from 0.0 to 1.0.

Output ONLY a number between 0.0 and 1.0, nothing else.
No explanations, no reasoning, just the score number.

Score: """

        score_text = self.client.generate(prompt, max_tokens=5, temperature=0.1)

        try:
            # Extract just the number from the response
            import re
            # Look for first floating point number in the response
            match = re.search(r'([0-9]*\.?[0-9]+)', score_text.strip())
            if match:
                score = float(match.group(1))
                return max(0.0, min(1.0, score))
            else:
                raise ValueError("No number found")
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse score from: {score_text[:50]}")
            return 0.5

    def _evaluate_generic(self, task: Task, output: str) -> float:
        """Generic evaluation for any task type."""
        prompt = f"""Evaluate this output for the given task.

Task Type: {self.task_type}
Input: {task.input_text[:500]}...

Output: {output}

Rate the quality from 0.0 (poor) to 1.0 (excellent).
Score: """

        score_text = self.client.generate(prompt, max_tokens=5, temperature=0.1)

        try:
            score = float(score_text.strip())
            return max(0.0, min(1.0, score))
        except ValueError:
            return 0.5

    def generate_pseudo_labels(self,
                              tasks: List[Task],
                              num_samples: int = 1) -> List[str]:
        """
        Generate pseudo-labels for tasks using the challenger model.

        Args:
            tasks: List of tasks
            num_samples: Number of samples to generate for voting

        Returns:
            List of pseudo-labels (one per task)
        """
        pseudo_labels = []
        print(f"\n           === Generating Pseudo-Labels ===")
        print(f"           Creating high-quality documentation for {len(tasks)} tasks")
        print(f"           Using {num_samples} sample(s) per task for quality selection")
        start_time = time.time()

        for i, task in enumerate(tasks):
            # Enhanced progress logging
            if i == 0:
                print(f"\n           Starting label generation...")
            elif i % 5 == 0:
                elapsed = time.time() - start_time
                avg_time = elapsed / i
                remaining = len(tasks) - i
                eta_seconds = remaining * avg_time
                print(f"           Progress: {i}/{len(tasks)} labels - Avg: {avg_time:.1f}s/label - ETA: {eta_seconds/60:.1f} min")
            if self.task_type == 'code_documentation':
                label = self._generate_documentation_label(task.input_text, num_samples)
            else:
                label = self._generate_generic_label(task, num_samples)
            pseudo_labels.append(label)

            # Show snippet of generated label every 4th task
            if (i+1) % 4 == 0 and label:
                # Show first meaningful line (skip empty lines)
                lines = [l.strip() for l in label.split('\n') if l.strip()]
                if lines:
                    snippet = lines[0][:150]
                    if len(lines[0]) > 150 or len(lines) > 1:
                        snippet += "..."
                    print(f"             → Label preview: {snippet}")

        # Final summary
        total_time = time.time() - start_time
        print(f"\n           ✓ Completed pseudo-label generation")
        print(f"           Generated {len(pseudo_labels)} labels in {total_time/60:.1f} minutes")
        print(f"           Average: {total_time/len(tasks):.1f} seconds per label\n")

        return pseudo_labels

    def _generate_documentation_label(self, code: str, num_samples: int) -> str:
        """Generate documentation pseudo-label via majority voting."""
        prompt = f"""You are a Python documentation expert. Write complete, professional documentation for the following code.
Do NOT write instructions about how to document. Write the ACTUAL documentation directly.

Code to document:
```python
{code}
```

Write the documentation below (including docstrings, type hints explanation, parameters, returns, examples):

"""

        # Generate multiple samples
        samples = []
        for _ in range(num_samples):
            doc = self.client.generate(
                prompt,
                max_tokens=512,
                temperature=0.7,
                stop=["```", "\n\n\n"]  # Stop if we hit code blocks or excessive newlines
            )
            samples.append(doc.strip())

        # For documentation, use the highest quality sample rather than voting
        scores = [self._evaluate_documentation(code, doc) for doc in samples]
        best_idx = scores.index(max(scores))

        return samples[best_idx]

    def _generate_generic_label(self, task: Task, num_samples: int) -> str:
        """Generate generic pseudo-label."""
        # Task-specific prompts can be added here
        prompt = f"Complete this {self.task_type} task:\n{task.input_text}\n\nOutput:"

        samples = []
        for _ in range(num_samples):
            output = self.client.generate(prompt, max_tokens=256)
            samples.append(output)

        # Simple majority voting or quality-based selection
        return samples[0]  # Simplified for now
"""Instruction optimizer that evolves routing instructions."""
from __future__ import annotations
import random
import uuid
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np

from ..core.llm_client import LLMClient
from ..config.evolution_config import EvolutionConfig, InstructionGene
from ..hierarchy import HierNode

class InstructionOptimizer:
    """
    Evolves and optimizes routing instructions using genetic algorithm.
    Implements mutation, crossover, and selection strategies.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: EvolutionConfig,
        hierarchy_root: HierNode
    ):
        self.llm = llm_client
        self.config = config
        self.hierarchy = hierarchy_root
        self.generation_counter = 0

    def initialize_population(self) -> List[InstructionGene]:
        """
        Create initial population of instruction variants.
        Uses LLM to generate diverse starting instructions.
        """
        population = []

        # Use task goal configuration for styles and base instruction
        task_goal = self.config.task_goal
        styles = task_goal.instruction_styles if task_goal else [
            "technical and precise",
            "simple and clear",
            "detailed with examples",
            "question-based",
            "action-oriented",
            "category-focused",
            "keyword-focused",
            "intent-focused"
        ]

        base_instruction = task_goal.base_instruction if task_goal else "Route the customer query to the most appropriate category in the hierarchy."

        # Generate base variant
        population.append(InstructionGene(
            id=str(uuid.uuid4()),
            content=base_instruction,
            generation=0
        ))

        # Generate styled variants
        for style in styles[:self.config.population_size - 1]:
            variant = self._generate_styled_instruction(base_instruction, style)
            if variant:
                population.append(InstructionGene(
                    id=str(uuid.uuid4()),
                    content=variant,
                    generation=0
                ))

        # Fill remaining slots with mutations if needed
        while len(population) < self.config.population_size:
            base_gene = random.choice(population)
            mutated = self.mutate(base_gene)
            population.append(mutated)

        return population

    def _generate_styled_instruction(self, base: str, style: str) -> Optional[str]:
        """Generate instruction variant in specific style."""
        task_goal = self.config.task_goal

        # Use custom prompt template if available
        if task_goal and task_goal.style_prompt_template:
            prompt = task_goal.style_prompt_template.format(
                style=style,
                base=base
            )
        else:
            # Default prompt for backward compatibility
            constraints = "\n".join([f"{i+1}. {c}" for i, c in enumerate(
                task_goal.instruction_constraints if task_goal else [
                    "Guide routing of customer queries to categories",
                    "Be clear and actionable",
                    "Follow the {style} approach".format(style=style),
                    "Be concise (1-2 sentences)"
                ]
            )])

            prompt = f"""Rewrite this instruction in a {style} style:

Original: {base}

The instruction should:
{constraints}

Return only the rewritten instruction, nothing else."""

        response = self.llm.complete(prompt, temperature=0.7)
        return response.content.strip() if response.content else None

    def evolve_generation(
        self,
        population: List[InstructionGene],
        fitness_scores: Dict[str, float]
    ) -> List[InstructionGene]:
        """
        Evolve population to next generation.

        Args:
            population: Current generation
            fitness_scores: Fitness score for each gene ID

        Returns:
            Next generation population
        """
        self.generation_counter += 1

        # Update fitness in genes
        for gene in population:
            gene.fitness = fitness_scores.get(gene.id, 0.0)

        # Sort by fitness
        population.sort(key=lambda x: x.fitness, reverse=True)

        next_generation = []

        # Elite selection - keep top performers
        elite = population[:self.config.elite_size]
        for gene in elite:
            next_generation.append(InstructionGene(
                id=str(uuid.uuid4()),
                content=gene.content,
                fitness=gene.fitness,
                generation=self.generation_counter,
                parent_ids=[gene.id]
            ))

        # Generate rest through crossover and mutation
        while len(next_generation) < self.config.population_size:
            if random.random() < self.config.crossover_rate and len(population) >= 2:
                # Crossover
                parent1, parent2 = self._select_parents(population)
                child = self.crossover(parent1, parent2)
            else:
                # Mutation
                parent = self._select_parent(population)
                child = self.mutate(parent)

            child.generation = self.generation_counter
            next_generation.append(child)

        return next_generation

    def mutate(self, gene: InstructionGene) -> InstructionGene:
        """
        Mutate instruction to create variation.
        Uses LLM to intelligently modify instruction.
        """
        task_goal = self.config.task_goal
        mutation_strategies = task_goal.mutation_strategies if task_goal else [
            "Add more specific guidance",
            "Simplify and make more concise",
            "Emphasize accuracy over speed",
            "Focus on identifying key terms",
            "Add confidence assessment",
            "Rephrase using different terminology",
            "Make more action-oriented",
            "Add handling for ambiguous cases"
        ]

        strategy = random.choice(mutation_strategies)

        task_goal = self.config.task_goal

        # Use custom mutation prompt if available
        if task_goal and task_goal.mutation_prompt_template:
            prompt = task_goal.mutation_prompt_template.format(
                strategy=strategy,
                instruction=gene.content
            )
        else:
            # Default prompt
            purpose = task_goal.objective if task_goal else "routing queries"
            prompt = f"""Modify this instruction using the strategy: {strategy}

Current instruction: {gene.content}

Create a variation that:
1. Applies the strategy: {strategy}
2. Maintains the core purpose ({purpose})
3. Is still clear and actionable
4. Introduces meaningful change

Return only the modified instruction, nothing else."""

        response = self.llm.complete(prompt, temperature=0.8)

        if response.content:
            return InstructionGene(
                id=str(uuid.uuid4()),
                content=response.content.strip(),
                generation=gene.generation,
                parent_ids=[gene.id]
            )

        # Fallback - minor text modification
        words = gene.content.split()
        if len(words) > 3:
            idx = random.randint(0, len(words) - 1)
            words[idx] = random.choice(["best", "most appropriate", "correct", "optimal"])

        return InstructionGene(
            id=str(uuid.uuid4()),
            content=" ".join(words),
            generation=gene.generation,
            parent_ids=[gene.id]
        )

    def crossover(
        self,
        parent1: InstructionGene,
        parent2: InstructionGene
    ) -> InstructionGene:
        """
        Combine two instructions to create offspring.
        Uses LLM to intelligently merge concepts.
        """
        task_goal = self.config.task_goal

        # Use custom crossover prompt if available
        if task_goal and task_goal.crossover_prompt_template:
            prompt = task_goal.crossover_prompt_template.format(
                parent1=parent1.content,
                parent2=parent2.content
            )
        else:
            # Default prompt
            purpose = task_goal.description if task_goal else "routing instructions"
            prompt = f"""Combine the best aspects of these two instructions for {purpose}:

Instruction 1: {parent1.content}
Instruction 2: {parent2.content}

Create a new instruction that:
1. Takes the strongest elements from each parent
2. Maintains clarity and actionability
3. Is concise (1-2 sentences)
4. Represents a meaningful combination, not just concatenation

Return only the combined instruction, nothing else."""

        response = self.llm.complete(prompt, temperature=0.7)

        if response.content:
            return InstructionGene(
                id=str(uuid.uuid4()),
                content=response.content.strip(),
                generation=self.generation_counter,
                parent_ids=[parent1.id, parent2.id]
            )

        # Fallback - take first half of one, second half of other
        words1 = parent1.content.split()
        words2 = parent2.content.split()
        mid1 = len(words1) // 2
        mid2 = len(words2) // 2

        combined = " ".join(words1[:mid1] + words2[mid2:])

        return InstructionGene(
            id=str(uuid.uuid4()),
            content=combined,
            generation=self.generation_counter,
            parent_ids=[parent1.id, parent2.id]
        )

    def _select_parents(
        self,
        population: List[InstructionGene]
    ) -> Tuple[InstructionGene, InstructionGene]:
        """Select two parents using tournament selection."""
        parent1 = self._select_parent(population)

        # Ensure different parents
        remaining = [g for g in population if g.id != parent1.id]
        parent2 = self._select_parent(remaining) if remaining else parent1

        return parent1, parent2

    def _select_parent(
        self,
        population: List[InstructionGene]
    ) -> InstructionGene:
        """Select parent using tournament selection."""
        tournament_size = min(3, len(population))
        tournament = random.sample(population, tournament_size)
        return max(tournament, key=lambda x: x.fitness)

    def analyze_population(
        self,
        population: List[InstructionGene]
    ) -> Dict[str, any]:
        """Analyze current population statistics."""
        fitness_values = [g.fitness for g in population]

        analysis = {
            "generation": self.generation_counter,
            "population_size": len(population),
            "best_fitness": max(fitness_values) if fitness_values else 0,
            "worst_fitness": min(fitness_values) if fitness_values else 0,
            "mean_fitness": np.mean(fitness_values) if fitness_values else 0,
            "std_fitness": np.std(fitness_values) if fitness_values else 0,
            "best_instruction": population[0].content if population else "",
        }

        # Diversity analysis - unique instruction starts
        instruction_starts = [g.content[:30] for g in population]
        analysis["diversity"] = len(set(instruction_starts)) / len(population)

        return analysis

    def generate_targeted_mutation(
        self,
        gene: InstructionGene,
        failure_analysis: Dict[str, any]
    ) -> InstructionGene:
        """
        Generate mutation specifically targeting observed failures.

        Args:
            gene: Instruction to mutate
            failure_analysis: Analysis of routing failures

        Returns:
            Targeted mutation
        """
        # Extract problem areas from failure analysis
        problems = []
        if "hardest_to_classify" in failure_analysis:
            for cat, _ in failure_analysis["hardest_to_classify"]:
                problems.append(f"Queries about '{cat}' are often misclassified")

        if "most_confused_with" in failure_analysis:
            for cat, _ in failure_analysis["most_confused_with"]:
                problems.append(f"Queries are incorrectly routed to '{cat}'")

        if not problems:
            # No specific problems, do regular mutation
            return self.mutate(gene)

        problem_desc = "\n".join(problems)

        prompt = f"""Improve this routing instruction to address specific problems:

Current instruction: {gene.content}

Observed problems:
{problem_desc}

Create an improved instruction that:
1. Addresses the identified routing errors
2. Maintains general routing capability
3. Is clear and actionable
4. Is concise (1-2 sentences)

Return only the improved instruction, nothing else."""

        response = self.llm.complete(prompt, temperature=0.7)

        if response.content:
            return InstructionGene(
                id=str(uuid.uuid4()),
                content=response.content.strip(),
                generation=gene.generation,
                parent_ids=[gene.id],
                metrics={"targeted_mutation": True}
            )

        return self.mutate(gene)  # Fallback to regular mutation

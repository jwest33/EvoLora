"""Dataset builder for generating synthetic training data."""
from __future__ import annotations
import json
import random
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

from ..core.llm_client import LLMClient
from ..hierarchy import HierNode
from ..config.evolution_config import EvolutionConfig
from ..config.task_goals import TaskType

class DatasetBuilder:
    """
    Orchestrates synthetic dataset creation from hierarchy descriptions.
    Implements zero-data bootstrap approach from R-Zero.
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
        self.nodes_by_path = self._build_node_map(hierarchy_root)
        self.leaf_paths = [path for path, node in self.nodes_by_path.items() if node.is_leaf()]

    def _build_node_map(self, root: HierNode) -> Dict[str, HierNode]:
        """Build path -> node mapping."""
        nodes = {}
        for node in root.iter_nodes():
            nodes[node.path] = node
        return nodes

    def generate_seed_dataset(self) -> pd.DataFrame:
        """
        Generate initial seed dataset from hierarchy descriptions.
        This bootstraps from zero data using only the hierarchy structure.
        """
        all_examples = []

        for leaf_path in self.leaf_paths:
            node = self.nodes_by_path[leaf_path]
            examples = self._generate_seed_examples(node)
            all_examples.extend(examples)

        # Create dataframe
        df = pd.DataFrame(all_examples)
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle

        return df

    def _generate_seed_examples(self, node: HierNode) -> List[Dict]:
        """Generate seed examples for a single node."""
        examples = []

        # Build context from parent chain
        context = self._build_node_context(node)

        # Adapt prompt based on task goal
        task_goal = self.config.task_goal
        if task_goal:
            if task_goal.type == TaskType.SEMANTIC_SEARCH:
                query_type = "search queries"
                query_purpose = "to find documents about"
                query_instruction = "Generate search queries that someone would use to find information about this topic."
            elif task_goal.type == TaskType.QUESTION_ANSWERING:
                query_type = "questions"
                query_purpose = "seeking answers about"
                query_instruction = "Generate questions that users would ask to get information about this topic."
            elif task_goal.type == TaskType.INTENT_CLASSIFICATION:
                query_type = "user requests"
                query_purpose = "expressing intent related to"
                query_instruction = "Generate user requests that express different intents related to this category."
            else:
                query_type = "queries"
                query_purpose = "for"
                query_instruction = f"Generate queries aligned with the task: {task_goal.objective}"
        else:
            query_type = "customer support queries"
            query_purpose = "for"
            query_instruction = "Generate realistic queries that clearly belong to this category."

        prompt = f"""Generate {self.config.seed_examples_per_node} diverse {query_type} {query_purpose} the following category:

Category: {node.name}
Full path: {node.path}
Description: {node.description}
Context: {context}

{query_instruction}
Vary the style, length, complexity, and specific aspects mentioned.

Respond in JSON format:
{{
    "queries": [
        {{
            "text": "query text",
            "specific_aspect": "what the query focuses on",
            "complexity": "simple|medium|complex"
        }}
    ]
}}"""

        response = self.llm.generate_json(prompt, temperature=0.7)

        if response and "queries" in response:
            for item in response["queries"]:
                examples.append({
                    "text": item["text"],
                    "label_path": node.path,
                    "node_name": node.name,
                    "complexity": item.get("complexity", "medium"),
                    "source": "seed",
                    "generation": 0
                })

        return examples

    def augment_dataset(
        self,
        base_df: pd.DataFrame,
        factor: int = 3
    ) -> pd.DataFrame:
        """
        Augment existing dataset through paraphrasing and variations.

        Args:
            base_df: Base dataset to augment
            factor: Multiplication factor for augmentation

        Returns:
            Augmented dataset
        """
        augmented = []

        for _, row in base_df.iterrows():
            # Keep original
            augmented.append(row.to_dict())

            # Generate variations
            variations = self._generate_variations(
                row["text"],
                row["label_path"],
                count=factor - 1
            )

            for var in variations:
                augmented.append({
                    "text": var["text"],
                    "label_path": row["label_path"],
                    "node_name": row["node_name"],
                    "complexity": var.get("complexity", row.get("complexity", "medium")),
                    "source": "augmented",
                    "generation": row.get("generation", 0) + 1,
                    "original_text": row["text"]
                })

        return pd.DataFrame(augmented)

    def _generate_variations(
        self,
        text: str,
        label_path: str,
        count: int
    ) -> List[Dict]:
        """Generate variations of a query."""
        variations = []

        # Adapt variation strategies based on task goal
        task_goal = self.config.task_goal
        if task_goal and task_goal.type == TaskType.SEMANTIC_SEARCH:
            strategies = [
                "Use different search terms",
                "Make more specific",
                "Make more general",
                "Use synonyms",
                "Change query structure"
            ]
        elif task_goal and task_goal.type == TaskType.QUESTION_ANSWERING:
            strategies = [
                "Rephrase as different question type",
                "Make more specific",
                "Ask from different angle",
                "Use technical terms",
                "Simplify language"
            ]
        elif task_goal and task_goal.type == TaskType.INTENT_CLASSIFICATION:
            strategies = [
                "Express same intent differently",
                "Use formal language",
                "Use casual language",
                "Add context",
                "Be more direct"
            ]
        else:
            strategies = [
            "Rephrase using different words but same meaning",
            "Make it more formal/professional",
            "Make it more casual/conversational",
            "Add more specific details",
            "Simplify and make more concise",
            "Express with frustration or urgency",
            "Frame as a question vs statement"
        ]

        selected_strategies = random.sample(strategies, min(count, len(strategies)))

        for strategy in selected_strategies:
            prompt = f"""Rewrite this customer query using the following strategy: {strategy}

Original query: {text}

Keep the core issue the same (it should still belong to category: {label_path}).
Make it sound natural and realistic.

Respond in JSON format:
{{
    "text": "rewritten query",
    "strategy_used": "{strategy}",
    "complexity": "simple|medium|complex"
}}"""

            response = self.llm.generate_json(prompt, temperature=0.8)

            if response and "text" in response:
                variations.append(response)

        return variations

    def generate_cross_boundary_examples(
        self,
        num_examples: int = 20
    ) -> pd.DataFrame:
        """
        Generate examples that sit at boundaries between categories.
        These are especially useful for testing routing robustness.
        """
        examples = []

        for _ in range(num_examples):
            # Pick two sibling nodes
            parent_paths = [p for p in self.nodes_by_path if self.nodes_by_path[p].children]
            if not parent_paths:
                continue

            parent_path = random.choice(parent_paths)
            parent = self.nodes_by_path[parent_path]

            if len(parent.children) < 2:
                continue

            node1, node2 = random.sample(parent.children, 2)

            # Adapt based on task goal
            task_goal = self.config.task_goal
            if task_goal:
                if task_goal.type == TaskType.SEMANTIC_SEARCH:
                    query_type = "search query"
                    scenario_type = "search scenario"
                elif task_goal.type == TaskType.QUESTION_ANSWERING:
                    query_type = "question"
                    scenario_type = "question-asking scenario"
                elif task_goal.type == TaskType.INTENT_CLASSIFICATION:
                    query_type = "user request"
                    scenario_type = "user intent scenario"
                else:
                    query_type = "query"
                    scenario_type = f"scenario for {task_goal.objective}"
            else:
                query_type = "customer query"
                scenario_type = "customer scenario"

            prompt = f"""Generate a {query_type} that sits at the boundary between two categories.

Category 1: {node1.name}
Description: {node1.description}

Category 2: {node2.name}
Description: {node2.description}

Create a query that:
1. Has valid elements from both categories
2. Slightly leans toward one category (you choose)
3. Would be challenging to route correctly
4. Represents a realistic {scenario_type}

Respond in JSON format:
{{
    "text": "boundary query",
    "primary_category": "path of category it leans toward",
    "secondary_category": "path of other relevant category",
    "boundary_score": 0.0-1.0 (how close to boundary),
    "reasoning": "why this is at the boundary"
}}"""

            response = self.llm.generate_json(prompt, temperature=0.8)

            if response and "text" in response:
                # Determine leaf path
                primary_node = node1 if node1.path in response.get("primary_category", "") else node2
                target_leaf = self._get_random_leaf_under(primary_node)

                examples.append({
                    "text": response["text"],
                    "label_path": target_leaf,
                    "node_name": self.nodes_by_path[target_leaf].name,
                    "complexity": "complex",
                    "source": "boundary",
                    "generation": 0,
                    "boundary_score": response.get("boundary_score", 0.5),
                    "secondary_category": response.get("secondary_category", "")
                })

        return pd.DataFrame(examples)

    def _build_node_context(self, node: HierNode) -> str:
        """Build context string from node's position in hierarchy."""
        path_parts = node.path.split("/")
        context_parts = []

        current_path = ""
        for part in path_parts:
            if current_path:
                current_path += "/" + part
            else:
                current_path = part

            if current_path in self.nodes_by_path:
                n = self.nodes_by_path[current_path]
                context_parts.append(f"{n.name}: {n.description}")

        return " > ".join(context_parts)

    def _get_random_leaf_under(self, node: HierNode) -> str:
        """Get random leaf path under node."""
        if node.is_leaf():
            return node.path

        leaves = []
        def collect_leaves(n):
            if n.is_leaf():
                leaves.append(n.path)
            else:
                for child in n.children:
                    collect_leaves(child)

        collect_leaves(node)
        return random.choice(leaves) if leaves else node.path

    def balance_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Balance dataset to have roughly equal examples per category.
        Uses oversampling for minority classes.
        """
        # Count examples per category
        category_counts = df["label_path"].value_counts()
        max_count = category_counts.max()

        balanced_dfs = []
        for category in category_counts.index:
            cat_df = df[df["label_path"] == category]

            if len(cat_df) < max_count:
                # Oversample
                n_needed = max_count - len(cat_df)
                oversampled = cat_df.sample(n=n_needed, replace=True, random_state=42)
                cat_df = pd.concat([cat_df, oversampled], ignore_index=True)

            balanced_dfs.append(cat_df)

        balanced_df = pd.concat(balanced_dfs, ignore_index=True)
        return balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

    def export_dataset(self, df: pd.DataFrame, output_path: Path):
        """Export dataset to CSV."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Select columns for export
        export_columns = ["text", "label_path"]
        if "complexity" in df.columns:
            export_columns.append("complexity")
        if "source" in df.columns:
            export_columns.append("source")

        df[export_columns].to_csv(output_path, index=False)
        print(f"Dataset exported to {output_path}")
        print(f"Total examples: {len(df)}")
        print(f"Categories: {df['label_path'].nunique()}")
        print(f"Distribution:\n{df['label_path'].value_counts()}")

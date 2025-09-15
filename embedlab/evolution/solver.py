"""Solver agent that attempts to route queries using evolved instructions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import numpy as np
from collections import defaultdict

from ..core.llm_client import LLMClient
from ..core.embedding_client import EmbeddingClient
from ..config.evolution_config import SolverState, EvolutionConfig, InstructionGene
from ..hierarchy import HierNode

@dataclass
class RoutingResult:
    """Result of routing a single query."""
    query: str
    predicted_path: str
    true_path: str
    confidence: float
    decisions: List[Dict]
    success: bool

class SolverAgent:
    """
    Attempts to solve routing challenges using current instructions.
    Inspired by R-Zero's Solver that learns from increasingly difficult tasks.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        embedding_client: EmbeddingClient,
        config: EvolutionConfig,
        hierarchy_root: HierNode
    ):
        self.llm = llm_client
        self.embedder = embedding_client
        self.config = config
        self.hierarchy = hierarchy_root
        self.state = SolverState()

        # Build node mapping
        self.nodes_by_path = self._build_node_map(hierarchy_root)

        # Cache for prototype embeddings
        self.prototype_cache = {}

    def _build_node_map(self, root: HierNode) -> Dict[str, HierNode]:
        """Build path -> node mapping."""
        nodes = {}
        for node in root.iter_nodes():
            nodes[node.path] = node
        return nodes

    def route_queries(
        self,
        queries: List[Tuple[str, str]],  # (text, true_path)
        instruction_gene: InstructionGene,
        use_embeddings: bool = True
    ) -> List[RoutingResult]:
        """
        Route a batch of queries using given instruction.

        Args:
            queries: List of (query_text, true_path) tuples
            instruction_gene: Instruction variant to use
            use_embeddings: Whether to use embeddings for routing

        Returns:
            List of routing results
        """
        results = []

        for query_text, true_path in queries:
            if use_embeddings:
                result = self._route_with_embeddings(
                    query_text,
                    true_path,
                    instruction_gene.content
                )
            else:
                result = self._route_with_llm(
                    query_text,
                    true_path,
                    instruction_gene.content
                )

            results.append(result)

            # Update state
            self.state.attempts += 1
            if result.success:
                self.state.successes += 1
            else:
                # Track failure patterns
                failure_key = f"{true_path}->{result.predicted_path}"
                self.state.failure_patterns[failure_key] = \
                    self.state.failure_patterns.get(failure_key, 0) + 1

        self.state.current_accuracy = self.state.successes / max(1, self.state.attempts)
        return results

    def _route_with_embeddings(
        self,
        query_text: str,
        true_path: str,
        instruction: str
    ) -> RoutingResult:
        """Route using embedding similarity."""
        decisions = []
        current = self.hierarchy
        predicted_path = current.path

        while not current.is_leaf():
            if not current.children:
                break

            # Get embeddings for query with instruction
            query_embedding = self.embedder.embed_query(
                query_text,
                instruction=instruction
            )

            # Get prototype embeddings for children
            child_scores = []
            for child in current.children:
                proto = self._get_prototype_embedding(child)
                if proto is not None:
                    similarity = float(self.embedder.similarity(query_embedding, proto)[0, 0])
                    child_scores.append((child, similarity))

            if not child_scores:
                break

            # Sort by similarity
            child_scores.sort(key=lambda x: x[1], reverse=True)
            best_child, best_score = child_scores[0]

            # Calculate confidence (margin between top 2)
            confidence = best_score
            if len(child_scores) > 1:
                margin = best_score - child_scores[1][1]
                confidence = min(1.0, margin * 2)  # Scale margin to confidence

            # Record decision
            decisions.append({
                "at_node": current.path,
                "candidates": [
                    {"path": c.path, "score": s}
                    for c, s in child_scores
                ],
                "choice": best_child.path,
                "confidence": confidence
            })

            # Move to best child
            current = best_child
            predicted_path = current.path

        success = predicted_path == true_path

        return RoutingResult(
            query=query_text,
            predicted_path=predicted_path,
            true_path=true_path,
            confidence=np.mean([d["confidence"] for d in decisions]) if decisions else 0.0,
            decisions=decisions,
            success=success
        )

    def _route_with_llm(
        self,
        query_text: str,
        true_path: str,
        instruction: str
    ) -> RoutingResult:
        """Route using LLM reasoning."""
        decisions = []
        current = self.hierarchy
        predicted_path = current.path

        while not current.is_leaf():
            if not current.children:
                break

            # Build prompt for routing decision
            children_desc = "\n".join([
                f"- {child.name}: {child.description}"
                for child in current.children
            ])

            prompt = f"""{instruction}

Current node: {current.name}
Query: {query_text}

Available subcategories:
{children_desc}

Which subcategory best matches this query? Respond in JSON format:
{{
    "choice": "exact name of chosen subcategory",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

            response = self.llm.generate_json(
                prompt,
                temperature=self.config.solver_temperature
            )

            if not response or "choice" not in response:
                break

            # Find chosen child
            chosen_child = None
            for child in current.children:
                if child.name.lower() == response["choice"].lower():
                    chosen_child = child
                    break

            if not chosen_child:
                # Fallback to first child
                chosen_child = current.children[0]

            # Record decision
            decisions.append({
                "at_node": current.path,
                "choice": chosen_child.path,
                "confidence": response.get("confidence", 0.5),
                "reasoning": response.get("reasoning", "")
            })

            # Move to chosen child
            current = chosen_child
            predicted_path = current.path

        success = predicted_path == true_path

        return RoutingResult(
            query=query_text,
            predicted_path=predicted_path,
            true_path=true_path,
            confidence=np.mean([d["confidence"] for d in decisions]) if decisions else 0.0,
            decisions=decisions,
            success=success
        )

    def _get_prototype_embedding(self, node: HierNode) -> Optional[np.ndarray]:
        """Get or compute prototype embedding for node."""
        if node.path in self.prototype_cache:
            return self.prototype_cache[node.path]

        # Use node description as prototype
        if node.description:
            embedding = self.embedder.embed_document(node.description)
            self.prototype_cache[node.path] = embedding
            return embedding

        return None

    def analyze_failures(self) -> Dict[str, any]:
        """Analyze failure patterns to inform instruction improvement."""
        analysis = {
            "overall_accuracy": self.state.current_accuracy,
            "total_attempts": self.state.attempts,
            "total_successes": self.state.successes
        }

        # Find most common failure patterns
        if self.state.failure_patterns:
            sorted_failures = sorted(
                self.state.failure_patterns.items(),
                key=lambda x: x[1],
                reverse=True
            )

            analysis["top_failures"] = [
                {
                    "pattern": pattern,
                    "count": count,
                    "true_category": pattern.split("->")[0],
                    "predicted_category": pattern.split("->")[1]
                }
                for pattern, count in sorted_failures[:5]
            ]

            # Identify problematic categories
            true_category_errors = defaultdict(int)
            pred_category_errors = defaultdict(int)

            for pattern, count in self.state.failure_patterns.items():
                true_cat, pred_cat = pattern.split("->")
                true_category_errors[true_cat] += count
                pred_category_errors[pred_cat] += count

            analysis["hardest_to_classify"] = sorted(
                true_category_errors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

            analysis["most_confused_with"] = sorted(
                pred_category_errors.items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]

        return analysis

    def calculate_reward(self) -> float:
        """
        Calculate Solver's reward based on routing performance.
        Reward increases with accuracy and confidence.
        """
        base_reward = self.state.current_accuracy

        # Bonus for high confidence on correct predictions
        # (would need to track this in more detail)
        confidence_bonus = 0.1 if self.state.current_accuracy > 0.7 else 0.0

        reward = base_reward + confidence_bonus
        return max(0.0, min(1.0, reward))

    def reset_state(self):
        """Reset Solver state for new generation."""
        self.state = SolverState()
        # Keep prototype cache for efficiency

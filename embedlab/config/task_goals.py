"""Task goal configurations for different retrieval and routing tasks."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union
from enum import Enum
from pathlib import Path
import yaml

class TaskType(Enum):
    """Types of tasks the system can optimize for."""
    HIERARCHICAL_ROUTING = "hierarchical_routing"
    SEMANTIC_SEARCH = "semantic_search"
    QUESTION_ANSWERING = "question_answering"
    INTENT_CLASSIFICATION = "intent_classification"
    DOCUMENT_RETRIEVAL = "document_retrieval"
    MULTI_HOP_REASONING = "multi_hop_reasoning"
    CUSTOM = "custom"

@dataclass
class TaskGoal:
    """Defines the goal and constraints for instruction evolution."""

    # Core task definition
    name: str
    type: TaskType
    description: str
    objective: str  # What the instruction should achieve

    # Instruction generation prompts
    base_instruction: str  # Starting instruction template
    instruction_styles: List[str] = field(default_factory=list)
    instruction_constraints: List[str] = field(default_factory=list)

    # Mutation strategies specific to this task
    mutation_strategies: List[str] = field(default_factory=list)

    # Evaluation criteria
    success_criteria: Dict[str, float] = field(default_factory=dict)
    evaluation_focus: List[str] = field(default_factory=list)

    # LLM prompt templates
    style_prompt_template: str = ""
    mutation_prompt_template: str = ""
    crossover_prompt_template: str = ""

    # Optional custom configuration
    custom_config: Dict = field(default_factory=dict)

# Predefined task goals
PREDEFINED_GOALS = {
    "hierarchical_routing": TaskGoal(
        name="Hierarchical Category Routing",
        type=TaskType.HIERARCHICAL_ROUTING,
        description="Route queries through a hierarchical category tree to find the most specific matching leaf node",
        objective="Accurately route customer queries to the most appropriate category in the hierarchy",
        base_instruction="Route the customer query to the most appropriate category in the hierarchy.",
        instruction_styles=[
            "technical and precise",
            "simple and clear",
            "detailed with examples",
            "question-based",
            "action-oriented",
            "category-focused",
            "keyword-focused",
            "intent-focused"
        ],
        instruction_constraints=[
            "Guide routing of customer queries to categories",
            "Be clear and actionable",
            "Be concise (1-2 sentences)"
        ],
        mutation_strategies=[
            "Add more specific guidance",
            "Simplify and make more concise",
            "Emphasize accuracy over speed",
            "Focus on identifying key terms",
            "Add confidence assessment",
            "Rephrase using different terminology",
            "Make more action-oriented",
            "Add handling for ambiguous cases"
        ],
        success_criteria={
            "accuracy": 0.85,
            "confidence": 0.7,
            "consistency": 0.8
        },
        evaluation_focus=["routing_accuracy", "decision_confidence", "handling_ambiguity"],
        style_prompt_template="""Rewrite this routing instruction in a {style} style:

Original: {base}

The instruction should:
1. Guide routing of customer queries to categories
2. Be clear and actionable
3. Follow the {style} approach
4. Be concise (1-2 sentences)

Return only the rewritten instruction, nothing else.""",
        mutation_prompt_template="""Modify this routing instruction using the strategy: {strategy}

Current instruction: {instruction}

Create a variation that:
1. Applies the strategy: {strategy}
2. Maintains the core purpose (routing queries)
3. Is still clear and actionable
4. Introduces meaningful change

Return only the modified instruction, nothing else."""
    ),

    "semantic_search": TaskGoal(
        name="Semantic Document Search",
        type=TaskType.SEMANTIC_SEARCH,
        description="Find the most semantically relevant documents for a given query",
        objective="Retrieve documents that best match the semantic meaning and intent of the search query",
        base_instruction="Find and rank documents based on their semantic relevance to the search query.",
        instruction_styles=[
            "relevance-focused",
            "context-aware",
            "similarity-based",
            "meaning-oriented",
            "comprehensive",
            "precision-focused"
        ],
        instruction_constraints=[
            "Focus on semantic similarity not just keywords",
            "Consider context and meaning",
            "Prioritize relevance over exact matches"
        ],
        mutation_strategies=[
            "Emphasize conceptual understanding",
            "Add context awareness",
            "Focus on user intent",
            "Improve relevance scoring",
            "Add synonym handling",
            "Consider related concepts",
            "Balance precision and recall"
        ],
        success_criteria={
            "precision": 0.8,
            "recall": 0.75,
            "mrr": 0.85  # Mean Reciprocal Rank
        },
        evaluation_focus=["retrieval_accuracy", "ranking_quality", "semantic_understanding"]
    ),

    "intent_classification": TaskGoal(
        name="User Intent Classification",
        type=TaskType.INTENT_CLASSIFICATION,
        description="Classify user queries into predefined intent categories",
        objective="Accurately identify the underlying intent behind user queries",
        base_instruction="Identify and classify the user's intent based on their query.",
        instruction_styles=[
            "intent-focused",
            "action-based",
            "goal-oriented",
            "pattern-recognition",
            "context-sensitive"
        ],
        instruction_constraints=[
            "Focus on understanding user goals",
            "Distinguish between similar intents",
            "Handle ambiguous queries"
        ],
        mutation_strategies=[
            "Improve intent disambiguation",
            "Add context clues recognition",
            "Focus on action words",
            "Consider query structure",
            "Add confidence scoring",
            "Handle edge cases"
        ],
        success_criteria={
            "accuracy": 0.9,
            "f1_score": 0.85,
            "confidence": 0.8
        },
        evaluation_focus=["classification_accuracy", "intent_disambiguation", "confidence_calibration"]
    ),

    "question_answering": TaskGoal(
        name="Question Answering Retrieval",
        type=TaskType.QUESTION_ANSWERING,
        description="Retrieve passages that contain answers to specific questions",
        objective="Find text segments that directly answer the user's question",
        base_instruction="Retrieve passages that contain direct answers to the given question.",
        instruction_styles=[
            "answer-focused",
            "fact-finding",
            "comprehension-based",
            "extractive",
            "evidence-based"
        ],
        instruction_constraints=[
            "Prioritize passages with explicit answers",
            "Consider question type and expected answer format",
            "Focus on factual accuracy"
        ],
        mutation_strategies=[
            "Improve answer identification",
            "Focus on question keywords",
            "Consider answer types",
            "Add evidence ranking",
            "Handle complex questions",
            "Improve fact extraction"
        ],
        success_criteria={
            "exact_match": 0.7,
            "f1_score": 0.8,
            "answer_relevance": 0.85
        },
        evaluation_focus=["answer_accuracy", "passage_relevance", "fact_extraction"]
    )
}

def get_task_goal(name: str) -> Optional[TaskGoal]:
    """Get a predefined task goal by name."""
    return PREDEFINED_GOALS.get(name)

def list_available_goals() -> List[str]:
    """List all available predefined task goals."""
    return list(PREDEFINED_GOALS.keys())

def create_custom_goal(
    name: str,
    objective: str,
    base_instruction: str,
    **kwargs
) -> TaskGoal:
    """Create a custom task goal with user-defined parameters."""
    return TaskGoal(
        name=name,
        type=TaskType.CUSTOM,
        description=kwargs.get("description", "Custom task goal"),
        objective=objective,
        base_instruction=base_instruction,
        instruction_styles=kwargs.get("instruction_styles", ["simple", "detailed", "technical"]),
        instruction_constraints=kwargs.get("instruction_constraints", []),
        mutation_strategies=kwargs.get("mutation_strategies", ["rephrase", "simplify", "add detail"]),
        success_criteria=kwargs.get("success_criteria", {"accuracy": 0.8}),
        evaluation_focus=kwargs.get("evaluation_focus", ["accuracy"]),
        custom_config=kwargs.get("custom_config", {})
    )

def load_goal_from_yaml(file_path: Union[str, Path], goal_name: str) -> Optional[TaskGoal]:
    """
    Load a task goal configuration from a YAML file.

    Args:
        file_path: Path to the YAML file
        goal_name: Name of the goal to load from the file

    Returns:
        TaskGoal object or None if not found
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return None

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    if goal_name not in config:
        return None

    goal_config = config[goal_name]

    # Convert type string to enum if needed
    task_type = TaskType.CUSTOM
    if 'type' in goal_config:
        type_str = goal_config['type']
        for t in TaskType:
            if t.value == type_str:
                task_type = t
                break

    return TaskGoal(
        name=goal_config.get('name', goal_name),
        type=task_type,
        description=goal_config.get('description', ''),
        objective=goal_config.get('objective', ''),
        base_instruction=goal_config.get('base_instruction', ''),
        instruction_styles=goal_config.get('instruction_styles', []),
        instruction_constraints=goal_config.get('instruction_constraints', []),
        mutation_strategies=goal_config.get('mutation_strategies', []),
        success_criteria=goal_config.get('success_criteria', {}),
        evaluation_focus=goal_config.get('evaluation_focus', []),
        style_prompt_template=goal_config.get('style_prompt_template', ''),
        mutation_prompt_template=goal_config.get('mutation_prompt_template', ''),
        crossover_prompt_template=goal_config.get('crossover_prompt_template', ''),
        custom_config=goal_config.get('custom_config', {})
    )

def list_goals_in_yaml(file_path: Union[str, Path]) -> List[str]:
    """List all goal names defined in a YAML file."""
    file_path = Path(file_path)
    if not file_path.exists():
        return []

    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Filter out non-goal entries (like comments or metadata keys)
    return [key for key in config.keys() if isinstance(config[key], dict)]

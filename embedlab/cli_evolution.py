"""CLI for R-Zero evolution experiments."""
import typer
from typing import Optional
from pathlib import Path
import json
from rich import print
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config.evolution_config import EvolutionConfig
from .config.task_goals import (
    list_available_goals,
    get_task_goal,
    create_custom_goal,
    load_goal_from_yaml,
    list_goals_in_yaml,
    TaskGoal
)
from .engine.evolution_loop import EvolutionEngine

app = typer.Typer(
    add_completion=False,
    help="""R-Zero Self-Evolving Reasoning System

A self-improving instruction optimization system that uses co-evolution
of Challenger and Solver agents to discover optimal routing instructions.

KEY FEATURES:
• Configurable task goals (routing, search, Q&A, intent classification)
• Automatic dataset reuse to save time and resources
• Custom server endpoints for LLM and embedding services
• Genetic algorithm-based instruction evolution
• Progressive difficulty curriculum

QUICK START:
1. Ensure LLM and embedding servers are running
2. Run: embedlab-evolve evolve --hierarchy data/h.yaml --output runs/exp1
3. Monitor progress: embedlab-evolve monitor --dir runs/exp1/checkpoints

Use --help with any command for detailed options."""
)
console = Console()

@app.command("evolve")
def evolve(
    hierarchy: Path = typer.Option(..., "--hierarchy", "-h", help="Path to hierarchy YAML file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output directory for results"),
    generations: int = typer.Option(50, "--generations", "-g", help="Number of evolution generations"),
    population: int = typer.Option(20, "--population", "-p", help="Population size for instructions"),
    task_goal: str = typer.Option("hierarchical_routing", "--task-goal", "-t", help="Task goal to optimize for (hierarchical_routing, semantic_search, intent_classification, question_answering, or custom)"),
    task_goal_file: Optional[Path] = typer.Option(None, "--task-goal-file", help="YAML file containing custom task goal definitions"),
    custom_objective: Optional[str] = typer.Option(None, "--custom-objective", help="Objective for custom task goal"),
    custom_instruction: Optional[str] = typer.Option(None, "--custom-instruction", help="Base instruction for custom task goal"),
    llm_endpoint: str = typer.Option("http://localhost:8000", "--llm", help="LLM server endpoint URL (e.g., http://localhost:8000)"),
    embedding_endpoint: str = typer.Option("http://localhost:8002", "--embedding", help="Embedding server endpoint URL (e.g., http://localhost:8002)"),
    checkpoint_interval: int = typer.Option(10, "--checkpoint", help="Save checkpoint every N generations"),
    regenerate: bool = typer.Option(False, "--regenerate", "-r", help="Force regeneration of dataset even if compatible datasets exist. By default, existing datasets are reused if they match the task goal."),
    verbose: bool = typer.Option(True, "--verbose", "-v", help="Verbose output")
):
    """
    Run R-Zero evolution to optimize routing instructions.

    This implements a self-evolving system where:
    - Challenger generates increasingly difficult queries
    - Solver attempts to route them with evolving instructions
    - Instructions improve through genetic algorithm

    DATASET MANAGEMENT:
    - By default, reuses existing datasets if they match the current task goal
    - Automatically regenerates if task goal or hierarchy changes
    - Use --regenerate to force new dataset generation
    - Dataset metadata is saved to track compatibility

    SERVER REQUIREMENTS:
    - LLM server: Provides language model capabilities (default: http://localhost:8000)
    - Embedding server: Provides text embedding capabilities (default: http://localhost:8002)
    - Both servers must be running before starting evolution

    EXAMPLES:
    # Use default servers and reuse existing datasets
    embedlab-evolve evolve --hierarchy data/h.yaml --output runs/exp1

    # Custom servers
    embedlab-evolve evolve --llm http://192.168.1.100:8000 --embedding http://192.168.1.100:8002 ...

    # Force dataset regeneration
    embedlab-evolve evolve --regenerate ...
    """
    console.print("[bold cyan]R-Zero Evolution System[/bold cyan]")
    console.print(f"[dim]Hierarchy: {hierarchy}[/dim]")
    console.print(f"[dim]Output: {output}[/dim]")
    console.print(f"[dim]Generations: {generations}[/dim]")
    console.print(f"[dim]Task Goal: {task_goal}[/dim]")
    if not regenerate:
        console.print(f"[dim]Dataset: Will use existing if available[/dim]")
    else:
        console.print(f"[yellow]Dataset: Forcing regeneration[/yellow]")

    # Set up task goal
    goal = None

    # First check if loading from YAML file
    if task_goal_file:
        goal = load_goal_from_yaml(task_goal_file, task_goal)
        if goal:
            console.print(f"[dim]Loaded goal '{task_goal}' from {task_goal_file}[/dim]")
        else:
            console.print(f"[red]Goal '{task_goal}' not found in {task_goal_file}[/red]")
            available = list_goals_in_yaml(task_goal_file)
            if available:
                console.print(f"[dim]Available goals in file: {', '.join(available)}[/dim]")
            raise typer.Exit(1)

    # If not from file, check predefined or custom
    if not goal:
        if task_goal == "custom":
            if not custom_objective or not custom_instruction:
                console.print("[red]Custom task goal requires --custom-objective and --custom-instruction[/red]")
                raise typer.Exit(1)
            goal = create_custom_goal(
                name="Custom Task",
                objective=custom_objective,
                base_instruction=custom_instruction
            )
            console.print(f"[dim]Custom Objective: {custom_objective}[/dim]")
        else:
            goal = get_task_goal(task_goal)
            if not goal:
                console.print(f"[red]Unknown task goal: {task_goal}[/red]")
                console.print(f"[dim]Available predefined goals: {', '.join(list_available_goals())}[/dim]")
                console.print(f"[dim]Or use --task-goal-file to load from a YAML file[/dim]")
                raise typer.Exit(1)

    # Create config
    config = EvolutionConfig(
        task_goal=goal,
        task_goal_name=task_goal,
        llm_endpoint=llm_endpoint,
        embedding_endpoint=embedding_endpoint,
        generations=generations,
        population_size=population,
        checkpoint_interval=checkpoint_interval,
        verbose=verbose
    )

    # Initialize engine
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Initializing evolution engine...", total=None)

        try:
            engine = EvolutionEngine(
                config=config,
                hierarchy_path=str(hierarchy),
                output_dir=str(output),
                regenerate_dataset=regenerate
            )
            progress.update(task, description="[magenta]Engine initialized")
        except Exception as e:
            console.print(f"[red]Failed to initialize: {e}[/red]")
            raise typer.Exit(1)

    # Run evolution
    console.print("\n[bold]Starting evolution process...[/bold]")

    try:
        results = engine.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Evolution interrupted by user[/yellow]")
        raise typer.Exit(0)
    except Exception as e:
        console.print(f"[red]Evolution failed: {e}[/red]")
        raise typer.Exit(1)

    # Display results
    display_results(results)

@app.command("generate-dataset")
def generate_dataset(
    hierarchy: Path = typer.Option(..., "--hierarchy", "-h", help="Path to hierarchy YAML file"),
    output: Path = typer.Option(..., "--output", "-o", help="Output CSV file path"),
    size_per_node: int = typer.Option(10, "--size", "-s", help="Examples per leaf node"),
    augmentation: int = typer.Option(3, "--augment", "-a", help="Augmentation factor"),
    task_goal: str = typer.Option("hierarchical_routing", "--task-goal", "-t", help="Task goal for dataset generation"),
    task_goal_file: Optional[Path] = typer.Option(None, "--task-goal-file", help="YAML file containing custom task goal definitions"),
    llm_endpoint: str = typer.Option("http://localhost:8000", "--llm", help="LLM server endpoint URL")
):
    """
    Generate synthetic dataset from hierarchy.

    Uses LLM to create diverse examples for each category based on the task goal.
    The generated dataset adapts to the specified task (e.g., routing queries,
    search queries, questions, or user intents).

    EXAMPLES:
    # Generate for hierarchical routing
    embedlab-evolve generate-dataset --hierarchy data/h.yaml --output data.csv

    # Generate for semantic search task
    embedlab-evolve generate-dataset --task-goal semantic_search ...

    # Use custom LLM server
    embedlab-evolve generate-dataset --llm http://192.168.1.100:8000 ...
    """
    console.print("[bold cyan]Dataset Generation[/bold cyan]")
    console.print(f"[dim]Task Goal: {task_goal}[/dim]")

    from .core.llm_client import LLMClient, LLMConfig
    from .generation.dataset_builder import DatasetBuilder
    from .hierarchy import load_hierarchy

    # Set up task goal (similar to evolve command)
    goal = None
    if task_goal_file:
        goal = load_goal_from_yaml(task_goal_file, task_goal)
        if goal:
            console.print(f"[dim]Loaded goal '{task_goal}' from {task_goal_file}[/dim]")
        else:
            console.print(f"[red]Goal '{task_goal}' not found in {task_goal_file}[/red]")
            raise typer.Exit(1)

    if not goal:
        goal = get_task_goal(task_goal)
        if not goal:
            console.print(f"[red]Unknown task goal: {task_goal}[/red]")
            console.print(f"[dim]Available goals: {', '.join(list_available_goals())}[/dim]")
            raise typer.Exit(1)

    # Setup
    config = EvolutionConfig(
        task_goal=goal,
        task_goal_name=task_goal,
        llm_endpoint=llm_endpoint,
        seed_examples_per_node=size_per_node,
        augmentation_factor=augmentation
    )

    llm = LLMClient(LLMConfig(base_url=llm_endpoint))
    hierarchy_root = load_hierarchy(str(hierarchy))

    builder = DatasetBuilder(llm, config, hierarchy_root)

    with console.status("[bold magenta]Generating seed dataset...") as status:
        # Generate seed
        seed_df = builder.generate_seed_dataset()
        status.update("[bold magenta]Augmenting dataset...")

        # Augment
        augmented_df = builder.augment_dataset(seed_df, factor=augmentation)
        status.update("[bold magenta]Adding boundary examples...")

        # Add boundary examples
        boundary_df = builder.generate_cross_boundary_examples(num_examples=20)

        # Combine and balance
        import pandas as pd
        full_df = pd.concat([augmented_df, boundary_df], ignore_index=True)
        balanced_df = builder.balance_dataset(full_df)

    # Save
    builder.export_dataset(balanced_df, output)

    # Show statistics
    table = Table(title="Dataset Statistics")
    table.add_column("Category", style="cyan")
    table.add_column("Count", style="magenta")

    for category, count in balanced_df["label_path"].value_counts().items():
        table.add_row(category, str(count))

    console.print(table)
    console.print(f"[magenta]✓ Dataset saved to {output}[/magenta]")

@app.command("test-routing")
def test_routing(
    hierarchy: Path = typer.Option(..., "--hierarchy", "-h", help="Path to hierarchy YAML file"),
    dataset: Path = typer.Option(..., "--dataset", "-d", help="Path to test dataset CSV"),
    instruction: Optional[str] = typer.Option(None, "--instruction", "-i", help="Instruction to use for routing"),
    llm_endpoint: str = typer.Option("http://localhost:8000", "--llm", help="LLM server endpoint URL"),
    embedding_endpoint: str = typer.Option("http://localhost:8002", "--embedding", help="Embedding server endpoint URL")
):
    """
    Test routing performance with given instruction.

    Evaluates how well a specific instruction performs on a test dataset.
    Uses both LLM and embedding servers for routing decisions.

    EXAMPLES:
    # Test with default instruction
    embedlab-evolve test-routing --hierarchy data/h.yaml --dataset test.csv

    # Test with custom instruction
    embedlab-evolve test-routing --instruction "Route based on keywords" ...

    # Use custom servers
    embedlab-evolve test-routing --llm http://192.168.1.100:8000 --embedding http://192.168.1.100:8002 ...
    """
    console.print("[bold cyan]Testing Routing Performance[/bold cyan]")

    import pandas as pd
    from .core.llm_client import LLMClient, LLMConfig
    from .core.embedding_client import EmbeddingClient, EmbeddingConfig
    from .evolution.solver import SolverAgent
    from .config.evolution_config import InstructionGene
    from .hierarchy import load_hierarchy

    # Setup
    config = EvolutionConfig(
        llm_endpoint=llm_endpoint,
        embedding_endpoint=embedding_endpoint
    )

    llm = LLMClient(LLMConfig(base_url=llm_endpoint))
    embedder = EmbeddingClient(EmbeddingConfig(base_url=embedding_endpoint))
    hierarchy_root = load_hierarchy(str(hierarchy))

    solver = SolverAgent(llm, embedder, config, hierarchy_root)

    # Load dataset
    df = pd.read_csv(dataset)
    queries = list(zip(df["text"], df["label_path"]))

    # Default instruction if not provided
    if not instruction:
        instruction = "Route the customer query to the most appropriate category in the hierarchy."

    gene = InstructionGene(
        id="test",
        content=instruction,
        generation=0
    )

    # Test routing
    with console.status("[bold magenta]Testing routing...") as status:
        results = solver.route_queries(queries[:50], gene, use_embeddings=True)  # Test first 50

    # Calculate metrics
    accuracy = sum(r.success for r in results) / len(results)
    avg_confidence = sum(r.confidence for r in results) / len(results)

    # Display results
    console.print(f"\n[bold]Results:[/bold]")
    console.print(f"Accuracy: [magenta]{accuracy:.2%}[/magenta]")
    console.print(f"Average Confidence: [yellow]{avg_confidence:.2%}[/yellow]")

    # Show confusion matrix
    from collections import defaultdict
    confusion = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r.true_path][r.predicted_path] += 1

    # Display failures
    failures = [r for r in results if not r.success]
    if failures:
        console.print(f"\n[bold]Sample Failures:[/bold]")
        for f in failures[:5]:
            console.print(f"[red]✗[/red] '{f.query[:50]}...'")
            console.print(f"   Expected: [magenta]{f.true_path}[/magenta]")
            console.print(f"   Got: [red]{f.predicted_path}[/red]")

@app.command("list-goals")
def list_goals():
    """
    List all available task goals for instruction evolution.

    Shows predefined task goals (hierarchical_routing, semantic_search,
    intent_classification, question_answering) and their configurations.

    Use these goals with --task-goal parameter in evolve and generate-dataset commands.
    """
    console.print("[bold cyan]Available Task Goals[/bold cyan]\n")

    goals = list_available_goals()

    for goal_name in goals:
        goal = get_task_goal(goal_name)
        console.print(f"[bold magenta]{goal_name}[/bold magenta]")
        console.print(f"  Type: {goal.type.value}")
        console.print(f"  Description: {goal.description}")
        console.print(f"  Objective: {goal.objective}")
        console.print(f"  Base Instruction: {goal.base_instruction}")
        console.print(f"  Success Criteria: {goal.success_criteria}")
        console.print()

    console.print("[dim]To use a specific goal, add --task-goal <name> to the evolve command[/dim]")
    console.print("[dim]For custom goals, use --task-goal custom with --custom-objective and --custom-instruction[/dim]")

@app.command("monitor")
def monitor(
    checkpoint_dir: Path = typer.Option("checkpoints", "--dir", "-d", help="Directory containing evolution checkpoints")
):
    """
    Monitor evolution progress from checkpoints.

    Displays evolution history, fitness progression, and best instructions
    from saved checkpoint files.

    EXAMPLES:
    # Monitor default checkpoint directory
    embedlab-evolve monitor

    # Monitor specific directory
    embedlab-evolve monitor --dir evolution_runs/exp1/checkpoints
    """
    console.print("[bold cyan]Evolution Monitor[/bold cyan]")

    # Find all checkpoint files
    checkpoint_files = sorted(checkpoint_dir.glob("checkpoint_gen_*.json"))

    if not checkpoint_files:
        console.print("[yellow]No checkpoints found[/yellow]")
        return

    # Load checkpoints
    checkpoints = []
    for f in checkpoint_files:
        with open(f) as file:
            checkpoints.append(json.load(file))

    # Display progress table
    table = Table(title="Evolution Progress")
    table.add_column("Generation", style="cyan")
    table.add_column("Best Fitness", style="magenta")
    table.add_column("Instruction Preview", style="yellow")

    for cp in checkpoints:
        gen = str(cp["generation"])
        fitness = f"{cp['best_instruction']['fitness']:.3f}" if cp.get("best_instruction") else "N/A"
        instruction = cp.get("best_instruction", {}).get("content", "N/A")[:50] + "..."

        table.add_row(gen, fitness, instruction)

    console.print(table)

    # Plot progress if possible
    if checkpoints and checkpoints[-1].get("history"):
        history = checkpoints[-1]["history"]
        console.print("\n[bold]Fitness Over Time:[/bold]")

        # Simple ASCII plot
        fitness_values = [h["best_fitness"] for h in history]
        max_fitness = max(fitness_values)
        scale = 20  # Height of plot

        for i, fitness in enumerate(fitness_values[-20:]):  # Last 20 generations
            bar_length = int((fitness / max_fitness) * scale)
            bar = "█" * bar_length
            console.print(f"Gen {i:3d}: {bar} {fitness:.3f}")

def display_results(results: dict):
    """Display evolution results."""
    console.print("\n[bold magenta]✓ Evolution Complete![/bold magenta]")

    # Best instruction
    console.print(f"\n[bold]Best Instruction:[/bold]")
    console.print(f"[cyan]{results.get('best_instruction', 'N/A')}[/cyan]")

    # Performance metrics
    console.print(f"\n[bold]Performance:[/bold]")
    console.print(f"Test Accuracy: [magenta]{results.get('test_accuracy', 0):.2%}[/magenta]")
    console.print(f"Test Confidence: [yellow]{results.get('test_confidence', 0):.2%}[/yellow]")
    console.print(f"Total Generations: [blue]{results.get('total_generations', 0)}[/blue]")

    # Top instructions table
    if "final_population" in results:
        table = Table(title="Top Instructions")
        table.add_column("Rank", style="cyan")
        table.add_column("Fitness", style="magenta")
        table.add_column("Instruction", style="yellow")

        for i, gene in enumerate(results["final_population"][:5], 1):
            table.add_row(
                str(i),
                f"{gene['fitness']:.3f}",
                gene["content"][:60] + "..."
            )

        console.print(table)

if __name__ == "__main__":
    app()

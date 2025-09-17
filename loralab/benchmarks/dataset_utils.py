"""Utilities for managing versioned datasets"""

from pathlib import Path
from typing import Optional
import json

def get_latest_dataset(dataset_dir: str = "data/tool_calling_datasets") -> Optional[str]:
    """
    Get the path to the most recent dataset.

    Args:
        dataset_dir: Directory containing versioned datasets

    Returns:
        Path to the latest dataset file, or None if no datasets exist
    """
    dataset_path = Path(dataset_dir)

    # First check for latest.json
    latest_file = dataset_path / "latest.json"
    if latest_file.exists():
        return str(latest_file)

    # Otherwise find the most recent timestamped file
    if not dataset_path.exists():
        return None

    json_files = list(dataset_path.glob("*.json"))

    # Filter out non-timestamped files
    timestamped_files = []
    for f in json_files:
        if f.stem.replace('_', '').isdigit() and len(f.stem) == 15:  # YYYYMMDD_HHMMSS format
            timestamped_files.append(f)

    if not timestamped_files:
        # Fall back to any JSON file sorted by modification time
        if json_files:
            return str(max(json_files, key=lambda f: f.stat().st_mtime))
        return None

    # Return the most recent based on filename (which contains timestamp)
    return str(max(timestamped_files))


def load_dataset(dataset_path: Optional[str] = None) -> dict:
    """
    Load a dataset, defaulting to the latest if no path specified.

    Args:
        dataset_path: Path to dataset file, or None to use latest

    Returns:
        Dictionary containing tools and examples

    Raises:
        FileNotFoundError: If no dataset is found
    """
    if dataset_path is None:
        dataset_path = get_latest_dataset()
        if dataset_path is None:
            raise FileNotFoundError(
                "No datasets found. Generate a dataset first using generate_tool_dataset.py"
            )
        print(f"Using latest dataset: {dataset_path}")

    with open(dataset_path, 'r') as f:
        return json.load(f)


def list_datasets(dataset_dir: str = "data/tool_calling_datasets") -> list:
    """
    List all available datasets with metadata.

    Args:
        dataset_dir: Directory containing datasets

    Returns:
        List of dictionaries with dataset information
    """
    dataset_path = Path(dataset_dir)
    if not dataset_path.exists():
        return []

    datasets = []
    for json_file in dataset_path.glob("*.json"):
        if json_file.name == "latest.json":
            continue

        try:
            # Load to get metadata
            with open(json_file, 'r') as f:
                data = json.load(f)

            # Try to load accompanying stats
            stats_file = json_file.parent / f"{json_file.stem}_stats.json"
            stats = {}
            if stats_file.exists():
                with open(stats_file, 'r') as f:
                    stats = json.load(f)

            datasets.append({
                "path": str(json_file),
                "name": json_file.name,
                "size": len(data.get("examples", [])),
                "created": json_file.stat().st_mtime,
                "stats": stats
            })
        except Exception:
            continue

    # Sort by creation time (newest first)
    datasets.sort(key=lambda x: x["created"], reverse=True)
    return datasets

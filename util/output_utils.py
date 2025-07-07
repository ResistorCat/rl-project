"""
Model management utilities for the Pokémon RL project.
"""

from pathlib import Path
from util.types import RLModel


def get_output_dir(
    task_type: str | None = None, model_type: RLModel | None = None
) -> Path:
    """
    Get the output directory path, creating it if it doesn't exist.

    Returns:
        Path: The models directory path
    """
    output_dir = Path("outputs")
    if task_type:
        output_dir /= task_type
    if model_type and task_type:
        output_dir /= model_type.value.lower()
    output_dir.mkdir(exist_ok=True)
    return output_dir

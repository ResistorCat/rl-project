"""
Model management utilities for the Pokémon RL project.
"""

from datetime import datetime
from pathlib import Path
from utils.types import RLModel
from utils.output_utils import get_output_dir


def get_model_path(model_type: RLModel):
    """
    Get the full path for saving a model.
    """
    return (
        get_output_dir(task_type="train", model_type=model_type)
        / f"{model_type.value}_model.zip"
    )


def get_monitor_dir():
    """
    Get the monitor directory path for training logs.

    Returns:
        Path: The monitor directory path (models/train)
    """
    monitor_dir = Path("models") / "train"
    monitor_dir.mkdir(parents=True, exist_ok=True)
    return monitor_dir


def get_monitor_file_path():
    """
    Get the full path for the monitor file.

    Returns:
        Path: The full path to the monitor file (models/train/latest.monitor.csv)
    """
    monitor_dir = get_monitor_dir()
    return monitor_dir / "latest"

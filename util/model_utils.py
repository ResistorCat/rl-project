"""
Model management utilities for the Pokémon RL project.
"""

from datetime import datetime
from pathlib import Path
from util.types import RLModel


def get_models_dir():
    """
    Get the models directory path, creating it if it doesn't exist.
    
    Returns:
        Path: The models directory path
    """
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    return models_dir


def generate_model_filename(model_type: RLModel, include_timestamp: bool = True):
    """
    Generate a filename for saving a trained model.
    
    Args:
        model_type: The type of model (PPO, DQN, etc.)
        include_timestamp: Whether to include timestamp in the filename
        
    Returns:
        str: The generated filename
    """
    base_name = f"{model_type.value}_pokemon_model"
    
    if include_timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{base_name}_{timestamp}"
    
    return base_name


def get_model_path(model_type: RLModel, include_timestamp: bool = True):
    """
    Get the full path for saving a model.
    
    Args:
        model_type: The type of model (PPO, DQN, etc.)
        include_timestamp: Whether to include timestamp in the filename
        
    Returns:
        Path: The full path where the model should be saved
    """
    models_dir = get_models_dir()
    filename = generate_model_filename(model_type, include_timestamp)
    return models_dir / filename


def list_saved_models(model_type: RLModel | None = None):
    """
    List all saved models, optionally filtered by model type.
    
    Args:
        model_type: Optional model type to filter by
        
    Returns:
        list: List of model file paths
    """
    models_dir = get_models_dir()
    
    if not models_dir.exists():
        return []
    
    if model_type:
        pattern = f"{model_type.value}_pokemon_model*"
    else:
        pattern = "*_pokemon_model*"
    
    return list(models_dir.glob(pattern))


def get_latest_model_path(model_type: RLModel):
    """
    Get the path to the most recently saved model of the given type.
    
    Args:
        model_type: The type of model to find
        
    Returns:
        Path | None: Path to the latest model, or None if no models found
    """
    models = list_saved_models(model_type)
    
    if not models:
        return None
    
    # Sort by modification time, most recent first
    models.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return models[0]


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

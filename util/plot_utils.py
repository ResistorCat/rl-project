"""
Plotting utilities for training analysis.
"""

import os
from pathlib import Path
import matplotlib.pyplot as plt
from util.types import RLModel
import pandas as pd


def plot_training_learning_curve(
    model_type: RLModel, monitor_path: str | Path, save_path: str | Path
):
    """
    Generate and save a learning curve plot from monitor data.

    Args:
        model_type: The type of model that was trained
        monitor_dir: Directory containing monitor CSV files

    Returns:
        str: Path to the saved plot file
    """
    try:
        # Load monitor results
        df = pd.read_csv(monitor_path, skiprows=1)

        if df.empty:
            raise ValueError("No training data found in monitor files")

        # Create the plot
        plt.figure(figsize=(12, 6))

        # Plot raw rewards
        plt.subplot(1, 2, 1)
        plt.plot(
            df.index, df["r"], alpha=0.6, color="lightblue", label="Episode Rewards"
        )
        plt.title(f"{model_type.value} - Episode Rewards")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot smoothed rewards (rolling average)
        plt.subplot(1, 2, 2)
        window_size = max(1, min(50, len(df) // 10))  # Adaptive window size
        if len(df) > 1:
            rolling_mean = df["r"].rolling(window=window_size, center=True).mean()
            plt.plot(
                df.index,
                rolling_mean,
                color="darkblue",
                linewidth=2,
                label=f"Rolling Average (window={window_size})",
            )
        else:
            # If only one episode, just plot the single point
            plt.plot(
                df.index,
                df["r"],
                "o",
                color="darkblue",
                markersize=8,
                label="Single Episode",
            )

        plt.title(f"{model_type.value} - Learning Curve (Smoothed)")
        plt.xlabel("Episode")
        plt.ylabel("Average Reward")
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Add overall title and statistics
        mean_reward = df["r"].mean()
        max_reward = df["r"].max()
        min_reward = df["r"].min()
        total_episodes = len(df)

        plt.suptitle(
            f"{model_type.value} Training Results - {total_episodes} Episodes\n"
            f"Mean: {mean_reward:.2f} | Max: {max_reward:.2f} | Min: {min_reward:.2f}",
            fontsize=14,
        )

        plt.tight_layout()
        # Save the plot
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()  # Close the figure to free memory
    except Exception as e:
        raise RuntimeError(f"Failed to generate learning curve plot: {e}")

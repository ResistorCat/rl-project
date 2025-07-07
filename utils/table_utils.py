"""
Table utilities for evaluation results.
"""

import pandas as pd
from pathlib import Path
from utils.types import RLModel
from utils.output_utils import get_output_dir


def create_evaluation_table(
    model_type: RLModel,
    battles_won: int,
    total_battles: int,
    mean_reward: float,
    std_reward: float,
):
    """
    Create a table with evaluation metrics using pandas DataFrame.

    Args:
        model_type: The type of model evaluated
        battles_won: Number of battles won by the model
        total_battles: Total number of battles
        mean_reward: Mean reward across all battles
        std_reward: Standard deviation of rewards

    Returns:
        tuple: (str: Path to saved LaTeX file, pd.DataFrame: Results data)
    """
    win_rate = (battles_won / total_battles) * 100
    loss_rate = ((total_battles - battles_won) / total_battles) * 100

    # Create DataFrame with evaluation results
    data = {
        "Metric": [
            "Model Type",
            "Total Battles",
            "Battles Won",
            "Win Rate (%)",
            "Loss Rate (%)",
            "Mean Reward",
            "Std Reward",
        ],
        "Value": [
            model_type.value,
            total_battles,
            battles_won,
            f"{win_rate:.1f}%",
            f"{loss_rate:.1f}%",
            f"{mean_reward:.2f}",
            f"{std_reward:.2f}",
        ],
    }

    df = pd.DataFrame(data)

    # Use pandas built-in LaTeX formatting
    latex_content = df.to_latex(
        index=False,
        escape=False,
        column_format="|l|c|",
        caption=f"Evaluation Results for {model_type.value} Model",
        label=f"tab:{model_type.value.lower()}_evaluation",
        position="h",
    )

    # Save the table
    output_dir = get_evaluate_output_dir()
    table_filename = f"{model_type.value}_evaluation_table.tex"
    table_path = output_dir / table_filename

    with open(table_path, "w") as f:
        f.write(latex_content)

    # Also save as CSV for easy data analysis
    csv_filename = f"{model_type.value}_evaluation_results.csv"
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)

    return str(table_path), df

def create_battle_results_dataframe(
    model_type: RLModel,
    battle_rewards: list,
    battle_results: list,
    battle_steps: list | None = None,
):
    """
    Create a detailed DataFrame with individual battle results.

    Args:
        model_type: The type of model evaluated
        battle_rewards: List of rewards from each battle
        battle_results: List of battle results (True for win, False for loss)
        battle_steps: Optional list of steps taken in each battle

    Returns:
        pd.DataFrame: Detailed battle results
    """
    battle_data = {
        "Battle_Number": list(range(1, len(battle_rewards) + 1)),
        "Reward": battle_rewards,
        "Result": ["Win" if result else "Loss" for result in battle_results],
        "Win": battle_results,
    }

    # Add steps if provided
    if battle_steps is not None:
        battle_data["Steps"] = battle_steps

    df = pd.DataFrame(battle_data)

    # Save detailed results to CSV
    output_dir = get_evaluate_output_dir()
    csv_filename = f"{model_type.value}_detailed_battle_results.csv"
    csv_path = output_dir / csv_filename
    df.to_csv(csv_path, index=False)

    return df

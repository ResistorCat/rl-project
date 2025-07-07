"""
Table utilities for evaluation results.
"""

import pandas as pd
from util.types import RLModel
from util.output_utils import get_output_dir


class EvaluationResults:
    """
    Class to handle evaluation results.
    """

    def __init__(self, model_type: RLModel):
        self.model_type = model_type
        self.output_dir = get_output_dir(task_type="evaluate", model_type=model_type)
        self.results: list[dict] = []

    def add_result(
        self,
        opponent_name: str,
        battles_won: int,
        total_battles: int,
        mean_reward: float,
        std_reward: float,
    ):
        """
        Add a new evaluation result.
        """
        win_rate = (battles_won / total_battles) * 100 if total_battles > 0 else 0
        loss_rate = (
            ((total_battles - battles_won) / total_battles) * 100
            if total_battles > 0
            else 0
        )

        result = {
            "player": opponent_name,
            "total_battles": total_battles,
            "battles_won": battles_won,
            "win_rate": win_rate,
            "loss_rate": loss_rate,
            "mean_reward": mean_reward,
            "std_reward": std_reward,
        }
        self.results.append(result)

    def save(self):
        """
        Save the DataFrame.
        """
        table = self._to_df()
        file_path = self.output_dir / f"{self.model_type.value}_evaluation_results.csv"
        table.to_csv(file_path, index=False)
        print(f"✅ Evaluation results saved to: {file_path}")

    def print(self):
        """
        Print the evaluation results.
        """
        table = self._to_df()
        print(table.to_string(index=False))

    def _to_df(self) -> pd.DataFrame:
        """
        Convert the results to a pandas DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing evaluation results
        """
        data = {
            "Metric": [
                "Total Battles",
                "Battles Won",
                "Win Rate (%)",
                "Loss Rate (%)",
                "Mean Reward",
                "Std Reward",
            ]
        }
        for result in self.results:
            data[result.get("player", "UnknownPlayer")] = [
                result.get("total_battles", 0),
                result.get("battles_won", 0),
                f"{result.get('win_rate', 0):.1f}%",
                f"{result.get('loss_rate', 0):.1f}%",
                f"{result.get('mean_reward', 0):.2f}",
                f"{result.get('std_reward', 0):.2f}",
            ]
        return pd.DataFrame(data)

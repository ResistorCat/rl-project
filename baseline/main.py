import asyncio, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from poke_env import AccountConfiguration

from utils.evaluation import evaluate_player, accept_challenges
from baseline.train import train
from baseline.players import BaselinePlayer

######################################
TRAIN           = True
EVALUATE        = False
CHALLENGE       = False

SAVE_MODEL_PATH = "models/baseline_dqn_pokemon_model"
SAVE_CSV_PATH   = "results/baseline_dqn"
######################################


if __name__ == "__main__":  
  if TRAIN:
    train(csv_path=SAVE_CSV_PATH, model_path=SAVE_MODEL_PATH, total_timesteps=50_000)
  baseline_player = BaselinePlayer(
    "models/baseline_dqn_pokemon_model",
    account_configuration=AccountConfiguration("Baseline Bot", None)
    )
  if EVALUATE:
    asyncio.run(evaluate_player(baseline_player))
  if CHALLENGE:
    asyncio.run(accept_challenges(baseline_player))
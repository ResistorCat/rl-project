import asyncio, sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.evaluation import evaluate_player, accept_challenges
from baseline.dqn_train import train

######################################
TRAIN           = False
EVALUATE        = True
CHALLENGE       = False

SAVE_MODEL_PATH = "models/baseline_dqn_pokemon_model"
SAVE_CSV_PATH   = "results/baseline_dqn"
######################################


if __name__ == "__main__":  
  if TRAIN:
    train(csv_path=SAVE_CSV_PATH, model_path=SAVE_MODEL_PATH, total_timesteps=50_000)
  if EVALUATE:
    asyncio.run(evaluate_player())
  if CHALLENGE:
    asyncio.run(accept_challenges())
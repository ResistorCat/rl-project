import asyncio

from utils_evaluate import evaluate_player, accept_challenges
from dqn_train import train

######################################
TRAIN           = True
EVALUATE        = False
CHALLENGE       = False

SAVE_MODEL_PATH = "dqn_pokemon_model"
SAVE_CSV_PATH   = "dqn_results"
######################################


if __name__ == "__main__":  
  if TRAIN:
    train(csv_path=SAVE_CSV_PATH, model_path=SAVE_MODEL_PATH, total_timesteps=25_000)
  if EVALUATE:
    asyncio.run(evaluate_player())
  if CHALLENGE:
    asyncio.run(accept_challenges())
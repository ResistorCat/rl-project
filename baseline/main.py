import asyncio

from baseline import evaluate_player, accept_challenges
from baseline import train

######################################
TRAIN           = True
EVALUATE        = True
CHALLENGE       = True

SAVE_MODEL_PATH = "dqn_pokemon_model"
SAVE_CSV_PATH   = "dqn_results"
######################################


if __name__ == "__main__":  
  if TRAIN:
    train(csv_path=SAVE_CSV_PATH, model_path=SAVE_MODEL_PATH)
  if EVALUATE:
    asyncio.run(evaluate_player())
  if CHALLENGE:
    asyncio.run(accept_challenges())
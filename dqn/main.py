import asyncio
from poke_env import AccountConfiguration

from dqn.train import train
from dqn.player import OurDQNPlayer
from utils.evaluation import accept_challenges, evaluate_player

######################################
TRAIN           = True
EVALUATE        = False
CHALLENGE       = False

SAVE_MODEL_PATH = "models/our_dqn_pokemon_model"
SAVE_CSV_PATH   = "results/our_dqn"
######################################


if __name__ == "__main__":  
  if TRAIN:
    train(
      verbose=1,
      learning_rate=0.0005,                   # Más conservador, mejor para generalizar
      buffer_size=150_000,                     # Suficiente para >500k pasos
      learning_starts=10_000,                  # Retrasa inicio para llenar buffer con variedad
      batch_size=64,                           # Más estable
      tau=1,                                   # DQN no usa soft update, está bien
      gamma=0.99,                              # Conserva
      train_freq=1,
      gradient_steps=4,                        # Más actualizaciones por paso → más eficiente
      target_update_interval=5000,             # Actualizaciones más espaciadas = más estables
      exploration_fraction=0.5,                # Reducción a la mitad, explora y explota
      exploration_initial_eps=1.0,
      exploration_final_eps=0.1,               # Más exploración mínima
      max_grad_norm=10,
      policy_kwargs=dict(net_arch=[256, 128]), # Red más profunda y con más capacidad
      device="cuda",
      total_timesteps=100_000,                 # Entrenamiento más prolongado
      csv_path = SAVE_CSV_PATH,
      model_path = SAVE_MODEL_PATH
    )
  our_first_dqn_player = OurDQNPlayer(
    "models/our_dqn_pokemon_model",
    account_configuration=AccountConfiguration("Our First DQN", None)
    )
  if EVALUATE:
    asyncio.run(evaluate_player(our_first_dqn_player))
  if CHALLENGE:
    asyncio.run(accept_challenges(our_first_dqn_player))
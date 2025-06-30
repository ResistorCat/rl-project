import asyncio
from dqn.train import train
from dqn.player import OurDQNPlayer
from utils.evaluation import accept_challenges

######################################
TRAIN           = True
EVALUATE        = False
CHALLENGE       = False

SAVE_MODEL_PATH = "models/our_dqn_pokemon_model"
SAVE_CSV_PATH   = "results/our_dqn"
######################################


if __name__ == "__main__":  
  if TRAIN:
    # train(csv_path=SAVE_CSV_PATH, model_path=SAVE_MODEL_PATH, total_timesteps=50_000)
    train(
      verbose=1,
      learning_rate=0.00025,                   # Más conservador, mejor para generalizar
      buffer_size=150_000,                     # Suficiente para >500k pasos
      learning_starts=10_000,                  # Retrasa inicio para llenar buffer con variedad
      batch_size=64,                           # Más estable
      tau=1,                                   # DQN no usa soft update, está bien
      gamma=0.99,                              # Conserva
      train_freq=1,
      gradient_steps=4,                        # Más actualizaciones por paso → más eficiente
      target_update_interval=2000,             # Actualizaciones más espaciadas = más estables
      exploration_fraction=0.4,                # Reducción más rápida, para actuar antes
      exploration_initial_eps=1.0,
      exploration_final_eps=0.05,              # Más exploración mínima
      max_grad_norm=10,
      policy_kwargs=dict(net_arch=[128, 128]), # Red más profunda y con más capacidad
      device="cuda",
      total_timesteps=500_000,                 # Entrenamiento más prolongado
      csv_path = SAVE_CSV_PATH,
      model_path = SAVE_MODEL_PATH
    )
  # if EVALUATE:
  #   asyncio.run(evaluate_player())
  if CHALLENGE:
    asyncio.run(accept_challenges())
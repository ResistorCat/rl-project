from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from poke_env.player import SingleAgentWrapper

from baseline.baseline_env import BaselineSinglesEnv
from baseline.players import SimpleRandomPlayer


def train(
  verbose=1,
  learning_rate=0.0005,
  buffer_size=70_000,
  learning_starts=5_000,
  batch_size=32,
  tau=1,
  gamma=0.99,
  train_freq=1,
  gradient_steps=1,
  target_update_interval=1_000,
  exploration_fraction=0.5,
  exploration_initial_eps=1,
  exploration_final_eps=0.01,
  max_grad_norm=10,
  policy_kwargs=dict(net_arch=[64, 64]),
  device="cuda",
  total_timesteps=100_000,
  csv_path: str = "results/baseline_dqn",
  model_path: str = "models/baseline_dqn_pokemon_model"
  ):

  env = BaselineSinglesEnv(
    battle_format=f"gen9randombattle",
    log_level=25,
    start_challenging=True,
    strict=False,
  )

  env = SingleAgentWrapper(env, SimpleRandomPlayer())
  env = Monitor(env, filename=csv_path)

  model = DQN(
    "MlpPolicy",
    env,
      verbose=verbose,
      learning_rate=learning_rate,
      buffer_size=buffer_size,
      learning_starts=learning_starts,
      batch_size=batch_size,
      tau=tau,
      gamma=gamma,
      train_freq=train_freq,
      gradient_steps=gradient_steps,
      target_update_interval=target_update_interval,
      exploration_fraction=exploration_fraction,
      exploration_initial_eps=exploration_initial_eps,
      exploration_final_eps=exploration_final_eps,
      max_grad_norm=max_grad_norm,
      policy_kwargs=policy_kwargs,
    device=device
  )

  model.learn(total_timesteps=total_timesteps)

  model.save(model_path)
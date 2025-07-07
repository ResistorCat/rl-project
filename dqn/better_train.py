from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from poke_env.player import SingleAgentWrapper
from stable_baselines3.common.callbacks import CheckpointCallback
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


from dqn.environment import FirstDQNSinglesEnv
from baseline.players import SimpleRandomPlayer

TOTAL_TIMESTEPS = {
    "random": 300_000,
    "max": 400_000,
    "heuristic": 500_000,
}

SAVE_DIR = "./checkpoints"

checkpoint_callback = CheckpointCallback(
  save_freq=100_000, save_path='./checkpoints/',
  name_prefix='dqn_model'
)

def train_against_opponent(opponent, phase_name, model=None):
    print(f"\n===== ENTRENANDO CONTRA: {phase_name.upper()} =====")

    env = FirstDQNSinglesEnv(
        battle_format=f"gen9randombattle",
        log_level=25,
        start_challenging=True,
        strict=False,
    )

    env = SingleAgentWrapper(env, opponent)
    env = Monitor(env, filename=f"results/our_dqn_against_{phase_name}")

    # Checkpoint automático cada 100k steps
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=f"{SAVE_DIR}/checkpoints_{phase_name}",
        name_prefix="dqn_model"
    )

    if model is None:
        model = DQN(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=0.0005,
            buffer_size=150_000,
            learning_starts=10_000,
            batch_size=64,
            tau=1,
            gamma=0.99,
            train_freq=1,
            gradient_steps=4,
            target_update_interval=5_000,
            exploration_fraction=0.5,
            exploration_initial_eps=1,
            exploration_final_eps=0.1,
            max_grad_norm=10,
            policy_kwargs=dict(net_arch=[256, 128]),
            device="cuda",
            tensorboard_log=f"{SAVE_DIR}/logs")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS[phase_name],
        callback=checkpoint_callback,
    )

    model.save(f"{SAVE_DIR}/dqn_final_{phase_name}")
    env.close()
    return model



# -----------------------------------------------
# Fase 1: contra RandomPlayer
# -----------------------------------------------
random_opponent = SimpleRandomPlayer(battle_format="gen9randombattle")
model = train_against_opponent(random_opponent, phase_name="random")

# -----------------------------------------------
# Fase 2: contra MaxBasePowerPlayer
# -----------------------------------------------
max_opponent = MaxBasePowerPlayer(battle_format="gen9randombattle")
model = train_against_opponent(max_opponent, phase_name="max", model=model)

# -----------------------------------------------
# Fase 3: contra SimpleHeuristicsPlayer
# -----------------------------------------------
heuristic_opponent = SimpleHeuristicsPlayer(battle_format="gen9randombattle")
model = train_against_opponent(heuristic_opponent, phase_name="heuristic", model=model)

print("\n✅ Entrenamiento completo")
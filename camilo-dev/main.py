import numpy as np
import asyncio
from gymnasium.spaces import Box
from gymnasium.utils.env_checker import check_env
from poke_env.environment import Move
from stable_baselines3 import DQN
from poke_env import AccountConfiguration, cross_evaluate



from poke_env.player import RandomPlayer, SingleAgentWrapper, SinglesEnv


class MySinglesEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        low = [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [3, 3, 3, 3, 4, 4, 4, 4, 1, 1]
        self.observation_spaces = {
            agent: Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        # print(battle.available_moves)
        for i, move in enumerate(battle.available_moves):
            # print(move)
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)

        # We count how many pokemons have fainted in each team
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        # Final vector with 10 components
        final_vector = np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
            ]
        )
        return np.float32(final_vector)


def single_agent_play_function(env: SingleAgentWrapper, n_battles: int):
    for _ in range(n_battles):
        done = False
        env.reset()
        while not done:
            action = env.action_space.sample()
            _, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated


def start_single_agent_env_run():
    GENERATIONS = [9]
    NUM_CHALLENGES = 3
    for gen in GENERATIONS:
        env = MySinglesEnv(
            battle_format=f"gen{gen}randombattle",
            log_level=25,
            start_challenging=False,
            strict=False,
        )
        env = SingleAgentWrapper(env, RandomPlayer())
        env.env.start_challenging(NUM_CHALLENGES)
        single_agent_play_function(env, NUM_CHALLENGES)
        env.close()







async def main():
    # Jugadores que usan RandomBattle (equipos generados aleatoriamente)
    eval_env = MySinglesEnv(
        battle_format=f"gen9randombattle",
        log_level=25,
        start_challenging=True,
        strict=False,
    )

    dqn_player = SingleAgentWrapper(eval_env, RandomPlayer())

    model = DQN.load("dqn_pokemon_model")


    # Ejecutar una batalla
    print("##### Comenzando evaluación #####")
    results = await cross_evaluate([dqn_player, RandomPlayer()], n_challenges=10)
    print("Resultados:", results)



if __name__ == "__main__":
    ## DQN ###
    env = MySinglesEnv(
        battle_format=f"gen9randombattle",
        log_level=25,
        start_challenging=True,
        strict=False,
    )


    env = SingleAgentWrapper(env, RandomPlayer())

    model = DQN(
      "MlpPolicy",
      env,
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
    )
    model.learn(total_timesteps=100_000)
    print("##################### DQN ENTRENADO #####################")
    model.save("dqn_pokemon_model")

    # asyncio.run(main())

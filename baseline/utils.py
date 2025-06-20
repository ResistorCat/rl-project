import numpy as np
from poke_env import AccountConfiguration, cross_evaluate
from poke_env.player import SingleAgentWrapper

from baseline import DQNPlayer
from baseline import SimpleRandomPlayer


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
        env = SimpleRandomPlayer(
            battle_format=f"gen{gen}randombattle",
            log_level=25,
            start_challenging=False,
            strict=False,
        )
        env = SingleAgentWrapper(env, SimpleRandomPlayer())
        env.env.start_challenging(NUM_CHALLENGES)
        single_agent_play_function(env, NUM_CHALLENGES)
        env.close()


async def evaluate_player(num_challenges: int = 100):

    dqn_player = DQNPlayer("dqn_pokemon_model")

    results = await cross_evaluate([dqn_player, SimpleRandomPlayer()], n_challenges=num_challenges)
    print("Resultados:", results)


async def accept_challenges(opponents: str | list[str] | None = None, num_challenges: int = 1):
  """
  
  """
  # Configuracion de la cuenta del Jugador / Bot. Autenticar si el servidor lo pide.
  player_config = AccountConfiguration("DQNPlayer", None)

  dqn_player = DQNPlayer(
      account_configuration=player_config,
      model_path="dqn_pokemon_model",
      battle_format="gen9randombattle",
  )

  print("Esperando desafíos... (Ctrl+C para detener)")
  await dqn_player.accept_challenges(opponents, n_challenges=num_challenges)


def simple_embed_battle(battle):
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
import asyncio
from poke_env import AccountConfiguration, cross_evaluate
from poke_env.player.baselines import MaxBasePowerPlayer, SimpleHeuristicsPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from baseline.players import BaselinePlayer, SimpleRandomPlayer
from dqn.player import OurDQNPlayer

##########################################
COMPARE_WITH_RANDOM_PLAYER        = True
COMPARE_WITH_MAX_PLAYER           = False
COMPARE_WITH_HEURISTICS_PLAYER    = False
COMPARE_WITH_BASELINE_DQN_PLAYER  = True
COMPARE_WITH_OUR_FIRST_DQN_PLAYER = True
COMPARE_WITH_OUR_PPO_PLAYER       = False

NUM_CHALLENGES = 10
##########################################


async def main(num_challenges: int = 5):


    random_opponent = SimpleRandomPlayer(
        account_configuration=AccountConfiguration("Random Bot", None),
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_timer_on_battle_start=True,
        # log_level=logging.INFO,
    )

    max_opponent = MaxBasePowerPlayer(
        account_configuration=AccountConfiguration("Max Bot", None),
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_timer_on_battle_start=True,
        # log_level=logging.INFO,
    )

    simple_heuristics_opponent = SimpleHeuristicsPlayer(
        account_configuration=AccountConfiguration("Heuristics Bot", None),
        battle_format="gen9randombattle",
        server_configuration=LocalhostServerConfiguration,
        start_timer_on_battle_start=True,
        # log_level=logging.INFO,
    )

    dqn_player = BaselinePlayer(
      "models/baseline_dqn_pokemon_model",
      account_configuration=AccountConfiguration("BL DQN Bot", None)
    )

    our_dqn_player = OurDQNPlayer(
      "models/our_dqn_pokemon_model",
      account_configuration=AccountConfiguration("Our DQN Bot", None)
    )

    players = []
    if COMPARE_WITH_RANDOM_PLAYER:
      players.append(random_opponent)
    if COMPARE_WITH_MAX_PLAYER:
      players.append(max_opponent)
    if COMPARE_WITH_HEURISTICS_PLAYER:
      players.append(simple_heuristics_opponent)
    if COMPARE_WITH_BASELINE_DQN_PLAYER:
      players.append(dqn_player)
    if COMPARE_WITH_OUR_FIRST_DQN_PLAYER:
      players.append(our_dqn_player)

    # Ejecutar una batalla
    results = await cross_evaluate(players, n_challenges=num_challenges)
    print("Resultados:", results)

    # print(f"{dqn_player.times_random_choice} de {dqn_player.times_made_a_choice} acciones fueron al azar. {dqn_player.times_random_choice / dqn_player.times_made_a_choice:.2}")


if __name__ == "__main__":
    asyncio.run(main(num_challenges=NUM_CHALLENGES))

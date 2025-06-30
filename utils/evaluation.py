from poke_env import AccountConfiguration, cross_evaluate
from poke_env.player import SingleAgentWrapper

from baseline.players import BaselinePlayer, SimpleRandomPlayer


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


async def evaluate_player(player: BaselinePlayer, num_challenges: int = 100):
    results = await cross_evaluate([player, SimpleRandomPlayer()], n_challenges=num_challenges)
    print("Resultados:", results)


async def accept_challenges(player: BaselinePlayer, opponents: str | list[str] | None = None, num_challenges: int = 1):
  """
  En opponents puede listarse los nombres de usuario permitidos para desafiar al bot.
  """
  # Configuracion de la cuenta del Jugador / Bot. Autenticar si el servidor lo pide.

  print("Esperando desafíos... (Ctrl+C para detener)")
  await player.accept_challenges(opponents, n_challenges=num_challenges)

import numpy as np
from gymnasium.spaces import Box
from poke_env.environment import Battle
from poke_env.player import (
    SinglesEnv,
    SingleAgentWrapper,
    Player,
    DefaultBattleOrder,
    BattleOrder,
)
from utils.model import simple_embed_battle, simple_action_to_order
import torch
import random


class PokeEnvSinglesWrapper(SinglesEnv):
    """
    A wrapper for the PokeEnv Singles environment that provides a custom observation space
    """

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
        """
        Calculate the reward for the agent based on the battle state.
        """
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle):
        """
        Embed the battle state into a vector representation.
        """
        # -1 indicates that the move does not have a base power or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        # print(battle.available_moves)
        for i, move in enumerate(battle.available_moves):
            # print(move)
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = (
                    battle.opponent_active_pokemon.damage_multiplier(move)
                )

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

    def get_wrapped_env(self, opponent):
        """
        Get the wrapped environment with a specific opponent.

        Args:
            opponent: The opponent to battle against

        Returns:
            SingleAgentWrapper environment
        """

        return SingleAgentWrapper(self, opponent)


class DQNPlayer(Player):
    def __init__(self, model):
        super().__init__(log_level=30)
        self.model = model

        self.observations_dim = 10
        self.actions_dim = 4 + 6  # 4 Moves and 6 Switches
        self.times_random_choice = 0
        self.times_made_a_choice = 0

    # Mismo método que la clase
    def embed_battle(self, battle):
        return simple_embed_battle(battle)

    def choose_move(self, battle):
        self.times_made_a_choice += 1

        # Protege contra el estado inicial del combate
        if (
            battle.active_pokemon is None
            or len(battle.available_moves) == 0
            and len(battle.available_switches) == 0
        ):
            self.times_random_choice += 1
            # print(">>>> Estado inicial incompleto, acción aleatoria")
            return self.choose_random_move(battle)
        obs = self.embed_battle(battle).reshape(1, -1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        q_values = self.model.q_net(obs_tensor).detach().numpy()[0]

        sorted_actions = np.argsort(q_values)[::-1]

        # Buscar la mejor acción válida
        for action in sorted_actions:
            try:
                order = self.action_to_order(action, battle)
                if not isinstance(order, DefaultBattleOrder):
                    # print(f">>>> Acción válida seleccionada: {action}")
                    return order
            except AssertionError as e:
                # print(f">>> Acción no me sirve: {e}")
                continue  # Acción inválida, probar la siguiente

        # action, _ = self.model.predict(obs, deterministic=True)

        # order = simple_action_to_order(action, battle)

        self.times_random_choice += 1
        # print(">>>> Elige acción por defecto")
        return self.choose_random_move(battle)

    def choose_random_move(self, battle: Battle) -> BattleOrder:
        available_orders = [BattleOrder(move) for move in battle.available_moves]
        available_orders.extend(
            [BattleOrder(switch) for switch in battle.available_switches]
        )

        if available_orders:
            return random.choice(available_orders)
        else:
            return Player.choose_default_move()

    @staticmethod
    def create_order(
        order,
        mega: bool = False,
        z_move: bool = False,
        dynamax: bool = False,
        terastallize: bool = False,
        move_target: int = None,
    ) -> BattleOrder:
        """Formats an move order corresponding to the provided pokemon or move.

        :param order: Move to make or Pokemon to switch to.
        :type order: Move or Pokemon
        :param mega: Whether to mega evolve the pokemon, if a move is chosen.
        :type mega: bool
        :param z_move: Whether to make a zmove, if a move is chosen.
        :type z_move: bool
        :param dynamax: Whether to dynamax, if a move is chosen.
        :type dynamax: bool
        :param terastallize: Whether to terastallize, if a move is chosen.
        :type terastallize: bool
        :param move_target: Target Pokemon slot of a given move
        :type move_target: int
        :return: Formatted move order
        :rtype: str
        """
        return BattleOrder(
            order,
            mega=mega,
            move_target=move_target,
            z_move=z_move,
            dynamax=dynamax,
            terastallize=terastallize,
        )

    def action_to_order(self, action, battle, fake=False, strict=True):
        return simple_action_to_order(action, battle, fake, strict)

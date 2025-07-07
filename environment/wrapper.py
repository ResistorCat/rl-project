import numpy as np
from gymnasium.spaces import Box
from poke_env.player import SinglesEnv, SingleAgentWrapper


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

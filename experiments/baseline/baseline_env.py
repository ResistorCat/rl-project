import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.player import SinglesEnv, BattleOrder
from poke_env.environment import Battle

from utils_model import simple_embed_battle, simple_action_to_order, simple_order_to_action

class BaselineSinglesEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        low =  [-1, -1, -1, -1, 0, 0, 0, 0, 0, 0]
        high = [ 3,  3,  3,  3, 4, 4, 4, 4, 1, 1]
        self.observation_spaces = {
            agent: Box(
                np.array(low, dtype=np.float32),
                np.array(high, dtype=np.float32),
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }

        act_size = 6 + 4

        self.action_spaces = {
            agent: Discrete(act_size) for agent in self.possible_agents
        }

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def embed_battle(self, battle):
        return simple_embed_battle(battle)
    
    @staticmethod
    def action_to_order(
        action: np.int64, battle: Battle, fake: bool = False, strict: bool = True
    ) -> BattleOrder:
        return simple_action_to_order(action, battle, fake, strict)
    
    @staticmethod
    def order_to_action(
        order: BattleOrder, battle: Battle, fake: bool = False, strict: bool = True
    ) -> np.int64:
        return simple_order_to_action(order, battle, fake, strict)
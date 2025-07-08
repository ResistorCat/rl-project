import numpy as np
from gymnasium.spaces import Box, Discrete
from poke_env.player import BattleOrder, DefaultBattleOrder, ForfeitBattleOrder
from poke_env.environment import SinglesEnv
from collections import defaultdict
from poke_env.battle import Battle

from dqn.observation_space import build_observation_bounds
from dqn.embedding import enhanced_embed_battle
from utils.model import simple_order_to_action, enhanced_action_to_order

class OurDQNSinglesEnv(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        low, high = build_observation_bounds(
            include_active_pokemon=True,
            include_status=False,
            include_types=False,
            include_boosts=False,
            include_fainted=False,
            status_one_hot=True
        )
        self.observation_spaces = {
            agent: Box(
                low,
                high,
                dtype=np.float32,
            )
            for agent in self.possible_agents
        }
        print("Espacio de observaciones:", len(low))

        act_size = 6 + 4

        self.action_spaces = {
            agent: Discrete(act_size) for agent in self.possible_agents
        }

        self._last_invalid_action = defaultdict(bool)
        self.action_to_order = self._wrapped_action_to_order

    def calc_reward(self, battle) -> float:
        base_reward = self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            victory_value=60.0
        )

        penalty = 0.0
        if self._last_invalid_action.get(battle, False):
            penalty = -4.0
            print(f"[Reward] Penalización aplicada por acción inválida en turno {battle.turn}")
            self._last_invalid_action[battle] = False  # Reiniciar para siguiente turno

        return base_reward + penalty

    def embed_battle(self, battle):
        return enhanced_embed_battle(
            battle,
            include_active_pokemon=True,
            include_status=False,
            include_types=False,
            include_boosts=False,
            include_fainted=False,
            status_one_hot=True
        )
    
    def _wrapped_action_to_order(self, action: int, battle, fake: bool = False, strict: bool = True) -> BattleOrder:
        try:
            return enhanced_action_to_order(action, battle, fake, strict)
        except AssertionError as e:
            print(f"[wrapped_action_to_order] Acción inválida: {e}")
            self._last_invalid_action[battle] = True
            return safe_default_order(battle)


    @staticmethod
    def order_to_action(
        order: BattleOrder, battle: Battle, fake: bool = False, strict: bool = True
    ) -> np.int64:
        return simple_order_to_action(order, battle, fake, strict)


def safe_default_order(battle: Battle) -> BattleOrder:
    if (
        len(battle.available_switches) == 0 and len(battle.available_moves) == 0
    ):
        return ForfeitBattleOrder()
    return DefaultBattleOrder()
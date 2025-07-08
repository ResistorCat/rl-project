"""
Shared types and constants for the Pokémon RL project.
"""

from enum import Enum


class RLModel(str, Enum):
    PPO = "ppo"
    DQN = "dqn"


class RLPlayer(str, Enum):
    DQN_RANDOM = "dqn_random"
    DQN_MAX = "dqn_max"
    RANDOM = "random"
    MAX = "max"

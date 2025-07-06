"""
Shared types and constants for the Pokémon RL project.
"""

from enum import Enum


class RLModel(str, Enum):
    PPO = "ppo"
    DQN = "dqn"
    RANDOM = "random"

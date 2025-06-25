import random
from stable_baselines3 import DQN
from poke_env.player import Player
from poke_env.environment import Battle
from poke_env.player import BattleOrder, DefaultBattleOrder
from poke_env import AccountConfiguration
import torch
import numpy as np

from utils_model import simple_embed_battle, simple_action_to_order

"""
 Acción (0, ..., 5)  significa Switch(index)
 Acción (6, 7, 8, 9) significa Move(index - 6)
"""

class DQNPlayer(Player):
    def __init__(self, model_path, account_configuration, battle_format="gen9randombattle"):
        super().__init__(battle_format=battle_format, account_configuration=account_configuration)
        self.model = DQN.load(model_path)

        self.observations_dim = 10
        self.actions_dim = 4 + 6 # 4 Moves and 6 Switches
        self.times_random_choice = 0
        self.times_made_a_choice = 0
    
    # Mismo método que la clase 
    def embed_battle(self, battle):
        return simple_embed_battle(battle)

    def choose_move(self, battle):
        self.times_made_a_choice += 1

        # Protege contra el estado inicial del combate
        if (
          battle.active_pokemon is None or
          len(battle.available_moves) == 0 and len(battle.available_switches) == 0
        ):
          self.times_random_choice += 1
          print(">>>> Estado inicial incompleto, acción aleatoria")
          return self.choose_random_move(battle)
        obs = self.embed_battle(battle).reshape(1, -1)
        obs_tensor = torch.tensor(obs, dtype=torch.float32)

        q_values = self.model.q_net(obs_tensor).detach().numpy()[0]

        sorted_actions = np.argsort(q_values)[::-1]

        # Buscar la mejor acción válida
        for action in sorted_actions:
          try:
            order = simple_action_to_order(action, battle)
            if not isinstance(order, DefaultBattleOrder):
              print(f">>>> Acción válida seleccionada: {action}")
              return order
          except AssertionError as e:
            print(f">>> Acción no me sirve: {e}")
            continue  # Acción inválida, probar la siguiente

        # action, _ = self.model.predict(obs, deterministic=True)

        # order = simple_action_to_order(action, battle)

        self.times_random_choice += 1
        print(">>>> Elige acción por defecto")  
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


class SimpleRandomPlayer(Player):
  """
  Versión simplificada de RandomPlayer. Quita Teracristalizar en generación 9.
  """

  def choose_move(self, battle: Battle) -> BattleOrder:        
    for _ in range(3):
      order = self.generate_move(battle)
      try:
        return order
      except Exception as e:
        if "Invalid choice" in str(e):
          self.logger.warning("Invalid choice, retrying..")
          continue
      return Player.choose_default_move()
  
  def generate_move(self, battle: Battle) -> BattleOrder:
    available_orders = [BattleOrder(move) for move in battle.available_moves]
    available_orders.extend(
      [BattleOrder(switch) for switch in battle.available_switches]
    )

    if available_orders:
      return available_orders[int(random.random() * len(available_orders))]
    else:
      return Player.choose_default_move()
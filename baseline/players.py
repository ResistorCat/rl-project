import random
from stable_baselines3 import DQN
from poke_env.player import Player
from poke_env.environment import Battle
from poke_env.player import BattleOrder

from baseline import simple_embed_battle


class DQNPlayer(Player):
    def __init__(self, model_path, account_configuration, battle_format="gen9randombattle"):
        super().__init__(battle_format=battle_format, account_configuration=account_configuration)
        self.model = DQN.load(model_path)

        self.obs_dim = 10
        self.times_random_choice = 0
        self.times_made_a_choice = 0
    
    # Mismo método que la clase 
    def embed_battle(self, battle):
        return simple_embed_battle(battle)

    def choose_move(self, battle):
        self.times_made_a_choice += 1
        obs = self.embed_battle(battle)
        action, _ = self.model.predict(obs, deterministic=True)
        
        # Si la acción es un índice de move
        if action < len(battle.available_moves):
            move = battle.available_moves[action]
            return self.create_order(move)
        # Si la acción corresponde a un switch
        switch_index = action - len(battle.available_moves)
        if switch_index < len(battle.available_switches):
            switch = battle.available_switches[switch_index]
            return self.create_order(switch)
        
        # Fallback: movimiento o switch aleatorio válido
        # print(">>>>>>> Selecciona una acción al azar")
        self.times_random_choice += 1
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
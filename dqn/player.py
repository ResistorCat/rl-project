from dqn.embedding import enhanced_embed_battle
from dqn.observation_space import get_embedding_dimension
from baseline.players import BaselinePlayer
from utils.model import get_valid_action_mask, enhanced_action_to_order
from poke_env.environment import Battle
import torch
import numpy as np
from poke_env.player import DefaultBattleOrder


class OurDQNPlayer(BaselinePlayer):
  def __init__(self, model_path, account_configuration, battle_format="gen9randombattle"):
    super().__init__(model_path, account_configuration, battle_format)

    self.observations_dim = get_embedding_dimension(status_one_hot=True)
  

  def embed_battle(self, battle):
    return enhanced_embed_battle(battle, status_one_hot=True)
  
  def choose_move(self, battle: Battle):
        self.times_made_a_choice += 1

        # Estado inicial o ilegible
        if (
          battle.active_pokemon is None or
          len(battle.available_moves) == 0 and len(battle.available_switches) == 0
        ):
          self.times_random_choice += 1
          print(">>>> Estado inicial incompleto, acción aleatoria")
          return self.choose_random_move(battle)
        
        try:
            # Obtener embedding y pasar por red
            obs = self.embed_battle(battle).reshape(1, -1)
            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            q_values = self.model.q_net(obs_tensor).detach().numpy()[0]

            # Obtener máscara de validez
            moves = [move for move in battle.active_pokemon.moves.values()]
            switches = [p for p in battle.team.values()]
            valid_mask = get_valid_action_mask(battle, moves, switches)
            if valid_mask.sum() == 0:
                raise RuntimeError("No hay acciones válidas según la máscara")

            # Aplicar máscara a Q-values
            masked_q_values = np.where(valid_mask, q_values, -np.inf)


            # Seleccionar la mejor acción válida
            action = int(np.argmax(masked_q_values))
            assert valid_mask[action], f"El agente eligió una acción inválida: {action}"
            
            order  = self.action_to_order(action, battle)

            if isinstance(order, DefaultBattleOrder):
                raise RuntimeError("Acción elegida resultó ser DefaultBattleOrder")

            print(f">>>> Acción válida seleccionada: {action}")
            return order

        except Exception as e:
            self.times_random_choice += 1
            print(f">>>> Error al elegir acción: {e}")
            return self.choose_random_move(battle)
    
  def action_to_order(self, action, battle, fake=False, strict=True):
    return enhanced_action_to_order(action, battle, fake, strict)
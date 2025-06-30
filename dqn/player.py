from dqn.embedding import enhanced_embed_battle
from dqn.observation_space import get_embedding_dimension
from baseline.players import BaselinePlayer

class OurDQNPlayer(BaselinePlayer):
  def __init__(self, model_path, account_configuration, battle_format="gen9randombattle"):
    super().__init__(model_path, account_configuration, battle_format)

    self.observations_dim = get_embedding_dimension(status_one_hot=True)

  

  def embed_battle(self, battle):
    return enhanced_embed_battle(battle)
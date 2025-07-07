import numpy as np
from poke_env.player import BattleOrder, DefaultBattleOrder, ForfeitBattleOrder, Player
from poke_env.environment import Pokemon, Battle, Move


def simple_embed_battle(battle):
        # -1 indicates that the move does not have a base power
        # or is not available
        moves_base_power = -np.ones(4)
        moves_dmg_multiplier = np.ones(4)
        # print(battle.available_moves)
        for i, move in enumerate(battle.available_moves):
            # print(move)
            moves_base_power[i] = (
                move.base_power / 100
            )  # Simple rescaling to facilitate learning
            if move.type:
                moves_dmg_multiplier[i] = battle.opponent_active_pokemon.damage_multiplier(move)

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


# Método estático de SinglesEnv
def simple_action_to_order(
        action: np.int64, battle: Battle, fake: bool = False, strict: bool = True
    ) -> BattleOrder:
        """
        Returns the BattleOrder relative to the given action.

        The action mapping is as follows:
        action = -2: default
        action = -1: forfeit
        0 <= action <= 5: switch
        6 <= action <= 9: move

        >>> Quitamos mega, movimiento Z, dynamax y teracristal <<<
        10 <= action <= 13: move and mega evolve
        14 <= action <= 17: move and z-move
        18 <= action <= 21: move and dynamax
        22 <= action <= 25: move and terastallize
        """
        print(f">>>>>> Acción a convertir es {action} <<<<<<")
        try:
            if action == -2:
                return DefaultBattleOrder()
            elif action == -1:
                return ForfeitBattleOrder()
            elif action < 6:
                print(f">>> Acciones disponiles para switch {list(battle.team.values())}")
                order = Player.create_order(list(battle.team.values())[action])
                if not fake:
                    assert not battle.trapped, "invalid action"
                    assert isinstance(order.order, Pokemon)
                    assert order.order.base_species in [
                        p.base_species for p in battle.available_switches
                    ], "invalid action"
            else:
                if not fake:
                    assert not battle.force_switch, "invalid action: forzado a cambiar de pokémon"
                    assert battle.active_pokemon is not None, "invalid action: no hay pokémon activo en combate"
                elif battle.active_pokemon is None:
                    return DefaultBattleOrder()
                mvs = (
                    battle.available_moves
                    if len(battle.available_moves) == 1
                    and battle.available_moves[0].id in ["struggle", "recharge"]
                    else list(battle.active_pokemon.moves.values())
                )
                if not fake:
                    assert (action - 6) % 4 in range(len(mvs)), f"invalid action: la acción elegida ({action} -> {(action - 6) % 4}) no esta en el rango de moves ({mvs}) cuando {battle.available_moves}"
                elif (action - 6) % 4 not in range(len(mvs)):
                    return DefaultBattleOrder()
                order = Player.create_order(
                    mvs[(action - 6) % 4],
                    mega=False,
                    z_move=False,
                    dynamax=False,
                    terastallize=False,
                )
                print(f"Move creado es {order}")
                if not fake:
                    assert isinstance(order.order, Move)
                    assert order.order.id in [
                        m.id for m in battle.available_moves
                    ], "invalid action: la order generada no se encuentra en los moves disponibles"
            print(f">>>>>> Acción convertida a {order} <<<<<<")
            return order
        except AssertionError as e:
            if not strict and str(e) == "invalid action":
                return DefaultBattleOrder()
            else:
                raise e

def is_action_valid(action: int, battle: Battle) -> bool:
  if 0 <= action <= 5:
    return action < len(battle.available_switches)
  elif 6 <= action <= 9:
    move_idx = action - 6
    return move_idx < len(battle.available_moves)
  else:
    return False  # Fuera de rango

# Usar para filtrar en evaluación y penalizar al agente en entrenamiento
def get_valid_actions(battle: Battle) -> list[int]:
    valid_actions = []

    # Switches: acciones 0 a 5
    for i in range(6):
      if i < len(battle.available_switches) and not battle.trapped:
        valid_actions.append(i)

    # Movimientos: acciones 6 a 9 (4 movimientos)
    if not battle.force_switch and battle.active_pokemon is not None:
      for i in range(4):
        if i < len(battle.available_moves):
          valid_actions.append(i + 6)

    return valid_actions

# Método estático de SinglesEnv
def simple_order_to_action(
        order: BattleOrder, battle: Battle, fake: bool = False, strict: bool = True
    ) -> np.int64:
        """
        Returns the action relative to the given BattleOrder.

        >>> Quitamos mega, movimiento Z, dynamax y teracristal <<<
        """
        try:
            if isinstance(order, DefaultBattleOrder):
                action = -2
            elif isinstance(order, ForfeitBattleOrder):
                action = -1
            elif order.order is None:
                raise ValueError()
            elif isinstance(order.order, Pokemon): # SWITCH
                if not fake:
                    assert not battle.trapped, "invalid order"
                    assert order.order.base_species in [
                        p.base_species for p in battle.available_switches
                    ], "invalid order"
                action = [p.base_species for p in battle.team.values()].index(
                    order.order.base_species
                )
            else: # MOVE
                if not fake:
                    assert not battle.force_switch, "invalid order"
                    assert battle.active_pokemon is not None, "invalid order"
                elif battle.active_pokemon is None:
                    return np.int64(-2)
                mvs = (
                    battle.available_moves
                    if len(battle.available_moves) == 1
                    and battle.available_moves[0].id in ["struggle", "recharge"]
                    else list(battle.active_pokemon.moves.values())
                )
                if not fake:
                    assert order.order.id in [m.id for m in mvs], "invalid order"
                action = [m.id for m in mvs].index(order.order.id)
                action = 6 + action
                if not fake:
                    assert order.order.id in [
                        m.id for m in battle.available_moves
                    ], "invalid order"
            return np.int64(action)
        except AssertionError as e:
            if not strict and str(e) == "invalid order":
                return np.int64(-2)
            else:
                raise e
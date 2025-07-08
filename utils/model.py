import numpy as np
from poke_env.player import DefaultBattleOrder, ForfeitBattleOrder, Player, SingleBattleOrder, BattleOrder
from poke_env.battle import Pokemon, Battle, Move
from tabulate import tabulate

def simple_embed_battle(battle: Battle):
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
        try:
            # print("moves disponibles ->", battle.available_moves)
            # print("switches disponibles ->", battle.available_switches)
            if action == -2:
                return DefaultBattleOrder()
            elif action == -1:
                return ForfeitBattleOrder()
            elif action < 6:
                if not fake:
                    assert not battle.trapped, "invalid action"

                # Forma segura de elegir el cambio válido
                switch_options = battle.available_switches

                if action >= len(switch_options):
                    # print(f"[Switch inválido: acción {action} no disponible]")
                    return DefaultBattleOrder()
                if "inior" in switch_options[action].species:
                    # print("orden:", BattleOrder(switch_options[action]))
                    # print("pokemon:", switch_options[action])
                    pass
                order = SingleBattleOrder(switch_options[action])

                if not fake:
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
                # print(f"Move creado es {order}")
                if not fake:
                    assert isinstance(order.order, Move)
                    assert order.order.id in [
                        m.id for m in battle.available_moves
                    ], "invalid action: la order generada no se encuentra en los moves disponibles"
            # print(f">>>>>> Acción convertida a {order} <<<<<<")
            return order
        except AssertionError as e:
            if not strict and "invalid action" in str(e):
                # print(e)
                return DefaultBattleOrder()
            else:
                raise e


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


def enhanced_action_to_order(
    action: np.int64, battle: Battle, fake: bool = False, strict: bool = True
  ) -> BattleOrder:
    # switches = battle.available_switches
    # moves    = battle.available_moves

    assert battle.active_pokemon is not None, f"invalid action: no active pokemon in battle"

    if battle.reviving:
        # Revivir al primero
        candidates = [p for p in battle.team.values() if p.fainted]
        return SingleBattleOrder(candidates[0])

    moves    = [move for move in battle.active_pokemon.moves.values()]
    switches = [p for p in battle.team.values()]
    valid_mask = get_valid_action_mask(battle, moves, switches)

    print_tabulate(battle.active_pokemon.moves, battle.team)
    if battle.active_pokemon.must_recharge:
        assert len(battle.available_moves) > 0, "invalid action: no available move"
        return SingleBattleOrder(battle.available_moves[0])
    
    if len(battle.available_moves) > 0 and battle.available_moves[0].id == "struggle":
        return SingleBattleOrder(battle.available_moves[0])

    if action == -2:
        return DefaultBattleOrder()
    elif action == -1:
        return ForfeitBattleOrder()

    if 0 <= action < len(valid_mask) and valid_mask[action]:
        if action < 6:
            # print(f"Switch {action} - De {len(switches)} switches disponibles")
            # for (i, p) in enumerate(switches):
                # print(f"{i}: {p.species}, fainted={p.fainted}, active={p.active}")
            assert not battle.trapped, "invalid action: can't switch - trapped"
            target = switches[action]
            assert not target.fainted, "invalid action: can't switch to fainted Pokémon"
            assert not target.active, "invalid action: can't switch to active Pokémon"
            if "minior" in target.species:
                target = [poke for poke in battle.available_switches if "minior" in poke.species][0]
            return SingleBattleOrder(target)
        else:
            # Aqui, por si necesitamos agregar validaciones de move
            return SingleBattleOrder(moves[action - 6])
    else:
        # FALLBACK
        if action < 6:
            switches = battle.available_switches
            assert 1 == 2, f"invalid action: switch {action} not available -> {switches}"
        else:
            assert not battle.force_switch, "invalid action: forced to switch"
            moves = battle.available_moves
            assert 1 == 2, f"invalid action: move {action} not available -> {moves}"
            assert battle.active_pokemon is not None, "invalid action: no active (my) pokemon in battle "
        return (
            SingleBattleOrder(switches[0]) if action < 6 and switches else
            SingleBattleOrder(moves[0]) if moves else
            DefaultBattleOrder()
        )


def enhanced_order_to_action(
    action: np.int64, battle: Battle, fake: bool = False, strict: bool = True
  ) -> BattleOrder:
    pass


def get_valid_action_mask(battle: Battle, moves: list[Move], switches: list[Pokemon]) -> np.ndarray:

    mask = np.zeros(10, dtype=bool)
    valid_available_ids = {m.id for m in battle.available_moves}
    valid_available_switch_ids = {p.species for p in battle.available_switches}

    last_request = battle.last_request
    # print(f"\n{last_request}\n")
    active = last_request.get("active", None)
    disabled = [True, True, True, True]
    disabled = dict()
    if active is not None:
        moves_from_request = active[0]["moves"]
        print(moves_from_request)
        for mv in moves_from_request:
            disabled[mv['id']] = mv.get('disabled', True)
        for mv in moves:
            if mv.id not in disabled.keys():
                disabled[mv.id] = True
    

    for i, poke in enumerate(switches[:6]):
        if len(battle.available_switches) == 0:
            mask[i] = 0
        elif poke.species in valid_available_switch_ids:
            mask[i] = not battle.trapped 
        elif not poke.fainted and not poke.active:
            mask[i] = not battle.trapped
    
    for i, move in enumerate(moves[:4]):
        if move.id in valid_available_ids:
            mask[6 + i] = not battle.force_switch and not disabled[move.id]  # No se puede atacar si debe cambiar
        else:
            mask[6 + i] = 0

    
    return mask


def print_tabulate(moves: dict = None, switches: dict = None):
    # if moves is not None:
    #     rows_moves = [(key, str(value)) for key, value in moves.items()]
    #     print(tabulate(rows_moves, headers=["Orden", "Movimiento"]))
    if switches is not None:
        rows_switches = [(key, str(value), value.species) for key, value in switches.items()]
        print(tabulate(rows_switches, headers=["Orden", "Cambio"]))
        print()

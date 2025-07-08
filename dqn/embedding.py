import numpy as np
from poke_env.battle import Battle, Pokemon

TYPE_LIST = [
    "normal", "fire", "water", "electric", "grass", "ice", "fighting",
    "poison", "ground", "flying", "psychic", "bug", "rock",
    "ghost", "dragon", "dark", "steel", "fairy", "three_question_marks"
]

STATUS_TO_INT = {"brn": 1, "par": 2, "psn": 3, "slp": 4, "frz": 5, "tox": 6}

def type_to_onehot(type_):
    vec = np.zeros(len(TYPE_LIST), dtype=np.float32)
    if type_:
        vec[TYPE_LIST.index(type_.name.lower())] = 1.0
    return vec


def get_move_features(battle: Battle):
    moves = []
    for i in range(4):  # hasta 4 movimientos
        if i < len(battle.active_pokemon.moves):
            move = list(battle.active_pokemon.moves.values())[i]
            is_disabled = True
            for available_move in battle.available_moves:
                if available_move.id == move.id:
                    is_disabled = False
            moves.extend([
                move.base_power / 100 if move.base_power else 0,
                move.accuracy / 100 if move.accuracy else 1,
                # int(move.priority),     # Para futuras versiones, puede ser util agregar prioridad
                battle.opponent_active_pokemon.damage_multiplier(move),
                int(move.current_pp > 0),
                int(not is_disabled)
            ])
        else:
            moves.extend([
                -1,  # Movimiento no disponible
                -1,  # Movimiento no disponible
                # -1,
                -1,
                 0,
                 0,
            ])
    return np.array(moves, dtype=np.float32)


def get_switches_features(battle: Battle):
    switches = []
    for i in range(6):  # hasta 6 Pokémon
        team = list(battle.team.values())
        if i < len(team):
            poke = team[i]
            switches.extend([
                poke.current_hp_fraction,
                int(not poke.active),
                int(not poke.fainted), # Es 1, si no está debilitado
                # Para futuras versiones, que incluya one-shot de status de los 6 Pokémon
                # TYPE_LIST.get(poke.type_1, -1),  # Para futuras versiones, puede incluir aqui el one-shot de los 6 Pokémon
                # TYPE_LIST.get(poke.type_2, -1) if poke.type_2 else -1,
            ])
        else:
            # padding para slot vacío: equipo con menos de 6 Pokémon
            switches.extend([
                0, # HP_fraction == 0
                0, # is active   == False
                0, # not_FAINTED == False
                # -1,
                # -1
            ])

    return np.array(switches, dtype=np.float32)


def get_active_pokemon_features(battle: Battle):
    def extract_stats(pokemon: Pokemon):
        if pokemon is None:
            return [0.0] * 6  # si no hay pokémon, se asume "vacío"
        base_stats = pokemon.base_stats
        return [
            base_stats["atk"] / 255.0,
            base_stats["spa"] / 255.0,
            base_stats["def"] / 255.0,
            base_stats["spd"] / 255.0,
            base_stats["spe"] / 255.0,
        ]

    # Pokémon del jugador
    my_poke = battle.active_pokemon
    my_stats = extract_stats(my_poke)
    my_hp = my_poke.current_hp_fraction if my_poke else 0.0
    my_trapped = float(battle.trapped)  # bool → float
    my_force_switch = float(battle.force_switch)

    # Pokémon del oponente
    op_poke = battle.opponent_active_pokemon
    op_stats = extract_stats(op_poke)
    op_hp = op_poke.current_hp_fraction if op_poke else 0.0

    return np.array(
        [my_hp] + my_stats + [my_trapped, my_force_switch] + [op_hp] + op_stats,
        dtype=np.float32
    )


def get_status_features(battle: Battle, one_hot: bool = False):
    def encode(status):
        if one_hot:
            vec = np.zeros(len(STATUS_TO_INT) + 1, dtype=np.float32)
            index = STATUS_TO_INT.get(status, 0)
            vec[index] = 1.0
            return vec
        else:
            return np.array([STATUS_TO_INT.get(status, 0)], dtype=np.float32)

    my_status = encode(battle.active_pokemon.status)
    op_status = encode(battle.opponent_active_pokemon.status)
    return np.concatenate([my_status, op_status])


def get_type_features(battle: Battle):
    my_type1 = type_to_onehot(battle.active_pokemon.type_1)
    my_type2 = type_to_onehot(battle.active_pokemon.type_2)
    op_type1 = type_to_onehot(battle.opponent_active_pokemon.type_1)
    op_type2 = type_to_onehot(battle.opponent_active_pokemon.type_2)
    return np.concatenate([my_type1 + my_type2, op_type1 + op_type2])


def get_boost_features(battle: Battle):
    boosts = battle.active_pokemon.boosts
    return np.array([
        boosts["atk"], boosts["def"], boosts["spa"],
        boosts["spd"], boosts["spe"]
    ], dtype=np.float32) / 6.0  # Normalizado a [-1, 1]


def get_fainted_features(battle: Battle):
    fainted_my_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6.0
    fainted_op_team = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6.0
    return np.array([fainted_my_team, fainted_op_team], dtype=np.float32)


def enhanced_embed_battle(
    battle: Battle,
    include_active_pokemon=True,
    include_status=True,
    include_types=True,
    include_boosts=True,
    include_fainted=True,
    status_one_hot=True
) -> np.ndarray:
    features = []

    # Siempre se incluyen los movimientos y switches
    features.append(get_move_features(battle))
    features.append(get_switches_features(battle))

    if include_active_pokemon:
        features.append(get_active_pokemon_features(battle))
    if include_status:
        features.append(get_status_features(battle, one_hot=status_one_hot))
    if include_types:
        features.append(get_type_features(battle))
    if include_boosts:
        features.append(get_boost_features(battle))
    if include_fainted:
        features.append(get_fainted_features(battle))

    return np.concatenate(features).astype(np.float32)
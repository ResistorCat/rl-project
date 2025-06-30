import numpy as np
from poke_env.environment import Battle, Pokemon

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
    base_powers = -np.ones(4, dtype=np.float32)
    dmg_multipliers = np.ones(4, dtype=np.float32)
    for i, move in enumerate(battle.available_moves):
        base_powers[i] = move.base_power / 100 if move.base_power else 0.0
        if move.type:
            dmg_multipliers[i] = battle.opponent_active_pokemon.damage_multiplier(move)
    return np.concatenate([base_powers, dmg_multipliers])


def get_hp_features(battle: Battle):
    # TODO: Agregar otras estadísticas normalizadas (Cómo normalizar?)
    my_hp = battle.active_pokemon.current_hp_fraction
    op_hp = battle.opponent_active_pokemon.current_hp_fraction
    return np.array([my_hp, op_hp], dtype=np.float32)


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
    include_hp=True,
    include_status=True,
    include_types=True,
    include_boosts=True,
    include_fainted=True,
    status_one_hot=False
) -> np.ndarray:
    features = []

    # Siempre se incluyen los movimientos
    features.append(get_move_features(battle))

    if include_hp:
        features.append(get_hp_features(battle))
    if include_status:
        features.append(get_status_features(battle, one_hot=status_one_hot))
    if include_types:
        features.append(get_type_features(battle))
    if include_boosts:
        features.append(get_boost_features(battle))
    if include_fainted:
        features.append(get_fainted_features(battle))

    return np.concatenate(features).astype(np.float32)
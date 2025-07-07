import numpy as np

def get_bounds_moves():
    move_low  = np.array([-1.0,  0.0, -1.0, 0.0, 0.0])
    move_high = np.array([ 4.0,  1.0,  4.0, 1.0, 1.0])
    # base_power, accuracy, damage_multiplier, current_pp > 0, disabled
    # Que pasa si el movimiento no tiene PP?

    moves_low = np.tile(move_low, 4)
    moves_highs = np.tile(move_high, 4)

    return (
        moves_low,
        moves_highs
        )

def get_bounds_switches():
    return (
        np.zeros(6 * 3, dtype=np.float32),
         np.ones(6 * 3, dtype=np.float32)
        )

def get_bounds_active_pokemon():
    low = np.array(
        [0.0] + [0.0]*5 + [0.0, 0.0] + [0.0] + [0.0]*5,
        dtype=np.float32
    )
    high = np.array(
        [1.0] + [1.0]*5 + [1.0, 1.0] + [1.0] + [1.0]*5,
        dtype=np.float32
    )
    return low, high

def get_bounds_status(one_hot: bool = True):
    if one_hot:
        return (
            np.zeros(14, dtype=np.float32),
            np.ones(14, dtype=np.float32)
        )
    else:
        return (
            np.zeros(2, dtype=np.float32),
            np.full(2, 6.0, dtype=np.float32)
        )

def get_bounds_types():
    # Categórico
    return (
        np.zeros(38, dtype=np.float32),
        np.ones(38, dtype=np.float32)
    )

def get_bounds_boosts():
    # Boots de estadísticas (normalizado por 6)
    return (
        np.full(5, -1.0, dtype=np.float32),
        np.full(5, 1.0, dtype=np.float32)
    )

def get_bounds_fainted():
    # Proporción de Pokémon debilitados en un equipo (normalizado por 6)
    return (
        np.zeros(2, dtype=np.float32),
        np.ones(2, dtype=np.float32)
    )

def build_observation_bounds(
    include_active_pokemon=True,
    include_status=True,
    include_types=True,
    include_boosts=True,
    include_fainted=True,
    status_one_hot=False
) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []

    # Moves (always included)
    low, high = get_bounds_moves()
    lows.append(low)
    highs.append(high)

    # Switches (always included)
    low, high = get_bounds_switches()
    lows.append(low)
    highs.append(high)


    if include_active_pokemon:
        low, high = get_bounds_active_pokemon()
        lows.append(low)
        highs.append(high)

    if include_status:
        low, high = get_bounds_status(status_one_hot)
        lows.append(low)
        highs.append(high)

    if include_types:
        low, high = get_bounds_types()
        lows.append(low)
        highs.append(high)

    if include_boosts:
        low, high = get_bounds_boosts()
        lows.append(low)
        highs.append(high)

    if include_fainted:
        low, high = get_bounds_fainted()
        lows.append(low)
        highs.append(high)

    return np.concatenate(lows).astype(np.float32), np.concatenate(highs).astype(np.float32)


def get_embedding_dimension(
    include_active_pokemon=True,
    include_status=True,
    include_types=True,
    include_boosts=True,
    include_fainted=True,
    status_one_hot=False
) -> int:
    low, _ = build_observation_bounds(
        include_active_pokemon=include_active_pokemon,
        include_status=include_status,
        include_types=include_types,
        include_boosts=include_boosts,
        include_fainted=include_fainted,
        status_one_hot=status_one_hot,
    )
    return len(low)


if __name__ == "__main__":
    print(get_embedding_dimension(status_one_hot=True))
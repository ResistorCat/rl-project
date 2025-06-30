import numpy as np

def get_bounds_moves():
    return (
        np.full(4,-1.0),  # low  base power
        np.full(4, 2.0),  # high base power
        np.full(4, 0.0),  # low  damage multiplier
        np.full(4, 4.0)   # high damage multiplier
        )

def get_bounds_hp():
    return (
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0])
        )

def get_bounds_status(one_hot: bool = False):
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
    include_hp=True,
    include_status=True,
    include_types=True,
    include_boosts=True,
    include_fainted=True,
    status_one_hot=False
) -> tuple[np.ndarray, np.ndarray]:
    lows, highs = [], []

    # Moves (always included)
    low1, high1, low2, high2 = get_bounds_moves()
    lows.append(low1)
    highs.append(high1)
    lows.append(low2)
    highs.append(high2)

    if include_hp:
        low, high = get_bounds_hp()
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
    include_hp=True,
    include_status=True,
    include_types=True,
    include_boosts=True,
    include_fainted=True,
    status_one_hot=False
) -> int:
    low, _ = build_observation_bounds(
        include_hp=include_hp,
        include_status=include_status,
        include_types=include_types,
        include_boosts=include_boosts,
        include_fainted=include_fainted,
        status_one_hot=status_one_hot,
    )
    return len(low)


if __name__ == "__main__":
    print(get_embedding_dimension(status_one_hot=True))
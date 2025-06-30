"""
Este código viene de `poke-env`
Se parchea un error para que reconozca Pokémon con distintas formas, como Minior ,correctamente.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Union

from poke_env.environment.double_battle import DoubleBattle
from poke_env.environment.move import Move
from poke_env.environment.pokemon import Pokemon

FORME_ALIASES = {
    "miniorred": "minior",
    "minioryellow": "minior",
    "miniorgreen": "minior",
    "miniorblue": "minior",
    "miniorindigo": "minior",
    "miniorviolet": "minior",
    "miniororange": "minior",
    "minior-meteor": "minior",
    # Otros Pokémon con formas similares pueden ir aquí también
}

def normalize_pokemon_name(name: str) -> str:
    name = name.lower()
    return FORME_ALIASES.get(name, name)

@dataclass
class BattleOrder:
    order: Optional[Union[Move, Pokemon]]
    mega: bool = False
    z_move: bool = False
    dynamax: bool = False
    terastallize: bool = False
    move_target: int = DoubleBattle.EMPTY_TARGET_POSITION

    DEFAULT_ORDER = "/choose default"

    def __str__(self) -> str:
        return self.message

    @property
    def message(self) -> str:
        if isinstance(self.order, Move):
            if self.order.id == "recharge":
                return "/choose move 1"

            message = f"/choose move {self.order.id}"
            if self.mega:
                message += " mega"
            elif self.z_move:
                message += " zmove"
            elif self.dynamax:
                message += " dynamax"
            elif self.terastallize:
                message += " terastallize"

            if self.move_target != DoubleBattle.EMPTY_TARGET_POSITION:
                message += f" {self.move_target}"
            return message
        elif isinstance(self.order, Pokemon):
            return f"/choose switch {normalize_pokemon_name(self.order.species)}"   # único cambio, adaptar el nombre
        else:
            return ""

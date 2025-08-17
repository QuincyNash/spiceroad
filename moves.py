from typing import List, Union
from constants import *
from dataclasses import dataclass, field


@dataclass(slots=True)
class PointMove:
    buy_index: int
    bonus_points: int


@dataclass(slots=True)
class DrawMove:
    draw_index: int


@dataclass(slots=True)
class PlayMove:
    playing: int
    num: int = 1
    spice_upgrades: List[Spice] = field(default_factory=list)


Move = Union[PointMove, DrawMove, PlayMove]


@dataclass(slots=True)
class FullMove:
    base_move: Move
    placing: Union[List[Spice], None] = None
    discarding: Union[SpiceCollection, None] = None

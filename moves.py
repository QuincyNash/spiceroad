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
    placing: List[Union[Spice, None]]


@dataclass(slots=True)
class PartialDrawMove:
    draw_index: int
    placing: Union[List[Spice], None] = None


@dataclass(slots=True)
class PlayMove:
    playing: int
    num: int = 1
    spice_upgrades: List[Spice] = field(default_factory=list)


@dataclass(slots=True)
class PartialPlayMove:
    playing: int
    num: Union[int, None] = None
    spice_upgrades: Union[List[Spice], None] = None


Move = Union[PointMove, DrawMove, PlayMove]


@dataclass(slots=True)
class FullMove:
    base_move: Move
    placing: Union[List[Spice], None] = None
    discarding: Union[SpiceCollection, None] = None

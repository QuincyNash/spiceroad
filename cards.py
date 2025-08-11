from typing import Dict, Union, List
from constants import *
from dataclasses import dataclass, field
from array import array


@dataclass
class TradeCard:
    old: SpiceCollection
    new: SpiceCollection
    id: int
    value_delta: int
    spice_delta: int
    max_delta: int
    spices: SpiceCollection = field(default_factory=lambda: array("b", (0, 0, 0, 0)))


@dataclass
class ObtainCard:
    new: SpiceCollection
    id: int
    value_delta: int
    spice_delta: int
    max_delta: int
    spices: SpiceCollection = field(default_factory=lambda: array("b", (0, 0, 0, 0)))


@dataclass
class UpgradeCard:
    num_conversions: int
    id: int
    value_delta: int
    max_delta: int
    spice_delta: int = 0
    spices: SpiceCollection = field(default_factory=lambda: array("b", (0, 0, 0, 0)))


TraderCard = Union[TradeCard, ObtainCard, UpgradeCard]


_trader_cards: List[TraderCard] = [
    ObtainCard(
        array("b", (2, 0, 0, 0)), max_delta=2, value_delta=2, spice_delta=2, id=5
    ),
    UpgradeCard(2, max_delta=2, value_delta=2, id=6),
    ObtainCard(
        array("b", (2, 0, 0, 0)), max_delta=2, value_delta=2, spice_delta=2, id=7
    ),
    UpgradeCard(2, max_delta=2, value_delta=2, id=8),
    TradeCard(
        array("b", (3, 0, 0, 0)),
        array("b", (0, 0, 0, 1)),
        max_delta=3,
        value_delta=1,
        spice_delta=-2,
        id=9,
    ),
    TradeCard(
        array("b", (0, 1, 0, 0)),
        array("b", (3, 0, 0, 0)),
        max_delta=3,
        value_delta=1,
        spice_delta=2,
        id=10,
    ),
    ObtainCard(
        array("b", (1, 1, 0, 0)), max_delta=3, value_delta=3, spice_delta=2, id=11
    ),
    ObtainCard(
        array("b", (0, 0, 1, 0)), max_delta=3, value_delta=3, spice_delta=1, id=12
    ),
    ObtainCard(
        array("b", (3, 0, 0, 0)), max_delta=3, value_delta=3, spice_delta=3, id=13
    ),
    UpgradeCard(3, max_delta=3, value_delta=3, id=14),
    TradeCard(
        array("b", (0, 0, 2, 0)),
        array("b", (2, 3, 0, 0)),
        max_delta=4,
        value_delta=2,
        spice_delta=3,
        id=15,
    ),
    TradeCard(
        array("b", (0, 0, 2, 0)),
        array("b", (2, 1, 0, 1)),
        max_delta=4,
        value_delta=2,
        spice_delta=2,
        id=16,
    ),
    TradeCard(
        array("b", (0, 0, 0, 1)),
        array("b", (3, 0, 1, 0)),
        max_delta=4,
        value_delta=2,
        spice_delta=3,
        id=17,
    ),
    TradeCard(
        array("b", (0, 2, 0, 0)),
        array("b", (3, 0, 1, 0)),
        max_delta=4,
        value_delta=2,
        spice_delta=2,
        id=18,
    ),
    TradeCard(
        array("b", (0, 3, 0, 0)),
        array("b", (2, 0, 2, 0)),
        max_delta=4,
        value_delta=2,
        spice_delta=1,
        id=19,
    ),
    TradeCard(
        array("b", (0, 0, 0, 1)),
        array("b", (2, 2, 0, 0)),
        max_delta=4,
        value_delta=2,
        spice_delta=3,
        id=20,
    ),
    TradeCard(
        array("b", (4, 0, 0, 0)),
        array("b", (0, 0, 2, 0)),
        max_delta=4,
        value_delta=2,
        spice_delta=-2,
        id=21,
    ),
    ObtainCard(
        array("b", (2, 1, 0, 0)), max_delta=4, value_delta=4, spice_delta=3, id=22
    ),
    ObtainCard(
        array("b", (4, 0, 0, 0)), max_delta=4, value_delta=4, spice_delta=4, id=23
    ),
    ObtainCard(
        array("b", (0, 0, 0, 1)), max_delta=4, value_delta=4, spice_delta=1, id=24
    ),
    ObtainCard(
        array("b", (0, 2, 0, 0)), max_delta=4, value_delta=4, spice_delta=2, id=25
    ),
    ObtainCard(
        array("b", (1, 0, 1, 0)), max_delta=4, value_delta=4, spice_delta=2, id=26
    ),
    TradeCard(
        array("b", (2, 0, 0, 0)),
        array("b", (0, 0, 1, 0)),
        max_delta=5,
        value_delta=1,
        spice_delta=-1,
        id=27,
    ),
    TradeCard(
        array("b", (1, 1, 0, 0)),
        array("b", (0, 0, 0, 1)),
        max_delta=5,
        value_delta=1,
        spice_delta=-1,
        id=28,
    ),
    TradeCard(
        array("b", (0, 0, 1, 0)),
        array("b", (0, 2, 0, 0)),
        max_delta=5,
        value_delta=1,
        spice_delta=1,
        id=29,
    ),
    TradeCard(
        array("b", (0, 2, 0, 0)),
        array("b", (2, 0, 0, 1)),
        max_delta=6,
        value_delta=2,
        spice_delta=1,
        id=30,
    ),
    TradeCard(
        array("b", (3, 0, 0, 0)),
        array("b", (0, 1, 1, 0)),
        max_delta=6,
        value_delta=2,
        spice_delta=-1,
        id=31,
    ),
    TradeCard(
        array("b", (0, 0, 2, 0)),
        array("b", (0, 2, 0, 1)),
        max_delta=6,
        value_delta=2,
        spice_delta=1,
        id=32,
    ),
    TradeCard(
        array("b", (0, 3, 0, 0)),
        array("b", (1, 0, 1, 1)),
        max_delta=6,
        value_delta=2,
        spice_delta=0,
        id=33,
    ),
    TradeCard(
        array("b", (0, 0, 0, 1)),
        array("b", (0, 3, 0, 0)),
        max_delta=6,
        value_delta=2,
        spice_delta=2,
        id=34,
    ),
    TradeCard(
        array("b", (0, 3, 0, 0)),
        array("b", (0, 0, 0, 2)),
        max_delta=6,
        value_delta=2,
        spice_delta=-1,
        id=35,
    ),
    TradeCard(
        array("b", (0, 0, 0, 1)),
        array("b", (1, 1, 1, 0)),
        max_delta=6,
        value_delta=2,
        spice_delta=2,
        id=36,
    ),
    TradeCard(
        array("b", (0, 0, 1, 0)),
        array("b", (1, 2, 0, 0)),
        max_delta=6,
        value_delta=2,
        spice_delta=2,
        id=37,
    ),
    TradeCard(
        array("b", (0, 0, 1, 0)),
        array("b", (4, 1, 0, 0)),
        max_delta=6,
        value_delta=3,
        spice_delta=4,
        id=38,
    ),
    TradeCard(
        array("b", (5, 0, 0, 0)),
        array("b", (0, 0, 0, 2)),
        max_delta=6,
        value_delta=3,
        spice_delta=-3,
        id=39,
    ),
    TradeCard(
        array("b", (4, 0, 0, 0)),
        array("b", (0, 0, 1, 1)),
        max_delta=6,
        value_delta=3,
        spice_delta=-2,
        id=40,
    ),
    TradeCard(
        array("b", (0, 0, 0, 2)),
        array("b", (0, 3, 2, 0)),
        max_delta=8,
        value_delta=4,
        spice_delta=3,
        id=41,
    ),
    TradeCard(
        array("b", (0, 0, 0, 2)),
        array("b", (1, 1, 3, 0)),
        max_delta=8,
        value_delta=4,
        spice_delta=3,
        id=42,
    ),
    TradeCard(
        array("b", (5, 0, 0, 0)),
        array("b", (0, 0, 3, 0)),
        max_delta=8,
        value_delta=4,
        spice_delta=-2,
        id=43,
    ),
    TradeCard(
        array("b", (2, 0, 1, 0)),
        array("b", (0, 0, 0, 2)),
        max_delta=9,
        value_delta=3,
        spice_delta=-1,
        id=44,
    ),
    TradeCard(
        array("b", (0, 0, 3, 0)),
        array("b", (0, 0, 0, 3)),
        max_delta=9,
        value_delta=3,
        spice_delta=0,
        id=45,
    ),
    TradeCard(
        array("b", (0, 3, 0, 0)),
        array("b", (0, 0, 3, 0)),
        max_delta=9,
        value_delta=3,
        spice_delta=0,
        id=46,
    ),
    TradeCard(
        array("b", (3, 0, 0, 0)),
        array("b", (0, 3, 0, 0)),
        max_delta=9,
        value_delta=3,
        spice_delta=0,
        id=47,
    ),
    TradeCard(
        array("b", (2, 0, 0, 0)),
        array("b", (0, 2, 0, 0)),
        max_delta=10,
        value_delta=2,
        spice_delta=0,
        id=48,
    ),
    TradeCard(
        array("b", (0, 2, 0, 0)),
        array("b", (0, 0, 2, 0)),
        max_delta=10,
        value_delta=2,
        spice_delta=0,
        id=49,
    ),
    TradeCard(
        array("b", (0, 0, 2, 0)),
        array("b", (0, 0, 0, 2)),
        max_delta=10,
        value_delta=2,
        spice_delta=0,
        id=50,
    ),
    TradeCard(
        array("b", (0, 0, 0, 1)),
        array("b", (0, 0, 2, 0)),
        max_delta=10,
        value_delta=2,
        spice_delta=1,
        id=51,
    ),
]
trader_cards: Dict[int, TraderCard] = {card.id: card for card in _trader_cards}
trader_card_ids = [card.id for card in _trader_cards]


@dataclass
class PointCard:
    spices: SpiceCollection
    points: int
    id: int


_point_cards: List[PointCard] = [
    PointCard(array("b", (2, 2, 0, 0)), 6, id=52),
    PointCard(array("b", (3, 2, 0, 0)), 7, id=53),
    PointCard(array("b", (0, 4, 0, 0)), 8, id=54),
    PointCard(array("b", (2, 0, 2, 0)), 8, id=55),
    PointCard(array("b", (2, 3, 0, 0)), 8, id=56),
    PointCard(array("b", (3, 0, 2, 0)), 9, id=57),
    PointCard(array("b", (0, 2, 2, 0)), 10, id=58),
    PointCard(array("b", (0, 5, 0, 0)), 10, id=59),
    PointCard(array("b", (2, 0, 0, 2)), 10, id=60),
    PointCard(array("b", (2, 0, 3, 0)), 11, id=61),
    PointCard(array("b", (3, 0, 0, 2)), 11, id=62),
    PointCard(array("b", (0, 0, 4, 0)), 12, id=63),
    PointCard(array("b", (0, 2, 0, 2)), 12, id=64),
    PointCard(array("b", (0, 3, 2, 0)), 12, id=65),
    PointCard(array("b", (0, 2, 3, 0)), 13, id=66),
    PointCard(array("b", (0, 0, 2, 2)), 14, id=67),
    PointCard(array("b", (0, 3, 0, 2)), 14, id=68),
    PointCard(array("b", (2, 0, 0, 3)), 14, id=69),
    PointCard(array("b", (0, 0, 5, 0)), 15, id=70),
    PointCard(array("b", (0, 0, 0, 4)), 16, id=71),
    PointCard(array("b", (0, 2, 0, 3)), 16, id=72),
    PointCard(array("b", (0, 0, 3, 2)), 17, id=73),
    PointCard(array("b", (0, 0, 2, 3)), 18, id=74),
    PointCard(array("b", (0, 0, 0, 5)), 20, id=75),
    PointCard(array("b", (2, 1, 0, 1)), 9, id=76),
    PointCard(array("b", (0, 2, 1, 1)), 12, id=77),
    PointCard(array("b", (1, 0, 2, 1)), 12, id=78),
    PointCard(array("b", (2, 2, 2, 0)), 13, id=79),
    PointCard(array("b", (2, 2, 0, 2)), 15, id=80),
    PointCard(array("b", (2, 0, 2, 2)), 17, id=81),
    PointCard(array("b", (0, 2, 2, 2)), 19, id=82),
    PointCard(array("b", (1, 1, 1, 1)), 12, id=83),
    PointCard(array("b", (3, 1, 1, 1)), 14, id=84),
    PointCard(array("b", (1, 3, 1, 1)), 16, id=85),
    PointCard(array("b", (1, 1, 3, 1)), 18, id=86),
    PointCard(array("b", (1, 1, 1, 3)), 20, id=87),
]
point_cards: Dict[int, PointCard] = {card.id: card for card in _point_cards}
point_card_ids = [card.id for card in _point_cards]

card_ids = [card.id for card in _trader_cards + _point_cards]

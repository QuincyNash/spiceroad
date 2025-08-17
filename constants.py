from __future__ import annotations
from typing import Dict, List, Tuple, Union, Literal
from array import array

FPS = 60
ANIMATION_SECONDS = 1

CARD_POINT_GOLD_BONUS = 3
CARD_POINT_SILVER_BONUS = 1
PLAYER_MAX_SPICES = 10
TRADER_CARD_NUM = 6
POINT_CARD_NUM = 5
POINT_CARDS_TO_END_GAME = 6

SPICE_SIZE = 15
SPICE_SPACING = 2
ARROW_SIZE = 20
COIN_SIZE = 40
FONT_SIZE = 20
BIG_FONT_SIZE = 35

CARD_GROUP_ROWS = 4
CARD_GROUP_COLS = 5
PLAYER_MAX_CARDS = CARD_GROUP_ROWS * CARD_GROUP_COLS


BORDER_SIZE = 20
CARD_SPACING = 10
HORIZONTAL_SECTION_SPACING = 10
VERTICAL_SECTION_SPACING = 50
CARD_COUNT_SPACING = 20

CARD_SIZE = 6 * SPICE_SIZE + 5 * SPICE_SPACING + 10

CARD_GROUP_HEIGHT = CARD_SIZE / 2 * CARD_GROUP_ROWS + CARD_SPACING / 2 * (
    CARD_GROUP_ROWS - 1
)
CARD_GROUP_WIDTH = CARD_GROUP_COLS * CARD_SIZE / 2 + CARD_SPACING / 2 * (
    CARD_GROUP_COLS - 1
)
SCREEN_WIDTH = BORDER_SIZE * 2 + CARD_SIZE * 6 + CARD_SPACING * 5
SCREEN_HEIGHT = (
    BORDER_SIZE * 2 + CARD_SIZE * 2 + VERTICAL_SECTION_SPACING * 3 + CARD_GROUP_HEIGHT
)

HOVER_BLACK_OPACITY = 25
SPICE_OPACITY_FACTOR = 0.6
NUM_CHOICE_BLACK_OPACITY = 180


COLOR_TUMERIC = (230, 200, 60)
COLOR_SAFFRON = (191, 50, 70)
COLOR_CARDAMOM = (90, 200, 100)
COLOR_CINNAMON = (150, 90, 60)
COLOR_GOLD = (212, 175, 55)
COLOR_SILVER = (150, 150, 150)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (255, 0, 0)

PLAYER_REST = [-128, -127]

Spice = Union[Literal[1], Literal[2], Literal[3], Literal[4]]
SpiceCollection = array
COLOR_SPICES: Dict[Spice, Tuple[int, int, int]] = {
    1: COLOR_TUMERIC,
    2: COLOR_SAFFRON,
    3: COLOR_CARDAMOM,
    4: COLOR_CINNAMON,
}
SPICES: List[Spice] = [1, 2, 3, 4]

NEXT_SPICE_UPGRADE: Dict[Spice, Spice] = {
    1: 2,
    2: 3,
    3: 4,
    4: 4,
}

DOUBLE_UPGRADE_POSSIBILITIES: List[Tuple[SpiceCollection, List[Spice]]] = [
    (array("b", (2, 0, 0, 0)), [1, 1]),
    (array("b", (1, 1, 0, 0)), [1, 2]),
    (array("b", (1, 0, 1, 0)), [1, 3]),
    (array("b", (0, 2, 0, 0)), [2, 2]),
    (array("b", (0, 1, 1, 0)), [2, 3]),
    (array("b", (0, 0, 2, 0)), [3, 3]),
]
TRIPLE_UPGRADE_POSSIBILITIES: List[Tuple[SpiceCollection, List[Spice]]] = [
    (array("b", (3, 0, 0, 0)), [1, 1, 1]),
    (array("b", (2, 1, 0, 0)), [1, 1, 2]),
    (
        array("b", (2, 0, 1, 0)),
        [1, 1, 3],
    ),
    (array("b", (1, 2, 0, 0)), [1, 2, 2]),
    (
        array("b", (1, 1, 1, 0)),
        [1, 2, 3],
    ),
    (
        array("b", (1, 0, 2, 0)),
        [1, 3, 3],
    ),
    (array("b", (0, 3, 0, 0)), [2, 2, 2]),
    (
        array("b", (0, 2, 1, 0)),
        [2, 2, 3],
    ),
    (
        array("b", (0, 1, 2, 0)),
        [2, 3, 3],
    ),
    (array("b", (0, 0, 3, 0)), [3, 3, 3]),
]

del array

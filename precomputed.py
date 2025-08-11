import time
from typing import Dict, List
from constants import *
from collections import Counter
from itertools import combinations_with_replacement
from utils import can_buy, floor_div, add_array_spices
from moves import PlayMove, DrawMove
from cards import trader_cards, ObtainCard, TradeCard, UpgradeCard
from array import array


ALL_REASONABLE_SPICE_PLACEMENTS: Dict[
    Tuple[int, int, int, int],
    Dict[int, List[List[Spice]]],
] = {}

for i in range(11):
    for spices in combinations_with_replacement(SPICES, r=i):
        spice_counter = Counter(spices)
        spice_tuple = (
            spice_counter[1],
            spice_counter[2],
            spice_counter[3],
            spice_counter[4],
        )

        possible: Dict[int, List[List[Spice]]] = {}

        for i2 in range(6):
            possible[i2] = []
            for comb_list, comb_tuple in REASONABLE_SPICE_PLACEMENTS[i2]:
                if can_buy(array("b", spice_tuple), comb_tuple):
                    possible[i2].append(comb_list)

        ALL_REASONABLE_SPICE_PLACEMENTS[spice_tuple] = possible


ALL_DISCARD_POSSIBILITIES: Dict[Tuple[int, int, int, int], List[SpiceCollection]] = {}
for i in range(11):
    for spices in combinations_with_replacement(SPICES, r=i):
        spice_counter = Counter(spices)
        for card in trader_cards.values():
            if card.spice_delta > 0 and not isinstance(card, UpgradeCard):
                spice_array = array(
                    "b",
                    (
                        spice_counter[1],
                        spice_counter[2],
                        spice_counter[3],
                        spice_counter[4],
                    ),
                )

                max_repeats = (
                    floor_div(spice_array, card.old)
                    if isinstance(card, TradeCard)
                    else 1
                )

                for repeats in range(1, max_repeats + 1):
                    if isinstance(card, ObtainCard):
                        add_array_spices(spice_array, card.new)
                    else:
                        add_array_spices(
                            spice_array,
                            array(
                                "b",
                                (
                                    (card.new[0] - card.old[0]),
                                    (card.new[1] - card.old[1]),
                                    (card.new[2] - card.old[2]),
                                    (card.new[3] - card.old[3]),
                                ),
                            ),
                        )

                        a_max, b_max, c_max, d_max = spice_array
                        total = sum(spice_array)
                        need_to_discard = min(max(total - 10, 0), 5)

                        results: List[SpiceCollection] = []
                        if need_to_discard > 0:
                            for dd in range(min(d_max, need_to_discard) + 1):
                                for dc in range(min(c_max, need_to_discard - dd) + 1):
                                    for db in range(
                                        min(b_max, need_to_discard - dd - dc) + 1
                                    ):
                                        da = need_to_discard - dd - dc - db
                                        if 0 <= da <= a_max:
                                            results.append(
                                                array(
                                                    "b",
                                                    (da, db, dc, dd),
                                                )
                                            )

                        ALL_DISCARD_POSSIBILITIES[
                            (
                                spice_array[0],
                                spice_array[1],
                                spice_array[2],
                                spice_array[3],
                            )
                        ] = results

print(ALL_DISCARD_POSSIBILITIES[(2, 3, 11, 0)])


ALL_DOUBLE_UPGRADE_POSSIBILITIES: Dict[
    int, Dict[Tuple[int, int, int, int], List[PlayMove]]
] = {
    6: {},
    8: {},
}
for id in [6, 8]:
    for i in range(11):
        for spices in combinations_with_replacement(SPICES, r=i):
            spice_counter = Counter(spices)
            spice_tuple = (
                spice_counter[1],
                spice_counter[2],
                spice_counter[3],
                spice_counter[4],
            )

            possible_combs: List[PlayMove] = []

            for collection, upgrades in DOUBLE_UPGRADE_POSSIBILITIES:
                fail = False
                new_spice_counter = spice_counter.copy()

                for i in range(len(upgrades)):
                    if new_spice_counter[upgrades[i]] == 0:
                        fail = True
                        break
                    new_spice_counter[upgrades[i]] -= 1
                    new_spice_counter[NEXT_SPICE_UPGRADE[upgrades[i]]] += 1

                if not fail:
                    possible_combs.append(PlayMove(id, spice_upgrades=upgrades))

            ALL_DOUBLE_UPGRADE_POSSIBILITIES[id][spice_tuple] = possible_combs


ALL_TRIPLE_UPGRADE_POSSIBILITIES: Dict[
    int, Dict[Tuple[int, int, int, int], List[PlayMove]]
] = {14: {}}
for i in range(11):
    for spices in combinations_with_replacement(SPICES, r=i):
        spice_counter = Counter(spices)
        spice_tuple = (
            spice_counter[1],
            spice_counter[2],
            spice_counter[3],
            spice_counter[4],
        )

        possible_combs: List[PlayMove] = []

        for collection, upgrades in TRIPLE_UPGRADE_POSSIBILITIES:
            fail = False
            new_spice_counter = spice_counter.copy()

            for i in range(len(upgrades)):
                if new_spice_counter[upgrades[i]] == 0:
                    fail = True
                    break
                new_spice_counter[upgrades[i]] -= 1
                new_spice_counter[NEXT_SPICE_UPGRADE[upgrades[i]]] += 1

            if not fail:
                possible_combs.append(PlayMove(14, spice_upgrades=upgrades))

        ALL_TRIPLE_UPGRADE_POSSIBILITIES[14][spice_tuple] = possible_combs


ALL_NONUPGRADE_PLAY_MOVES: Dict[int, Dict[int, List[PlayMove]]] = {}
for card in trader_cards.values():
    if isinstance(card, ObtainCard):
        ALL_NONUPGRADE_PLAY_MOVES[card.id] = {1: [PlayMove(card.id)]}
    elif isinstance(card, TradeCard):
        ALL_NONUPGRADE_PLAY_MOVES[card.id] = {}
        for i in range(11):
            ALL_NONUPGRADE_PLAY_MOVES[card.id][i] = [
                PlayMove(card.id, j) for j in range(1, i + 1)
            ]

ALL_DRAW_MOVES: Dict[int, List[DrawMove]] = {}
for i in range(TRADER_CARD_NUM):
    ALL_DRAW_MOVES[i] = [DrawMove(j, []) for j in range(i + 1)]

ALL_REST_MOVES = [PlayMove(PLAYER_REST[0]), PlayMove(PLAYER_REST[1])]

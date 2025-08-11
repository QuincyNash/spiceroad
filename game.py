from __future__ import annotations
from typing import List
from constants import *
from precomputed import (
    ALL_REASONABLE_SPICE_PLACEMENTS,
    ALL_DOUBLE_UPGRADE_POSSIBILITIES,
    ALL_TRIPLE_UPGRADE_POSSIBILITIES,
    ALL_NONUPGRADE_PLAY_MOVES,
    ALL_DRAW_MOVES,
    ALL_REST_MOVES,
)
from dataclasses import dataclass, field
from utils import (
    RandomizedSet,
    can_buy,
    add_array_spices,
    floor_div,
    sub_array_spices,
    upgrade_array_spice,
    add_array_spice,
    sub_array_spice,
)
from cards import (
    ObtainCard,
    TradeCard,
    UpgradeCard,
    point_cards,
    point_card_ids,
    trader_cards,
    trader_card_ids,
    SpiceCollection,
)
from moves import PartialDrawMove, PointMove, DrawMove, PlayMove, Move, PartialPlayMove
import random
from array import array


@dataclass
class Player:
    points: int
    spices: SpiceCollection
    cards: Dict[int, PlayerCard]
    point_cards_bought: int

    def copy(self) -> Player:
        return Player(
            self.points,
            self.spices[:],
            {i: self.cards[i].copy() for i in self.cards},
            self.point_cards_bought,
        )


@dataclass
class PlayerCard:
    index: int
    usable: bool = True

    def copy(self) -> PlayerCard:
        return PlayerCard(self.index, self.usable)


@dataclass
class FaceupTraderCard:
    id: int
    spices: SpiceCollection = field(default_factory=lambda: array("b", (0, 0, 0, 0)))

    def copy(self) -> FaceupTraderCard:
        return FaceupTraderCard(self.id, self.spices[:])


class Game:
    def __init__(self, seed: Union[int, None] = None) -> None:
        if seed is not None:
            random.seed(seed)

        self.point_card_stack = RandomizedSet(point_card_ids)
        self.trader_card_stack = RandomizedSet(trader_card_ids)
        self.coins = [4, 4]

        self.point_cards: List[int] = random.sample(point_card_ids, POINT_CARD_NUM)
        self.trader_cards: List[FaceupTraderCard] = [
            FaceupTraderCard(id)
            for id in random.sample(trader_card_ids[4:], TRADER_CARD_NUM)
        ]

        self.player1 = Player(
            0, array("b", (3, 0, 0, 0)), {5: PlayerCard(0), 6: PlayerCard(1)}, 0
        )
        self.player2 = Player(
            0, array("b", (3, 3, 2, 2)), {7: PlayerCard(0), 8: PlayerCard(1)}, 0
        )

        self.point_card_stack.remove_many(self.point_cards)
        self.trader_card_stack.remove_many(
            [c.id for c in self.trader_cards] + [5, 6, 7, 8]
        )

        self.players = [self.player1, self.player2]

    def copy(self) -> Game:
        game = Game.__new__(Game)

        game.point_card_stack = self.point_card_stack.copy()
        game.trader_card_stack = self.trader_card_stack.copy()
        game.coins = self.coins[:]
        game.point_cards = self.point_cards[:]
        game.trader_cards = [tc.copy() for tc in self.trader_cards]

        game.player1 = self.player1.copy()
        game.player2 = self.player2.copy()
        game.players = [game.player1, game.player2]

        return game

    def create_rest_move(self, turn: int) -> PlayMove:
        return PlayMove(PLAYER_REST[turn])

    def create_upgrade_move(
        self, partial_move: PartialPlayMove, spice: Spice
    ) -> PlayMove:
        return PlayMove(partial_move.playing, 1, [spice])

    def create_draw_move(
        self, partial_move: PartialDrawMove, spices: List[Spice]
    ) -> DrawMove:
        num_blanks = (
            0 if partial_move.placing is None else len(partial_move.placing) - 1
        )

        placing: List[Union[Spice, None]] = [None for _ in range(num_blanks)]
        placing.extend(spices)

        return DrawMove(partial_move.draw_index, placing)

    def create_full_play_move(
        self, partial_move: PartialPlayMove, num: int
    ) -> PlayMove:
        return PlayMove(partial_move.playing, num)

    def find_point_move(self, index: int, moves: List[Move]) -> Union[PointMove, None]:
        for move in moves:
            if isinstance(move, PointMove) and move.buy_index == index:
                return move

    def find_draw_move(
        self, index: int, moves: List[Move]
    ) -> Union[DrawMove, Tuple[PartialDrawMove, List[DrawMove]]]:
        found: List[DrawMove] = []
        for move in moves:
            if isinstance(move, DrawMove) and move.draw_index == index:
                found.append(move)

        if len(found) == 1:
            return found[0]
        return (PartialDrawMove(draw_index=index), found)

    def find_play_move(
        self, id: int, moves: List[Move]
    ) -> Union[PlayMove, Tuple[PartialPlayMove, List[PlayMove]], None]:
        found: List[PlayMove] = []
        for move in moves:
            if isinstance(move, PlayMove) and move.playing == id:
                if isinstance(trader_cards[id], ObtainCard):
                    return move
                found.append(move)

        if len(found) == 1:
            return found[0]
        if isinstance(trader_cards[id], UpgradeCard):
            return PartialPlayMove(id, num=1), found
        return PartialPlayMove(id, spice_upgrades=[]), found

    def coin_bonus(self, index: int) -> int:
        if index == 0 and self.coins[0] > 0:
            return CARD_POINT_GOLD_BONUS
        if (index == 0 and self.coins[0] == 0 and self.coins[1] > 0) or (
            index == 1 and self.coins[0] > 0 and self.coins[1] > 0
        ):
            return CARD_POINT_SILVER_BONUS
        return 0

    def all_base_moves(self, player_id: int) -> List[Move]:
        moves: List[Move] = [ALL_REST_MOVES[player_id]]
        player = self.players[player_id]

        for index, card_id in enumerate(self.point_cards):
            if can_buy(player.spices, point_cards[card_id].spices):
                moves.append(PointMove(index, self.coin_bonus(index)))

        for id, player_card in player.cards.items():
            if player_card.usable:
                card = trader_cards[id]
                if isinstance(card, ObtainCard):
                    moves.extend(ALL_NONUPGRADE_PLAY_MOVES[card.id][1])
                elif isinstance(card, TradeCard):
                    moves.extend(
                        ALL_NONUPGRADE_PLAY_MOVES[card.id][
                            floor_div(player.spices, card.old)
                        ]
                    )
                else:
                    conversions = (
                        ALL_DOUBLE_UPGRADE_POSSIBILITIES
                        if card.num_conversions == 2
                        else ALL_TRIPLE_UPGRADE_POSSIBILITIES
                    )
                    moves.extend(conversions[card.id][tuple(player.spices)])

        if len(player.cards) < PLAYER_MAX_CARDS:
            num_cards_buyable = max(sum(player.spices), TRADER_CARD_NUM - 1)
            moves.extend(ALL_DRAW_MOVES[num_cards_buyable])

        return moves

    def all_placement_moves(self, player_id: int, index: int) -> List[List[Spice]]:
        spice_placements = ALL_REASONABLE_SPICE_PLACEMENTS[
            tuple(self.players[player_id].spices)
        ]
        return spice_placements[index]

    def all_moves(self, player_id: int) -> List[Move]:
        moves: List[Move] = []
        player = self.players[player_id]
        num_player_spices = sum(player.spices)

        for i, card_id in enumerate(self.point_cards):
            card = point_cards[card_id]

            if can_buy(player.spices, card.spices):
                moves.append(PointMove(i, self.coin_bonus(i)))

        all_cards_usable = True
        for id, player_card in player.cards.items():
            if player_card.usable:
                card = trader_cards[id]
                if isinstance(card, ObtainCard):
                    if num_player_spices + card.spice_delta <= PLAYER_MAX_SPICES:
                        moves.append(PlayMove(card.id))
                elif isinstance(card, TradeCard):
                    num_conversions = 1
                    while (
                        num_player_spices + num_conversions * card.spice_delta
                        <= PLAYER_MAX_SPICES
                        and all(
                            player.spices[i] >= card.old[i] * num_conversions
                            for i in range(4)
                        )
                    ):
                        moves.append(PlayMove(card.id, num_conversions))
                        num_conversions += 1
                else:
                    conversions = (
                        ALL_DOUBLE_UPGRADE_POSSIBILITIES
                        if card.num_conversions == 2
                        else ALL_TRIPLE_UPGRADE_POSSIBILITIES
                    )
                    moves.extend(conversions[card.id][tuple(player.spices)])
            else:
                all_cards_usable = False

        if len(player.cards) < PLAYER_MAX_CARDS:
            spice_placements = ALL_REASONABLE_SPICE_PLACEMENTS[tuple(player.spices)]
            for i, card in enumerate(self.trader_cards):
                spice_delta = sum(card.spices) - i
                if num_player_spices + spice_delta <= PLAYER_MAX_SPICES:
                    for comb_list in spice_placements[i]:
                        moves.append(DrawMove(i, comb_list))  # type: ignore

        if not all_cards_usable or len(moves) == 0:
            moves.append(PlayMove(PLAYER_REST[player_id]))

        return moves

    def make_move(self, player_id: int, move: Move, *, is_partial=False) -> None:
        player = self.players[player_id]

        if isinstance(move, PointMove):
            card = point_cards[self.point_cards[move.buy_index]]
            player.points += card.points + move.bonus_points
            player.point_cards_bought += 1
            sub_array_spices(player.spices, card.spices)
            self.point_cards.pop(move.buy_index)
            self.point_cards.append(self.point_card_stack.pop_random())

            if move.bonus_points == CARD_POINT_GOLD_BONUS:
                self.coins[0] -= 1
            elif move.bonus_points == CARD_POINT_SILVER_BONUS:
                self.coins[1] -= 1

        elif isinstance(move, PlayMove):
            if move.playing in PLAYER_REST:
                for card in player.cards.values():
                    card.usable = True
                return

            card = trader_cards[move.playing]

            if isinstance(card, ObtainCard):
                add_array_spices(player.spices, card.new)
            elif isinstance(card, TradeCard):
                add_array_spices(
                    player.spices,
                    array(
                        "b",
                        (
                            move.num * (card.new[0] - card.old[0]),
                            move.num * (card.new[1] - card.old[1]),
                            move.num * (card.new[2] - card.old[2]),
                            move.num * (card.new[3] - card.old[3]),
                        ),
                    ),
                )
            else:
                for spice in move.spice_upgrades:
                    upgrade_array_spice(player.spices, spice)

            if not is_partial:
                player.cards[move.playing].usable = False

        else:
            for i, spice in enumerate(move.placing):
                if spice is not None:
                    add_array_spice(self.trader_cards[i].spices, spice)
                    sub_array_spice(player.spices, spice)

            if not is_partial:
                card = self.trader_cards[move.draw_index]

                add_array_spices(player.spices, card.spices)

                self.trader_cards.pop(move.draw_index)
                new_card = self.trader_card_stack.pop_random()
                self.trader_cards.append(FaceupTraderCard(new_card))

                player.cards[card.id] = PlayerCard(len(player.cards))

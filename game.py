from __future__ import annotations
from typing import List
from constants import *
from precomputed import (
    ALL_DISCARD_POSSIBILITIES,
    ALL_REASONABLE_SPICE_PLACEMENTS,
    MAX_REASONABLE_INDEX_BUYABLE,
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
    floor_div,
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
from moves import PointMove, PlayMove, Move
import random
from array import array


@dataclass(slots=True)
class Player:
    spices: SpiceCollection
    cards: Dict[int, PlayerCard]
    points: int = 0
    point_cards_bought: int = 0

    def copy(self) -> Player:
        return Player(
            self.spices[:],
            {i: self.cards[i].copy() for i in self.cards},
            self.points,
            self.point_cards_bought,
        )


@dataclass(slots=True)
class PlayerCard:
    index: int
    usable: bool = True

    def copy(self) -> PlayerCard:
        return PlayerCard(self.index, self.usable)


@dataclass(slots=True)
class FaceupTraderCard:
    id: int
    spices: SpiceCollection = field(default_factory=lambda: array("b", (0, 0, 0, 0)))

    def copy(self) -> FaceupTraderCard:
        return FaceupTraderCard(self.id, self.spices[:])


class Game:
    def __init__(self, seed: Union[int, None] = None) -> None:
        if seed is not None:
            random.seed(seed)

        self.finished = False
        self.point_card_stack = RandomizedSet(point_card_ids)
        self.trader_card_stack = RandomizedSet(trader_card_ids)
        self.coins = [4, 4]

        self.point_cards: List[int] = random.sample(point_card_ids, POINT_CARD_NUM)
        self.trader_cards: List[FaceupTraderCard] = [
            FaceupTraderCard(id)
            for id in random.sample(trader_card_ids[4:], TRADER_CARD_NUM)
        ]

        self.player1 = Player(
            array("b", (3, 0, 0, 0)), {5: PlayerCard(0), 6: PlayerCard(1)}
        )
        self.player2 = Player(
            array("b", (4, 0, 0, 0)), {7: PlayerCard(0), 8: PlayerCard(1)}
        )

        self.point_card_stack.remove_many(self.point_cards)
        self.trader_card_stack.remove_many(
            [c.id for c in self.trader_cards] + [5, 6, 7, 8]
        )

        self.players = [self.player1, self.player2]

    def copy(self) -> Game:
        game = Game.__new__(Game)

        game.finished = self.finished
        game.point_card_stack = self.point_card_stack.copy()
        game.trader_card_stack = self.trader_card_stack.copy()
        game.coins = self.coins[:]
        game.point_cards = self.point_cards[:]
        game.trader_cards = [tc.copy() for tc in self.trader_cards]

        game.player1 = self.player1.copy()
        game.player2 = self.player2.copy()
        game.players = [game.player1, game.player2]

        return game

    def coin_bonus(self, index: int) -> int:
        if index == 0 and self.coins[0] > 0:
            return CARD_POINT_GOLD_BONUS
        if (index == 0 and self.coins[0] == 0 and self.coins[1] > 0) or (
            index == 1 and self.coins[0] > 0 and self.coins[1] > 0
        ):
            return CARD_POINT_SILVER_BONUS
        return 0

    def max_repeats(self, player_id: int, card_id: int) -> int:
        card = trader_cards[card_id]
        if isinstance(card, TradeCard):
            return floor_div(self.players[player_id].spices, card.old)
        return 1

    def all_base_moves(
        self, player_id: int, *, allow_all_placements=False
    ) -> List[Move]:
        moves: List[Move] = [ALL_REST_MOVES[player_id]]
        player = self.players[player_id]

        for index, card_id in enumerate(self.point_cards):
            if can_buy(player.spices, point_cards[card_id].spices):
                moves.append(PointMove(index, self.coin_bonus(index)))

        for id, player_card in player.cards.items():
            if player_card.usable:
                card = trader_cards[id]
                if isinstance(card, UpgradeCard):
                    conversions = (
                        ALL_DOUBLE_UPGRADE_POSSIBILITIES
                        if card.num_conversions == 2
                        else ALL_TRIPLE_UPGRADE_POSSIBILITIES
                    )
                    moves.extend(conversions[card.id][tuple(player.spices)])
                else:
                    moves.extend(
                        ALL_NONUPGRADE_PLAY_MOVES[card.id][
                            self.max_repeats(player_id, id)
                        ]
                    )

        if len(player.cards) < PLAYER_MAX_CARDS:
            if allow_all_placements:
                num_cards_buyable = min(sum(player.spices), TRADER_CARD_NUM - 1)
            else:
                num_cards_buyable = MAX_REASONABLE_INDEX_BUYABLE[tuple(player.spices)]
            moves.extend(ALL_DRAW_MOVES[num_cards_buyable])

        return moves

    def all_placement_moves(self, player_id: int, index: int) -> List[List[Spice]]:
        return ALL_REASONABLE_SPICE_PLACEMENTS[tuple(self.players[player_id].spices)][
            index
        ]

    def one_spice_type(self, player_id: int) -> bool:
        a = self.players[player_id].spices
        return (a[0] == 0) + (a[1] == 0) + (a[2] == 0) + (a[3] == 0) == 3

    def only_spice_type(self, player_id: int) -> Spice:
        a = self.players[player_id].spices
        if a[0] > 0:
            return 1
        elif a[1] > 0:
            return 2
        elif a[2] > 0:
            return 3
        else:
            return 4

    def needs_to_discard(self, player_id: int) -> bool:
        return sum(self.players[player_id].spices) > PLAYER_MAX_SPICES

    def discard_simple(self, player_id: int) -> SpiceCollection:
        spice = self.only_spice_type(player_id)
        num = sum(self.players[player_id].spices) - PLAYER_MAX_SPICES
        return array("b", (num if i == spice - 1 else 0 for i in range(4)))

    def discard_array(self, spice: Spice) -> SpiceCollection:
        return array("b", (1 if i == spice - 1 else 0 for i in range(4)))

    def discard_nums(self, player_id: int, card_id: int) -> List[int]:
        card = trader_cards[card_id]
        spice_count = sum(self.players[player_id].spices)
        nums = [
            max(spice_count + card.spice_delta * i - PLAYER_MAX_SPICES, 0)
            for i in range(1, PLAYER_MAX_SPICES + 1)
        ]
        return nums

    def all_discard_moves(self, player_id: int) -> List[SpiceCollection]:
        return ALL_DISCARD_POSSIBILITIES.get(tuple(self.players[player_id].spices), [])

    def make_base_move(self, player_id: int, move: Move) -> None:
        player = self.players[player_id]
        player_spices = self.players[player_id].spices

        if isinstance(move, PointMove):
            card = point_cards[self.point_cards[move.buy_index]]
            card_spices = card.spices
            player.points += card.points + move.bonus_points
            player.point_cards_bought += 1

            player_spices[0] -= card_spices[0]
            player_spices[1] -= card_spices[1]
            player_spices[2] -= card_spices[2]
            player_spices[3] -= card_spices[3]

            self.point_cards.pop(move.buy_index)
            self.point_cards.append(self.point_card_stack.pop_random())

            if move.bonus_points == CARD_POINT_GOLD_BONUS:
                self.coins[0] -= 1
            elif move.bonus_points == CARD_POINT_SILVER_BONUS:
                self.coins[1] -= 1

            if player.point_cards_bought == POINT_CARDS_TO_END_GAME:
                self.finished = True
                for p in self.players:
                    spices = p.spices
                    p.points += spices[1] + spices[2] + spices[3]

        elif isinstance(move, PlayMove):
            move_playing = move.playing
            if move_playing in PLAYER_REST:
                for card in player.cards.values():
                    card.usable = True
                return

            card = trader_cards[move_playing]

            if isinstance(card, ObtainCard):
                card_new = card.new
                player_spices[0] += card_new[0]
                player_spices[1] += card_new[1]
                player_spices[2] += card_new[2]
                player_spices[3] += card_new[3]
            elif isinstance(card, TradeCard):
                card_new = card.new
                card_old = card.old
                move_num = move.num
                player_spices[0] += move_num * (card_new[0] - card_old[0])
                player_spices[1] += move_num * (card_new[1] - card_old[1])
                player_spices[2] += move_num * (card_new[2] - card_old[2])
                player_spices[3] += move_num * (card_new[3] - card_old[3])
            else:
                for spice in move.spice_upgrades:
                    player_spices[spice - 1] -= 1
                    player_spices[spice] += 1

            player.cards[move_playing].usable = False

        else:
            player_trader_cards = self.trader_cards
            card = player_trader_cards[move.draw_index]
            card_spices = card.spices
            player_cards = player.cards

            player_spices[0] += card_spices[0]
            player_spices[1] += card_spices[1]
            player_spices[2] += card_spices[2]
            player_spices[3] += card_spices[3]

            player_trader_cards.pop(move.draw_index)
            new_card = self.trader_card_stack.pop_random()
            player_trader_cards.append(FaceupTraderCard(new_card))

            player_cards[card.id] = PlayerCard(len(player_cards))

    def make_placement_move(
        self, player_id: int, placing: List[Union[Spice, None]]
    ) -> None:
        player_spices = self.players[player_id].spices
        player_trader_cards = self.trader_cards
        for i, spice in enumerate(placing):
            if spice is not None:
                player_trader_cards[i].spices[spice - 1] += 1
                player_spices[spice - 1] -= 1

    def make_upgrade_move(self, player_id: int, spices: List[Spice]) -> None:
        for spice in spices:
            self.players[player_id].spices[spice - 1] -= 1
            self.players[player_id].spices[spice] += 1

    def make_discard_move(self, player_id: int, discarding: SpiceCollection) -> None:
        player_spices = self.players[player_id].spices
        player_spices[0] -= discarding[0]
        player_spices[1] -= discarding[1]
        player_spices[2] -= discarding[2]
        player_spices[3] -= discarding[3]

from constants import *
from typing import cast, List
from cards import point_cards, trader_cards, UpgradeCard
from game import Game, Spice
from moves import PointMove, DrawMove, PlayMove
from animation import Animation
from graphics import Graphics
import pygame

game = Game()
graphics = Graphics()
canvas, clock = graphics.init()

quit = False
animation: Union[Animation, None] = None
state: Union[
    Literal["main"],
    Literal["num"],
    Literal["placement"],
    Literal["upgrade"],
    Literal["discard"],
] = "main"
active_card: Union[int, None] = None
draw_index: Union[int, None] = None
placement_index: Union[int, None] = None
upgraded_count: Union[int, None] = None
game_finished: bool = False

turn = 0


def discard_if_needed():
    global state

    if game.needs_to_discard(turn) and game.one_spice_type(turn):
        game.make_discard_move(turn, game.discard_simple(turn))
        state = "main"
    else:
        state = "discard" if game.needs_to_discard(turn) else "main"


while not quit:
    if state == "main" and not game_finished:
        game_moves = game.all_base_moves(turn, allow_all_placements=True)
    else:
        game_moves = []

    selecting_spices = turn if state in ["placement", "upgrade", "discard"] else None

    allowed_spices: List[Spice] = SPICES
    if state == "upgrade":
        allowed_spices: List[Spice] = [1, 2, 3]

    spice_positions, card_positions = graphics.get_rendering_positions(
        game,
        game_moves,
        selecting_spices=selecting_spices,
        allowed_spices=allowed_spices,
    )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True
        if not animation and not game_finished and event.type == pygame.MOUSEBUTTONUP:
            if state == "main":
                for id, position in card_positions.items():
                    if position.hovered:
                        if id in point_cards:
                            for i, card in enumerate(game.point_cards):
                                if card == id:
                                    index = i

                            move = PointMove(index, game.coin_bonus(index))
                            game_copy = game.copy()
                            game.make_base_move(turn, move)
                            animation = Animation(
                                game_copy, game, move, canvas, clock, graphics
                            )
                        elif id in PLAYER_REST:
                            move = PlayMove(PLAYER_REST[turn])
                            game_copy = game.copy()
                            game.make_base_move(turn, move)
                            animation = Animation(
                                game_copy, game, move, canvas, clock, graphics
                            )
                        elif id in game.players[turn].cards:
                            max_repeats = game.max_repeats(turn, id)
                            card = trader_cards[id]
                            if not isinstance(card, UpgradeCard) and max_repeats == 1:
                                move = PlayMove(id)
                                game_copy = game.copy()
                                game.make_base_move(turn, move)
                                discard_if_needed()
                                animation = Animation(
                                    game_copy, game, move, canvas, clock, graphics
                                )
                            else:
                                active_card = id
                                if isinstance(card, UpgradeCard):
                                    state = "upgrade"
                                    upgraded_count = 0
                                else:
                                    state = "num"

                        else:
                            index = 0
                            for i, card in enumerate(game.trader_cards):
                                if card.id == id:
                                    index = i

                            if index == 0 or game.one_spice_type(turn):
                                move = DrawMove(index)
                                game_copy = game.copy()
                                placement_type = game.only_spice_type(turn)
                                game.make_placement_move(turn, [placement_type] * index)
                                game.make_base_move(turn, move)
                                discard_if_needed()
                                animation = Animation(
                                    game_copy, game, move, canvas, clock, graphics
                                )
                            else:
                                state = "placement"
                                draw_index = index
                                placement_index = 0

                        break

            elif state == "num" and active_card is not None:
                max_repeats = game.max_repeats(turn, active_card)
                for position in graphics.get_num_choice_rendering_positions(
                    max_repeats
                ):
                    if position.hovered:
                        move = PlayMove(active_card, position.num)
                        game_copy = game.copy()
                        game.make_base_move(turn, move)
                        discard_if_needed()
                        animation = Animation(
                            game_copy, game, move, canvas, clock, graphics
                        )
                        break

            elif (
                state == "upgrade"
                and active_card is not None
                and upgraded_count is not None
            ):
                for spice in SPICES:
                    for position in spice_positions[spice]:
                        if position.hovered:
                            game_copy = game.copy()
                            card = cast(UpgradeCard, trader_cards[active_card])
                            upgraded_count += 1

                            move = PlayMove(active_card, spice_upgrades=[spice])
                            if upgraded_count == card.num_conversions:
                                game.make_base_move(turn, move)
                                state = "main"
                            else:
                                game.make_upgrade_move(turn, [spice])
                            animation = Animation(
                                game_copy, game, move, canvas, clock, graphics
                            )

            elif (
                state == "placement"
                and draw_index is not None
                and placement_index is not None
            ):
                for spice in SPICES:
                    for position in spice_positions[spice]:
                        if position.hovered:
                            game_copy = game.copy()
                            placement_index += 1

                            placing: List[Union[Spice, None]] = [
                                None for _ in range(placement_index - 1)
                            ]
                            placing.append(spice)
                            game.make_placement_move(turn, placing)

                            move = DrawMove(draw_index)
                            if placement_index == draw_index:
                                game.make_base_move(turn, move)
                                discard_if_needed()

                            animation = Animation(
                                game_copy, game, move, canvas, clock, graphics
                            )

            elif state == "discard":
                for spice in SPICES:
                    for position in spice_positions[spice]:
                        if position.hovered:
                            game_copy = game.copy()
                            game.make_discard_move(turn, game.discard_array(spice))
                            if game.needs_to_discard(turn) and game.one_spice_type(
                                turn
                            ):
                                game.make_discard_move(turn, game.discard_simple(turn))
                                state = "main"
                            else:
                                state = (
                                    "discard" if game.needs_to_discard(turn) else "main"
                                )
                            animation = Animation(
                                game_copy, game, move, canvas, clock, graphics
                            )

    if animation and not animation.finished:
        animation.update()
        animation.render()
    else:
        if animation:
            animation = None
            if state == "main":
                if game.finished:
                    game_finished = True
                turn = (turn + 1) % 2
                active_card, draw_index, placement_index, upgraded_count = [None] * 4

        graphics.render(game, canvas, (spice_positions, card_positions))
        if state == "num" and active_card is not None:
            discard_nums = game.discard_nums(turn, active_card)
            graphics.render_num_choice(
                active_card, game.max_repeats(turn, active_card), discard_nums, canvas
            )

    pygame.display.flip()
    clock.tick(FPS)

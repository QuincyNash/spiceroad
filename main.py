from constants import *
from typing import cast, List
from cards import point_cards, trader_cards, UpgradeCard
from game import Game, Spice
from moves import DrawMove, PartialDrawMove, PartialPlayMove, PlayMove
from animation import Animation
from graphics import Graphics
import pygame

game = Game()
graphics = Graphics()
canvas, clock = graphics.init()

animation: Union[Animation, None] = None
quit = False
partial_move: Union[PartialPlayMove, PartialDrawMove, None] = None
possibilities: Union[List[PlayMove], List[DrawMove], None] = None

turn = 0

while not quit:
    game_moves = game.all_moves(turn)
    selecting_spices = (
        turn
        if partial_move
        and (isinstance(partial_move, PartialDrawMove) or partial_move.num is not None)
        else None
    )

    allowed_spices: List[Spice] = SPICES
    if isinstance(partial_move, PartialPlayMove):
        allowed_spices: List[Spice] = [1, 2, 3]
    elif isinstance(partial_move, PartialDrawMove) and possibilities is not None:
        moves = [move for move in possibilities if isinstance(move, DrawMove)]
        spice_number = (
            len(partial_move.placing) if partial_move.placing is not None else 0
        )
        allowed_spices: List[Spice] = list(
            set(
                (move.placing[spice_number] if spice_number < len(move.placing) else 1)
                for move in moves
            )
        )  # type:ignore
        print(allowed_spices)

    spice_positions, card_positions = graphics.get_rendering_positions(
        game,
        game_moves if partial_move is None else [],
        selecting_spices=selecting_spices,
        allowed_spices=allowed_spices,
    )

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            quit = True
        if not animation and event.type == pygame.MOUSEBUTTONUP:
            if partial_move is not None and possibilities is not None:
                if (
                    isinstance(partial_move, PartialPlayMove)
                    and partial_move.num is None
                ):
                    positions = graphics.get_num_choice_rendering_positions(
                        [
                            poss.num
                            for poss in possibilities
                            if isinstance(poss, PlayMove)
                        ]
                    )
                    for position in positions:
                        if position.hovered:
                            game_copy = game.copy()
                            move = game.create_full_play_move(
                                partial_move, position.num
                            )
                            game.make_move(turn, move)
                            animation = Animation(
                                game_copy, game, move, canvas, clock, graphics
                            )
                elif isinstance(partial_move, PartialPlayMove):
                    for spice in SPICES:
                        for position in spice_positions[spice]:
                            if position.hovered:
                                game_copy = game.copy()
                                if not partial_move.spice_upgrades:
                                    partial_move.spice_upgrades = [spice]
                                else:
                                    partial_move.spice_upgrades.append(spice)

                                card = trader_cards[partial_move.playing]
                                is_partial = (
                                    card.num_conversions
                                    != len(partial_move.spice_upgrades)
                                    if isinstance(card, UpgradeCard)
                                    else False
                                )

                                move = game.create_upgrade_move(partial_move, spice)
                                game.make_move(turn, move, is_partial=is_partial)
                                animation = Animation(
                                    game_copy, game, move, canvas, clock, graphics
                                )
                elif isinstance(partial_move, PartialDrawMove):
                    for spice in SPICES:
                        for position in spice_positions[spice]:
                            if position.hovered:
                                draw_moves = [
                                    move
                                    for move in possibilities
                                    if isinstance(move, DrawMove)
                                ]
                                game_copy = game.copy()
                                if not partial_move.placing:
                                    partial_move.placing = [spice]
                                else:
                                    partial_move.placing.append(spice)

                                possibilities = [
                                    poss
                                    for poss in possibilities
                                    if isinstance(poss, DrawMove)
                                    and poss.placing[len(partial_move.placing) - 1]
                                    == partial_move.placing[-1]
                                ]

                                if len(possibilities) == 1:
                                    placing = cast(
                                        List[Spice], possibilities[0].placing
                                    )
                                    move = game.create_draw_move(
                                        partial_move,
                                        placing[len(partial_move.placing) - 1 :],
                                    )
                                    game.make_move(turn, move)
                                    animation = Animation(
                                        game_copy,
                                        game,
                                        move,
                                        canvas,
                                        clock,
                                        graphics,
                                    )
                                else:
                                    is_partial = len(draw_moves[0].placing) != len(
                                        partial_move.placing
                                    )

                                    move = game.create_draw_move(partial_move, [spice])
                                    game.make_move(turn, move, is_partial=is_partial)
                                    animation = Animation(
                                        game_copy, game, move, canvas, clock, graphics
                                    )
            else:
                for id, position in card_positions.items():
                    if position.hovered:
                        if id in point_cards:
                            index = 0
                            for i, card in enumerate(game.point_cards):
                                if card == id:
                                    index = i

                            move = game.find_point_move(index, game_moves)
                            if move:
                                game_copy = game.copy()
                                game.make_move(turn, move)
                                animation = Animation(
                                    game_copy, game, move, canvas, clock, graphics
                                )

                        elif id in game.players[turn].cards:
                            move = game.find_play_move(id, game_moves)
                            if isinstance(move, PlayMove):
                                game_copy = game.copy()
                                game.make_move(turn, move)
                                animation = Animation(
                                    game_copy, game, move, canvas, clock, graphics
                                )
                            elif move is not None:
                                partial_move, possibilities = move

                        elif id in PLAYER_REST:
                            move = game.create_rest_move(turn)
                            game_copy = game.copy()
                            game.make_move(turn, move)
                            animation = Animation(
                                game_copy, game, move, canvas, clock, graphics
                            )

                        else:
                            index = 0
                            for i, card in enumerate(game.trader_cards):
                                if card.id == id:
                                    index = i

                            move = game.find_draw_move(index, game_moves)
                            if isinstance(move, DrawMove):
                                game_copy = game.copy()
                                game.make_move(turn, move)
                                animation = Animation(
                                    game_copy, game, move, canvas, clock, graphics
                                )
                            else:
                                partial_move, possibilities = move

    if animation and not animation.finished:
        animation.update()
        animation.render()
    else:
        if animation:
            finished = True
            if (
                isinstance(partial_move, PartialPlayMove)
                and partial_move.spice_upgrades is not None
            ):
                card = trader_cards[partial_move.playing]
                if isinstance(card, UpgradeCard):
                    if card.num_conversions != len(partial_move.spice_upgrades):
                        finished = False
            if isinstance(partial_move, PartialDrawMove) and possibilities is not None:
                draw_moves = [
                    move for move in possibilities if isinstance(move, DrawMove)
                ]
                if partial_move.placing is None or len(partial_move.placing) != len(
                    draw_moves[0].placing
                ):
                    finished = False

            animation = None
            if finished:
                turn = (turn + 1) % 2
                partial_move, possibilities = None, None

        graphics.render(game, canvas, (spice_positions, card_positions))
        if (
            isinstance(partial_move, PartialPlayMove)
            and partial_move.num is None
            and possibilities is not None
        ):
            nums = [poss.num for poss in possibilities if isinstance(poss, PlayMove)]
            graphics.render_num_choice(partial_move.playing, nums, canvas)

    pygame.display.flip()
    clock.tick(FPS)

import random
from typing import List, Set, Tuple
from constants import *
from dataclasses import dataclass, field
from moves import DrawMove, Move, PointMove
from utils import Vector
from game import Game
from cards import (
    SpiceCollection,
    TradeCard,
    ObtainCard,
    UpgradeCard,
    point_cards,
    trader_cards,
)
import pygame


@dataclass
class RenderingPosition:
    pos: Vector
    coin_bonus: int = 0
    card_border: Tuple[int, int, int] = COLOR_BLACK
    shrink_factor: float = 1
    hovered: bool = False
    size: Vector = field(default_factory=lambda: Vector(CARD_SIZE, CARD_SIZE))


@dataclass
class NumChoiceRenderingPosition:
    pos: Vector
    num: int
    hovered: bool = False


SpiceRenderingPositions = Dict[
    Spice,
    List[RenderingPosition],
]
CardRenderingPositions = Dict[
    int,
    RenderingPosition,
]


class Graphics:
    def __init__(self) -> None:
        pygame.init()
        self.gold_coin = pygame.image.load("images/gold.png")
        self.gold_coin = pygame.transform.smoothscale(
            self.gold_coin, (COIN_SIZE, COIN_SIZE)
        )
        self.silver_coin = pygame.image.load("images/silver.png")
        self.silver_coin = pygame.transform.smoothscale(
            self.silver_coin, (COIN_SIZE, COIN_SIZE)
        )
        self.arrow = pygame.image.load("images/arrow.png")
        self.down_arrow = pygame.transform.smoothscale(
            self.arrow, (ARROW_SIZE, ARROW_SIZE)
        )
        self.up_arrow = pygame.transform.rotate(self.down_arrow, 180)
        self.font = pygame.font.SysFont("Arial", FONT_SIZE)
        self.big_font = pygame.font.SysFont("Arial", BIG_FONT_SIZE)

    def init(self) -> Tuple[pygame.Surface, pygame.time.Clock]:
        canvas = pygame.display.set_mode(
            (SCREEN_WIDTH, SCREEN_HEIGHT), pygame.DOUBLEBUF
        )
        clock = pygame.time.Clock()
        return canvas, clock

    def render_spice(
        self,
        canvas: pygame.Surface,
        spice: Spice,
        pos: Vector,
        shrink_factor: float = 1,
        opacity: float = 1,
        shaded=False,
    ) -> None:
        if shaded:
            factor = SPICE_OPACITY_FACTOR
            color = (
                round(COLOR_SPICES[spice][0] * factor + 255 * (1 - factor)),
                round(COLOR_SPICES[spice][1] * factor + 255 * (1 - factor)),
                round(COLOR_SPICES[spice][2] * factor + 255 * (1 - factor)),
            )
        else:
            color = COLOR_SPICES[spice]

        pygame.draw.rect(
            canvas,
            color + (round(255 * opacity),),
            (pos.x, pos.y, SPICE_SIZE / shrink_factor, SPICE_SIZE / shrink_factor),
        )

        if shaded:
            pygame.draw.rect(
                canvas,
                COLOR_BLACK,
                (pos.x, pos.y, SPICE_SIZE / shrink_factor, SPICE_SIZE / shrink_factor),
                1,
            )

    # Uses x and y as the center of the row
    def render_spice_row(
        self,
        canvas: pygame.Surface,
        spice_collection: SpiceCollection,
        pos: Vector,
        shrink_factor: float = 1,
        opacity: float = 1,
    ) -> None:
        spice_size = SPICE_SIZE / shrink_factor

        spices: List[Spice] = [
            SPICES[i] for i in range(3, -1, -1) for _ in range(spice_collection[i])
        ]

        row_width = len(spices) * spice_size + (len(spices) - 1) * SPICE_SPACING
        left_x = pos.x - row_width / 2
        for i, spice in enumerate(spices):
            x_pos = left_x + i * (spice_size + SPICE_SPACING)
            y_pos = pos.y - spice_size / 2
            self.render_spice(
                canvas, spice, Vector(x_pos, y_pos), shrink_factor, opacity
            )

    def render_trader_card(
        self,
        canvas: pygame.Surface,
        card_id: int,
        pos: Vector,
        shrink_factor: float = 1,
        border_color: Tuple[int, int, int] = COLOR_BLACK,
        opacity: float = 1,
        shaded: bool = False,
    ) -> None:
        card = trader_cards[card_id]
        card_size = CARD_SIZE / shrink_factor
        arrow_size = ARROW_SIZE / shrink_factor

        surface = pygame.Surface((card_size, card_size), pygame.SRCALPHA)
        surface.fill((255, 255, 255, 0))

        if shaded:
            pygame.draw.rect(
                surface,
                COLOR_BLACK + (round(HOVER_BLACK_OPACITY * opacity),),
                (0, 0, card_size, card_size),
            )
        pygame.draw.rect(
            surface,
            border_color + (round(255 * opacity),),
            (0, 0, card_size, card_size),
            1,
        )

        if isinstance(card, ObtainCard):
            self.render_spice_row(
                surface,
                card.new,
                Vector(card_size / 2, card_size / 2),
                shrink_factor,
                opacity,
            )
        elif isinstance(card, TradeCard):
            self.render_spice_row(
                surface,
                card.old,
                Vector(card_size / 2, card_size / 4),
                shrink_factor,
                opacity,
            )
            self.render_spice_row(
                surface,
                card.new,
                Vector(card_size / 2, 3 * card_size / 4),
                shrink_factor,
                opacity,
            )

            arrow = pygame.transform.smoothscale_by(self.down_arrow, 1 / shrink_factor)
            arrow.set_alpha(round(255 * opacity))
            surface.blit(
                arrow,
                (
                    (card_size - arrow_size) / 2,
                    (card_size - arrow_size) / 2,
                ),
            )
        else:
            left_x = (card_size - card.num_conversions * arrow_size) / 2
            for i in range(card.num_conversions):
                x_pos = left_x + i * arrow_size

                up_arrow = pygame.transform.smoothscale_by(
                    self.up_arrow, 1 / shrink_factor
                )
                up_arrow.set_alpha(round(255 * opacity))
                surface.blit(
                    up_arrow,
                    (x_pos, (card_size - arrow_size) / 2),
                )

        canvas.blit(surface, (pos.x, pos.y))

    def render_point_card(
        self,
        canvas: pygame.Surface,
        card_id: int,
        pos: Vector,
        coin_bonus: int = 0,
        border_color: Tuple[int, int, int] = COLOR_BLACK,
        opacity: float = 1,
        shaded=False,
    ) -> None:
        surface = pygame.Surface((CARD_SIZE, CARD_SIZE), pygame.SRCALPHA)
        surface.fill((255, 255, 255, 0))
        card = point_cards[card_id]

        if shaded:
            pygame.draw.rect(
                surface,
                COLOR_BLACK + (round(HOVER_BLACK_OPACITY * opacity),),
                (0, 0, CARD_SIZE, CARD_SIZE),
            )
        pygame.draw.rect(
            surface,
            border_color + (round(255 * opacity),),
            (0, 0, CARD_SIZE, CARD_SIZE),
            1,
        )
        self.render_spice_row(
            surface,
            card.spices,
            Vector(CARD_SIZE / 2, 2 * CARD_SIZE / 3),
            opacity=opacity,
        )
        text_str = (
            str(card.points) if coin_bonus == 0 else f"{card.points} (+{coin_bonus})"
        )
        text = self.font.render(text_str, True, COLOR_BLACK)
        text.set_alpha(round(255 * opacity))
        text_rect = text.get_rect(center=(CARD_SIZE / 2, CARD_SIZE / 3))
        surface.blit(text, text_rect)

        canvas.blit(surface, (pos.x, pos.y))

    def render_rest_button(
        self, canvas: pygame.Surface, pos: Vector, size: Vector, shaded: bool = False
    ):
        text = self.font.render("Rest", True, COLOR_BLACK)
        rect = (
            pos.x,
            pos.y,
            size.x,
            size.y,
        )
        if shaded:
            pygame.draw.rect(
                canvas,
                (
                    255 - HOVER_BLACK_OPACITY,
                    255 - HOVER_BLACK_OPACITY,
                    255 - HOVER_BLACK_OPACITY,
                ),
                rect,
            )
        pygame.draw.rect(canvas, COLOR_BLACK, rect, 1)
        canvas.blit(
            text,
            (
                pos.x + BORDER_SIZE / 4,
                pos.y + BORDER_SIZE / 4,
            ),
        )

    def get_rendering_positions(
        self,
        game: Game,
        game_moves: List[Move] = [],
        selecting_spices: Union[int, None] = None,
        allowed_spices: List[Spice] = SPICES,
    ) -> Tuple[SpiceRenderingPositions, CardRenderingPositions]:
        clickable_ids: Set[Union[Spice, int]] = set(
            (
                game.point_cards[move.buy_index]
                if isinstance(move, PointMove)
                else (
                    game.trader_cards[move.draw_index].id
                    if isinstance(move, DrawMove)
                    else move.playing
                )
            )
            for move in game_moves
        )
        mouse_x, mouse_y = pygame.mouse.get_pos()

        spice_positions: SpiceRenderingPositions = {
            1: [],
            2: [],
            3: [],
            4: [],
        }
        card_positions: CardRenderingPositions = {}

        for player_id, x in enumerate(
            [
                BORDER_SIZE
                + 2 * SPICE_SIZE
                + SPICE_SPACING
                + HORIZONTAL_SECTION_SPACING,
                SCREEN_WIDTH - BORDER_SIZE - CARD_GROUP_WIDTH,
            ]
        ):
            text = self.font.render("Rest", True, COLOR_BLACK)
            rect = text.get_rect()
            card_positions[PLAYER_REST[player_id]] = RenderingPosition(
                Vector(
                    x,
                    SCREEN_HEIGHT
                    - 5 * BORDER_SIZE / 4
                    - CARD_GROUP_HEIGHT
                    - VERTICAL_SECTION_SPACING
                    - rect.h / 2,
                ),
                size=Vector(rect.w + BORDER_SIZE / 2, rect.h + BORDER_SIZE / 2),
            )

        for i, id in enumerate(game.point_cards):
            x_pos = (
                SCREEN_WIDTH - BORDER_SIZE - CARD_SIZE - (CARD_SIZE + CARD_SPACING) * i
            )

            coin_bonus = game.coin_bonus(i)
            if coin_bonus == CARD_POINT_GOLD_BONUS:
                card_border = COLOR_GOLD
            elif coin_bonus == CARD_POINT_SILVER_BONUS:
                card_border = COLOR_SILVER
            else:
                card_border = COLOR_BLACK

            card_positions[id] = RenderingPosition(
                Vector(x_pos, BORDER_SIZE),
                coin_bonus=coin_bonus,
                card_border=card_border,
            )

        for i, card in enumerate(game.trader_cards):
            x_pos = (
                SCREEN_WIDTH - BORDER_SIZE - CARD_SIZE - (CARD_SIZE + CARD_SPACING) * i
            )
            y_pos = BORDER_SIZE + CARD_SIZE + VERTICAL_SECTION_SPACING
            card_positions[card.id] = RenderingPosition(Vector(x_pos, y_pos))

            spices: List[Spice] = [
                SPICES[i] for i in range(3, -1, -1) for _ in range(card.spices[i])
            ]

            for i, spice in enumerate(spices):
                y = y_pos + (SPICE_SIZE + SPICE_SPACING) * i + SPICE_SPACING
                spice_positions[spice].append(
                    RenderingPosition(Vector(x_pos + SPICE_SPACING, y))
                )

        for player_id, x_group in enumerate(
            [
                BORDER_SIZE
                + 2 * SPICE_SIZE
                + SPICE_SPACING
                + HORIZONTAL_SECTION_SPACING,
                SCREEN_WIDTH - BORDER_SIZE - CARD_GROUP_WIDTH,
            ]
        ):
            player = game.players[player_id]
            y_group = SCREEN_HEIGHT - BORDER_SIZE - CARD_GROUP_HEIGHT
            player_cards = sorted(
                player.cards.keys(), key=lambda id: player.cards[id].index
            )

            for y in range(CARD_GROUP_ROWS):
                for x in range(CARD_GROUP_COLS):
                    if CARD_GROUP_COLS * y + x < len(player.cards):
                        id = player_cards[CARD_GROUP_COLS * y + x]
                        x_pos = x_group + (CARD_SIZE / 2 + CARD_SPACING / 2) * x
                        y_pos = y_group + (CARD_SIZE / 2 + CARD_SPACING / 2) * y

                        border = COLOR_BLACK if player.cards[id].usable else COLOR_RED

                        card_positions[id] = RenderingPosition(
                            Vector(x_pos, y_pos),
                            card_border=border,
                            shrink_factor=2,
                        )

            spice_group_x = (
                x_group - 2 * SPICE_SIZE - SPICE_SPACING - HORIZONTAL_SECTION_SPACING
            )
            spices: List[Spice] = [
                SPICES[i] for i in range(3, -1, -1) for _ in range(player.spices[i])
            ]

            for i, spice in enumerate(spices):
                x_pos = spice_group_x + (SPICE_SIZE + SPICE_SPACING) * (i % 2)
                y_pos = y_group + (SPICE_SIZE + SPICE_SPACING) * (i // 2)
                spice_positions[spice].append(
                    RenderingPosition(
                        Vector(x_pos, y_pos),
                        hovered=(
                            False
                            if selecting_spices is None
                            else (
                                player_id == selecting_spices
                                and spice in allowed_spices
                            )
                        ),
                    )
                )

        if selecting_spices is not None:
            for spice in allowed_spices:
                for pos in spice_positions[spice]:
                    if pos.hovered and not (
                        pos.pos.x < mouse_x < pos.pos.x + SPICE_SIZE
                        and pos.pos.y < mouse_y < pos.pos.y + SPICE_SIZE
                    ):
                        pos.hovered = False
        else:
            for card_id, pos in card_positions.items():
                if (
                    pos.pos.x < mouse_x < pos.pos.x + pos.size.x / pos.shrink_factor
                    and pos.pos.y < mouse_y < pos.pos.y + pos.size.y / pos.shrink_factor
                    and card_id in clickable_ids
                ):
                    card_positions[card_id].hovered = True

        return spice_positions, card_positions

    def render_supplemental(self, game: Game, canvas: pygame.Surface):
        text_str = f"{game.player1.points} - {game.player2.points}"
        text = self.big_font.render(text_str, True, COLOR_BLACK)
        text_rect = text.get_rect(
            center=(BORDER_SIZE + CARD_SIZE / 2, BORDER_SIZE + CARD_SIZE / 4)
        )
        canvas.blit(text, text_rect)

        gold_rect = self.gold_coin.get_rect(
            center=(BORDER_SIZE + CARD_SIZE / 4, BORDER_SIZE + 3 * CARD_SIZE / 4)
        )
        canvas.blit(self.gold_coin, gold_rect)
        gold_text = self.font.render(str(game.coins[0]), True, COLOR_BLACK)
        gold_text_rect = gold_text.get_rect(
            center=(BORDER_SIZE + CARD_SIZE / 4, BORDER_SIZE + 3 * CARD_SIZE / 4)
        )
        canvas.blit(gold_text, gold_text_rect)

        silver_rect = self.silver_coin.get_rect(
            center=(BORDER_SIZE + 3 * CARD_SIZE / 4, BORDER_SIZE + 3 * CARD_SIZE / 4)
        )
        canvas.blit(self.silver_coin, silver_rect)
        silver_text = self.font.render(str(game.coins[1]), True, COLOR_BLACK)
        silver_text_rect = silver_text.get_rect(
            center=(BORDER_SIZE + 3 * CARD_SIZE / 4, BORDER_SIZE + 3 * CARD_SIZE / 4)
        )
        canvas.blit(silver_text, silver_text_rect)

        for i, x_pos in enumerate(
            [
                BORDER_SIZE
                + 2 * SPICE_SIZE
                + SPICE_SPACING
                + HORIZONTAL_SECTION_SPACING,
                SCREEN_WIDTH - BORDER_SIZE - CARD_GROUP_WIDTH,
            ]
        ):
            text_str = f"{game.players[i].point_cards_bought} point {'card' if game.players[i].point_cards_bought else 'cards'}"
            text = self.font.render(text_str, True, COLOR_BLACK)
            rest_rect = self.font.render("Rest", True, COLOR_BLACK).get_rect()
            canvas.blit(
                text,
                (
                    x_pos + rest_rect.w + BORDER_SIZE / 2 + CARD_COUNT_SPACING,
                    SCREEN_HEIGHT
                    - BORDER_SIZE
                    - CARD_GROUP_HEIGHT
                    - VERTICAL_SECTION_SPACING
                    - rest_rect.h / 2,
                ),
            )

    def render(
        self,
        game: Game,
        canvas: pygame.Surface,
        rendering_positions: Tuple[SpiceRenderingPositions, CardRenderingPositions],
    ) -> None:
        canvas.fill(COLOR_WHITE)
        self.render_supplemental(game, canvas)

        spice_positions, card_positions = rendering_positions

        for id, position in card_positions.items():
            if id in point_cards:
                self.render_point_card(
                    canvas,
                    id,
                    Vector(position.pos.x, position.pos.y),
                    coin_bonus=position.coin_bonus,
                    border_color=position.card_border,
                    shaded=position.hovered,
                )
            elif id in trader_cards:
                self.render_trader_card(
                    canvas,
                    id,
                    Vector(position.pos.x, position.pos.y),
                    shrink_factor=position.shrink_factor,
                    border_color=position.card_border,
                    shaded=position.hovered,
                )
            else:
                self.render_rest_button(
                    canvas,
                    Vector(position.pos.x, position.pos.y),
                    Vector(position.size.x, position.size.y),
                    shaded=position.hovered,
                )

        for spice in SPICES:
            for position in spice_positions[spice]:
                self.render_spice(
                    canvas,
                    spice,
                    Vector(position.pos.x, position.pos.y),
                    position.shrink_factor,
                    shaded=position.hovered,
                )

    def get_num_choice_rendering_positions(
        self, nums: List[int]
    ) -> List[NumChoiceRenderingPosition]:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        positions: List[NumChoiceRenderingPosition] = []

        for i, num in enumerate(nums):
            x_pos = (
                SCREEN_WIDTH - len(nums) * (CARD_SIZE) - (len(nums) - 1) * CARD_SPACING
            ) / 2 + (CARD_SIZE + CARD_SPACING) * i
            y_pos = (SCREEN_HEIGHT - CARD_SIZE) / 2
            hovered = (
                x_pos < mouse_x < x_pos + CARD_SIZE
                and y_pos < mouse_y < y_pos + CARD_SIZE
            )

            positions.append(
                NumChoiceRenderingPosition(Vector(x_pos, y_pos), num, hovered)
            )

        return positions

    def render_num_choice(
        self, playing: int, nums: List[int], canvas: pygame.Surface
    ) -> None:
        surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        surface.fill((255, 255, 255, 0))

        pygame.draw.rect(
            surface,
            COLOR_BLACK + (NUM_CHOICE_BLACK_OPACITY,),
            (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT),
        )

        for pos in self.get_num_choice_rendering_positions(nums):
            num, pos, hovered = pos.num, pos.pos, pos.hovered

            if hovered:
                pygame.draw.rect(
                    surface,
                    (
                        255 - HOVER_BLACK_OPACITY,
                        255 - HOVER_BLACK_OPACITY,
                        255 - HOVER_BLACK_OPACITY,
                    ),
                    (pos.x, pos.y, CARD_SIZE, CARD_SIZE),
                )
            else:
                pygame.draw.rect(
                    surface, COLOR_WHITE, (pos.x, pos.y, CARD_SIZE, CARD_SIZE)
                )

            pygame.draw.rect(
                surface, COLOR_BLACK, (pos.x, pos.y, CARD_SIZE, CARD_SIZE), 1
            )

            self.render_trader_card(
                surface,
                playing,
                Vector(pos.x + CARD_SIZE / 4, pos.y + CARD_SIZE / 16),
                2,
            )
            text = self.font.render(f"x{num}", True, COLOR_BLACK)
            text_rect = text.get_rect(
                center=(pos.x + CARD_SIZE / 2, pos.y + 12 * CARD_SIZE / 16)
            )
            surface.blit(text, text_rect)

        canvas.blit(surface, (0, 0))

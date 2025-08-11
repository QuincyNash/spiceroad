from typing import Tuple, Union
from constants import *
from dataclasses import dataclass, field
from utils import Vector
from game import Game
from cards import point_cards, trader_cards
from moves import Move
from graphics import Graphics
import pygame


@dataclass(slots=True)
class AnimatingObject:
    id: Union[Spice, int]
    old_pos: Vector
    new_pos: Vector
    old_shrink_factor: float = 1
    new_shrink_factor: float = 1
    old_border: Tuple[int, int, int] = COLOR_BLACK
    new_border: Tuple[int, int, int] = COLOR_BLACK
    fade_in: bool = False
    fade_out: bool = False
    coin_bonus: int = 0
    movement: Vector = field(default_factory=lambda: Vector(0, 0))
    pos: Vector = field(default_factory=lambda: Vector(0, 0))
    size: Vector = field(default_factory=lambda: Vector(0, 0))
    shrink_factor: float = 1
    border: Tuple[float, float, float] = COLOR_BLACK
    opacity: float = 1
    opacity_change: float = 0
    shrink_factor_change: float = 0
    border_change: Tuple[float, float, float] = (0, 0, 0)

    def __post_init__(self) -> None:
        self.pos = self.old_pos.copy()
        self.border = self.old_border
        self.shrink_factor = self.old_shrink_factor

        self.movement = (self.new_pos - self.old_pos) / (ANIMATION_SECONDS * FPS)
        self.shrink_factor_change = (
            self.new_shrink_factor - self.old_shrink_factor
        ) / (ANIMATION_SECONDS * FPS)
        self.border_change = (
            (self.new_border[0] - self.old_border[0]) / (ANIMATION_SECONDS * FPS),
            (self.new_border[1] - self.old_border[1]) / (ANIMATION_SECONDS * FPS),
            (self.new_border[2] - self.old_border[2]) / (ANIMATION_SECONDS * FPS),
        )

        if self.fade_in:
            self.opacity = 0
            self.opacity_change = 1 / (ANIMATION_SECONDS * FPS)
        elif self.fade_out:
            self.opacity_change = -1 / (ANIMATION_SECONDS * FPS)


class Animation:
    def __init__(
        self,
        old_game: Game,
        new_game: Game,
        move: Move,
        canvas: pygame.Surface,
        clock: pygame.time.Clock,
        graphics: Graphics,
    ) -> None:
        self.old_game = old_game
        self.new_game = new_game
        self.move = move
        self.canvas = canvas
        self.clock = clock
        self.graphics = graphics
        self.frame = 0
        self.finished = False

        self.animated_objects: List[AnimatingObject] = []
        old_spice_positions, old_card_positions = self.graphics.get_rendering_positions(
            old_game
        )
        new_spice_positions, new_card_positions = self.graphics.get_rendering_positions(
            new_game
        )

        for id in old_card_positions:
            if id in new_card_positions:
                self.animated_objects.append(
                    AnimatingObject(
                        id,
                        old_card_positions[id].pos,
                        new_card_positions[id].pos,
                        size=old_card_positions[id].size,
                        old_border=old_card_positions[id].card_border,
                        new_border=new_card_positions[id].card_border,
                        old_shrink_factor=old_card_positions[id].shrink_factor,
                        new_shrink_factor=new_card_positions[id].shrink_factor,
                        coin_bonus=old_card_positions[id].coin_bonus,
                        fade_out=False,
                    )
                )
            else:
                self.animated_objects.append(
                    AnimatingObject(
                        id,
                        old_card_positions[id].pos,
                        old_card_positions[id].pos,
                        size=old_card_positions[id].size,
                        old_border=old_card_positions[id].card_border,
                        new_border=old_card_positions[id].card_border,
                        old_shrink_factor=old_card_positions[id].shrink_factor,
                        new_shrink_factor=old_card_positions[id].shrink_factor,
                        coin_bonus=old_card_positions[id].coin_bonus,
                        fade_out=True,
                    )
                )

        for id in new_card_positions:
            if id not in old_card_positions:
                self.animated_objects.append(
                    AnimatingObject(
                        id,
                        new_card_positions[id].pos,
                        new_card_positions[id].pos,
                        size=new_card_positions[id].size,
                        old_border=new_card_positions[id].card_border,
                        new_border=new_card_positions[id].card_border,
                        old_shrink_factor=new_card_positions[id].shrink_factor,
                        new_shrink_factor=new_card_positions[id].shrink_factor,
                        coin_bonus=new_card_positions[id].coin_bonus,
                        fade_in=True,
                    )
                )

        for spice in SPICES:
            for position in old_spice_positions[spice]:
                fade_out = position not in new_spice_positions[spice]
                self.animated_objects.append(
                    AnimatingObject(
                        spice,
                        old_pos=position.pos,
                        new_pos=position.pos,
                        fade_out=fade_out,
                    )
                )

            for position in new_spice_positions[spice]:
                if position not in old_spice_positions[spice]:
                    self.animated_objects.append(
                        AnimatingObject(
                            spice,
                            old_pos=position.pos,
                            new_pos=position.pos,
                            fade_in=True,
                        )
                    )

    def render(self):
        self.canvas.fill(COLOR_WHITE)
        self.graphics.render_supplemental(self.old_game, self.canvas)

        for object in self.animated_objects:
            rounded_border = (
                round(object.border[0]),
                round(object.border[1]),
                round(object.border[2]),
            )
            if object.id in SPICES:
                surface = pygame.Surface(
                    (
                        SPICE_SIZE / object.shrink_factor,
                        SPICE_SIZE / object.shrink_factor,
                    ),
                    pygame.SRCALPHA,
                )
                self.graphics.render_spice(
                    surface,
                    object.id,
                    Vector(0, 0),
                    object.shrink_factor,
                    object.opacity,
                )
                self.canvas.blit(surface, (object.pos.x, object.pos.y))
            elif object.id in point_cards:
                self.graphics.render_point_card(
                    self.canvas,
                    object.id,
                    object.pos,
                    object.coin_bonus,
                    rounded_border,
                    object.opacity,
                )
            elif object.id in trader_cards:
                self.graphics.render_trader_card(
                    self.canvas,
                    object.id,
                    object.pos,
                    object.shrink_factor,
                    rounded_border,
                    object.opacity,
                )
            else:
                self.graphics.render_rest_button(self.canvas, object.pos, object.size)

    def update(self):
        for object in self.animated_objects:
            object.pos += object.movement
            object.shrink_factor += object.shrink_factor_change
            object.opacity += object.opacity_change
            object.border = (
                object.border[0] + object.border_change[0],
                object.border[1] + object.border_change[1],
                object.border[2] + object.border_change[2],
            )

        self.frame += 1
        if self.frame == ANIMATION_SECONDS * FPS:
            self.finished = True

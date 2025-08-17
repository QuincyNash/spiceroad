from __future__ import annotations
from dataclasses import dataclass
from constants import *
import random


@dataclass(slots=True)
class Vector:
    x: float
    y: float

    def copy(self) -> Vector:
        return Vector(self.x, self.y)

    def __add__(self, other: Vector) -> Vector:
        return Vector(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vector) -> Vector:
        return Vector(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> Vector:
        return Vector(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> Vector:
        return Vector(self.x / other, self.y / other)


class RandomizedSet(object):
    def __init__(self, items: List[int]):
        self.item_to_position: Dict[int, int] = {}
        self.items: List[int] = []

        for item in items:
            self.add(item)

    def __contains__(self, item: int):
        return item in self.item_to_position

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)

    def __str__(self):
        return self.items.__str__()

    def copy(self) -> RandomizedSet:
        new = RandomizedSet.__new__(RandomizedSet)
        new.item_to_position = self.item_to_position.copy()
        new.items = self.items[:]
        return new

    def add(self, item: int):
        if item in self.item_to_position:
            return
        self.items.append(item)
        self.item_to_position[item] = len(self.items) - 1

    def remove(self, item: int) -> None:
        position = self.item_to_position.pop(item)
        last_item = self.items.pop()
        if position != len(self.items):
            self.items[position] = last_item
            self.item_to_position[last_item] = position

    def remove_many(self, items: List[int]) -> None:
        for item in items:
            self.remove(item)

    def pop_random(self) -> int:
        index = random.randint(0, len(self.items) - 1)
        items = self.items
        choice = items[index]

        last_item = items.pop()
        if index != len(items):
            items[index] = last_item
            self.item_to_position[last_item] = index

        del self.item_to_position[choice]

        return choice


def can_buy(owned: SpiceCollection, buying: SpiceCollection) -> bool:
    return (
        owned[0] >= buying[0]
        and owned[1] >= buying[1]
        and owned[2] >= buying[2]
        and owned[3] >= buying[3]
    )


def floor_div(a: SpiceCollection, b: SpiceCollection) -> int:
    m = 127
    if b[0] != 0:
        q = a[0] // b[0]
        if q < m:
            m = q
    if b[1] != 0:
        q = a[1] // b[1]
        if q < m:
            m = q
    if b[2] != 0:
        q = a[2] // b[2]
        if q < m:
            m = q
    if b[3] != 0:
        q = a[3] // b[3]
        if q < m:
            m = q
    return m if m != 127 else 0


def add_array_spices(a: SpiceCollection, b: SpiceCollection) -> None:
    a[0] += b[0]
    a[1] += b[1]
    a[2] += b[2]
    a[3] += b[3]

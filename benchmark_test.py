from constants import *
from pytest_benchmark.fixture import BenchmarkFixture as Benchmark
from game import Game, PlayerCard
from moves import PlayMove
from graphics import Graphics
from array import array
from utils import floor_div


def test_game_test(benchmark: Benchmark):
    game = Game()
    # for card_id in range(9, 30):
    #     game.player2.cards[card_id] = PlayerCard(0)

    # game.all_placement_moves(0, 0)

    benchmark(game.all_placement_moves, 1, 0)


# def test_game_moves(benchmark: Benchmark):
#     game = Game()
#     benchmark(game.all_moves, 0)


# def test_copy_game_and_make_move(benchmark: Benchmark):
#     game = Game()
#     move = game.all_moves(0)[0]

#     def run():
#         game.copy().make_move(0, move)

#     benchmark(run)


# def test_game_copy(benchmark: Benchmark):
#     game = Game()
#     benchmark(game.copy)


# def test_rendering(benchmark: Benchmark):
#     game = Game()
#     graphics = Graphics()
#     benchmark(graphics.get_rendering_positions, game)

game = Game()
game.player1.spices = array("b", (3, 3, 2, 2))
print(game.all_placement_moves(0, 3))

# total = 0
# for m in range(1, 6):
#     for a in range(m + 1):
#         for b in range(m + 1 - a):
#             for c in range(m + 1 - a - b):
#                 d = m - a - b - c
#                 print(a, b, c, d)
#                 total += 1

# print(total)

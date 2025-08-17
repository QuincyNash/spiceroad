from constants import *
from pytest_benchmark.fixture import BenchmarkFixture as Benchmark
from game import Game
from array import array


def test_game_base_moves(benchmark: Benchmark):
    game = Game()
    benchmark(game.all_base_moves, 0)


def test_game_placement_moves(benchmark: Benchmark):
    game = Game()
    benchmark(game.all_placement_moves, 0, 5)


def test_game_discard_moves(benchmark: Benchmark):
    game = Game()
    benchmark(game.all_discard_moves, 0)


def test_make_base_move(benchmark: Benchmark):
    game = Game()
    games = [game.copy() for _ in range(100000)]
    games_iter = iter(games)
    move = game.all_base_moves(0)[4]

    def run(g: Game):
        g.make_base_move(0, move)

    benchmark.pedantic(
        lambda: run(next(games_iter)), rounds=len(games) - 100, warmup_rounds=100
    )


def test_make_placement_move(benchmark: Benchmark):
    game = Game()
    games = [game.copy() for _ in range(100000)]
    games_iter = iter(games)
    placement = [None, 1, 2, 3, 4]

    def run(g: Game):
        g.make_placement_move(0, placement)

    benchmark.pedantic(
        lambda: run(next(games_iter)), rounds=len(games) - 100, warmup_rounds=100
    )


def test_make_discard_move(benchmark: Benchmark):
    game = Game()
    games = [game.copy() for _ in range(100000)]
    games_iter = iter(games)
    discarding = array("b", (1, 1, 1, 1))

    def run(g: Game):
        g.make_discard_move(0, discarding)

    benchmark.pedantic(
        lambda: run(next(games_iter)), rounds=len(games) - 100, warmup_rounds=100
    )


def test_game_copy(benchmark: Benchmark):
    game = Game()
    benchmark(game.copy)


game = Game()
move = game.all_base_moves(0)[4]
print(move)

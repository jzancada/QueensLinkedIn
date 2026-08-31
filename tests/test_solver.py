"""Solver tests: the rules it enforces, and the trace it produces while doing it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from queens.board import Board, Mark
from queens.solver import (StepKind, conflict, is_solution, render, solve,
                           solve_steps)
from queens.vision import detect_file

DOC = Path(__file__).resolve().parent.parent / "doc"

# A queen fits in every quadrant without any two touching: the smallest board
# that exercises all four rules at once.
FOUR_QUADRANTS = ["AABB",
                  "AABB",
                  "CCDD",
                  "CCDD"]

# Region A wraps around the corner, so it holds two cells that are neither in
# the same column nor touching: the only way to isolate the region rule.
L_SHAPED = ["ABBB",
            "ABBB",
            "AAAC",
            "DDDC"]

# Regions A and B live inside column 0, so both need that column and neither
# can have it. Unsolvable for a reason that has nothing to do with the size.
TWO_REGIONS_ONE_COLUMN = ["ACCC",
                          "ACCC",
                          "BDDD",
                          "BDDD"]


def make_board(rows: list[str]) -> Board:
    """Build a board from region letters, one string per row."""
    letters = sorted({ch for row in rows for ch in row})
    region = np.array([[letters.index(ch) for ch in row] for row in rows], dtype=np.int32)
    return Board(n=len(rows), region=region, colors=[(0, 0, 0)] * len(letters))


def test_solves_a_small_board():
    board = make_board(FOUR_QUADRANTS)
    queens = solve(board)

    assert queens is not None
    assert is_solution(board, queens)
    assert sorted(queens) == list(range(4))                  # one per column
    assert len({int(board.region[r, c]) for r, c in enumerate(queens)}) == 4


def test_solves_the_detected_board():
    """The real point of the solver: the board that came out of the screenshot."""
    result = detect_file(DOC / "Example3.png", debug=False)
    assert result.ok, result.error

    queens = solve(result.board)
    assert queens is not None
    assert is_solution(result.board, queens)


def test_an_unsolvable_board_is_reported_not_faked():
    board = make_board(TWO_REGIONS_ONE_COLUMN)
    assert solve(board) is None

    steps = list(solve_steps(board))
    assert steps[-1].kind is StepKind.EXHAUSTED
    assert not any(s.kind is StepKind.SOLVED for s in steps)


def test_queens_may_share_a_diagonal_at_distance_two():
    """The rule that separates this from the classic N-queens problem."""
    board = make_board(FOUR_QUADRANTS)

    assert conflict(board, [0], row=1, col=1) is not None        # touching
    assert conflict(board, [0], row=2, col=2) is None            # same diagonal, far


@pytest.mark.parametrize("row, col, expected", [
    (1, 0, "column"),        # straight below the queen at (0,0)
    (1, 1, "touches"),       # diagonal neighbour
    (2, 2, "region"),        # region A again, two rows away
])
def test_every_rejection_names_the_rule_it_broke(row, col, expected):
    """The reason is what the solver panel displays, so it must be specific."""
    board = make_board(L_SHAPED)
    reason = conflict(board, [0], row, col)

    assert reason is not None and expected in reason


def test_the_trace_can_be_replayed_onto_the_board():
    """Each step carries the state after it, which is what lets the panel animate.

    Replaying only the snapshots must reproduce the search exactly, with no
    bookkeeping on the consumer's side.
    """
    board = make_board(FOUR_QUADRANTS)
    steps = list(solve_steps(board))

    assert steps[0].kind is StepKind.TRY
    assert steps[-1].kind is StepKind.SOLVED
    assert any(s.kind is StepKind.BACKTRACK for s in steps)      # it does search

    for step in steps:
        if step.kind is StepKind.PLACE:
            assert step.queens[-1] == step.col
            assert len(step.queens) == step.row + 1
            # Nothing is ever placed on a cell that breaks a rule.
            assert conflict(board, step.queens[:-1], step.row, step.col) is None
        elif step.kind in (StepKind.TRY, StepKind.REJECT, StepKind.BACKTRACK):
            assert len(step.queens) == step.row                  # the row is open

    assert is_solution(board, steps[-1].queens)


def test_the_solution_can_be_dropped_on_the_board():
    board = make_board(FOUR_QUADRANTS)
    queens = solve(board)

    board.set_queens(queens)
    assert int((board.marks == Mark.QUEEN).sum()) == board.n
    for row, col in enumerate(queens):
        assert board.marks[row, col] == Mark.QUEEN


def test_render_shows_the_regions_and_the_queens():
    board = make_board(FOUR_QUADRANTS)
    assert render(board).splitlines()[0] == "A A B B"

    text = render(board, solve(board))
    assert text.count("Q") == board.n

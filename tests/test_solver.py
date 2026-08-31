"""Solver tests: the rules it enforces, and the trace it produces while doing it."""

from __future__ import annotations

from pathlib import Path

import pytest

from queens.board import Mark
from queens.solver import (StepKind, conflict, is_solution, render, solve,
                           solve_steps, stranded)
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

# Region C is the single cell (3,3): any queen in column 3 strands it, which is
# the cheapest possible case for the look-ahead to catch.
ONE_CELL_REGION = ["ABBB",
                   "ABBB",
                   "AABB",
                   "DDDC"]

# Region A swallows the whole of column 0: once its queen goes somewhere else,
# that column has nowhere left to put one.
COLUMN_INSIDE_A_REGION = ["AABB",
                          "AABB",
                          "ACCB",
                          "ACDD"]

# Regions A and B live inside column 0, so both need that column and neither
# can have it. Unsolvable for a reason that has nothing to do with the size.
TWO_REGIONS_ONE_COLUMN = ["ACCC",
                          "ACCC",
                          "BDDD",
                          "BDDD"]


def test_solves_a_small_board(make_board):
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


def test_an_unsolvable_board_is_reported_not_faked(make_board):
    board = make_board(TWO_REGIONS_ONE_COLUMN)
    assert solve(board) is None

    steps = list(solve_steps(board))
    assert steps[-1].kind is StepKind.EXHAUSTED
    assert not any(s.kind is StepKind.SOLVED for s in steps)


def test_queens_may_share_a_diagonal_at_distance_two(make_board):
    """The rule that separates this from the classic N-queens problem."""
    board = make_board(FOUR_QUADRANTS)

    assert conflict(board, [0], row=1, col=1) is not None        # touching
    assert conflict(board, [0], row=2, col=2) is None            # same diagonal, far


@pytest.mark.parametrize("row, col, expected", [
    (1, 0, "column"),        # straight below the queen at (0,0)
    (1, 1, "touches"),       # diagonal neighbour
    (2, 2, "region"),        # region A again, two rows away
])
def test_every_rejection_names_the_rule_it_broke(row, col, expected, make_board):
    """The reason is what the solver panel displays, so it must be specific."""
    board = make_board(L_SHAPED)
    reason = conflict(board, [0], row, col)

    assert reason is not None and expected in reason


def test_the_trace_can_be_replayed_onto_the_board(make_board):
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
        elif step.kind in (StepKind.TRY, StepKind.REJECT,
                           StepKind.PRUNE, StepKind.BACKTRACK):
            assert len(step.queens) == step.row                  # the row is open

    assert is_solution(board, steps[-1].queens)


def test_pruning_does_not_change_the_answer(make_board):
    """A prune may only remove branches that hold no solution.

    So the first solution in depth-first order is the very same one, and this
    is the test that would catch a look-ahead that cuts too much.
    """
    board = make_board(FOUR_QUADRANTS)
    assert solve(board, prune=True) == solve(board, prune=False)

    result = detect_file(DOC / "Example3.png", debug=False)
    assert result.ok, result.error
    assert solve(result.board, prune=True) == solve(result.board, prune=False)


def test_pruning_cuts_the_search_by_an_order_of_magnitude():
    """The reason the look-ahead exists: a watchable trace, not a faster answer."""
    result = detect_file(DOC / "Example3.png", debug=False)
    assert result.ok, result.error

    def attempts(prune):
        return sum(s.kind is StepKind.TRY for s in solve_steps(result.board, prune))

    assert attempts(prune=True) * 10 < attempts(prune=False)


def test_a_stranded_region_is_spotted_before_the_search_finds_out(make_board):
    """Region C is a single cell, so one queen in its column strands it."""
    board = make_board(ONE_CELL_REGION)

    assert stranded(board, []) is None
    assert stranded(board, [0]) is None                  # C still reachable
    assert "region 2" in (stranded(board, [3]) or "")    # C is the third letter


def test_a_stranded_column_is_spotted_too(make_board):
    """The dual of the region check, and it falls out of the same sweep.

    Region A owns the whole of column 0, so the queen that takes A anywhere
    else leaves that column with no cell it could still use.
    """
    board = make_board(COLUMN_INSIDE_A_REGION)
    assert stranded(board, []) is None
    assert "column 0" in (stranded(board, [1]) or "")


def test_a_prune_reads_as_a_queen_placed_and_withdrawn():
    """That is how the panel will show it: the move is legal, its future is not."""
    result = detect_file(DOC / "Example3.png", debug=False)
    assert result.ok, result.error

    steps = list(solve_steps(result.board, prune=True))
    prunes = [i for i, s in enumerate(steps) if s.kind is StepKind.PRUNE]
    assert prunes

    for i in prunes:
        step = steps[i]
        assert step.reason.split()[0] in ("region", "column")
        assert len(step.queens) == step.row                     # the queen is gone
        assert steps[i - 1].kind is StepKind.PLACE              # it had just landed
        assert steps[i - 1].queens[-1] == step.col


def test_the_solution_can_be_dropped_on_the_board(make_board):
    board = make_board(FOUR_QUADRANTS)
    queens = solve(board)

    board.set_queens(queens)
    assert int((board.marks == Mark.QUEEN).sum()) == board.n
    for row, col in enumerate(queens):
        assert board.marks[row, col] == Mark.QUEEN


def test_render_shows_the_regions_and_the_queens(make_board):
    board = make_board(FOUR_QUADRANTS)
    assert render(board).splitlines()[0] == "A A B B"

    text = render(board, solve(board))
    assert text.count("Q") == board.n

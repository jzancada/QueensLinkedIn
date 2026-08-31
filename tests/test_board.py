"""Board tests: the rules as the player breaks them, not as the solver avoids them.

The solver never produces an illegal state, so these rules only ever get
exercised by a board being played by hand — which is exactly why they are
tested apart from it.
"""

from __future__ import annotations

import numpy as np

from queens.board import Mark

QUADRANTS = ["AABB",
             "AABB",
             "CCDD",
             "CCDD"]


def test_a_board_with_no_queens_has_nothing_to_complain_about(make_board):
    board = make_board(QUADRANTS)

    assert board.queens() == []
    assert board.conflicts() == []
    assert not board.is_solved()            # empty is not solved, it is unstarted


def test_every_rule_is_reported_by_name(make_board):
    """The caption shows this text, so it has to say which rule broke."""
    checks = [
        ((0, 0), (0, 3), "row 0 has two queens"),
        ((0, 0), (3, 0), "column 0 has two queens"),
        ((0, 0), (1, 1), "two queens touch"),          # diagonal neighbours
        ((0, 0), (1, 1), "two queens touch"),
        ((0, 1), (1, 0), "two queens touch"),
    ]
    for first, second, expected in checks:
        board = make_board(QUADRANTS)
        board.marks[first] = board.marks[second] = Mark.QUEEN

        conflicts = board.conflicts()
        assert len(conflicts) == 1
        assert conflicts[0][2] == expected


def test_two_queens_in_one_region_are_caught(make_board):
    """Region A is the top-left quadrant; (0,0) and (1,1) touch, (0,0) and
    (1,0) share a column — so the region rule needs a board where it is the
    only thing broken."""
    board = make_board(["AAAB",
                        "BBAB",
                        "CCDD",
                        "CCDD"])
    board.marks[0, 0] = board.marks[1, 2] = Mark.QUEEN

    conflicts = board.conflicts()
    assert len(conflicts) == 1
    assert "region 0 has two queens" in conflicts[0][2]


def test_queens_on_the_same_diagonal_are_fine_two_apart(make_board):
    board = make_board(QUADRANTS)
    board.marks[0, 0] = board.marks[2, 2] = Mark.QUEEN

    assert board.conflicts() == []


def test_a_full_legal_board_is_solved(make_board):
    board = make_board(QUADRANTS)
    board.set_queens([1, 3, 0, 2])

    assert board.is_solved()
    assert board.queens() == [(0, 1), (1, 3), (2, 0), (3, 2)]


def test_a_full_board_with_a_conflict_is_not_solved(make_board):
    """Four queens are not enough: they have to be four legal ones."""
    board = make_board(QUADRANTS)
    board.set_queens([0, 2, 0, 2])

    assert len(board.queens()) == board.n
    assert not board.is_solved()


def test_crosses_are_not_queens(make_board):
    board = make_board(QUADRANTS)
    board.marks[0, 0] = Mark.CROSS
    board.marks[1, 1] = Mark.CROSS

    assert board.queens() == []
    assert board.conflicts() == []


def test_a_copy_shares_the_puzzle_but_not_the_game(make_board):
    board = make_board(QUADRANTS)
    board.marks[0, 0] = Mark.QUEEN

    clone = board.copy()
    clone.marks[3, 3] = Mark.QUEEN
    clone.region[0, 0] = 3

    assert board.marks[3, 3] == Mark.EMPTY          # the game did not leak back
    assert board.region[0, 0] == 0
    assert np.array_equal(clone.marks[0], board.marks[0])

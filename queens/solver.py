"""Backtracking solver, written to be watched rather than just run.

The whole search is exposed as a stream of `Step` events — every attempt, every
rejection with its reason, every queen placed and every one taken back. That
stream is what the solver panel will animate; `solve()` is just the same search
with the events thrown away.

Rows are filled top to bottom, one queen per row by construction, so a partial
state is simply the column chosen for each row so far.

    python -m queens.solver doc/Example3.png --trace
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

from .board import Board


class StepKind(Enum):
    """What just happened in the search."""

    TRY = "try"                  # about to test a cell
    REJECT = "reject"            # the cell breaks a rule, see `reason`
    PLACE = "place"              # queen placed, descend to the next row
    BACKTRACK = "backtrack"      # the queen is taken back, the row below failed
    SOLVED = "solved"            # every row has a queen
    EXHAUSTED = "exhausted"      # the whole tree was searched, no solution


@dataclass(frozen=True)
class Step:
    """One event of the search, with the state of the board at that moment.

    `queens[r]` is the column of the queen in row r, and its length is how many
    rows are filled. It is a snapshot taken *after* the event, so a PLACE
    already includes its queen and a BACKTRACK no longer includes it: replaying
    the stream is enough to redraw the board, no bookkeeping needed.
    """

    kind: StepKind
    row: int
    col: int
    queens: tuple[int, ...]
    reason: str = ""

    def __str__(self) -> str:
        where = f"({self.row},{self.col})"
        if self.kind is StepKind.SOLVED:
            return f"SOLVED with queens at {self.queens}"
        if self.kind is StepKind.EXHAUSTED:
            return "EXHAUSTED: the board has no solution"
        text = f"{self.kind.value.upper():<9} {where}"
        return f"{text}  {self.reason}" if self.reason else text


def conflict(board: Board, queens: tuple[int, ...] | list[int],
             row: int, col: int) -> str | None:
    """Why a queen cannot go in this cell, or None if it can.

    Only the queens already placed are considered — rows below are still empty,
    which is what makes the check cheap. The message is not decoration: it is
    what the panel shows to explain a rejection.
    """
    region = int(board.region[row, col])
    for r, c in enumerate(queens):
        if c == col:
            return f"column {col} is taken by the queen at ({r},{c})"
        if abs(row - r) <= 1 and abs(col - c) <= 1:
            return f"touches the queen at ({r},{c})"
        if int(board.region[r, c]) == region:
            return f"region {region} already has a queen at ({r},{c})"
    return None


def _search(board: Board, queens: list[int]) -> Iterator[Step]:
    """Fill row `len(queens)`. Returns True as soon as a solution is complete."""
    row = len(queens)
    if row == board.n:
        yield Step(StepKind.SOLVED, row - 1, queens[-1], tuple(queens))
        return True

    for col in range(board.n):
        yield Step(StepKind.TRY, row, col, tuple(queens))

        reason = conflict(board, queens, row, col)
        if reason:
            yield Step(StepKind.REJECT, row, col, tuple(queens), reason)
            continue

        queens.append(col)
        yield Step(StepKind.PLACE, row, col, tuple(queens))
        if (yield from _search(board, queens)):
            return True

        queens.pop()
        yield Step(StepKind.BACKTRACK, row, col, tuple(queens),
                   f"no queen fits in row {row + 1} onwards")

    return False


def solve_steps(board: Board) -> Iterator[Step]:
    """Run the search, yielding every step until the first solution.

    The stream always ends in SOLVED or EXHAUSTED, so a consumer can stop on
    either and never has to guess whether more is coming.
    """
    queens: list[int] = []
    if not (yield from _search(board, queens)):
        yield Step(StepKind.EXHAUSTED, -1, -1, ())


def solve(board: Board) -> tuple[int, ...] | None:
    """The first solution as a column per row, or None if there is none."""
    for step in solve_steps(board):
        if step.kind is StepKind.SOLVED:
            return step.queens
    return None


def is_solution(board: Board, queens: tuple[int, ...] | list[int]) -> bool:
    """Check a solution from scratch, without trusting how it was produced."""
    if len(queens) != board.n:
        return False
    return all(conflict(board, queens[:row], row, col) is None
               for row, col in enumerate(queens))


def render(board: Board, queens: tuple[int, ...] | list[int] = ()) -> str:
    """The board as text: region letters, with `Q` where a queen stands."""
    letters = "ABCDEFGHIJKLMNOP"
    lines = []
    for row in range(board.n):
        cells = ["Q" if row < len(queens) and queens[row] == col
                 else letters[int(board.region[row, col])]
                 for col in range(board.n)]
        lines.append(" ".join(cells))
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    from .vision import detect_file

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="screenshot containing a Queens board")
    parser.add_argument("--trace", action="store_true",
                        help="print every step of the search, not just the result")
    args = parser.parse_args(argv)

    result = detect_file(args.image, debug=False)
    if not result.ok:
        print(f"DETECTION FAILED: {result.error}")
        return 1

    board = result.board
    print(render(board))
    print()

    queens: tuple[int, ...] | None = None
    tried = 0
    for step in solve_steps(board):
        if step.kind is StepKind.TRY:
            tried += 1
        if args.trace:
            print(step)
        if step.kind is StepKind.SOLVED:
            queens = step.queens

    if queens is None:
        print(f"No solution after {tried} attempts.")
        return 1

    print(render(board, queens))
    print(f"\nSolved in {tried} attempts: "
          + ", ".join(f"row {r} -> column {c}" for r, c in enumerate(queens)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

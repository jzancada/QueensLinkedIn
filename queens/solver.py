"""Backtracking solver, written to be watched rather than just run.

The whole search is exposed as a stream of `Step` events — every attempt, every
rejection with its reason, every queen placed and every one taken back. That
stream is what the solver panel will animate; `solve()` is just the same search
with the events thrown away.

Rows are filled top to bottom, one queen per row by construction, so a partial
state is simply the column chosen for each row so far. On top of that there is
one look-ahead: after each placement, any region or column left with no legal
cell in the rows below is already lost, so the branch is cut there instead of
thousands of attempts later. It can be turned off to watch the naive search for
comparison — same answer, an order of magnitude more steps.

    python -m queens.solver doc/Example3.png --trace
    python -m queens.solver doc/Example3.png --no-prune
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from enum import Enum
from typing import Iterator

import numpy as np

from .board import Board


class StepKind(Enum):
    """What just happened in the search."""

    TRY = "try"                  # about to test a cell
    REJECT = "reject"            # the cell breaks a rule, see `reason`
    PLACE = "place"              # queen placed, descend to the next row
    PRUNE = "prune"              # legal, but it strands a region or a column
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


def _region_ids(board: Board) -> set[int]:
    return {int(v) for v in np.unique(board.region)}


def stranded(board: Board, queens: tuple[int, ...] | list[int]) -> str | None:
    """Name a region or column that can no longer get a queen, or None.

    Every region and every column needs exactly one queen, and only the rows
    below are left: one with no legal cell down there is already lost, however
    far the search still is from finding out the hard way. Both come out of the
    same sweep over the free cells, so the check costs one pass.

    It is a *necessary* condition only — a region still having a free cell does
    not make the board solvable — so it can only cut branches that hold no
    solution: the answer is the same, just reached far sooner. On the 9x9 of
    `doc/` it takes the search from 7299 attempts to 549, and the regions carry
    most of that: alone they reach 2151.
    """
    rows_left = range(len(queens), board.n)
    taken_regions = {int(board.region[r, c]) for r, c in enumerate(queens)}
    taken_cols = set(queens)
    free_regions: set[int] = set()
    free_cols: set[int] = set()

    for row in rows_left:
        for col in range(board.n):
            rid = int(board.region[row, col])
            if (col in free_cols or col in taken_cols) and \
                    (rid in free_regions or rid in taken_regions):
                continue                     # this cell can no longer teach us anything
            if conflict(board, queens, row, col) is None:
                free_regions.add(rid)
                free_cols.add(col)

    where = f"rows {len(queens)}-{board.n - 1}"
    lost_regions = sorted(_region_ids(board) - taken_regions - free_regions)
    if lost_regions:
        return f"region {lost_regions[0]} has no cell left in {where}"
    lost_cols = sorted(set(range(board.n)) - taken_cols - free_cols)
    if lost_cols:
        return f"column {lost_cols[0]} has no cell left in {where}"
    return None


def _search(board: Board, queens: list[int], prune: bool) -> Iterator[Step]:
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

        # The queen goes down first and only then is questioned: seeing it
        # placed and immediately withdrawn is what makes the prune legible.
        lost = stranded(board, queens) if prune else None
        if lost:
            queens.pop()
            yield Step(StepKind.PRUNE, row, col, tuple(queens), lost)
            continue

        if (yield from _search(board, queens, prune)):
            return True

        queens.pop()
        yield Step(StepKind.BACKTRACK, row, col, tuple(queens),
                   f"no queen fits in row {row + 1} onwards")

    return False


def solve_steps(board: Board, prune: bool = True) -> Iterator[Step]:
    """Run the search, yielding every step until the first solution.

    The stream always ends in SOLVED or EXHAUSTED, so a consumer can stop on
    either and never has to guess whether more is coming. With `prune` off the
    search is plain backtracking, which is worth watching side by side: same
    answer, an order of magnitude more steps to get there.
    """
    queens: list[int] = []
    if not (yield from _search(board, queens, prune)):
        yield Step(StepKind.EXHAUSTED, -1, -1, ())


def solve(board: Board, prune: bool = True) -> tuple[int, ...] | None:
    """The first solution as a column per row, or None if there is none."""
    for step in solve_steps(board, prune):
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
    parser.add_argument("--no-prune", dest="prune", action="store_false",
                        help="plain backtracking, without the look-ahead")
    args = parser.parse_args(argv)

    result = detect_file(args.image, debug=False)
    if not result.ok:
        print(f"DETECTION FAILED: {result.error}")
        return 1

    board = result.board
    print(render(board))
    print()

    queens: tuple[int, ...] | None = None
    tried = pruned = 0
    for step in solve_steps(board, args.prune):
        if step.kind is StepKind.TRY:
            tried += 1
        elif step.kind is StepKind.PRUNE:
            pruned += 1
        if args.trace:
            print(step)
        if step.kind is StepKind.SOLVED:
            queens = step.queens

    how = "with the look-ahead" if args.prune else "with plain backtracking"
    if queens is None:
        print(f"No solution after {tried} attempts {how}.")
        return 1

    print(render(board, queens))
    print(f"\nSolved {how} in {tried} attempts"
          + (f", {pruned} of them cut early" if pruned else "") + ":")
    print("  " + ", ".join(f"row {r} -> column {c}" for r, c in enumerate(queens)))
    return 0


if __name__ == "__main__":
    sys.exit(main())

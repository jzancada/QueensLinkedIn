"""Board model: regions, colors, borders and play state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np


class Mark(IntEnum):
    """What the player has placed on a cell."""

    EMPTY = 0
    CROSS = 1
    QUEEN = 2


@dataclass
class Board:
    """A digitized Queens board.

    `region` holds the region id of every cell and is all the solver needs.
    `colors` and `cells` come from the detection and are only used by the
    interface: the colors to paint, and the rectangles so that the vision
    panel's inspector can translate a point of the image back to a cell.
    """

    n: int
    region: np.ndarray                      # (n, n) int, region id per cell
    colors: list[tuple[int, int, int]]      # BGR per region
    cells: np.ndarray | None = None         # (n, n, 4) rect x,y,w,h in the image
    marks: np.ndarray = field(init=False)   # (n, n) Mark

    def __post_init__(self) -> None:
        self.marks = np.full((self.n, self.n), Mark.EMPTY, dtype=np.uint8)

    def is_border(self, row: int, col: int, drow: int, dcol: int) -> bool:
        """Is there a region border between this cell and the given neighbour?

        Off-board counts as a border: the outer edge is drawn thick, just like
        the inner region borders.
        """
        r, c = row + drow, col + dcol
        if not (0 <= r < self.n and 0 <= c < self.n):
            return True
        return bool(self.region[row, col] != self.region[r, c])

    def copy(self) -> "Board":
        """An independent board with the same regions.

        The panels must not fight over one set of marks: the solver rewrites
        them on every step, and it would wipe out the game the player has going
        in the other tab.
        """
        clone = Board(self.n, self.region.copy(), list(self.colors),
                      None if self.cells is None else self.cells.copy())
        clone.marks = self.marks.copy()
        return clone

    def set_queens(self, queens) -> None:
        """Replace the marks with one queen per row, at the given columns.

        `queens[r]` is the column of the queen in row r, the same shape the
        solver works in, so a search state can be dropped on the board as is —
        including a partial one, where the rows below are simply left empty.
        """
        self.marks[:] = Mark.EMPTY
        for row, col in enumerate(queens):
            self.marks[row, col] = Mark.QUEEN

    def queens(self) -> list[tuple[int, int]]:
        """Every cell the player has put a queen on, top to bottom."""
        return [(int(r), int(c)) for r, c in np.argwhere(self.marks == Mark.QUEEN)]

    def conflicts(self) -> list[tuple[tuple[int, int], tuple[int, int], str]]:
        """Pairs of queens that break a rule, with the rule they break.

        The solver's `conflict()` cannot be reused here: it assumes one queen
        per row, placed top to bottom, which is true of a search state but not
        of a board being played, where two queens may well sit in the same row.

        Only the first rule broken by a pair is reported — a message per pair,
        not per rule, is what the player can act on.
        """
        found = []
        placed = self.queens()
        for i, (row, col) in enumerate(placed):
            for other in placed[i + 1:]:
                orow, ocol = other
                if row == orow:
                    reason = f"row {row} has two queens"
                elif col == ocol:
                    reason = f"column {col} has two queens"
                elif abs(row - orow) <= 1 and abs(col - ocol) <= 1:
                    reason = "two queens touch"
                elif self.region[row, col] == self.region[orow, ocol]:
                    reason = f"region {int(self.region[row, col])} has two queens"
                else:
                    continue
                found.append(((row, col), other, reason))
        return found

    def is_solved(self) -> bool:
        """N queens and no conflict at all — which is the whole of the rules.

        One per row, per column and per region does not need checking on top:
        with N queens and no two sharing any of the three, there is one in each
        by counting alone.
        """
        return len(self.queens()) == self.n and not self.conflicts()

    def region_sizes(self) -> np.ndarray:
        """Number of cells in each region, indexed by id."""
        return np.bincount(self.region.ravel(), minlength=len(self.colors))

    def regions_are_connected(self) -> bool:
        """Does every region form a single connected piece (4-neighbourhood)?

        A region split into two pieces betrays that the color grouping has
        merged distinct regions that happen to look alike.
        """
        for rid in range(len(self.colors)):
            cells = np.argwhere(self.region == rid)
            if len(cells) == 0:
                return False
            pending = [tuple(cells[0])]
            seen = {tuple(cells[0])}
            while pending:
                r, c = pending.pop()
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nb = (r + dr, c + dc)
                    if (0 <= nb[0] < self.n and 0 <= nb[1] < self.n
                            and nb not in seen and self.region[nb] == rid):
                        seen.add(nb)
                        pending.append(nb)
            if len(seen) != len(cells):
                return False
        return True

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

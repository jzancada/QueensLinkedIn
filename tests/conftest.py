"""Qt must run headless under pytest, or the UI tests would pop up windows.

Set before anything imports QtGui: the platform plugin is chosen when the
QApplication is created, and there is no going back afterwards.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np                                          # noqa: E402
import pytest                                               # noqa: E402

from queens.board import Board                              # noqa: E402


@pytest.fixture
def make_board():
    """Build a board out of region letters, one string per row.

        make_board(["AABB",
                    "AABB",
                    "CCDD",
                    "CCDD"])

    Colors are stand-ins: what the solver and the rules work on is the region
    matrix, and reading one off a screenshot in a test would prove nothing.
    """
    def build(rows: list[str]) -> Board:
        letters = sorted({ch for row in rows for ch in row})
        region = np.array([[letters.index(ch) for ch in row] for row in rows],
                          dtype=np.int32)
        return Board(n=len(rows), region=region, colors=[(0, 0, 0)] * len(letters))

    return build

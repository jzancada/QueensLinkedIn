"""The board to solve by hand.

Step 3 only puts the board on screen and lets marks be placed on it, in the
order LinkedIn uses: a click crosses the cell out, another turns it into a
queen, a third clears it. Checking the rules while you play, and the hints,
come in step 5.
"""

from __future__ import annotations

from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget

from ..board import Board, Mark
from .board_view import BoardView

NEXT_MARK = {Mark.EMPTY: Mark.CROSS, Mark.CROSS: Mark.QUEEN, Mark.QUEEN: Mark.EMPTY}


class PlayTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.board: Board | None = None

        self.view = BoardView()
        self.view.cell_clicked.connect(self.cycle_mark)

        self.caption = QLabel()
        self.caption.setStyleSheet("color: #666666;")

        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.caption)
        self.set_board(None)

    def set_board(self, board: Board | None) -> None:
        self.board = board
        self.view.set_board(board)
        self.describe()

    def cycle_mark(self, row: int, col: int) -> None:
        if not self.board:
            return
        self.board.marks[row, col] = NEXT_MARK[Mark(self.board.marks[row, col])]
        self.view.update()
        self.describe()

    def describe(self) -> None:
        if not self.board:
            self.caption.setText("Open a screenshot to get a board.")
            return
        queens = int((self.board.marks == Mark.QUEEN).sum())
        self.caption.setText(
            f"{queens} of {self.board.n} queens placed. "
            "Click a cell to cross it out, again for a queen, again to clear it.")

"""The board to solve by hand, with the rules checked as you play.

A click crosses a cell out, another turns it into a queen, a third clears it —
the order LinkedIn uses. Every change is judged straight away: the queens that
break a rule are washed in red and the caption names the rule, because being
told *which* rule is the difference between a hint and a scolding.

Nothing is forbidden. An illegal queen can be placed and left there; the panel
says what is wrong with it and lets the player work it out.
"""

from __future__ import annotations

from PySide6.QtWidgets import (QHBoxLayout, QLabel, QPushButton, QVBoxLayout,
                               QWidget)

from ..board import Board, Mark
from ..solver import solve
from .board_view import BoardView

NEXT_MARK = {Mark.EMPTY: Mark.CROSS, Mark.CROSS: Mark.QUEEN, Mark.QUEEN: Mark.EMPTY}

SOLVED_STYLE = "color: #1f9d55; font-weight: bold;"
ERROR_STYLE = "color: #d64545;"
PLAIN_STYLE = "color: #666666;"


class PlayTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.board: Board | None = None
        self._solution: tuple[int, ...] | None = None

        self.view = BoardView()
        self.view.cell_clicked.connect(self.cycle_mark)

        self.caption = QLabel()
        self.caption.setWordWrap(True)

        self.hint_button = QPushButton("Hint")
        self.hint_button.setToolTip("Place one queen of the solution, or point out a wrong one.")
        self.hint_button.clicked.connect(self.hint)

        self.solve_button = QPushButton("Solve")
        self.solve_button.clicked.connect(self.reveal)

        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear)

        buttons = QHBoxLayout()
        buttons.addWidget(self.hint_button)
        buttons.addWidget(self.solve_button)
        buttons.addWidget(self.clear_button)
        buttons.addStretch()

        layout = QVBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(self.caption)
        layout.addLayout(buttons)
        self.set_board(None)

    def set_board(self, board: Board | None) -> None:
        self.board = board
        self._solution = None                  # only worked out if a hint is asked for
        self.view.set_board(board)
        for button in (self.hint_button, self.solve_button, self.clear_button):
            button.setEnabled(board is not None)
        self.describe()

    # --- playing ------------------------------------------------------------

    def cycle_mark(self, row: int, col: int) -> None:
        if not self.board:
            return
        self.board.marks[row, col] = NEXT_MARK[Mark(self.board.marks[row, col])]
        self.describe()

    def clear(self) -> None:
        if self.board:
            self.board.marks[:] = Mark.EMPTY
            self.describe()

    def solution(self) -> tuple[int, ...] | None:
        """The solved board, worked out once and kept."""
        if self._solution is None and self.board:
            self._solution = solve(self.board)
        return self._solution

    def reveal(self) -> None:
        solution = self.solution()
        if not self.board:
            return
        if solution is None:
            self.say("This board has no solution.", ERROR_STYLE)
            return
        self.board.set_queens(solution)
        self.describe()

    def hint(self) -> None:
        """One step, not the answer: a wrong queen if there is one, else a right one.

        A queen the solution does not have is worth more than another correct
        one — the player will not find the mistake by adding queens on top of it.
        """
        solution = self.solution()
        if not self.board or solution is None:
            self.say("This board has no solution.", ERROR_STYLE)
            return

        wrong = [(row, col) for row, col in self.board.queens() if solution[row] != col]
        if wrong:
            row, col = wrong[0]
            self.view.set_errors(wrong)
            self.say(f"The queen at ({row}, {col}) is not in the solution.", ERROR_STYLE)
            return

        missing = [row for row in range(self.board.n)
                   if self.board.marks[row, solution[row]] != Mark.QUEEN]
        if not missing:
            self.describe()
            return

        row = missing[0]
        self.board.marks[row, solution[row]] = Mark.QUEEN
        self.describe()
        self.say(f"A queen goes at ({row}, {solution[row]}).", PLAIN_STYLE)

    # --- what the player is told --------------------------------------------

    def describe(self) -> None:
        if not self.board:
            self.view.set_errors(())
            self.say("Open a screenshot to get a board.", PLAIN_STYLE)
            return

        conflicts = self.board.conflicts()
        self.view.set_errors({cell for a, b, _ in conflicts for cell in (a, b)})
        placed = len(self.board.queens())
        count = f"{placed} of {self.board.n} queens"

        if self.board.is_solved():
            self.say(f"Solved — {count}, no conflicts.", SOLVED_STYLE)
        elif conflicts:
            (row, col), (orow, ocol), reason = conflicts[0]
            more = f"  (+{len(conflicts) - 1} more)" if len(conflicts) > 1 else ""
            self.say(f"{count} — {reason}: ({row}, {col}) and ({orow}, {ocol}).{more}",
                     ERROR_STYLE)
        else:
            self.say(f"{count} placed, no conflicts. Click to cross out, "
                     "again for a queen, again to clear.", PLAIN_STYLE)

    def say(self, text: str, style: str) -> None:
        self.caption.setText(text)
        self.caption.setStyleSheet(style)

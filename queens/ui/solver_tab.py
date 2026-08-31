"""The solver panel: the search, watched as it happens.

`solve_steps` already yields the search as a stream of events, each carrying
the board state after it, so this panel needs no logic of its own: it pulls one
step, drops its queens on the board and colors the cell the step is about. Play
it, or walk it one step at a time.

The board here is a copy. The solver rewrites the marks on every step, and the
game the player has going in the third tab must not be wiped out by it.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QCheckBox, QHBoxLayout, QLabel, QPlainTextEdit,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from ..board import Board
from ..solver import Step, StepKind, solve_steps
from .board_view import BoardView

# Steps per second at each end of the speed slider. The slow end has to be slow
# enough to follow a rejection by eye; the fast end fast enough that the naive
# search (7299 attempts on a 9x9) still finishes while you watch.
MIN_SPEED, MAX_SPEED = 1, 400

# Above this rate the timer stops being the limit and several steps are taken
# per tick: asking Qt for a 2 ms timer just drops frames.
TICKS_PER_SECOND = 60


class SolverTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.board: Board | None = None
        self.steps = None                  # the running generator, or None
        self.counts: dict[StepKind, int] = {}
        self.finished = False

        self.view = BoardView()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)

        self.run_button = QPushButton("Play")
        self.run_button.setShortcut(Qt.Key.Key_Space)
        self.run_button.clicked.connect(self.toggle_run)

        self.step_button = QPushButton("Step")
        self.step_button.clicked.connect(self.single_step)

        self.reset_button = QPushButton("Reset")
        self.reset_button.clicked.connect(self.restart)

        self.speed = QSlider(Qt.Orientation.Horizontal)
        self.speed.setRange(MIN_SPEED, MAX_SPEED)
        self.speed.setValue(20)
        self.speed.valueChanged.connect(self.apply_speed)

        self.prune = QCheckBox("Region and column look-ahead")
        self.prune.setChecked(True)
        self.prune.setToolTip("Off, it is plain backtracking: same answer, "
                              "an order of magnitude more steps.")
        self.prune.toggled.connect(self.restart)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setMinimumHeight(40)
        self.tally = QLabel()
        self.tally.setStyleSheet("color: #666666;")

        self.trace = QPlainTextEdit()
        self.trace.setReadOnly(True)
        self.trace.setFont(QFont("Consolas", 9))
        self.trace.setMaximumBlockCount(500)      # the last steps are the ones that matter

        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(self.step_button)
        buttons.addWidget(self.reset_button)

        side = QVBoxLayout()
        side.addLayout(buttons)
        side.addWidget(QLabel("Speed"))
        side.addWidget(self.speed)
        side.addWidget(self.prune)
        side.addWidget(self.status)
        side.addWidget(self.tally)
        side.addWidget(QLabel("Trace"))
        side.addWidget(self.trace, 1)

        panel = QWidget()
        panel.setLayout(side)
        panel.setFixedWidth(340)

        layout = QHBoxLayout(self)
        layout.addWidget(self.view, 1)
        layout.addWidget(panel)

        self.apply_speed()
        self.set_board(None)

    # --- state --------------------------------------------------------------

    def set_board(self, board: Board | None) -> None:
        """Take a fresh board to solve, on its own copy."""
        self.board = board.copy() if board else None
        self.view.set_board(self.board)
        self.restart()

    def restart(self) -> None:
        """Back to the empty board, ready to search again."""
        self.timer.stop()
        self.run_button.setText("Play")
        self.counts = {}
        self.finished = False
        self.trace.clear()

        if not self.board:
            self.steps = None
            self.status.setText("Open a screenshot to get a board to solve.")
            self.tally.setText("")
            self.set_enabled(False)
            return

        self.board.set_queens([])
        self.view.clear_focus()
        self.view.update()
        self.steps = solve_steps(self.board, self.prune.isChecked())
        self.status.setText("Ready. Play runs the search; Step walks it one at a time.")
        self.update_tally()
        self.set_enabled(True)

    def set_enabled(self, enabled: bool) -> None:
        for widget in (self.run_button, self.step_button, self.reset_button,
                       self.speed, self.prune):
            widget.setEnabled(enabled)

    # --- running ------------------------------------------------------------

    def apply_speed(self) -> None:
        rate = self.speed.value()
        self.timer.setInterval(max(1000 // TICKS_PER_SECOND, 1000 // rate))

    def steps_per_tick(self) -> int:
        return max(1, self.speed.value() // TICKS_PER_SECOND)

    def toggle_run(self) -> None:
        if self.timer.isActive():
            self.timer.stop()
            self.run_button.setText("Play")
        elif not self.finished:
            self.timer.start()
            self.run_button.setText("Pause")

    def single_step(self) -> None:
        self.timer.stop()
        self.run_button.setText("Play")
        self.advance()

    def tick(self) -> None:
        for _ in range(self.steps_per_tick()):
            if not self.advance():
                return

    def advance(self) -> bool:
        """Consume one step and show it. False when the search is over."""
        if self.steps is None or self.finished:
            return False
        try:
            step = next(self.steps)
        except StopIteration:
            self.stop("The search ended.")
            return False

        self.show_step(step)
        return not self.finished

    def show_step(self, step: Step) -> None:
        self.counts[step.kind] = self.counts.get(step.kind, 0) + 1
        self.board.set_queens(step.queens)

        if step.row >= 0:
            self.view.set_focus(step.row, step.col, step.kind.value)
        else:
            self.view.clear_focus()
        self.view.update()

        self.trace.appendPlainText(str(step))
        self.update_tally()

        if step.kind is StepKind.SOLVED:
            attempts = self.counts.get(StepKind.TRY, 0)
            self.stop(f"Solved in {attempts} attempts.")
        elif step.kind is StepKind.EXHAUSTED:
            self.stop("No solution: the whole tree was searched.")
        else:
            self.status.setText(str(step))

    def stop(self, message: str) -> None:
        self.finished = True
        self.timer.stop()
        self.run_button.setText("Play")
        self.status.setText(message)

    def update_tally(self) -> None:
        self.tally.setText(", ".join(
            f"{self.counts.get(kind, 0)} {kind.value}"
            for kind in (StepKind.TRY, StepKind.PLACE, StepKind.PRUNE,
                         StepKind.BACKTRACK)))

"""The vision panel: the ten stages of the detection, one at a time.

Nothing here computes anything. `vision.detect` already records every stage
with its notes precisely so this panel can be a viewer and no more — including
when the detection fails, where the stages reached are the whole explanation.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QPlainTextEdit,
                               QSplitter, QVBoxLayout, QWidget)

from ..vision import Detection
from .image_view import ImageView

HINT = "Click the image to inspect a cell."


class VisionTab(QWidget):
    def __init__(self) -> None:
        super().__init__()
        self.detection: Detection | None = None

        self.stage_list = QListWidget()
        self.stage_list.currentRowChanged.connect(self.show_stage)

        self.notes = QPlainTextEdit()
        self.notes.setReadOnly(True)
        self.notes.setFont(QFont("Consolas", 9))
        self.notes.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)

        self.view = ImageView("Open a screenshot to begin  (Ctrl+O)")
        self.view.clicked.connect(self.inspect)

        self.caption = QLabel()
        self.caption.setWordWrap(True)
        self.caption.setStyleSheet("color: #666666;")

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("Stages"))
        left_layout.addWidget(self.stage_list, 1)
        left_layout.addWidget(QLabel("Notes"))
        left_layout.addWidget(self.notes, 1)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self.view, 1)
        right_layout.addWidget(self.caption)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 700])

        layout = QHBoxLayout(self)
        layout.addWidget(splitter)

    def show_detection(self, detection: Detection) -> None:
        """Load a fresh detection and jump to the stage worth looking at.

        On success that is the last one, the borders, which is where a wrong
        reading shows up at a glance; on a failure it is also the last one,
        because that is where the pipeline stopped.
        """
        self.detection = detection

        self.stage_list.blockSignals(True)
        self.stage_list.clear()
        self.stage_list.addItems([stage.name for stage in detection.stages])
        self.stage_list.blockSignals(False)

        if detection.stages:
            self.stage_list.setCurrentRow(len(detection.stages) - 1)
        else:
            self.show_stage(-1)

    def show_stage(self, index: int) -> None:
        stages = self.detection.stages if self.detection else []
        if not 0 <= index < len(stages):
            self.view.set_image(None)
            self.notes.setPlainText(self.detection.error if self.detection else "")
            self.caption.setText("")
            return

        stage = stages[index]
        self.view.set_image(stage.image)
        self.notes.setPlainText("\n".join(stage.notes))
        self.caption.setText(self._caption(index))

    def _caption(self, index: int) -> str:
        """One line under the image: where we are, and what went wrong if it did."""
        stages = self.detection.stages
        position = f"Stage {index + 1} of {len(stages)}"
        if self.detection.ok:
            return f"{position} — {HINT}" if self._inspectable(index) else position
        if index == len(stages) - 1:
            return f"{position} — detection stopped here: {self.detection.error}"
        return position

    def _inspectable(self, index: int) -> bool:
        """Can a click on this stage be translated into a cell?

        Only on the stages drawn over the whole screenshot: the board crop is
        the same picture in different coordinates, and a click there would name
        a cell several rows off.
        """
        if not self.detection or not self.detection.ok:
            return False
        original = self.detection.stages[0].image
        return self.detection.stages[index].image.shape == original.shape

    def inspect(self, x: int, y: int) -> None:
        """Report the cell under a click, in the board's own terms."""
        if not self._inspectable(self.stage_list.currentRow()):
            return

        board = self.detection.board
        for row in range(board.n):
            for col in range(board.n):
                cx, cy, w, h = board.cells[row, col]
                if cx <= x < cx + w and cy <= y < cy + h:
                    color = board.colors[int(board.region[row, col])]
                    self.caption.setText(
                        f"Cell ({row}, {col}) — region {board.region[row, col]}, "
                        f"BGR {tuple(color)}, rect ({cx}, {cy}) {w}x{h}")
                    return
        self.caption.setText(f"({x}, {y}) is outside the board.")

"""The main window: one screenshot in, three panels over it.

The window opens the file and hands the result around; each panel decides what
to make of it. Step 2 of the interface: the vision panel is live, the solver
and play panels are still placeholders.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (QFileDialog, QLabel, QMainWindow, QTabWidget,
                               QVBoxLayout, QWidget)

from ..vision import Detection, detect_file
from .play_tab import PlayTab
from .vision_tab import VisionTab

DOC = Path(__file__).resolve().parent.parent.parent / "doc"


def _placeholder(text: str) -> QWidget:
    """A tab that is not written yet, saying so rather than looking broken."""
    page = QWidget()
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet("color: #888888; font-size: 14px;")
    layout = QVBoxLayout(page)
    layout.addStretch()
    layout.addWidget(label)
    layout.addStretch()
    return page


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Queens")
        self.resize(1000, 720)

        self.detection: Detection | None = None

        self.vision = VisionTab()
        self.play = PlayTab()

        self.tabs = QTabWidget()
        self.tabs.addTab(self.vision, "1 · Vision")
        self.tabs.addTab(_placeholder("The solver panel goes here:\n"
                                      "the search, step by step."), "2 · Solver")
        self.tabs.addTab(self.play, "3 · Play")
        self.setCentralWidget(self.tabs)

        open_action = QAction("&Open screenshot…", self)
        open_action.setShortcut(QKeySequence.StandardKey.Open)
        open_action.triggered.connect(self.choose_file)
        self.menuBar().addMenu("&File").addAction(open_action)
        self.addToolBar("Main").addAction(open_action)

        self.statusBar().showMessage("No screenshot loaded.")

    def choose_file(self) -> None:
        start = str(DOC if DOC.is_dir() else Path.home())
        path, _ = QFileDialog.getOpenFileName(self, "Open screenshot", start,
                                              "Images (*.png *.jpg *.jpeg *.bmp)")
        if path:
            self.load(path)

    def load(self, path: str | Path) -> None:
        """Run the detection on a screenshot and show what came of it.

        A failure is not an error dialog: the stages recorded up to the point
        of failure are exactly what explains it, and the Vision panel is where
        they will be shown.
        """
        self.detection = detect_file(path)
        self.setWindowTitle(f"Queens — {Path(path).name}")

        stages = self.detection.stages
        self.vision.show_detection(self.detection)
        self.play.set_board(self.detection.board)

        if self.detection.ok:
            board = self.detection.board
            self.statusBar().showMessage(
                f"Board {board.n}x{board.n} with {len(board.colors)} regions, "
                f"in {len(stages)} stages.")
        else:
            self.statusBar().showMessage(
                f"Detection failed: {self.detection.error}  "
                f"({len(stages)} stages recorded)")

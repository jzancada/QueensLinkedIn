"""The board as the game draws it: region colors, thick borders, marks.

This is the one widget both remaining panels are built on — the solver animates
its search on it and the player clicks on it — so it knows nothing about either
one. It paints a `Board` and reports which cell was clicked; the rules live in
`board.py` and `solver.py`, where they already are.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen, QPolygon
from PySide6.QtWidgets import QSizePolicy, QWidget

from ..board import Board, Mark

# What a cell can be singled out for. The keys are the `StepKind` values of the
# solver, so the solver panel can hand its steps straight over.
FOCUS_COLORS = {
    "try": QColor("#2f6fed"),          # being tested right now
    "reject": QColor("#d64545"),       # breaks a rule
    "place": QColor("#1f9d55"),        # accepted, the search moves on
    "prune": QColor("#c9800a"),        # legal, but it strands a region
    "backtrack": QColor("#8b5cf6"),    # taken back, the rows below failed
}

BORDER = QColor("#111111")
GRID = QColor(0, 0, 0, 60)
MARK = QColor("#1a1a1a")


def _qcolor(bgr) -> QColor:
    """OpenCV keeps colors as BGR; Qt wants them the other way round."""
    b, g, r = (int(v) for v in bgr)
    return QColor(r, g, b)


class BoardView(QWidget):
    cell_clicked = Signal(int, int)     # row, column

    def __init__(self) -> None:
        super().__init__()
        self.board: Board | None = None
        self._focus: tuple[int, int, str] | None = None
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 240)

    def set_board(self, board: Board | None) -> None:
        self.board = board
        self._focus = None
        self.update()

    def set_focus(self, row: int, col: int, kind: str) -> None:
        """Single out one cell, in the color that goes with `kind`."""
        self._focus = (row, col, kind)
        self.update()

    def clear_focus(self) -> None:
        self._focus = None
        self.update()

    # --- geometry -----------------------------------------------------------

    def cell_size(self) -> int:
        """Side of a cell in pixels: whole, so the grid never blurs."""
        if not self.board:
            return 0
        return max(8, (min(self.width(), self.height()) - 16) // self.board.n)

    def board_rect(self) -> QRect:
        """The square the board occupies, centered in the widget."""
        if not self.board:
            return QRect()
        side = self.cell_size() * self.board.n
        return QRect((self.width() - side) // 2, (self.height() - side) // 2, side, side)

    def cell_rect(self, row: int, col: int) -> QRect:
        origin, cell = self.board_rect().topLeft(), self.cell_size()
        return QRect(origin.x() + col * cell, origin.y() + row * cell, cell, cell)

    def cell_at(self, pos: QPoint) -> tuple[int, int] | None:
        """Which cell a widget point falls on, or None if it misses the board."""
        rect = self.board_rect()
        if not self.board or not rect.contains(pos):
            return None
        cell = self.cell_size()
        return ((pos.y() - rect.y()) // cell, (pos.x() - rect.x()) // cell)

    def mousePressEvent(self, event) -> None:      # noqa: N802  (Qt naming)
        where = self.cell_at(event.position().toPoint())
        if where is not None:
            self.cell_clicked.emit(*where)

    # --- painting -----------------------------------------------------------

    def paintEvent(self, event) -> None:           # noqa: N802  (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().window())
        if not self.board:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No board yet.")
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        board = self.board

        for row in range(board.n):
            for col in range(board.n):
                rect = self.cell_rect(row, col)
                painter.fillRect(rect, _qcolor(board.colors[int(board.region[row, col])]))
                self._draw_mark(painter, rect, Mark(board.marks[row, col]))

        # Borders last, so no fill of a neighbouring cell paints over them.
        for row in range(board.n):
            for col in range(board.n):
                self._draw_edges(painter, row, col)

        if self._focus:
            row, col, kind = self._focus
            color = FOCUS_COLORS.get(kind, QColor("#2f6fed"))
            painter.setPen(QPen(color, max(3, self.cell_size() // 12)))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.cell_rect(row, col).adjusted(3, 3, -3, -3))

    def _draw_edges(self, painter: QPainter, row: int, col: int) -> None:
        """A thick stroke where the region changes, a hairline where it does not.

        Only the top and left edges of each cell are drawn, plus the board's own
        right and bottom: every inner edge belongs to two cells, and drawing it
        twice makes the thin lines look uneven.
        """
        rect = self.cell_rect(row, col)
        thick = max(2, self.cell_size() // 12)

        edges = [((-1, 0), rect.topLeft(), rect.topRight()),
                 ((0, -1), rect.topLeft(), rect.bottomLeft())]
        if row == self.board.n - 1:
            edges.append(((1, 0), rect.bottomLeft(), rect.bottomRight()))
        if col == self.board.n - 1:
            edges.append(((0, 1), rect.topRight(), rect.bottomRight()))

        for (drow, dcol), start, end in edges:
            border = self.board.is_border(row, col, drow, dcol)
            painter.setPen(QPen(BORDER if border else GRID, thick if border else 1))
            painter.drawLine(start, end)

    def _draw_mark(self, painter: QPainter, rect: QRect, mark: Mark) -> None:
        if mark is Mark.QUEEN:
            self._draw_queen(painter, rect)
        elif mark is Mark.CROSS:
            self._draw_cross(painter, rect)

    def _draw_queen(self, painter: QPainter, rect: QRect) -> None:
        """A crown, drawn rather than typed: a glyph would depend on the fonts
        installed, and it has to read the same on nine different colors."""
        box = rect.adjusted(rect.width() // 4, rect.height() // 4,
                            -rect.width() // 4, -rect.height() // 4)
        w, h = box.width(), box.height()
        x, y = box.x(), box.y()
        crown = QPolygon([
            QPoint(x, y + h), QPoint(x, y + int(h * 0.15)),
            QPoint(x + w // 4, y + int(h * 0.55)), QPoint(x + w // 2, y),
            QPoint(x + 3 * w // 4, y + int(h * 0.55)), QPoint(x + w, y + int(h * 0.15)),
            QPoint(x + w, y + h),
        ])
        painter.setPen(QPen(QColor(255, 255, 255, 200), 3))
        painter.setBrush(MARK)
        painter.drawPolygon(crown)

    def _draw_cross(self, painter: QPainter, rect: QRect) -> None:
        box = rect.adjusted(rect.width() // 3, rect.height() // 3,
                            -rect.width() // 3, -rect.height() // 3)
        painter.setPen(QPen(MARK, max(2, rect.width() // 14),
                            Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(box.topLeft(), box.bottomRight())
        painter.drawLine(box.topRight(), box.bottomLeft())

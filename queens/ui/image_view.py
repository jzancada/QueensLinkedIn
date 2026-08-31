"""A widget that shows an OpenCV image, scaled to fit and never distorted."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPixmap
from PySide6.QtWidgets import QSizePolicy, QWidget


def to_qimage(bgr: np.ndarray) -> QImage:
    """Wrap an OpenCV BGR array as a QImage.

    `.copy()` is not wasteful here, it is the point: without it the QImage
    would keep pointing at the numpy buffer and would show garbage the moment
    the array is freed.
    """
    bgr = np.ascontiguousarray(bgr)
    height, width = bgr.shape[:2]
    return QImage(bgr.data, width, height, bgr.strides[0],
                  QImage.Format.Format_BGR888).copy()


class ImageView(QWidget):
    """Shows one image centered, keeping its aspect ratio at any window size.

    A plain QLabel with a scaled pixmap would do almost this, but the panels
    need to map a click back to a pixel of the original image — the inspector
    that tells which cell was clicked — and that is far easier when the widget
    knows exactly where it drew.
    """

    clicked = Signal(int, int)      # coordinates in the image, not in the widget

    def __init__(self, placeholder: str = "") -> None:
        super().__init__()
        self._pixmap: QPixmap | None = None
        self._placeholder = placeholder
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(320, 240)

    def set_image(self, bgr: np.ndarray | None) -> None:
        self._pixmap = None if bgr is None else QPixmap.fromImage(to_qimage(bgr))
        self.update()

    def target_rect(self) -> QRect:
        """Where the image is being drawn, in widget coordinates."""
        if self._pixmap is None:
            return QRect()
        size = self._pixmap.size().scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio)
        return QRect(QPoint((self.width() - size.width()) // 2,
                            (self.height() - size.height()) // 2), size)

    def image_pos(self, pos: QPoint) -> tuple[int, int] | None:
        """Translate a widget point back to a pixel of the image, or None."""
        rect = self.target_rect()
        if self._pixmap is None or not rect.contains(pos) or rect.isEmpty():
            return None
        scale = self._pixmap.width() / rect.width()
        return (int((pos.x() - rect.x()) * scale), int((pos.y() - rect.y()) * scale))

    def mousePressEvent(self, event) -> None:  # noqa: N802  (Qt naming)
        where = self.image_pos(event.position().toPoint())
        if where is not None:
            self.clicked.emit(*where)

    def paintEvent(self, event) -> None:      # noqa: N802  (Qt naming)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#1e1e1e"))

        if self._pixmap is None:
            painter.setPen(QColor("#888888"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, self._placeholder)
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.drawPixmap(self.target_rect(), self._pixmap)

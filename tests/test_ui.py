"""Interface tests: the window builds, and it says what the pipeline found.

Headless (see conftest), so what is checked is the wiring — which image reaches
which widget and what the status bar ends up saying — not how it looks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

# `queens.ui` first, always: importing it is what pins the right ICU before Qt
# is loaded (see the package docstring), and PySide6 may not import without it.
pytest.importorskip("queens.ui")
pytest.importorskip("PySide6")

from PySide6.QtCore import QPoint, QSize                    # noqa: E402
from PySide6.QtWidgets import QApplication                  # noqa: E402

from queens.ui.image_view import ImageView, to_qimage       # noqa: E402
from queens.ui.main_window import MainWindow                # noqa: E402

DOC = Path(__file__).resolve().parent.parent / "doc"


@pytest.fixture(scope="module")
def app():
    """One QApplication for the whole module: Qt allows no more than one."""
    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    window = MainWindow()
    yield window
    window.close()


def test_the_window_opens_with_three_tabs(window):
    assert window.tabs.count() == 3
    assert [window.tabs.tabText(i)[0] for i in range(3)] == ["1", "2", "3"]
    assert "No screenshot" in window.statusBar().currentMessage()


def test_loading_a_screenshot_reports_the_board(window):
    window.load(DOC / "Example3.png")

    assert window.detection.ok
    message = window.statusBar().currentMessage()
    assert "9x9" in message and "9 regions" in message
    assert "Example3.png" in window.windowTitle()


def test_the_vision_panel_lists_every_stage(window):
    """Ten stages, and it opens on the last one — the borders tell at a glance
    whether the reading is right."""
    window.load(DOC / "Example3.png")
    panel = window.vision

    assert panel.stage_list.count() == 10
    assert panel.stage_list.item(0).text().startswith("1.")
    assert panel.stage_list.currentRow() == 9
    assert "Thick stroke" in panel.notes.toPlainText()

    panel.stage_list.setCurrentRow(3)
    assert "ACCEPTED" in panel.notes.toPlainText()       # the candidates and their verdicts
    assert "Stage 4 of 10" in panel.caption.text()


def test_clicking_the_image_names_the_cell(window):
    """The inspector is what ties a pixel of the screenshot back to the model."""
    window.load(DOC / "Example3.png")
    panel = window.vision
    board = window.detection.board

    x, y, w, h = board.cells[4, 6]
    panel.inspect(x + w // 2, y + h // 2)

    assert "Cell (4, 6)" in panel.caption.text()
    assert f"region {board.region[4, 6]}" in panel.caption.text()


def test_a_click_on_the_board_crop_is_not_mistaken_for_a_cell(window):
    """That stage is the same board in different coordinates, so it is off limits."""
    window.load(DOC / "Example3.png")
    panel = window.vision
    panel.stage_list.setCurrentRow(5)                    # 6. Board crop

    before = panel.caption.text()
    panel.inspect(10, 10)
    assert panel.caption.text() == before
    assert "Click the image" not in before


def test_a_failed_detection_is_reported_not_swallowed(window, tmp_path):
    """The window must survive a bad image and say why, without a board."""
    blank = tmp_path / "blank.png"
    import cv2
    cv2.imwrite(str(blank), np.full((300, 300, 3), 255, dtype=np.uint8))

    window.load(blank)

    assert not window.detection.ok
    assert "failed" in window.statusBar().currentMessage().lower()

    # The stages reached are the explanation, so the panel must still show them
    # and say where the pipeline stopped.
    panel = window.vision
    assert panel.stage_list.count() == len(window.detection.stages)
    assert "stopped here" in panel.caption.text()


def test_the_image_keeps_its_aspect_ratio_and_maps_clicks_back(app):
    """The click mapping is what the cell inspector will be built on."""
    view = ImageView()
    view.resize(400, 400)
    view.set_image(np.zeros((100, 200, 3), dtype=np.uint8))     # twice as wide as tall

    rect = view.target_rect()
    assert rect.size() == QSize(400, 200)                       # fits the width, centered
    assert rect.y() == 100

    assert view.image_pos(QPoint(0, 0)) is None                 # outside the image
    assert view.image_pos(rect.topLeft()) == (0, 0)
    assert view.image_pos(rect.topLeft() + QPoint(200, 100)) == (100, 50)


def test_an_opencv_image_survives_the_trip_to_qt():
    """BGR in, BGR out: a channel swap here would silently recolor everything."""
    bgr = np.zeros((2, 2, 3), dtype=np.uint8)
    bgr[:, :] = (255, 0, 0)                                     # pure blue in BGR

    image = to_qimage(bgr)
    assert image.pixelColor(0, 0).getRgb()[:3] == (0, 0, 255)   # pure blue in RGB

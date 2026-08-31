"""Detection tests: the same board must come out of any reasonable screenshot."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from queens.vision import detect, detect_file

DOC = Path(__file__).resolve().parent.parent / "doc"
EXAMPLES = ["Example1.png", "Example2.png", "Example3.png"]


@pytest.fixture(scope="module")
def reference():
    """The region matrix of Example3, used as the expected answer."""
    result = detect_file(DOC / "Example3.png", debug=False)
    assert result.ok, result.error
    return result.board.region


@pytest.mark.parametrize("name", EXAMPLES)
def test_every_screenshot_yields_the_same_board(name, reference):
    """The three screenshots are the same board cropped three ways.

    This is the quality gate of the whole pipeline: what changes between them
    is the surrounding context, so any difference in the result would mean the
    detection is reading the context instead of the board.
    """
    result = detect_file(DOC / name, debug=False)
    assert result.ok, result.error

    board = result.board
    assert board.n == 9
    assert len(board.colors) == 9
    assert board.regions_are_connected()
    assert np.array_equal(board.region, reference)


@pytest.mark.parametrize("factor", [0.5, 0.75, 1.25, 2.0, 3.0])
def test_detection_survives_rescaling(factor, reference):
    """A different browser zoom is a realistic input, so scale must not matter.

    Downscaling fades the thin inner lines and merges neighbouring cells;
    upscaling turns the browser chrome's text into dozens of holes. The lattice
    fit and the hole-uniformity filter are what hold both cases together.
    """
    img = cv2.imread(str(DOC / "Example3.png"))
    interpolation = cv2.INTER_AREA if factor < 1 else cv2.INTER_CUBIC
    scaled = cv2.resize(img, None, fx=factor, fy=factor, interpolation=interpolation)

    result = detect(scaled, debug=False)
    assert result.ok, result.error
    assert np.array_equal(result.board.region, reference)


def test_the_board_is_not_the_largest_contour():
    """Guards the subtlest trap in the pipeline.

    In Example3 the browser chrome is larger than the board and passes every
    shape filter (aspect ratio 1.08, solidity 1.00, four vertices). Picking the
    largest contour would silently detect the wrong thing, so this pins down
    that the accepted frame is the board's, not the chrome's.
    """
    result = detect_file(DOC / "Example3.png")
    assert result.ok, result.error

    stage = next(s for s in result.stages if s.name.startswith("5."))
    assert "(144,236), 416x416" in stage.notes[0]


def test_stages_are_recorded_even_on_failure():
    """A failure must still explain itself: that is what the vision panel shows."""
    header_only = cv2.imread(str(DOC / "Example3.png"))[0:200, :]

    result = detect(header_only)
    assert not result.ok
    assert result.error
    assert len(result.stages) >= 4      # up to the candidate contours at least


def test_blank_image_is_rejected():
    """No dark pixels at all must be reported, not crash."""
    result = detect(np.full((400, 400, 3), 255, dtype=np.uint8))
    assert not result.ok
    assert result.error

"""Detection of the Queens board in a screenshot, with OpenCV.

The pipeline records every intermediate stage (image + notes) as it works. That
record is not optional debugging: it is what feeds the vision panel, where you
can watch the digital input being derived from the PNG.

When something fails, `Detection.error` explains what failed and `stages` keeps
everything reached up to that point. A half-detected board is never returned:
silently handing back a corrupt board would be the worst possible outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .board import Board

# Gray level below which a pixel counts as a board "line". The frame is nearly
# black and the page sits above 200, so there is plenty of margin.
DARK_THRESHOLD = 100

# A candidate contour needs at least this many holes to be the board. This is
# the decisive filter: in the screenshots under doc/ the board has 81 holes and
# the runner-up (the browser chrome) only reaches 12.
MIN_HOLES = 20

# Maximum relative spread of a candidate's hole sizes. The board's holes are its
# cells and all measure the same; a blob whose holes vary wildly is text, not a
# grid. Needed because the browser chrome covers more area than the board and
# would otherwise win on an upscaled screenshot.
MAX_HOLE_SPREAD = 0.25

# Maximum distance in OpenCV's 8-bit Lab space for two cells to count as the
# same region. In the doc/ screenshots sampling yields identical colors within a
# region, and the closest pair of regions (purple and pink) sits at 23.4: we
# must stay below that, with enough margin to absorb rescaling noise.
COLOR_TOLERANCE = 15


@dataclass
class Stage:
    """One intermediate stage of the pipeline, ready to be displayed."""

    name: str
    image: np.ndarray          # BGR
    notes: list[str] = field(default_factory=list)


@dataclass
class Detection:
    board: Board | None
    stages: list[Stage]
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.board is not None


class _Recorder:
    """Collects the stages. When disabled it copies nothing, to save memory."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.stages: list[Stage] = []

    def add(self, name: str, image: np.ndarray, notes: list[str] | None = None) -> None:
        if not self.enabled:
            return
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        self.stages.append(Stage(name, image.copy(), list(notes or ())))


def _label(img: np.ndarray, text: str, org: tuple[int, int],
           color: tuple[int, int, int]) -> None:
    """Text with a white outline, legible over any background."""
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)


def _cluster_1d(values: np.ndarray, gap: float) -> list[float]:
    """Group nearby values and return the center of each group, sorted."""
    centers: list[float] = []
    group: list[float] = []
    for v in np.sort(values):
        if group and v - group[-1] > gap:
            centers.append(float(np.mean(group)))
            group = []
        group.append(float(v))
    if group:
        centers.append(float(np.mean(group)))
    return centers


def _dominant_color(roi: np.ndarray) -> tuple[int, int, int]:
    """Most frequent color of the ROI, quantized to 4 bits per channel.

    The mean is deliberately avoided: if the cell already has a queen or a cross
    drawn on it, the mean is skewed by the glyph. The mode is immune to any
    drawing covering less than half the area.
    """
    q = (roi.reshape(-1, 3) >> 4).astype(np.int32)
    key = (q[:, 0] << 8) | (q[:, 1] << 4) | q[:, 2]
    dominant = int(np.bincount(key).argmax())
    mask = key == dominant
    return tuple(int(v) for v in roi.reshape(-1, 3)[mask].mean(axis=0))


def _group_by_color(colors: list[tuple[int, int, int]],
                    tolerance: float) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Group colors by proximity in Lab. Returns (id per cell, color per id).

    Incremental nearest-centroid rather than k-means: the colors are flat, and
    the number of regions is not known up front — checking that it comes out as
    N is precisely one of the validations.
    """
    swatches = np.array(colors, dtype=np.uint8).reshape(-1, 1, 3)
    lab = cv2.cvtColor(swatches, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)

    ids: list[int] = []
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    for i, point in enumerate(lab):
        if centroids:
            distances = [np.linalg.norm(point - c) for c in centroids]
            best = int(np.argmin(distances))
            if distances[best] <= tolerance:
                ids.append(best)
                members[best].append(i)
                centroids[best] = lab[members[best]].mean(axis=0)
                continue
        ids.append(len(centroids))
        centroids.append(point.copy())
        members.append([i])

    palette = [tuple(int(v) for v in np.array(colors)[m].mean(axis=0)) for m in members]
    return ids, palette


def _hole_spread(hole_rects: list[tuple[int, int, int, int]]) -> float:
    """Relative spread of the holes' side lengths (0 = all identical).

    A board's holes are its cells and they all measure the same. This is what
    rules out the browser chrome: on an upscaled screenshot its text glyphs
    become dozens of holes, enough to pass the hole count, but they range from
    5 px to over a thousand. Without this the chrome wins, because it covers
    more area than the board.

    Median absolute deviation, not standard deviation: on a downscaled
    screenshot some neighbouring cells merge into one double-sized hole, and a
    mean-based measure would blow up over that minority and reject the real
    board. The MAD ignores them, while text — where nearly every hole differs —
    still scores high.
    """
    if not hole_rects:
        return float("inf")
    sides = np.array([s for (_, _, w, h) in hole_rects for s in (w, h)], dtype=float)
    median = np.median(sides)
    if median <= 0:
        return float("inf")
    return float(np.median(np.abs(sides - median)) / median)


def _find_board(dark: np.ndarray, img: np.ndarray, rec: _Recorder):
    """Locate the outer frame contour. Returns (index, contours, hierarchy).

    It leans on RETR_CCOMP: because the inner lines touch the frame, the whole
    board is a single top-level contour and its cells are its holes. Counting
    holes is what separates it from any other dark square.
    """
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    if hierarchy is None:          # an image with no dark pixels at all
        rec.add("4. Candidate contours", img, ["No dark contour found in the image."])
        return None, contours, np.empty((0, 4), dtype=np.int32)
    hierarchy = hierarchy[0]

    holes_of: dict[int, list] = {}
    for i, h in enumerate(hierarchy):
        if h[3] != -1:
            holes_of.setdefault(h[3], []).append(cv2.boundingRect(contours[i]))

    min_area = 0.02 * img.shape[0] * img.shape[1]
    overlay = img.copy()
    survivors, notes = [], []

    for i, h in enumerate(hierarchy):
        if h[3] != -1:                       # top-level contours only
            continue
        contour = contours[i]
        area = cv2.contourArea(contour)
        if area < 500:                       # noise, not even worth reporting
            continue
        x, y, w, hh = cv2.boundingRect(contour)
        approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)
        aspect = w / hh
        solidity = area / (w * hh)
        hole_rects = holes_of.get(i, [])
        holes = len(hole_rects)
        spread = _hole_spread(hole_rects)

        reason = None
        if area < min_area:
            reason = f"area {area:.0f} < 2% of the image"
        elif len(approx) != 4:
            reason = f"{len(approx)} vertices, not 4"
        elif not cv2.isContourConvex(approx):
            reason = "not convex"
        elif not 0.9 <= aspect <= 1.1:
            reason = f"aspect ratio {aspect:.2f} outside [0.9, 1.1]"
        elif solidity <= 0.9:
            reason = f"solidity {solidity:.2f} <= 0.90"
        elif holes < MIN_HOLES:
            reason = f"only {holes} holes, {MIN_HOLES} required"
        elif spread > MAX_HOLE_SPREAD:
            reason = f"holes vary in size ({spread:.2f}), not a grid"

        color = (0, 0, 255) if reason else (0, 170, 0)
        cv2.drawContours(overlay, [approx], -1, color, 2)
        _label(overlay, reason or f"CANDIDATE ok, {holes} holes", (x + 4, max(14, y - 6)), color)
        notes.append(f"({x},{y}) {w}x{hh}  AR={aspect:.2f} solidity={solidity:.2f} "
                     f"holes={holes} spread={spread:.2f}  ->  {reason or 'ACCEPTED'}")

        if reason is None:
            survivors.append((area, i))

    rec.add("4. Candidate contours", overlay,
            ["Every top-level contour with its verdict:", *notes])

    if not survivors:
        return None, contours, hierarchy
    return max(survivors)[1], contours, hierarchy


def detect(img: np.ndarray, debug: bool = True) -> Detection:
    """Detect the Queens board in a BGR image."""
    rec = _Recorder(debug)

    def fail(msg: str) -> Detection:
        return Detection(None, rec.stages, msg)

    if img is None or img.ndim != 3:
        return fail("The image is not valid BGR.")

    rec.add("1. Original", img, [f"{img.shape[1]} x {img.shape[0]} px"])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, dark = cv2.threshold(gray, DARK_THRESHOLD, 255, cv2.THRESH_BINARY_INV)
    rec.add("2. Dark mask", dark,
            [f"Pixels with gray < {DARK_THRESHOLD}: {int(dark.sum() // 255)}",
             "The frame is nearly black; the page sits above 200."])

    dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    rec.add("3. After MORPH_CLOSE", dark,
            ["Closes the gaps left by the antialiasing of the lines,",
             "so that the frame becomes one continuous contour."])

    board_idx, contours, hierarchy = _find_board(dark, img, rec)
    if board_idx is None:
        return fail("No contour looks like a board (see stage 4).")

    bx, by, bw, bh = cv2.boundingRect(contours[board_idx])
    frame = img.copy()
    cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (0, 170, 0), 2)
    rec.add("5. Outer square", frame,
            [f"Frame at ({bx},{by}), {bw}x{bh} px"])

    rec.add("6. Board crop", img[by:by + bh, bx:bx + bw],
            ["The board isolated from the rest of the screenshot."])

    # --- cells: they are the holes of the frame contour ---------------------
    holes = [contours[i] for i, h in enumerate(hierarchy) if h[3] == board_idx]
    if not holes:
        return fail("The frame contains no cells.")

    rects = np.array([cv2.boundingRect(c) for c in holes])
    areas = rects[:, 2] * rects[:, 3]
    rects = rects[areas > 0.3 * np.median(areas)]       # drop antialiasing slivers

    # The holes are NOT trusted one by one, only in aggregate. On a downscaled
    # screenshot the thin inner lines fade above the threshold and neighbouring
    # cells merge into a single hole; on an upscaled one, interpolation splits
    # some. Either way the *median* hole is still one cell, and the holes still
    # span the whole board, so a regular lattice can be fitted over their
    # extent. Requiring exactly N*N holes would break at any browser zoom.
    x0, y0 = rects[:, 0].min(), rects[:, 1].min()
    x1 = (rects[:, 0] + rects[:, 2]).max()
    y1 = (rects[:, 1] + rects[:, 3]).max()
    side = float(np.median(np.concatenate([rects[:, 2], rects[:, 3]])))
    if side < 4:
        return fail(f"Cells of {side:.1f} px are too small to sample reliably.")

    n = int(round(((x1 - x0) / side + (y1 - y0) / side) / 2))
    if not 4 <= n <= 16:
        return fail(f"Implausible board size: N = {n}.")

    cell_w, cell_h = (x1 - x0) / n, (y1 - y0) / n
    cells = np.zeros((n, n, 4), dtype=np.int32)
    for row in range(n):
        for col in range(n):
            cx0, cy0 = round(x0 + col * cell_w), round(y0 + row * cell_h)
            cx1, cy1 = round(x0 + (col + 1) * cell_w), round(y0 + (row + 1) * cell_h)
            cells[row, col] = (cx0, cy0, cx1 - cx0, cy1 - cy0)

    grid = img.copy()
    for row in range(n):
        for col in range(n):
            x, y, w, h = cells[row, col]
            cv2.rectangle(grid, (x, y), (x + w, y + h), (255, 0, 255), 1)
    rec.add("7. Cell grid", grid,
            [f"{len(rects)} holes found, median side {side:.1f} px",
             f"Board spans ({x0},{y0})-({x1},{y1})  ->  N = {n}",
             f"Fitted cell: {cell_w:.1f} x {cell_h:.1f} px",
             "The lattice is fitted over the holes' extent rather than taken",
             "hole by hole, so merged or split holes do not shift the grid."])

    # --- color of each cell -------------------------------------------------
    sampling = img.copy()
    colors: list[tuple[int, int, int]] = []
    for row in range(n):
        for col in range(n):
            x, y, w, h = cells[row, col]
            mx, my = int(w * 0.2), int(h * 0.2)
            roi = img[y + my:y + h - my, x + mx:x + w - mx]
            color = _dominant_color(roi)
            colors.append(color)
            cv2.rectangle(sampling, (x + mx, y + my), (x + w - mx, y + h - my), (0, 0, 0), 1)
            cv2.rectangle(sampling, (x + mx + 1, y + my + 1),
                          (x + mx + 10, y + my + 10), color, -1)
    rec.add("8. Color sampling", sampling,
            ["ROI of the central 60% of each cell, to avoid the border.",
             "The quantized mode is used rather than the mean, so a queen or",
             "a cross already drawn does not alter the cell's color."])

    # --- regions ------------------------------------------------------------
    ids, palette = _group_by_color(colors, COLOR_TOLERANCE)
    region = np.array(ids, dtype=np.int32).reshape(n, n)

    regions_img = img.copy()
    for row in range(n):
        for col in range(n):
            x, y, w, h = cells[row, col]
            _label(regions_img, str(region[row, col]),
                   (x + w // 2 - 4, y + h // 2 + 5), (0, 0, 0))
    rec.add("9. Regions", regions_img,
            [f"{len(palette)} regions grouped in Lab (tolerance {COLOR_TOLERANCE}).",
             *[f"  region {i}: BGR {c}" for i, c in enumerate(palette)]])

    board = Board(n=n, region=region, colors=palette, cells=cells)

    if len(palette) != n:
        return fail(f"Expected {n} regions but got {len(palette)}.")
    if not board.regions_are_connected():
        return fail("Some region is not connected: the color grouping failed.")

    borders = img.copy()
    for row in range(n):
        for col in range(n):
            x, y, w, h = cells[row, col]
            for drow, dcol, p0, p1 in (
                (-1, 0, (x, y), (x + w, y)),
                (1, 0, (x, y + h), (x + w, y + h)),
                (0, -1, (x, y), (x, y + h)),
                (0, 1, (x + w, y), (x + w, y + h)),
            ):
                if board.is_border(row, col, drow, dcol):
                    cv2.line(borders, p0, p1, (0, 0, 0), 3)
                else:
                    cv2.line(borders, p0, p1, (160, 160, 160), 1)
    rec.add("10. Borders", borders,
            ["Thick stroke where the region changes, thin where it does not.",
             "This is the model's final input, and what makes the board read",
             "like LinkedIn's rather than like a uniform grid."])

    return Detection(board, rec.stages)


def detect_file(path: str, debug: bool = True) -> Detection:
    """Detect the board in an image file."""
    img = cv2.imread(str(path))
    if img is None:
        return Detection(None, [], f"Could not read the image: {path}")
    return detect(img, debug)

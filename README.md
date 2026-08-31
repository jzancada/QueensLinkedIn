# LinkedIn Queens — vision, solver and step-by-step

Reads a board of LinkedIn's **Queens** puzzle from a PNG or a screenshot,
detects it with **OpenCV**, solves it by **backtracking**, and lets you play it.

The goal is not just to solve it: it is to **watch it being solved**. Nothing
should be a black box, so the project has two observable panels — one to see how
the digital input is derived from the image, and another to watch the
backtracking move forward and backtrack, step by step, on the board.

## Status

| Part | Status |
|---|---|
| OpenCV detection (`queens/vision.py`) | working, and stable from 0.5× to 3× rescaling |
| Board model (`queens/board.py`) | done |
| Backtracking solver | pending |
| PySide6 interface (3 tabs) | pending |

## How the detection works

The board is a thick black frame on a light background, and its inner lines
touch that frame: with `RETR_CCOMP` the whole board comes out as **a single
contour** whose **holes are its cells**. Counting holes is what tells it apart
from any other dark square in the screenshot — and it is needed, because the
largest contour in the image is **not** the board but the browser chrome, which
passes every shape filter (aspect ratio 1.08, solidity 1.00, 4 vertices) and is
rejected only for having 12 holes instead of 81.

Those holes are not trusted one by one, only in aggregate: a **regular lattice
is fitted** over their extent. That is what keeps the detection stable across
browser zoom levels — downscaling fades the thin inner lines and merges
neighbouring cells, upscaling splits them, and neither shifts the grid.

Each cell's color is the **quantized mode** of its central 60 %, not the mean,
so a queen or a cross already drawn on the board does not skew it. Regions are
grouped by proximity in Lab, and the result is validated before being returned:
there must be exactly N regions and each one must be connected.

## Install

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code, `.vscode/settings.json` already points at `.venv` and enables the
pytest Test Explorer; accept the recommended extensions when prompted.

## Running

There is no GUI yet, but the pipeline records all ten of its stages, and they
can be dumped to disk:

```bat
python -m queens.dump_stages doc/Example3.png
```

This writes `out/01_original.png` … `out/10_borders.png` and prints each
stage's notes. The two most informative are **`04_candidate_contours.png`**,
where every candidate is labelled with the reason it was rejected, and
**`10_borders.png`**, whose thick strokes should line up exactly with the real
color boundaries.

```bat
python -m pytest
```

## Test screenshots

`doc/Example1..3.png` are the same 9×9 board cropped three different ways. The
detection test is that all three yield **the same** region matrix. They have
been sanitized: no avatar, no notification badge.

## Rules of the game

1. Exactly one queen per row.
2. Exactly one per column.
3. Exactly one per color region.
4. Two queens may not touch, **not even diagonally** — but they may share a
   diagonal at distance ≥ 2. This is the difference from the classic N-queens
   problem.

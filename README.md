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
| Backtracking solver (`queens/solver.py`) | working, and every step is observable |
| PySide6 interface (`queens/ui/`) | done: the three panels |

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

## How the solver works

Plain backtracking, filling one row at a time: a partial state is just the
column chosen for each row so far, and the one-queen-per-row rule comes for
free from that shape.

What it adds is that the search is a **stream of steps** — every attempt, every
rejection *with the rule it broke*, every queen placed and every one taken
back. Each step carries the board state after it, so the solver panel will be
able to animate the search by replaying the stream, with no bookkeeping of its
own. `solve()` is the same search with the steps thrown away.

On top of that there is a **look-ahead**: after each placement, every region
and every column that has no legal cell left in the rows below is already lost,
so the branch is cut right there instead of thousands of attempts later. It
shows in the trace as a queen placed and immediately withdrawn, with the reason
attached — *`column 8 has no cell left in rows 4-8`*.

On the 9×9 board of `doc/` that takes the search from **7299 attempts to 549**:
the regions alone bring it down to 2151, and the columns do the rest. The point
is not speed — the naive search is instant too — but a trace short enough to be
watched. `--no-prune` runs it without, to compare.

## Install

```bat
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

In VS Code, `.vscode/settings.json` already points at `.venv` and enables the
pytest Test Explorer; accept the recommended extensions when prompted.

## Running

The three panels are the three ways of looking at the same board:

1. **Vision** — the ten stages with their notes, and a click on the image names
   the cell underneath: region, color and rectangle.
2. **Solver** — the search played on the board, at any speed or one step at a
   time, coloring the cell each step is about and saying why it was rejected or
   cut. The checkbox turns the look-ahead off, which is the comparison the
   panel exists to make watchable.
3. **Play** — the board to solve by hand: a click crosses a cell out, another
   turns it into a queen, a third clears it. Every change is judged at once and
   the queens breaking a rule are washed in red, with the rule named. Nothing
   is forbidden — an illegal queen can be placed and left there. *Hint* points
   out a wrong queen if there is one, and only otherwise adds a right one.

```bat
python -m queens.ui
python -m queens.ui doc/Example3.png
```

> On a Python that comes from Anaconda, Qt used to die on import with
> `WinError 127`: it finds Anaconda's ICU 73 before Windows' own, and Qt's
> build needs the latter. `queens/ui/__init__.py` pins the right one, so
> nothing has to be installed or uninstalled.

The pipeline also records all ten of its detection stages, and they can be
dumped to disk without any GUI:

```bat
python -m queens.dump_stages doc/Example3.png
```

This writes `out/01_original.png` … `out/10_borders.png` and prints each
stage's notes. The two most informative are **`04_candidate_contours.png`**,
where every candidate is labelled with the reason it was rejected, and
**`10_borders.png`**, whose thick strokes should line up exactly with the real
color boundaries.

The solver reads a screenshot straight through the detection and prints the
board twice, as regions and with the queens on it:

```bat
python -m queens.solver doc/Example3.png
python -m queens.solver doc/Example3.png --trace
python -m queens.solver doc/Example3.png --trace --no-prune
```

`--trace` prints every step of the search, each with the rule that rejected it
or the region that stranded it — over a thousand lines for a 9×9 board even
pruned, which is exactly the point of showing it on a panel instead.

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

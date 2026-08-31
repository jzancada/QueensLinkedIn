"""Dump every detection stage to disk, to inspect the pipeline from a terminal.

A stand-in for the vision panel until the GUI exists — and useful afterwards
too, for diffing runs or attaching images to a bug report.

    python -m queens.dump_stages doc/Example3.png
    python -m queens.dump_stages doc/Example1.png -o out/example1
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2

from .vision import detect_file


def _slug(name: str) -> str:
    """'4. Candidate contours' -> '04_candidate_contours'"""
    number, _, rest = name.partition(".")
    return f"{int(number):02d}_" + re.sub(r"[^a-z0-9]+", "_", rest.strip().lower())


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("image", help="screenshot containing a Queens board")
    parser.add_argument("-o", "--out", default="out", help="output directory (default: out)")
    args = parser.parse_args(argv)

    result = detect_file(args.image)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for stage in result.stages:
        path = out / f"{_slug(stage.name)}.png"
        cv2.imwrite(str(path), stage.image)
        print(f"\n=== {stage.name}  ->  {path}")
        for note in stage.notes:
            print(f"    {note}")

    print()
    if not result.ok:
        print(f"DETECTION FAILED: {result.error}")
        print(f"The {len(result.stages)} stages above show how far it got.")
        return 1

    board = result.board
    print(f"Board {board.n}x{board.n} with {len(board.colors)} regions "
          f"of sizes {[int(s) for s in board.region_sizes()]}")
    print()
    for row in board.region:
        print("   " + " ".join(str(v) for v in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())

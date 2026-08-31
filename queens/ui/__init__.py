"""PySide6 interface: three panels over the same pipeline.

1. Vision — how the board is derived from the screenshot, stage by stage.
2. Solver — the backtracking search, moving forward and backtracking.
3. Play — the board to solve by hand.

The interface owns no logic of its own: it displays what `vision`, `board` and
`solver` produce. Anything it needs to show, those modules already record.
"""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path


def _pin_system_icu() -> None:
    """Load Windows' own ICU before Qt gets the chance to pick another one.

    Qt 6.11's Windows build imports `icuuc.dll` by its unversioned symbols and
    the PySide6 wheel ships no ICU of its own: it counts on the one in
    System32. If the interpreter has another ICU on its DLL search path — an
    Anaconda base environment puts ICU 73 there, whose symbols all carry a
    `_73` suffix — the loader finds that one first and importing QtCore dies
    with a bare "WinError 127: procedure not found".

    Loading the right one first is enough: from then on the loader reuses the
    module already in the process. Elsewhere this is a harmless no-op, since
    System32 is what Qt would have found anyway.
    """
    if sys.platform != "win32":
        return
    icu = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "icuuc.dll"
    try:
        if icu.is_file():
            ctypes.WinDLL(str(icu))
    except OSError:
        pass          # nothing lost: Qt will search for it the usual way


_pin_system_icu()

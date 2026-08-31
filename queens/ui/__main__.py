"""Launch the interface.

    python -m queens.ui
    python -m queens.ui doc/Example3.png
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv if argv is None else [sys.argv[0], *argv]
    app = QApplication(argv)

    window = MainWindow()
    if len(argv) > 1:
        window.load(argv[1])
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())

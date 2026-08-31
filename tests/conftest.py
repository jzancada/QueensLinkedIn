"""Qt must run headless under pytest, or the UI tests would pop up windows.

Set before anything imports QtGui: the platform plugin is chosen when the
QApplication is created, and there is no going back afterwards.
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

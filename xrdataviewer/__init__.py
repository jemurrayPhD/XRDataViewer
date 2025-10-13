"""XRDataViewer application package."""

import os

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide2")

from .app import MainWindow, main

__all__ = ["MainWindow", "main"]

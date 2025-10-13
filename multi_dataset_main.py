#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility launcher for the XRDataViewer application."""

import os


# Force pyqtgraph to use the same Qt binding as the rest of the application.
os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide2")


from xrdataviewer.app import main


if __name__ == "__main__":
    main()

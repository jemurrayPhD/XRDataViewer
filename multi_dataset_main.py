#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility launcher for the XRDataViewer application."""

from xrdataviewer.app import main as xrdataviewer_main


def main() -> None:
    """Launch the XRDataViewer application."""

    xrdataviewer_main()


if __name__ == "__main__":
    main()

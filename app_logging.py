"""Application-wide logging helpers for XRDataViewer."""
from __future__ import annotations

from typing import List

from PySide2 import QtCore


class ActionLogger(QtCore.QObject):
    """Collects user-facing log entries and emits updates for listeners."""

    log_added = QtCore.Signal(str)
    reset = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._entries: List[str] = []

    def log(self, message: str):
        timestamp = QtCore.QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
        entry = f"[{timestamp}] {message}"
        self._entries.append(entry)
        self.log_added.emit(entry)

    def entries(self) -> List[str]:
        return list(self._entries)

    def to_text(self) -> str:
        return "\n".join(self._entries)

    def clear(self):
        self._entries.clear()
        self.reset.emit()


ACTION_LOGGER = ActionLogger()


def log_action(message: str):
    """Record a user-facing action in the shared logger."""
    ACTION_LOGGER.log(message)

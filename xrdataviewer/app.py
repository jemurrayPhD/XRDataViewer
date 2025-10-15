from __future__ import annotations

from pathlib import Path
import os
from typing import Callable, Dict, Optional

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PySide2")

import pyqtgraph as pg
from PySide2 import QtCore, QtGui, QtWidgets, QtUiTools

from app_logging import ACTION_LOGGER, log_action

from .appearance import build_stylesheet
from .colormaps import register_scientific_colormaps
from .datasets import DatasetsPane, SliceDataTab
from .interactive import InteractiveBridgeServer, InteractiveProcessingTab
from .logging.panel import LoggingDockWidget
from .preferences import PreferencesDialog, PreferencesManager
from .processing import ProcessingDockContainer, ProcessingDockWidget, ProcessingManager
from .views.multiview import MultiViewGrid
from .views.overlay import OverlayView
from .views.sequential import SequentialView
from .widget_sizes import enable_widget_size_overlays

def _supports_button_word_wrap() -> bool:
    """Return True if the Qt build exposes a wordWrap property on buttons."""

    checkbox = QtWidgets.QCheckBox()
    radio = QtWidgets.QRadioButton()
    try:
        has_checkbox = checkbox.metaObject().indexOfProperty("wordWrap") >= 0
        has_radio = radio.metaObject().indexOfProperty("wordWrap") >= 0
    finally:
        checkbox.deleteLater()
        radio.deleteLater()
    return has_checkbox and has_radio


class StartupSplash(QtWidgets.QWidget):
    """Frameless splash screen that surfaces Jupyter startup progress."""

    startupComplete = QtCore.Signal(bool)

    def __init__(self) -> None:
        super().__init__(None, QtCore.Qt.SplashScreen | QtCore.Qt.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setWindowModality(QtCore.Qt.ApplicationModal)

        self._completed = False

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 18)
        outer.setSpacing(0)

        frame = QtWidgets.QFrame()
        frame.setObjectName("startupFrame")
        frame.setStyleSheet(
            "#startupFrame {"
            "background-color: rgba(255, 255, 255, 235);"
            "border-radius: 16px;"
            "border: 1px solid rgba(0, 0, 0, 40);"
            "}"
        )
        outer.addWidget(frame)

        layout = QtWidgets.QVBoxLayout(frame)
        layout.setContentsMargins(32, 32, 32, 28)
        layout.setSpacing(16)

        self.logo_label = QtWidgets.QLabel()
        self.logo_label.setAlignment(QtCore.Qt.AlignCenter)
        self.logo_label.setPixmap(self._build_logo())
        layout.addWidget(self.logo_label, alignment=QtCore.Qt.AlignCenter)

        title = QtWidgets.QLabel("Starting XRDataViewer")
        title.setAlignment(QtCore.Qt.AlignCenter)
        title.setStyleSheet("font-size: 22px; font-weight: 600; color: #1f2d3d;")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel("JupyterLab startup log")
        subtitle.setAlignment(QtCore.Qt.AlignLeft)
        subtitle.setStyleSheet("font-weight: 600; color: #375a7f;")
        layout.addWidget(subtitle)

        self.log_view = QtWidgets.QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setObjectName("startupLog")
        self.log_view.document().setMaximumBlockCount(500)
        self.log_view.setStyleSheet(
            "#startupLog {"
            "background: rgba(246, 248, 251, 240);"
            "border: 1px solid rgba(55, 90, 127, 60);"
            "border-radius: 8px;"
            "color: #1f2d3d;"
            "font-family: 'Source Code Pro', 'Consolas', 'Courier New', monospace;"
            "font-size: 12px;"
            "padding: 8px;"
            "}"
        )
        self.log_view.setFixedHeight(200)
        self.log_view.setPlainText("Waiting for embedded JupyterLab…")
        layout.addWidget(self.log_view)

        self.status_label = QtWidgets.QLabel("Initializing…")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("color: #4a4a4a; font-size: 13px;")
        layout.addWidget(self.status_label)

        self.resize(560, 420)

    def _build_logo(self) -> QtGui.QPixmap:
        pixmap = QtGui.QPixmap(320, 140)
        pixmap.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pixmap)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        gradient = QtGui.QLinearGradient(0, 0, pixmap.width(), pixmap.height())
        gradient.setColorAt(0.0, QtGui.QColor("#1D976C"))
        gradient.setColorAt(1.0, QtGui.QColor("#1FA2FF"))
        painter.setBrush(QtGui.QBrush(gradient))
        painter.setPen(QtCore.Qt.NoPen)
        painter.drawRoundedRect(pixmap.rect(), 24, 24)

        font = QtGui.QFont("Segoe UI", 64, QtGui.QFont.Bold)
        painter.setFont(font)
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff")))
        text_rect = pixmap.rect().adjusted(0, 0, 0, -10)
        painter.drawText(text_rect, QtCore.Qt.AlignCenter, "XRDV")
        painter.end()
        return pixmap

    def showEvent(self, event: QtGui.QShowEvent) -> None:  # type: ignore[override]
        super().showEvent(event)
        self._center_on_screen()

    def _center_on_screen(self) -> None:
        screen = QtWidgets.QApplication.primaryScreen()
        if not screen:
            return
        geometry = screen.availableGeometry()
        frame_geom = self.frameGeometry()
        frame_geom.moveCenter(geometry.center())
        self.move(frame_geom.topLeft())

    def startup_callbacks(self) -> Dict[str, Callable]:
        return {
            "starting": self._on_starting,
            "message": self.append_log,
            "ready": self._on_ready,
            "failed": self._on_failed,
        }

    def append_log(self, message: str) -> None:
        if not message:
            return
        if self.log_view.toPlainText().strip() == "Waiting for embedded JupyterLab…":
            self.log_view.clear()
        cursor = self.log_view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        cursor.insertText(message + "\n")
        self.log_view.setTextCursor(cursor)
        self.log_view.ensureCursorVisible()

    def _on_starting(self) -> None:
        self.status_label.setText("Launching embedded JupyterLab…")

    def _on_ready(self, url: str) -> None:
        if url:
            self.append_log(f"JupyterLab ready at {url}")
        else:
            self.append_log("JupyterLab ready.")
        self.status_label.setStyleSheet("color: #1c7c54; font-size: 13px;")
        self.status_label.setText("Embedded JupyterLab ready. Opening XRDataViewer…")
        self._schedule_complete(True, delay=600)

    def _on_failed(self, message: str) -> None:
        if message:
            self.append_log(f"Startup failed: {message}")
        self.status_label.setStyleSheet("color: #a33; font-size: 13px;")
        self.status_label.setText("Embedded JupyterLab failed to start. Opening XRDataViewer…")
        self._schedule_complete(False, delay=800)

    def notify_no_jupyter(self) -> None:
        self.append_log("QtWebEngine not available; embedded JupyterLab disabled.")
        self.status_label.setText("QtWebEngine unavailable. Skipping embedded JupyterLab.")
        self._schedule_complete(True, delay=400)

    def _schedule_complete(self, success: bool, delay: int = 0) -> None:
        if self._completed:
            return
        self._completed = True

        def emit_complete() -> None:
            self.startupComplete.emit(success)

        QtCore.QTimer.singleShot(delay, emit_complete)



class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, startup_splash: Optional[StartupSplash] = None):
        super().__init__()
        self.setWindowTitle("Dataset Multi-Viewer")

        self.preferences = PreferencesManager()
        self.preferences.changed.connect(self._on_preferences_changed)
        self._current_stylesheet = ""
        self._button_word_wrap_supported = _supports_button_word_wrap()
        self.processing_manager = ProcessingManager()
        self._startup_splash = startup_splash

        placeholders = self._load_central_layout()
        dataset_host = placeholders["dataset_host"]
        processing_host = placeholders["processing_host"]
        self.tabs = placeholders["tabs"]
        self._main_splitter = placeholders["main_splitter"]
        self._left_splitter = placeholders["left_splitter"]

        self.datasets = DatasetsPane(dataset_host)
        dataset_layout = dataset_host.layout() or QtWidgets.QVBoxLayout(dataset_host)
        dataset_layout.setContentsMargins(0, 0, 0, 0)
        dataset_layout.setSpacing(0)
        dataset_layout.addWidget(self.datasets)

        self.bridge_server = InteractiveBridgeServer(self.datasets, self)
        self.bridge_server.start()

        self.processing_dock = ProcessingDockWidget(self.processing_manager)
        self.processing_panel = ProcessingDockContainer(
            "Processing Pipelines", self.processing_dock, processing_host
        )
        processing_layout = processing_host.layout() or QtWidgets.QVBoxLayout(processing_host)
        processing_layout.setContentsMargins(0, 0, 0, 0)
        processing_layout.setSpacing(0)
        processing_layout.addWidget(self.processing_panel)

        if self._left_splitter is not None:
            self._left_splitter.setStretchFactor(0, 1)
            self._left_splitter.setStretchFactor(1, 1)
            QtCore.QTimer.singleShot(0, lambda: self._left_splitter.setSizes([340, 420]))

        if self._main_splitter is not None:
            self._main_splitter.setStretchFactor(1, 1)

        self._build_menus()

        self.tab_multiview = MultiViewGrid(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_multiview, "MultiView")
        self.tab_sequential = SequentialView(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_sequential, "Sequential View")
        self.tab_overlay = OverlayView(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_overlay, "Overlay")
        self.tab_overlay.set_processing_manager(self.processing_manager)
        self.tab_slice = SliceDataTab(self.datasets)
        self.tabs.addTab(self.tab_slice, "Slice Data")
        startup_callbacks: Optional[Dict[str, Callable]] = None
        if self._startup_splash is not None:
            startup_callbacks = self._startup_splash.startup_callbacks()

        self.tab_interactive = InteractiveProcessingTab(
            self.datasets,
            self.bridge_server,
            preferences=self.preferences,
            startup_callbacks=startup_callbacks,
        )
        self.tabs.addTab(self.tab_interactive, "Interactive Processing")

        self.log_dock = LoggingDockWidget(ACTION_LOGGER, self)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.setFloating(False)
        self.log_dock.resize(800, 200)
        self.log_dock.setCollapsed(True)

        self.resize(1220, 780)
        self.setMinimumSize(920, 640)

        if self._startup_splash is not None and not self.tab_interactive.has_embedded_jupyter:
            self._startup_splash.notify_no_jupyter()

        self._on_preferences_changed(self.preferences.data())
        try:
            enable_widget_size_overlays()
        except RuntimeError:
            pass

    def closeEvent(self, event: QtGui.QCloseEvent):  # type: ignore[override]
        try:
            self.tab_interactive.shutdown()
        except Exception:
            pass
        try:
            self.bridge_server.stop()
        except Exception:
            pass
        super().closeEvent(event)

    def _build_menus(self):
        menubar = self.menuBar()
        if not isinstance(menubar, QtWidgets.QMenuBar):
            menubar = QtWidgets.QMenuBar(self)
            self.setMenuBar(menubar)
        prefs_menu = QtWidgets.QMenu("Preferences", menubar)
        menubar.addMenu(prefs_menu)

        act_edit = QtWidgets.QAction("Edit preferences…", self)
        act_edit.triggered.connect(self._edit_preferences)
        prefs_menu.addAction(act_edit)

        prefs_menu.addSeparator()

        act_load = QtWidgets.QAction("Load preferences…", self)
        act_load.triggered.connect(self._load_preferences)
        prefs_menu.addAction(act_load)

        act_save = QtWidgets.QAction("Save preferences…", self)
        act_save.triggered.connect(self._save_preferences)
        prefs_menu.addAction(act_save)

    def _edit_preferences(self):
        dialog = PreferencesDialog(self.preferences, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            data = dialog.result_data()
            self.preferences.update(data)
            log_action("Updated preferences")

    def _load_preferences(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load preferences",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.preferences.load_from_file(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        log_action(f"Loaded preferences from {path}")

    def _save_preferences(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save preferences",
            "preferences.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            self.preferences.save_to_file(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))
            return
        log_action(f"Saved preferences to {path}")

    def _load_central_layout(self) -> Dict[str, QtWidgets.QWidget]:
        ui_path = Path(__file__).resolve().parent / "ui" / "main_window.ui"
        if not ui_path.exists():
            raise RuntimeError(f"Main window UI not found at {ui_path}")

        ui_file = QtCore.QFile(str(ui_path))
        if not ui_file.open(QtCore.QIODevice.ReadOnly):
            raise RuntimeError(f"Unable to open UI definition {ui_path}")

        loader = QtUiTools.QUiLoader()
        loader.setWorkingDirectory(str(ui_path.parent))

        try:
            central = loader.load(ui_file, self)
        finally:
            ui_file.close()

        if central is None:
            raise RuntimeError(f"Failed to load UI from {ui_path}: {loader.errorString()}")

        self.setCentralWidget(central)

        main_splitter = central.findChild(QtWidgets.QSplitter, "mainSplitter")
        left_splitter = central.findChild(QtWidgets.QSplitter, "leftSplitter")
        tabs = central.findChild(QtWidgets.QTabWidget, "mainTabs")
        dataset_host = central.findChild(QtWidgets.QWidget, "datasetsHost")
        processing_host = central.findChild(QtWidgets.QWidget, "processingHost")

        if None in (main_splitter, left_splitter, tabs, dataset_host, processing_host):
            raise RuntimeError("Main window UI is missing required widgets")

        return {
            "central": central,
            "main_splitter": main_splitter,
            "left_splitter": left_splitter,
            "tabs": tabs,
            "dataset_host": dataset_host,
            "processing_host": processing_host,
        }


def main() -> None:
    """Start the Qt application with the splash screen and main window."""

    app = QtWidgets.QApplication([])
    enable_widget_size_overlays()
    pg.setConfigOptions(imageAxisOrder="row-major")
    splash = StartupSplash()
    splash_holder: Dict[str, Optional[StartupSplash]] = {"widget": splash}

    def _clear_splash(*_args: object) -> None:
        splash_holder["widget"] = None

    splash.destroyed.connect(_clear_splash)
    splash.show()
    QtWidgets.QApplication.processEvents()

    window: Optional[MainWindow] = None

    def _finish_startup(*_args: object) -> None:
        nonlocal window
        splash_widget = splash_holder["widget"]
        if splash_widget is not None and splash_widget.isVisible():
            splash_widget.close()
        if window is not None and not window.isVisible():
            window.show()

    splash.startupComplete.connect(_finish_startup)
    QtCore.QTimer.singleShot(20000, _finish_startup)

    window = MainWindow(startup_splash=splash)
    splash_widget = splash_holder["widget"]
    if splash_widget is None or not splash_widget.isVisible():
        window.show()
    app.exec_()

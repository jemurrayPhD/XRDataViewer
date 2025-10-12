#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import io
import base64
import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple
from dataclasses import replace
from functools import partial
import itertools
import copy
import traceback
import importlib
import pkgutil
import sys
import os
import re
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
from types import ModuleType

import numpy as np
import xarray as xr

from PySide2 import QtCore, QtWidgets, QtGui
try:  # Optional dependency for embedded JupyterLab wrapper
    from PySide2 import QtWebEngineWidgets  # type: ignore
except Exception:  # pragma: no cover - QtWebEngine may be unavailable
    QtWebEngineWidgets = None
import pyqtgraph as pg
try:
    import pyqtgraph.opengl as gl
except Exception:  # pragma: no cover - optional dependency
    gl = None

from contextlib import redirect_stdout, redirect_stderr

from xr_plot_widget import (
    CentralPlotWidget,
    PlotAnnotationConfig,
    ScientificAxisItem,
    apply_plotitem_annotation,
    plotitem_annotation_state,
)
from xr_coords import guess_phys_coords
from data_processing import (
    ParameterDefinition,
    ProcessingPipeline,
    ProcessingStep,
    apply_processing_step,
    get_processing_function,
    list_processing_functions,
    summarize_parameters,
)
from app_logging import ACTION_LOGGER, ActionLogger, log_action

try:  # Optional dependency used when exporting movies
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV may not be installed
    cv2 = None  # type: ignore

# ---------------------------------------------------------------------------
# Helper: open_dataset
# ---------------------------------------------------------------------------
_FORCE_ENGINE = None  # override to force a specific xarray engine if desired

def open_dataset(path: Path) -> xr.Dataset:
    if path.suffix.lower() == '.zarr' or path.name.lower().endswith('.zarr'):
        return xr.open_zarr(str(path))
    if _FORCE_ENGINE:
        return xr.open_dataset(str(path), engine=_FORCE_ENGINE)
    return xr.open_dataset(str(path))


class EmbeddedJupyterManager(QtCore.QObject):
    """Launch and manage an embedded JupyterLab server for the interactive tab."""

    urlReady = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    message = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._token: Optional[str] = None
        self._url: Optional[str] = None
        self._stop_event = threading.Event()
        self._ready_emitted = False
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._watcher_thread: Optional[threading.Thread] = None

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        if self.is_running():
            if self._url:
                self.urlReady.emit(self._url)
            return True

        command = self._resolve_command()
        if command is None:
            self.failed.emit(
                "JupyterLab executable not found. Install jupyterlab to use the embedded environment."
            )
            return False

        self._stop_event.clear()
        self._ready_emitted = False
        desired_port = 8888
        if not self._is_port_available(desired_port):
            self.failed.emit(
                "Port 8888 is already in use. Stop the existing service or free the port before launching the embedded JupyterLab server."
            )
            return False

        self._port = desired_port
        self._token = None
        self._url = f"http://127.0.0.1:{self._port}/lab"

        args = list(command)
        args.extend(
            [
                "--no-browser",
                f"--ServerApp.port={self._port}",
                "--ServerApp.ip=127.0.0.1",
                "--ServerApp.port_retries=0",
                "--ServerApp.token=",
                "--ServerApp.password=",
                "--ServerApp.open_browser=False",
                "--ServerApp.allow_remote_access=False",
                "--ServerApp.disable_check_xsrf=True",
                f"--ServerApp.root_dir={os.getcwd()}",
            ]
        )

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            self._process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                env=env,
            )
        except Exception as exc:
            self._process = None
            self.failed.emit(f"Failed to start JupyterLab: {exc}")
            return False

        self._stdout_thread = threading.Thread(
            target=self._forward_stream,
            args=(self._process.stdout,),
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._forward_stream,
            args=(self._process.stderr,),
            daemon=True,
        )
        self._watcher_thread = threading.Thread(target=self._watch_process, daemon=True)
        self._stdout_thread.start()
        self._stderr_thread.start()
        self._watcher_thread.start()
        threading.Thread(target=self._probe_ready, daemon=True).start()

        self.message.emit("Starting JupyterLab server…")
        return True

    def stop(self):
        self._stop_event.set()
        if self._process is None:
            return
        proc = self._process
        self._process = None
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
        for attr in ("_stdout_thread", "_stderr_thread", "_watcher_thread"):
            thread = getattr(self, attr, None)
            if thread and thread.is_alive():
                try:
                    thread.join(timeout=1)
                except Exception:
                    pass
            setattr(self, attr, None)
        self._port = None
        self._token = None
        self._url = None
        self._ready_emitted = False

    def url(self) -> Optional[str]:
        return self._url

    def _resolve_command(self) -> Optional[List[str]]:
        for name in ("jupyter-lab", "jupyter-lab.exe", "jupyter-lab.cmd"):
            path = shutil.which(name)
            if path:
                return [path]
        generic = shutil.which("jupyter")
        if generic:
            return [generic, "lab"]
        return None

    def _is_port_available(self, port: int) -> bool:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            result = sock.connect_ex(("127.0.0.1", int(port)))
            return result != 0

    def _forward_stream(self, stream):
        """Relay output from the Jupyter process to the UI signals."""
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                if self._stop_event.is_set():
                    break
                text = line.strip()
                if text:
                    self.message.emit(text)
                    if not self._ready_emitted:
                        match = re.search(r"http[s]?://[^\s]+", text)
                        if match:
                            candidate = match.group(0)
                            if self._token:
                                if f"token={self._token}" not in candidate:
                                    continue
                            self._ready_emitted = True
                            if candidate:
                                self._url = candidate
                            self.urlReady.emit(self._url)
            stream.close()
        except Exception:
            pass

    def _probe_ready(self):
        if not self._url:
            return
        for _ in range(60):
            if self._stop_event.is_set() or not self.is_running():
                return
            try:
                with urllib.request.urlopen(self._url, timeout=1):
                    if not self._ready_emitted:
                        self._ready_emitted = True
                        self.urlReady.emit(self._url)
                    if self._url:
                        self.message.emit(f"JupyterLab ready at {self._url}.")
                    else:
                        self.message.emit("JupyterLab ready.")
                    return
            except Exception:
                time.sleep(0.5)
        if not self._ready_emitted and not self._stop_event.is_set():
            self.failed.emit("Timed out waiting for JupyterLab to start. Check the log for details.")

    def _watch_process(self):
        if self._process is None:
            return
        try:
            code = self._process.wait()
        except Exception:
            return
        if self._stop_event.is_set():
            return
        if code != 0:
            self.failed.emit(f"JupyterLab exited unexpectedly (code {code}).")
        else:
            self.message.emit("JupyterLab server stopped.")

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


class InteractiveBridgeServer(QtCore.QObject):
    """Lightweight HTTP bridge to register datasets from external sessions."""

    datasetRegistered = QtCore.Signal(str)
    _executeRequested = QtCore.Signal(object)

    def __init__(self, library: 'DatasetsPane', parent=None):
        super().__init__(parent)
        self.library = library
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._port: Optional[int] = None
        self._executeRequested.connect(self._on_execute_requested)

    def is_running(self) -> bool:
        return self._httpd is not None

    def start(self) -> Optional[str]:
        if self._httpd is not None:
            return self.register_url()

        server_ref = self

        class BridgeHandler(BaseHTTPRequestHandler):
            def _send_json(self, code: int, payload: Dict[str, object]):
                data = json.dumps(payload).encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_POST(self):  # noqa: N802 - Qt naming style
                if self.path not in ("/register", "/register/"):
                    self._send_json(404, {"ok": False, "error": "Unknown endpoint"})
                    return

                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except Exception:
                    self._send_json(400, {"ok": False, "error": "Missing content length"})
                    return

                raw = self.rfile.read(max(0, length))
                try:
                    payload = json.loads(raw.decode("utf-8")) if raw else {}
                except Exception:
                    self._send_json(400, {"ok": False, "error": "Invalid JSON payload"})
                    return

                encoded = payload.get("payload")
                if not encoded:
                    self._send_json(400, {"ok": False, "error": "Missing dataset payload"})
                    return

                try:
                    decoded = base64.b64decode(encoded)
                except Exception:
                    self._send_json(400, {"ok": False, "error": "Payload is not valid base64"})
                    return

                buffer = io.BytesIO(decoded)
                try:
                    dataset = xr.open_dataset(buffer)
                    dataset.load()
                except Exception as exc:
                    buffer.close()
                    self._send_json(400, {"ok": False, "error": f"Unable to decode dataset: {exc}"})
                    return

                kind = str(payload.get("kind") or "dataset").lower()
                label = str(payload.get("label") or "").strip()

                if kind == "dataarray":
                    var_name = payload.get("var_name")
                    fallback = None
                    try:
                        if var_name and var_name in dataset.data_vars:
                            arr = dataset[var_name]
                        else:
                            names = list(dataset.data_vars)
                            if not names:
                                raise KeyError("dataset contains no data variables")
                            fallback = names[0]
                            arr = dataset[names[0]]
                        arr = arr.copy(deep=True)
                    except Exception as exc:
                        try:
                            dataset.close()
                        except Exception:
                            pass
                        buffer.close()
                        self._send_json(400, {"ok": False, "error": f"Unable to extract data array: {exc}"})
                        return
                    finally:
                        try:
                            dataset.close()
                        except Exception:
                            pass
                        buffer.close()

                    arr.name = var_name or fallback or arr.name or "variable"
                    chosen_label = label or str(arr.name or "interactive_array")
                    try:
                        final_label = server_ref.handle_dataarray(chosen_label, arr)
                    except Exception as exc:
                        self._send_json(500, {"ok": False, "error": str(exc)})
                        return
                else:
                    chosen_label = label or str(
                        dataset.attrs.get("title")
                        or dataset.attrs.get("name")
                        or "interactive_dataset"
                    )
                    try:
                        final_label = server_ref.handle_dataset(chosen_label, dataset)
                    except Exception as exc:
                        try:
                            dataset.close()
                        except Exception:
                            pass
                        buffer.close()
                        self._send_json(500, {"ok": False, "error": str(exc)})
                        return
                    try:
                        dataset.close()
                    except Exception:
                        pass
                    buffer.close()

                self._send_json(200, {"ok": True, "label": final_label})

            def log_message(self, *_args):  # pragma: no cover - silence HTTP logs
                return

        try:
            httpd = ThreadingHTTPServer(("127.0.0.1", 0), BridgeHandler)
        except Exception:
            return None

        httpd.daemon_threads = True
        self._httpd = httpd
        self._port = httpd.server_address[1]
        self._thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        self._thread.start()

        url = self.register_url()
        if url:
            os.environ["XRVIEWER_BRIDGE_URL"] = url
            log_action(f"Interactive bridge listening on {url}")
        return url

    def stop(self):
        url = self.register_url()
        if self._httpd:
            try:
                self._httpd.shutdown()
            except Exception:
                pass
            try:
                self._httpd.server_close()
            except Exception:
                pass
            self._httpd = None
        if self._thread and self._thread.is_alive():
            try:
                self._thread.join(timeout=1.0)
            except Exception:
                pass
        self._thread = None
        self._port = None
        if url and os.environ.get("XRVIEWER_BRIDGE_URL") == url:
            os.environ.pop("XRVIEWER_BRIDGE_URL", None)

    def register_url(self) -> Optional[str]:
        if self._port is None:
            return None
        return f"http://127.0.0.1:{self._port}/register"

    def handle_dataset(self, label: str, dataset: xr.Dataset) -> str:
        result = self._run_on_main(lambda: self.library.register_interactive_dataset(label, dataset))
        self.datasetRegistered.emit(result)
        return result

    def handle_dataarray(self, label: str, dataarray: xr.DataArray) -> str:
        result = self._run_on_main(lambda: self.library.register_interactive_dataarray(label, dataarray))
        self.datasetRegistered.emit(result)
        return result

    def _run_on_main(self, callback):
        payload = {"callback": callback, "event": threading.Event()}
        self._executeRequested.emit(payload)
        payload["event"].wait()
        if "error" in payload:
            raise payload["error"]
        return payload.get("result", "")

    @QtCore.Slot(object)
    def _on_execute_requested(self, payload):
        event = payload.get("event")
        if event is None:
            return
        callback = payload.get("callback")
        try:
            if callable(callback):
                payload["result"] = callback()
        except Exception as exc:
            payload["error"] = exc
        finally:
            event.set()


if QtWebEngineWidgets is not None:
    class QuietWebEnginePage(QtWebEngineWidgets.QWebEnginePage):
        """QWebEnginePage that captures JS console output without spamming stderr."""

        consoleMessage = QtCore.Signal(int, str, int, str)

        def javaScriptConsoleMessage(
            self,
            level: QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel,
            message: str,
            line_number: int,
            source_id: str,
        ) -> None:
            self.consoleMessage.emit(int(level), message, int(line_number), source_id)

class CodeServerManager(QtCore.QObject):
    """Run a local code-server or openvscode-server instance when available."""

    urlReady = QtCore.Signal(str)
    message = QtCore.Signal(str)
    failed = QtCore.Signal(str)
    stopped = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._process: Optional[subprocess.Popen] = None
        self._port: Optional[int] = None
        self._url: Optional[str] = None
        self._kind: Optional[str] = None
        self._stop_event = threading.Event()

    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> bool:
        if self.is_running():
            if self._url:
                self.urlReady.emit(self._url)
            return True

        command = self._resolve_command()
        if command is None:
            self.failed.emit(
                "No code-server or openvscode-server executable found. Install one to embed VS Code here."
            )
            return False

        path, kind = command
        self._kind = kind
        self._port = self._find_free_port()
        self._url = f"http://127.0.0.1:{self._port}/"
        self._stop_event.clear()

        if kind == "code-server":
            args = [path, "--bind-addr", f"127.0.0.1:{self._port}", "--auth", "none", "--disable-telemetry"]
        else:
            args = [path, "--host", "127.0.0.1", "--port", str(self._port), "--without-connection-token"]

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")

        try:
            self._process = subprocess.Popen(
                args,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=os.getcwd(),
                env=env,
            )
        except Exception as exc:
            self._process = None
            self.failed.emit(f"Failed to start {kind}: {exc}")
            return False

        threading.Thread(target=self._forward_stream, args=(self._process.stdout,), daemon=True).start()
        threading.Thread(target=self._forward_stream, args=(self._process.stderr,), daemon=True).start()
        threading.Thread(target=self._probe_ready, daemon=True).start()
        threading.Thread(target=self._watch_process, daemon=True).start()

        self.message.emit("Starting local VS Code server…")
        return True

    def stop(self):
        self._stop_event.set()
        if self._process is None:
            return
        proc = self._process
        self._process = None
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass
        finally:
            self.stopped.emit()
        self._port = None
        self._url = None
        self._kind = None

    def url(self) -> Optional[str]:
        return self._url

    def _resolve_command(self) -> Optional[Tuple[str, str]]:
        for name in ("code-server", "code-server.cmd", "code-server.exe"):
            path = shutil.which(name)
            if path:
                return path, "code-server"
        for name in ("openvscode-server", "openvscode-server.cmd", "openvscode-server.exe"):
            path = shutil.which(name)
            if path:
                return path, "openvscode"
        return None

    def _find_free_port(self) -> int:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
        sock.close()
        return int(port)

    def _forward_stream(self, stream):
        if stream is None:
            return
        try:
            for line in iter(stream.readline, ""):
                if self._stop_event.is_set():
                    break
                text = line.strip()
                if text:
                    self.message.emit(text)
                    match = re.search(r"http[s]?://[^\s]+", text)
                    if match and match.group(0).startswith("http"):
                        self._url = match.group(0)
                        self.urlReady.emit(self._url)
            stream.close()
        except Exception:
            pass

    def _probe_ready(self):
        if not self._url:
            return
        for _ in range(60):
            if self._stop_event.is_set() or not self.is_running():
                return
            try:
                with urllib.request.urlopen(self._url, timeout=1):
                    self.urlReady.emit(self._url)
                    self.message.emit("Local VS Code server ready.")
                    return
            except Exception:
                time.sleep(0.5)
        if not self._stop_event.is_set():
            self.failed.emit("Timed out waiting for the VS Code server to start.")

    def _watch_process(self):
        if self._process is None:
            return
        try:
            code = self._process.wait()
        except Exception:
            return
        if self._stop_event.is_set():
            return
        if code != 0:
            self.failed.emit(f"VS Code server exited unexpectedly (code {code}).")
        else:
            self.message.emit("VS Code server stopped.")
        self.stopped.emit()

    def __del__(self):
        try:
            self.stop()
        except Exception:
            pass


if QtWebEngineWidgets is not None:
    class _ConsoleAwarePage(QtWebEngineWidgets.QWebEnginePage):
        consoleMessage = QtCore.Signal(int, str, int, str)

        def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):  # type: ignore[override]
            super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)
            self.consoleMessage.emit(level, message, lineNumber, sourceID)


class VsCodeWebWidget(QtWidgets.QWidget):
    """Optional embedded VS Code (web) workspace."""

    statusMessage = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        intro = QtWidgets.QLabel(
            "Launch a VS Code compatible environment. Provide a URL, start a local code-server, "
            "or open a desktop instance."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #555;")
        layout.addWidget(intro)

        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("URL:"))
        self.url_edit = QtWidgets.QLineEdit("https://vscode.dev/")
        self.url_edit.setPlaceholderText("https://vscode.dev/ or http://127.0.0.1:port/")
        controls.addWidget(self.url_edit, 1)
        self.btn_open = QtWidgets.QPushButton("Open")
        controls.addWidget(self.btn_open)
        self.btn_start_local = QtWidgets.QPushButton("Start local server")
        controls.addWidget(self.btn_start_local)
        self.btn_stop_local = QtWidgets.QPushButton("Stop server")
        controls.addWidget(self.btn_stop_local)
        self.btn_external = QtWidgets.QPushButton("Launch desktop VS Code")
        controls.addWidget(self.btn_external)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._code_path = shutil.which("code")
        if not self._code_path:
            self.btn_external.setEnabled(False)
            self.btn_external.setToolTip("VS Code command-line launcher not detected in PATH.")

        self.server_manager = CodeServerManager(self)
        self.server_manager.urlReady.connect(self._on_server_ready)
        self.server_manager.message.connect(self._relay_status)
        self.server_manager.failed.connect(self._on_server_failed)
        self.server_manager.stopped.connect(self._on_server_stopped)

        self._js_warning_shown = False
        if QtWebEngineWidgets is not None:
            self._profile = QtWebEngineWidgets.QWebEngineProfile(self)
            self._profile.setHttpUserAgent(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            )
            page = _ConsoleAwarePage(self._profile, self)
            page.consoleMessage.connect(self._handle_console_message)
            self.web_view = QtWebEngineWidgets.QWebEngineView()
            self.web_view.setPage(page)
            self.web_view.setUrl(QtCore.QUrl(self.url_edit.text()))
            layout.addWidget(self.web_view, 1)
            page.loadFinished.connect(lambda ok: self._toggle_error_banner(not ok))
            self._error_banner = QtWidgets.QLabel(self.web_view)
            self._error_banner.setStyleSheet(
                "background: rgba(30, 30, 30, 0.85); color: white; padding: 8px; border-radius: 4px;"
            )
            self._error_banner.setAlignment(QtCore.Qt.AlignCenter)
            self._error_banner.hide()
        else:
            self.web_view = None
            self._error_banner = None
            placeholder = QtWidgets.QTextBrowser()
            placeholder.setReadOnly(True)
            placeholder.setOpenExternalLinks(True)
            placeholder.setHtml(
                "<h2>VS Code web view unavailable</h2>"
                "<p>QtWebEngineWidgets is not installed. Use the desktop launcher or open a browser manually.</p>"
            )
            layout.addWidget(placeholder, 1)
            self.btn_open.setEnabled(False)

        self.lbl_status = QtWidgets.QLabel(" ")
        self.lbl_status.setStyleSheet("color: #666;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        self.btn_open.clicked.connect(self._load_requested_url)
        self.btn_external.clicked.connect(self._launch_desktop_code)
        self.btn_start_local.clicked.connect(self._start_local_server)
        self.btn_stop_local.clicked.connect(self._stop_local_server)
        self._update_server_buttons()
        self._shutdown_called = False
        self.destroyed.connect(lambda *_: self.shutdown())

    def _relay_status(self, text: str):
        if text:
            self.set_status(text)

    def set_status(self, text: str):
        if not text:
            text = " "
        self.lbl_status.setText(text)
        self.statusMessage.emit(text)

    def _load_requested_url(self):
        if self.web_view is None:
            return
        url = QtCore.QUrl.fromUserInput(self.url_edit.text().strip())
        if not url.isValid():
            QtWidgets.QMessageBox.warning(self, "Invalid URL", "Please enter a valid URL to load.")
            return
        self._toggle_error_banner(False)
        self.web_view.setUrl(url)
        self.set_status(f"Loading {url.toString()}…")

    def _launch_desktop_code(self):
        if not self._code_path:
            QtWidgets.QMessageBox.information(
                self,
                "VS Code unavailable",
                "The 'code' launcher was not found in PATH. Install Visual Studio Code or add it to PATH.",
            )
            return
        args = [self._code_path]
        url = self.url_edit.text().strip()
        if url and url.startswith("http"):
            args.extend(["--new-window", url])
        try:
            subprocess.Popen(args)
            self.set_status("Desktop VS Code launched.")
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Launch failed", str(exc))

    def _start_local_server(self):
        if self.server_manager.start():
            self._update_server_buttons()

    def _stop_local_server(self):
        self.server_manager.stop()
        self._update_server_buttons()

    def _on_server_ready(self, url: str):
        self._update_server_buttons()
        if url:
            self.url_edit.setText(url)
            self.set_status(f"Connected to {url}")
            if self.web_view is not None:
                self._toggle_error_banner(False)
                self.web_view.setUrl(QtCore.QUrl(url))

    def _on_server_failed(self, message: str):
        self._update_server_buttons()
        if message:
            QtWidgets.QMessageBox.warning(self, "VS Code server", message)
            self.set_status(message)

    def _on_server_stopped(self):
        self._update_server_buttons()
        self.set_status("Local VS Code server stopped.")

    def _handle_console_message(self, level: int, message: str, line: int, source: str):
        if QtWebEngineWidgets is None:
            return
        cleaned = (message or "").strip()
        location = source or "unknown source"
        if line:
            location = f"{location}:{line}"
        if level == QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
            log_action(f"VS Code JS error ({location}): {cleaned}")
            if "Unexpected token" in cleaned:
                hint = (
                    "The embedded browser hit a syntax error while loading the VS Code workspace. "
                    "This usually means the page requires a newer Chromium engine than the bundled Qt version. "
                    "Open the URL in an external browser or upgrade QtWebEngine."
                )
                self._toggle_error_banner(True, hint)
                if not getattr(self, "_js_warning_shown", False):
                    self.set_status(hint)
                    self._js_warning_shown = True
            else:
                self._toggle_error_banner(False)
                self.set_status(cleaned or "JavaScript error in VS Code view.")
        else:
            if cleaned:
                log_action(f"VS Code JS message ({location}): {cleaned}")

    def _toggle_error_banner(self, show: bool, text: Optional[str] = None):
        if not self._error_banner:
            return
        if not show:
            self._error_banner.hide()
            return
        self._error_banner.setText(text or "Unable to load the requested page.")
        self._error_banner.resize(self.web_view.size())
        self._error_banner.move(0, 0)
        self._error_banner.show()

    def resizeEvent(self, event):  # type: ignore[override]
        super().resizeEvent(event)
        if self._error_banner and self._error_banner.isVisible() and self.web_view is not None:
            self._error_banner.resize(self.web_view.size())

    def _update_server_buttons(self):
        running = self.server_manager.is_running()
        self.btn_start_local.setEnabled(not running)
        self.btn_stop_local.setEnabled(running)

    def shutdown(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        self.server_manager.stop()
        if QtWebEngineWidgets is not None:
            if getattr(self, "web_view", None) is not None:
                page = self.web_view.page()
                self.web_view.setPage(None)
                if page is not None:
                    try:
                        page.deleteLater()
                    except Exception:
                        pass
                try:
                    self.web_view.deleteLater()
                except Exception:
                    pass
                self.web_view = None
            banner = getattr(self, "_error_banner", None)
            if banner is not None:
                try:
                    banner.deleteLater()
                except Exception:
                    pass
                self._error_banner = None
            profile = getattr(self, "_profile", None)
            if profile is not None:
                try:
                    profile.deleteLater()
                except Exception:
                    pass
                self._profile = None

def _nan_aware_reducer(func):
    def wrapped(arr, axis=None):
        data = np.asarray(arr, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = func(data, axis=axis)
        if axis is None:
            if isinstance(result, np.ndarray) and result.shape == ():
                return result.item()
            return result
        try:
            axes = axis
            if axes is None:
                return result
            if isinstance(axes, (list, tuple, set)):
                axes = tuple(int(a) for a in axes)
            else:
                axes = (int(axes),)
            ndim = data.ndim
            normalized = []
            for ax in axes:
                if ax < 0:
                    normalized.append((ax + ndim) % ndim)
                else:
                    normalized.append(ax)
            mask = np.isnan(data)
            for ax in sorted(normalized, reverse=True):
                mask = np.all(mask, axis=ax)
            if isinstance(result, np.ndarray) and np.any(mask):
                result = np.array(result, dtype=float, copy=True)
                result[mask] = np.nan
        except Exception:
            pass
        return result


    return wrapped


def _compose_snapshot(widget: QtWidgets.QWidget, label: str = "") -> QtGui.QImage:
    pixmap = widget.grab()
    image = pixmap.toImage()
    label = (label or "").strip()
    if not label:
        return image
    return _image_with_label(image, label)


def _image_with_label(image: QtGui.QImage, label: str) -> QtGui.QImage:
    label = (label or "").strip()
    if not label:
        return image
    font = QtGui.QFont()
    font.setPointSize(11)
    metrics = QtGui.QFontMetrics(font)
    margin = 8
    label_height = metrics.height() + 2 * margin
    width = image.width()
    result = QtGui.QImage(width, image.height() + label_height, QtGui.QImage.Format_ARGB32)
    result.fill(QtGui.QColor("white"))
    painter = QtGui.QPainter(result)
    painter.fillRect(QtCore.QRect(0, 0, width, label_height), QtGui.QColor("white"))
    painter.setPen(QtGui.QColor("black"))
    painter.setFont(font)
    painter.drawText(QtCore.QRect(0, 0, width, label_height), QtCore.Qt.AlignCenter, label)
    painter.drawImage(0, label_height, image)
    painter.end()
    return result


class ColorButton(QtWidgets.QPushButton):
    colorChanged = QtCore.Signal(QtGui.QColor)

    def __init__(self, color: object = None, parent=None):
        super().__init__(parent)
        self._color = self._coerce_color(color)
        if not self._color.isValid():
            self._color = QtGui.QColor("#1b1b1b")
        self.setFixedWidth(60)
        self.clicked.connect(self._choose_color)
        self._update_style()

    def color(self) -> QtGui.QColor:
        return QtGui.QColor(self._color)

    def setColor(self, color: object):
        qcolor = self._coerce_color(color)
        if not qcolor.isValid() or qcolor == self._color:
            return
        self._color = qcolor
        self._update_style()
        self.colorChanged.emit(QtGui.QColor(self._color))

    def _coerce_color(self, value: object) -> QtGui.QColor:
        if isinstance(value, QtGui.QColor):
            return QtGui.QColor(value)
        if isinstance(value, QtGui.QBrush):
            return QtGui.QColor(value.color())
        if isinstance(value, str):
            return QtGui.QColor(value)
        if isinstance(value, tuple) or isinstance(value, list):
            if len(value) == 3:
                r, g, b = value
                return QtGui.QColor(int(r), int(g), int(b))
            if len(value) >= 4:
                r, g, b, a = value[:4]
                return QtGui.QColor(int(r), int(g), int(b), int(a))
        if isinstance(value, (int, float)):
            c = int(value)
            return QtGui.QColor(c)
        return QtGui.QColor(value) if value is not None else QtGui.QColor()

    def _choose_color(self):
        chosen = QtWidgets.QColorDialog.getColor(self._color, self, "Select background color")
        if chosen.isValid():
            self.setColor(chosen)

    def _update_style(self):
        self.setStyleSheet(
            "QPushButton { border: 1px solid #888; border-radius: 3px; background-color: %s; }"
            % self._color.name()
        )


class PlotAnnotationDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        *,
        initial: Optional[PlotAnnotationConfig] = None,
        allow_apply_all: bool = True,
        template_hint: Optional[str] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle("Set plot annotations")
        self._result: Optional[PlotAnnotationConfig] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        form = QtWidgets.QFormLayout()
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)
        self.edit_title = QtWidgets.QLineEdit(initial.title if initial else "")
        form.addRow("Plot title", self.edit_title)
        self.edit_xlabel = QtWidgets.QLineEdit(initial.xlabel if initial else "")
        form.addRow("X-axis label", self.edit_xlabel)
        self.edit_ylabel = QtWidgets.QLineEdit(initial.ylabel if initial else "")
        form.addRow("Y-axis label", self.edit_ylabel)
        self.edit_colorbar = QtWidgets.QLineEdit(initial.colorbar_label if initial else "")
        form.addRow("Colorbar label", self.edit_colorbar)
        layout.addLayout(form)

        aesthetics = QtWidgets.QGroupBox("Aesthetics")
        grid = QtWidgets.QGridLayout(aesthetics)
        grid.setContentsMargins(8, 8, 8, 8)
        grid.setSpacing(6)
        grid.addWidget(QtWidgets.QLabel("Font family"), 0, 0)
        self.font_combo = QtWidgets.QFontComboBox()
        if initial and initial.font_family:
            self.font_combo.setCurrentFont(QtGui.QFont(initial.font_family))
        grid.addWidget(self.font_combo, 0, 1, 1, 2)

        grid.addWidget(QtWidgets.QLabel("Title size"), 1, 0)
        self.spin_title = QtWidgets.QSpinBox()
        self.spin_title.setRange(6, 72)
        self.spin_title.setValue(initial.title_size if initial else 14)
        grid.addWidget(self.spin_title, 1, 1)

        grid.addWidget(QtWidgets.QLabel("Axis label size"), 2, 0)
        self.spin_axis = QtWidgets.QSpinBox()
        self.spin_axis.setRange(6, 60)
        self.spin_axis.setValue(initial.axis_size if initial else 12)
        grid.addWidget(self.spin_axis, 2, 1)

        grid.addWidget(QtWidgets.QLabel("Tick size"), 3, 0)
        self.spin_tick = QtWidgets.QSpinBox()
        self.spin_tick.setRange(6, 48)
        self.spin_tick.setValue(initial.tick_size if initial else 10)
        grid.addWidget(self.spin_tick, 3, 1)

        grid.addWidget(QtWidgets.QLabel("Colorbar size"), 4, 0)
        self.spin_colorbar = QtWidgets.QSpinBox()
        self.spin_colorbar.setRange(6, 60)
        self.spin_colorbar.setValue(initial.colorbar_size if initial else 12)
        grid.addWidget(self.spin_colorbar, 4, 1)

        grid.addWidget(QtWidgets.QLabel("Background"), 5, 0)
        self.btn_color = ColorButton(initial.background if initial else QtGui.QColor("#1b1b1b"))
        grid.addWidget(self.btn_color, 5, 1)
        grid.setColumnStretch(2, 1)
        layout.addWidget(aesthetics)

        if allow_apply_all:
            self.chk_apply_all = QtWidgets.QCheckBox("Apply to all plots in this tab")
            self.chk_apply_all.setChecked(bool(initial.apply_to_all) if initial else False)
            layout.addWidget(self.chk_apply_all)
        else:
            self.chk_apply_all = QtWidgets.QCheckBox()
            self.chk_apply_all.hide()

        if template_hint:
            hint = QtWidgets.QLabel(template_hint)
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #555;")
            layout.addWidget(hint)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def annotation_config(self) -> Optional[PlotAnnotationConfig]:
        return self._result

    def accept(self):
        font = self.font_combo.currentFont()
        config = PlotAnnotationConfig(
            title=self.edit_title.text(),
            xlabel=self.edit_xlabel.text(),
            ylabel=self.edit_ylabel.text(),
            colorbar_label=self.edit_colorbar.text(),
            font_family=font.family(),
            title_size=self.spin_title.value(),
            axis_size=self.spin_axis.value(),
            tick_size=self.spin_tick.value(),
            colorbar_size=self.spin_colorbar.value(),
            background=self.btn_color.color(),
            apply_to_all=self.chk_apply_all.isChecked(),
        )
        self._result = config
        super().accept()

def _save_snapshot(widget: QtWidgets.QWidget, path: Path, label: str = "") -> bool:
    image = _compose_snapshot(widget, label)
    try:
        return image.save(str(path))
    except Exception:
        return False


def _qimage_to_array(image: QtGui.QImage) -> np.ndarray:
    converted = image.convertToFormat(QtGui.QImage.Format_RGBA8888)
    width = converted.width()
    height = converted.height()
    bytes_per_line = converted.bytesPerLine()
    ptr = converted.bits()
    ptr.setsize(converted.byteCount())
    buffer = np.frombuffer(ptr, np.uint8).reshape((height, bytes_per_line // 4, 4))
    return np.array(buffer[:, :width, :], copy=True)


def _ask_layout_label(
    parent: QtWidgets.QWidget, title: str, default_text: str = ""
) -> Tuple[bool, str]:
    text, ok = QtWidgets.QInputDialog.getText(
        parent,
        title,
        "Enter a label to display above the saved layout (optional):",
        text=str(default_text or ""),
    )
    if not ok:
        return False, ""
    return True, str(text).strip()


def _process_events():
    app = QtWidgets.QApplication.instance()
    if app is not None:
        app.processEvents(QtCore.QEventLoop.AllEvents, 50)


def _sanitize_filename(text: str) -> str:
    safe = [ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(text)]
    cleaned = "".join(safe).strip("_")
    return cleaned or "image"


def _ensure_extension(path: str, default_suffix: str) -> Path:
    p = Path(path)
    if not p.suffix:
        p = p.with_suffix(default_suffix)
    return p


# ---------------------------------------------------------------------------
# Dataset / variable references used for drag & drop
# ---------------------------------------------------------------------------
class DataSetRef(QtCore.QObject):
    def __init__(self, path: Path):
        super().__init__()
        self.path = Path(path)

    def to_mime(self) -> str:
        return "DatasetRef:" + json.dumps({"path": str(self.path)})

    @staticmethod
    def from_mime(txt: str) -> Optional["DataSetRef"]:
        if not txt or not txt.startswith("DatasetRef:"):
            return None
        try:
            data = json.loads(txt.split(":", 1)[1])
            return DataSetRef(Path(data["path"]))
        except Exception:
            return None

    def load(self) -> xr.Dataset:
        return open_dataset(self.path)


class VarRef(QtCore.QObject):
    def __init__(self, path: Path, var: str, hint: str = ""):
        super().__init__()
        self.path = Path(path)
        self.var = var
        self.hint = hint

    def to_mime(self) -> str:
        return "VarRef:" + json.dumps({"path": str(self.path), "var": self.var, "hint": self.hint})

    @staticmethod
    def from_mime(txt: str) -> Optional["VarRef"]:
        if not txt or not txt.startswith("VarRef:"):
            return None
        try:
            data = json.loads(txt.split(":", 1)[1])
            return VarRef(Path(data["path"]), data["var"], data.get("hint", ""))
        except Exception:
            return None

    def load(self):
        ds = open_dataset(self.path)
        if self.var not in ds.data_vars or ds[self.var].ndim != 2:
            raise RuntimeError(f"{self.var!r} is not a 2D variable in {self.path}")
        da = ds[self.var]
        coords = guess_phys_coords(da)
        return da, coords


class MemoryDatasetRegistry:
    _datasets: Dict[str, Tuple[xr.Dataset, str]] = {}
    _counter = itertools.count()

    @classmethod
    def register(cls, dataset: xr.Dataset, label: str) -> str:
        key = f"memory:{next(cls._counter)}"
        cls._datasets[key] = (dataset, label)
        return key

    @classmethod
    def has(cls, key: str) -> bool:
        return key in cls._datasets

    @classmethod
    def get_dataset(cls, key: str) -> Optional[xr.Dataset]:
        entry = cls._datasets.get(key)
        if not entry:
            return None
        dataset, _ = entry
        try:
            return dataset.copy(deep=True)
        except Exception:
            return dataset

    @classmethod
    def get_label(cls, key: str) -> str:
        entry = cls._datasets.get(key)
        if not entry:
            return key
        return entry[1]


class PreferencesManager(QtCore.QObject):
    changed = QtCore.Signal(dict)

    def __init__(self):
        super().__init__()
        self._data = self._default_prefs()

    def _default_prefs(self) -> Dict[str, object]:
        return {
            "general": {
                "autoscale_on_load": True,
                "default_layout_label": "",
            },
            "colormaps": {
                "default": "viridis",
                "variables": {},
            },
            "misc": {
                "default_export_dir": "",
            },
        }

    def data(self) -> Dict[str, object]:
        return copy.deepcopy(self._data)

    def update(self, data: Dict[str, object]):
        normalized = self._default_prefs()
        general = data.get("general", {}) if isinstance(data, dict) else {}
        if isinstance(general, dict):
            normalized["general"].update(general)
        colormaps = data.get("colormaps", {}) if isinstance(data, dict) else {}
        if isinstance(colormaps, dict):
            normalized["colormaps"].update({k: v for k, v in colormaps.items() if k in ("default", "variables")})
            variables = (
                colormaps.get("variables", {})
                if isinstance(colormaps.get("variables"), dict)
                else {}
            )
            normalized["colormaps"]["variables"] = {
                str(var): str(cmap)
                for var, cmap in variables.items()
                if str(var).strip() and str(cmap).strip()
            }
        misc = data.get("misc", {}) if isinstance(data, dict) else {}
        if isinstance(misc, dict):
            normalized["misc"].update(misc)
        self._data = normalized
        self.changed.emit(self.data())

    def autoscale_on_load(self) -> bool:
        return bool(self._data.get("general", {}).get("autoscale_on_load", True))

    def default_layout_label(self) -> str:
        return str(self._data.get("general", {}).get("default_layout_label", ""))

    def default_export_directory(self) -> str:
        return str(self._data.get("misc", {}).get("default_export_dir", ""))

    def preferred_colormap(self, variable: Optional[str]) -> Optional[str]:
        variables = self._data.get("colormaps", {}).get("variables", {})
        if isinstance(variables, dict) and variable:
            cmap = variables.get(variable)
            if cmap:
                return str(cmap)
        default = self._data.get("colormaps", {}).get("default")
        if default:
            return str(default)
        return None

    def load_from_file(self, path: Path):
        try:
            data = json.loads(Path(path).read_text())
        except Exception as exc:
            raise RuntimeError(f"Failed to load preferences: {exc}")
        if not isinstance(data, dict):
            raise RuntimeError("Invalid preferences file")
        self.update(data)

    def save_to_file(self, path: Path):
        try:
            Path(path).write_text(json.dumps(self._data, indent=2))
        except Exception as exc:
            raise RuntimeError(f"Failed to save preferences: {exc}")


class MemoryDatasetRef(QtCore.QObject):
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def to_mime(self) -> str:
        return "MemoryDatasetRef:" + self.key

    @staticmethod
    def from_mime(txt: str) -> Optional["MemoryDatasetRef"]:
        if not txt or not txt.startswith("MemoryDatasetRef:"):
            return None
        key = txt.split(":", 1)[1]
        if not MemoryDatasetRegistry.has(key):
            return None
        return MemoryDatasetRef(key)

    def load(self) -> xr.Dataset:
        dataset = MemoryDatasetRegistry.get_dataset(self.key)
        if dataset is None:
            raise RuntimeError("Dataset is no longer available in memory")
        return dataset

    def display_name(self) -> str:
        return MemoryDatasetRegistry.get_label(self.key)


class MemoryVarRef(QtCore.QObject):
    def __init__(self, dataset_key: str, var: str, hint: str = ""):
        super().__init__()
        self.dataset_key = dataset_key
        self.var = var
        self.hint = hint

    def to_mime(self) -> str:
        payload = {"key": self.dataset_key, "var": self.var, "hint": self.hint}
        return "MemoryVarRef:" + json.dumps(payload)

    @staticmethod
    def from_mime(txt: str) -> Optional["MemoryVarRef"]:
        if not txt or not txt.startswith("MemoryVarRef:"):
            return None
        try:
            payload = json.loads(txt.split(":", 1)[1])
        except Exception:
            return None
        key = payload.get("key", "")
        if not MemoryDatasetRegistry.has(key):
            return None
        return MemoryVarRef(key, payload.get("var", ""), payload.get("hint", ""))

    def load(self):
        dataset = MemoryDatasetRegistry.get_dataset(self.dataset_key)
        if dataset is None or self.var not in dataset.data_vars:
            raise RuntimeError("Variable is no longer available in memory")
        da = dataset[self.var]
        if getattr(da, "ndim", 0) != 2:
            raise RuntimeError("Variable is not two-dimensional")
        coords = guess_phys_coords(da)
        return da, coords


class MemorySliceRef(QtCore.QObject):
    def __init__(
        self,
        dataset_key: str,
        var: str,
        alias: str,
        label: str = "",
        hint: str = "",
    ):
        super().__init__()
        self.dataset_key = dataset_key
        self.var = var
        self.alias = alias
        self.label = label
        self.hint = hint

    def to_mime(self) -> str:
        payload = {
            "key": self.dataset_key,
            "var": self.var,
            "alias": self.alias,
            "label": self.label,
            "hint": self.hint,
        }
        return "MemorySliceRef:" + json.dumps(payload)

    @staticmethod
    def from_mime(txt: str) -> Optional["MemorySliceRef"]:
        if not txt or not txt.startswith("MemorySliceRef:"):
            return None
        try:
            payload = json.loads(txt.split(":", 1)[1])
        except Exception:
            return None
        key = payload.get("key")
        var = payload.get("var")
        if not key or not var or not MemoryDatasetRegistry.has(key):
            return None
        alias = payload.get("alias") or var
        label = payload.get("label", "")
        hint = payload.get("hint", "")
        return MemorySliceRef(str(key), str(var), str(alias), str(label), str(hint))

    def load(self):
        dataset = MemoryDatasetRegistry.get_dataset(self.dataset_key)
        if dataset is None or self.var not in dataset.data_vars:
            raise RuntimeError("Variable is no longer available in memory")
        da = dataset[self.var]
        arr = da.copy(deep=True)
        alias = da.attrs.get("_source_var", self.alias or self.var)
        if alias:
            arr.name = alias
        coords = guess_phys_coords(arr)
        return arr, coords, alias

    def dataset_label(self) -> str:
        return MemoryDatasetRegistry.get_label(self.dataset_key)

    def display_label(self) -> str:
        base = self.dataset_label()
        if self.label:
            return f"{base} — {self.label}"
        return base

class HighDimVarRef(QtCore.QObject):
    def __init__(self, var: str, hint: str = "", *, path: Optional[Path] = None, memory_key: Optional[str] = None):
        super().__init__()
        self.path = Path(path) if path is not None else None
        self.memory_key = memory_key
        self.var = var
        self.hint = hint

    def to_mime(self) -> str:
        payload = {"var": self.var, "hint": self.hint}
        if self.path is not None:
            payload["source"] = "disk"
            payload["path"] = str(self.path)
        elif self.memory_key is not None:
            payload["source"] = "memory"
            payload["key"] = self.memory_key
        else:
            payload["source"] = "unknown"
        return "HighDimVarRef:" + json.dumps(payload)

    @staticmethod
    def from_mime(txt: str) -> Optional["HighDimVarRef"]:
        if not txt or not txt.startswith("HighDimVarRef:"):
            return None
        try:
            payload = json.loads(txt.split(":", 1)[1])
        except Exception:
            return None
        source = payload.get("source")
        if source == "disk":
            path_txt = payload.get("path")
            if not path_txt:
                return None
            return HighDimVarRef(payload.get("var", ""), payload.get("hint", ""), path=Path(path_txt))
        if source == "memory":
            key = payload.get("key")
            if not key or not MemoryDatasetRegistry.has(key):
                return None
            return HighDimVarRef(payload.get("var", ""), payload.get("hint", ""), memory_key=key)
        return None

    def _load_dataset(self) -> xr.Dataset:
        if self.memory_key is not None:
            dataset = MemoryDatasetRegistry.get_dataset(self.memory_key)
            if dataset is None:
                raise RuntimeError("Dataset is no longer available in memory")
            return dataset
        if self.path is not None:
            return open_dataset(self.path)
        raise RuntimeError("Unknown dataset source")

    def load_dataset(self) -> xr.Dataset:
        return self._load_dataset()

    def load_dataarray(self) -> xr.DataArray:
        dataset = self._load_dataset()
        if self.var not in dataset.data_vars:
            raise RuntimeError(f"{self.var!r} not present in dataset")
        return dataset[self.var]

    def dataset_label(self) -> str:
        if self.memory_key is not None:
            return MemoryDatasetRegistry.get_label(self.memory_key)
        if self.path is not None:
            return self.path.name
        return self.var

# ---------------------------------------------------------------------------
# Parameter editing helpers
# ---------------------------------------------------------------------------


class ParameterForm(QtWidgets.QWidget):
    parametersChanged = QtCore.Signal()

    def __init__(self, parameters: Iterable[ParameterDefinition], parent=None):
        super().__init__(parent)
        self._definitions = list(parameters)
        self._widgets: Dict[str, QtWidgets.QWidget] = {}

        layout = QtWidgets.QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        if not self._definitions:
            lbl = QtWidgets.QLabel("No parameters")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            layout.addRow(lbl)
            return

        for definition in self._definitions:
            widget: Optional[QtWidgets.QWidget] = None
            if definition.kind == "float":
                spin = QtWidgets.QDoubleSpinBox()
                spin.setDecimals(6)
                lo = float(definition.minimum) if definition.minimum is not None else -1e9
                hi = float(definition.maximum) if definition.maximum is not None else 1e9
                spin.setRange(lo, hi)
                if definition.step is not None:
                    spin.setSingleStep(float(definition.step))
                spin.setValue(float(definition.default))
                spin.valueChanged.connect(lambda *_: self.parametersChanged.emit())
                widget = spin
            elif definition.kind == "int":
                spin_i = QtWidgets.QSpinBox()
                lo = int(definition.minimum) if definition.minimum is not None else -1_000_000
                hi = int(definition.maximum) if definition.maximum is not None else 1_000_000
                spin_i.setRange(lo, hi)
                if definition.step is not None:
                    spin_i.setSingleStep(int(definition.step))
                spin_i.setValue(int(definition.default))
                spin_i.valueChanged.connect(lambda *_: self.parametersChanged.emit())
                widget = spin_i
            elif definition.kind == "enum":
                combo = QtWidgets.QComboBox()
                if definition.choices:
                    for label, value in definition.choices:
                        combo.addItem(label, value)
                combo.setCurrentIndex(max(combo.findData(definition.default), 0))
                combo.currentIndexChanged.connect(lambda *_: self.parametersChanged.emit())
                widget = combo
            else:
                line = QtWidgets.QLineEdit(str(definition.default))
                line.textChanged.connect(lambda *_: self.parametersChanged.emit())
                widget = line

            self._widgets[definition.name] = widget
            layout.addRow(definition.label, widget)

    def values(self) -> Dict[str, object]:
        values: Dict[str, object] = {}
        for definition in self._definitions:
            widget = self._widgets.get(definition.name)
            if widget is None:
                continue
            if isinstance(widget, QtWidgets.QDoubleSpinBox):
                values[definition.name] = float(widget.value())
            elif isinstance(widget, QtWidgets.QSpinBox):
                values[definition.name] = int(widget.value())
            elif isinstance(widget, QtWidgets.QComboBox):
                data = widget.currentData()
                values[definition.name] = data if data is not None else widget.currentText()
            elif isinstance(widget, QtWidgets.QLineEdit):
                values[definition.name] = widget.text()
        return values

    def set_values(self, params: Dict[str, object]):
        for definition in self._definitions:
            widget = self._widgets.get(definition.name)
            if widget is None:
                continue
            value = params.get(definition.name, definition.default)
            block = widget.blockSignals(True)
            try:
                if isinstance(widget, QtWidgets.QDoubleSpinBox):
                    widget.setValue(float(value))
                elif isinstance(widget, QtWidgets.QSpinBox):
                    widget.setValue(int(value))
                elif isinstance(widget, QtWidgets.QComboBox):
                    idx = widget.findData(value)
                    if idx < 0:
                        idx = widget.findText(str(value))
                    widget.setCurrentIndex(max(idx, 0))
                elif isinstance(widget, QtWidgets.QLineEdit):
                    widget.setText(str(value))
            finally:
                widget.blockSignals(block)


# ---------------------------------------------------------------------------
# Processing manager and dialogs
# ---------------------------------------------------------------------------


class ProcessingManager(QtCore.QObject):
    pipelines_changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self._pipelines: Dict[str, ProcessingPipeline] = {}

    def list_pipelines(self) -> List[ProcessingPipeline]:
        return [self._clone_pipeline(p) for p in self._pipelines.values()]

    def pipeline_names(self) -> List[str]:
        return sorted(self._pipelines.keys())

    def get_pipeline(self, name: str) -> Optional[ProcessingPipeline]:
        if name not in self._pipelines:
            return None
        return self._clone_pipeline(self._pipelines[name])

    def save_pipeline(self, pipeline: ProcessingPipeline):
        name = pipeline.name.strip()
        if not name:
            raise ValueError("Pipeline name cannot be empty")
        self._pipelines[name] = self._clone_pipeline(pipeline)
        self.pipelines_changed.emit()

    def delete_pipeline(self, name: str):
        if name in self._pipelines:
            del self._pipelines[name]
            self.pipelines_changed.emit()

    def _clone_pipeline(self, pipeline: ProcessingPipeline) -> ProcessingPipeline:
        return ProcessingPipeline(
            name=pipeline.name,
            steps=[ProcessingStep(step.key, dict(step.params)) for step in pipeline.steps],
        )


class PipelineEditorDialog(QtWidgets.QDialog):
    def __init__(self, manager: ProcessingManager, pipeline: ProcessingPipeline, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.pipeline = ProcessingPipeline(
            name=pipeline.name,
            steps=[ProcessingStep(step.key, dict(step.params)) for step in pipeline.steps],
        )
        self._raw_data: Optional[np.ndarray] = None
        self._processed_data: Optional[np.ndarray] = None
        self.setWindowTitle(f"Edit Pipeline – {self.pipeline.name or 'Untitled'}")
        self.resize(800, 700)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        top_row = QtWidgets.QHBoxLayout()
        self.lbl_data = QtWidgets.QLabel("No data loaded")
        self.lbl_data.setStyleSheet("color: #555;")
        top_row.addWidget(self.lbl_data, 1)
        self.btn_load_data = QtWidgets.QPushButton("Load data…")
        self.btn_load_data.clicked.connect(self._load_data)
        top_row.addWidget(self.btn_load_data, 0)
        cmap_label = QtWidgets.QLabel("Color map:")
        cmap_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        top_row.addWidget(cmap_label, 0)
        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        candidate_maps = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "thermal",
        ]
        for name in candidate_maps:
            try:
                pg.colormap.get(name)
            except Exception:
                continue
            label = name.title()
            self.cmb_colormap.addItem(label, name)
        if self.cmb_colormap.count() == 0:
            self.cmb_colormap.addItem("Default", "default")
        self.cmb_colormap.currentIndexChanged.connect(self._on_colormap_changed)
        top_row.addWidget(self.cmb_colormap, 0)
        layout.addLayout(top_row)

        self.image_view = pg.ImageView()
        try:
            self.image_view.getView().invertY(False)
        except Exception:
            pass
        self.roi = pg.RectROI([10, 10], [40, 40], pen=pg.mkPen('#ffaa00', width=2))
        self.roi.addScaleHandle((1, 1), (0, 0))
        self.roi.addScaleHandle((0, 0), (1, 1))
        try:
            self.roi.setVisible(False)
        except Exception:
            pass

        roi_button = getattr(self.image_view.ui, "roiBtn", None)
        if roi_button is not None:
            try:
                roi_button.clicked.disconnect()
            except Exception:
                pass
            roi_button.setCheckable(True)
            roi_button.setChecked(False)
            roi_button.toggled.connect(self._on_roi_button_toggled)
            roi_button.setToolTip("Toggle ROI preview (right-click the ROI plot for reduction options)")
        roi_plot_widget = getattr(self.image_view.ui, "roiPlot", None)
        if roi_plot_widget is not None:
            roi_plot_widget.hide()
            roi_plot_widget.setMaximumHeight(0)
        layout.addWidget(self.image_view, 2)

        self.roi_box = QtWidgets.QGroupBox("ROI preview")
        roi_layout = QtWidgets.QVBoxLayout(self.roi_box)
        roi_layout.setContentsMargins(8, 8, 8, 8)
        roi_layout.setSpacing(6)

        self._roi_axis_options: List[tuple[str, int, str, str]] = [
            ("Collapse rows (Y) → profile across X", 0, "rows (Y)", "X"),
            ("Collapse columns (X) → profile across Y", 1, "columns (X)", "Y"),
        ]
        self._roi_axis_index: int = 0
        self._roi_reducers = {
            "mean": ("Mean", _nan_aware_reducer(lambda arr, axis=None: np.nanmean(arr, axis=axis))),
            "median": ("Median", _nan_aware_reducer(lambda arr, axis=None: np.nanmedian(arr, axis=axis))),
            "min": ("Minimum", _nan_aware_reducer(lambda arr, axis=None: np.nanmin(arr, axis=axis))),
            "max": ("Maximum", _nan_aware_reducer(lambda arr, axis=None: np.nanmax(arr, axis=axis))),
            "std": ("Std. dev", _nan_aware_reducer(lambda arr, axis=None: np.nanstd(arr, axis=axis))),
            "ptp": (
                "Peak-to-peak",
                _nan_aware_reducer(
                    lambda arr, axis=None: np.nanmax(arr, axis=axis) - np.nanmin(arr, axis=axis)
                ),
            ),
        }
        self._roi_method_key: str = "mean"

        self.lbl_roi_axis = QtWidgets.QLabel()
        self.lbl_roi_axis.setStyleSheet("color: #555;")
        roi_layout.addWidget(self.lbl_roi_axis)

        hint = QtWidgets.QLabel("Right-click the ROI plot to change the reduction axis or statistic.")
        hint.setStyleSheet("color: #777;")
        roi_layout.addWidget(hint)

        self.roi_plot = pg.PlotWidget()
        self.roi_plot.showGrid(x=True, y=True, alpha=0.3)
        self.roi_plot.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.roi_plot.customContextMenuRequested.connect(self._show_roi_context_menu)
        self.roi_curve = self.roi_plot.plot([], [], pen=pg.mkPen('#ffaa00', width=2))
        roi_layout.addWidget(self.roi_plot, 1)

        self.roi_box.hide()
        layout.addWidget(self.roi_box, 1)

        self.steps_scroll = QtWidgets.QScrollArea()
        self.steps_scroll.setWidgetResizable(True)
        self.steps_container = QtWidgets.QWidget()
        self.steps_layout = QtWidgets.QVBoxLayout(self.steps_container)
        self.steps_layout.setContentsMargins(0, 0, 0, 0)
        self.steps_layout.setSpacing(8)
        self.steps_scroll.setWidget(self.steps_container)
        layout.addWidget(self.steps_scroll, 1)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Close)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._forms: List[tuple[ProcessingStep, ParameterForm]] = []
        self._roi_enabled = False
        self._rebuild_forms()
        self._update_roi_axis_label()
        try:
            self.roi.sigRegionChanged.connect(self._update_roi_preview)
        except Exception:
            pass

    # ----- helpers -----
    def result_pipeline(self) -> ProcessingPipeline:
        return ProcessingPipeline(
            name=self.pipeline.name,
            steps=[ProcessingStep(step.key, dict(step.params)) for step in self.pipeline.steps],
        )

    def _rebuild_forms(self):
        while self.steps_layout.count():
            item = self.steps_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._forms.clear()

        if not self.pipeline.steps:
            lbl = QtWidgets.QLabel("Pipeline has no steps.")
            lbl.setAlignment(QtCore.Qt.AlignCenter)
            lbl.setStyleSheet("color: #777;")
            self.steps_layout.addWidget(lbl)
            return

        for idx, step in enumerate(self.pipeline.steps, start=1):
            spec = get_processing_function(step.key)
            title = spec.label if spec else step.key
            box = QtWidgets.QGroupBox(f"Step {idx}: {title}")
            vbox = QtWidgets.QVBoxLayout(box)
            if spec is None:
                lbl = QtWidgets.QLabel("Unknown processing function")
                lbl.setAlignment(QtCore.Qt.AlignLeft)
                vbox.addWidget(lbl)
            else:
                form = ParameterForm(spec.parameters)
                form.set_values(step.params)
                form.parametersChanged.connect(self._on_parameters_changed)
                vbox.addWidget(form)
                self._forms.append((step, form))
            self.steps_layout.addWidget(box)
        self.steps_layout.addStretch(1)
        self._apply_pipeline()

    def _update_steps_from_forms(self):
        for step, form in self._forms:
            step.params = form.values()

    def _apply_pipeline(self):
        if self._raw_data is None:
            return
        data = np.asarray(self._raw_data, float)
        try:
            for step in self.pipeline.steps:
                data = apply_processing_step(step.key, data, step.params)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Processing failed", str(e))
            return
        self._processed_data = data
        try:
            self.image_view.setImage(data, autoLevels=True)
            self._apply_selected_colormap()
        except Exception:
            pass
        self._update_roi_preview()

    def _apply_selected_colormap(self):
        if not hasattr(self, "cmb_colormap"):
            return
        name = self.cmb_colormap.currentData()
        if not name or name == "default":
            return
        try:
            cmap = pg.colormap.get(str(name))
        except Exception:
            return
        try:
            self.image_view.setColorMap(cmap)
        except Exception:
            pass

    def _update_roi_axis_label(self):
        if not self._roi_axis_options:
            self.roi_box.setTitle("ROI preview")
            self.lbl_roi_axis.setText("")
            return
        index = max(0, min(self._roi_axis_index, len(self._roi_axis_options) - 1))
        axis, collapsed, remaining = self._roi_axis_options[index][1:]
        self.roi_box.setTitle(f"ROI preview – reducing {collapsed}")
        self.lbl_roi_axis.setText(f"Reducing over {collapsed} to plot along {remaining}.")
        self.roi_plot.setLabel("bottom", f"{remaining} index")
        self.roi_plot.setTitle(f"ROI profile along {remaining}")

    def _current_roi_axis(self) -> int:
        if not self._roi_axis_options:
            return 0
        index = max(0, min(self._roi_axis_index, len(self._roi_axis_options) - 1))
        axis = self._roi_axis_options[index][1]
        return int(axis)

    def _on_colormap_changed(self):
        self._apply_selected_colormap()

    def _show_roi_context_menu(self, pos: QtCore.QPoint):
        if not self._roi_axis_options:
            return
        menu = QtWidgets.QMenu(self.roi_plot)

        axis_menu = menu.addMenu("Reduce over")
        for idx, (label, *_rest) in enumerate(self._roi_axis_options):
            action = axis_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(idx == self._roi_axis_index)
            action.triggered.connect(partial(self._set_roi_axis_index, idx))

        stat_menu = menu.addMenu("Statistic")
        for key, (label, _) in self._roi_reducers.items():
            action = stat_menu.addAction(label)
            action.setCheckable(True)
            action.setChecked(key == self._roi_method_key)
            action.triggered.connect(partial(self._set_roi_method, key))

        menu.exec_(self.roi_plot.mapToGlobal(pos))

    def _set_roi_axis_index(self, index: int):
        if index == self._roi_axis_index:
            return
        self._roi_axis_index = max(0, min(index, len(self._roi_axis_options) - 1))
        self._update_roi_axis_label()
        self._update_roi_preview()

    def _set_roi_method(self, key: str):
        if key not in self._roi_reducers or key == self._roi_method_key:
            return
        self._roi_method_key = key
        self._update_roi_preview()

    def _on_roi_button_toggled(self, checked: bool):
        view = getattr(self.image_view, "view", None)
        if view is None:
            try:
                view = self.image_view.getView()
            except Exception:
                view = None
        if view is not None:
            try:
                if checked:
                    if self.roi.scene() is None:
                        view.addItem(self.roi)
                else:
                    view.removeItem(self.roi)
            except Exception:
                pass
        try:
            self.roi.setVisible(checked)
        except Exception:
            pass
        self._roi_enabled = bool(checked)
        self.roi_box.setVisible(self._roi_enabled)
        if self._roi_enabled:
            self._reset_roi_to_image()
            self._update_roi_preview()
        else:
            self.roi_curve.setData([], [])

    def _extract_roi_array(self) -> Optional[np.ndarray]:
        if self._processed_data is None:
            return None
        if not hasattr(self, "roi") or self.roi is None:
            return None
        image_item = getattr(self.image_view, "imageItem", None)
        if image_item is None:
            return None
        try:
            roi_data = self.roi.getArrayRegion(self._processed_data, image_item)
        except Exception:
            try:
                roi_data = self.roi.getArraySlice(self._processed_data, image_item)
                if isinstance(roi_data, tuple):
                    roi_data = roi_data[0]
            except Exception:
                return None
        if roi_data is None:
            return None
        return np.asarray(roi_data)

    def _update_roi_preview(self):
        if not self._roi_enabled or self._processed_data is None:
            self.roi_curve.setData([], [])
            return
        roi_array = self._extract_roi_array()
        if roi_array is None or roi_array.size == 0:
            self.roi_curve.setData([], [])
            return
        method_key = self._roi_method_key
        reducer_entry = self._roi_reducers.get(method_key)
        if reducer_entry is None:
            self.roi_curve.setData([], [])
            return
        _, reducer = reducer_entry
        axis = self._current_roi_axis()
        axis = max(0, min(axis, roi_array.ndim - 1))
        with np.errstate(all="ignore"):
            profile = reducer(roi_array, axis=axis)
        if profile is None:
            self.roi_curve.setData([], [])
            return
        profile = np.asarray(profile).ravel()
        if profile.size == 0:
            self.roi_curve.setData([], [])
            return
        x = np.arange(profile.size)
        self.roi_curve.setData(x, profile)
        self.roi_plot.enableAutoRange()

    def _reset_roi_to_image(self, shape: Optional[Tuple[int, int]] = None):
        if not self._roi_enabled or not hasattr(self, "roi") or self.roi is None:
            return
        if shape is None:
            if self._processed_data is None:
                return
            shape = self._processed_data.shape
        if not shape or len(shape) < 2:
            return
        height, width = int(shape[0]), int(shape[1])
        if width <= 0 or height <= 0:
            return
        rect_width = max(2, width // 2)
        rect_height = max(2, height // 2)
        pos_x = max(0, (width - rect_width) // 2)
        pos_y = max(0, (height - rect_height) // 2)
        try:
            self.roi.blockSignals(True)
            self.roi.setPos((pos_x, pos_y))
            self.roi.setSize((rect_width, rect_height))
        finally:
            try:
                self.roi.blockSignals(False)
            except Exception:
                pass
        self._update_roi_preview()

    # ----- slots -----
    def _on_parameters_changed(self):
        self._update_steps_from_forms()
        self._apply_pipeline()

    def _load_data(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select dataset",
            "",
            "NetCDF / Zarr (*.nc *.zarr);;All files (*)",
        )
        if not path:
            return
        p = Path(path)
        try:
            ds = open_dataset(p)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            return
        choices = [var for var in ds.data_vars if ds[var].ndim == 2]
        if not choices:
            QtWidgets.QMessageBox.information(self, "No 2D variables", "The dataset has no 2D variables to preview.")
            return
        var, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Select variable",
            "Variable:",
            choices,
            0,
            False,
        )
        if not ok or not var:
            return
        da = ds[var]
        self._raw_data = np.asarray(da.values, float)
        self._processed_data = np.asarray(self._raw_data)
        self.lbl_data.setText(f"{p.name}:{var}")
        self._reset_roi_to_image(self._raw_data.shape if hasattr(self._raw_data, "shape") else None)
        self._apply_pipeline()

# ---------------------------------------------------------------------------
# Processing dock widget
# ---------------------------------------------------------------------------


class ProcessingDockContainer(QtWidgets.QWidget):
    def __init__(self, title: str, widget: QtWidgets.QWidget, parent=None):
        super().__init__(parent)
        self._title = title
        self._content_widget = widget
        self._floating_window: Optional[QtWidgets.QDialog] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QtWidgets.QWidget()
        header_layout = QtWidgets.QHBoxLayout(header)
        header_layout.setContentsMargins(6, 6, 6, 6)
        header_layout.setSpacing(6)
        lbl = QtWidgets.QLabel(title)
        font = lbl.font()
        font.setBold(True)
        lbl.setFont(font)
        header_layout.addWidget(lbl)
        header_layout.addStretch(1)

        self.btn_float = QtWidgets.QToolButton()
        self.btn_float.setText("Float")
        self.btn_float.setAutoRaise(True)
        self.btn_float.setToolTip("Undock processing pane to a floating window")
        self.btn_float.clicked.connect(self._on_float_clicked)
        header_layout.addWidget(self.btn_float)

        self.btn_toggle = QtWidgets.QToolButton()
        self.btn_toggle.setCheckable(True)
        self.btn_toggle.setChecked(True)
        self.btn_toggle.setAutoRaise(True)
        self.btn_toggle.setArrowType(QtCore.Qt.DownArrow)
        self.btn_toggle.setToolTip("Hide processing pane")
        self.btn_toggle.toggled.connect(self._on_toggle_toggled)
        header_layout.addWidget(self.btn_toggle)

        layout.addWidget(header)

        self._content_frame = QtWidgets.QWidget()
        self._content_layout = QtWidgets.QVBoxLayout(self._content_frame)
        self._content_layout.setContentsMargins(0, 0, 0, 0)
        self._content_layout.setSpacing(0)
        self._content_layout.addWidget(self._content_widget)
        layout.addWidget(self._content_frame, 1)

        self._placeholder = QtWidgets.QLabel(
            "Processing pane is undocked. Click Dock to return it to the sidebar."
        )
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setWordWrap(True)
        self._placeholder.hide()
        layout.addWidget(self._placeholder, 1)

        self._update_toggle_visuals()
        self._update_content_visibility()
        self._update_float_button()


    def _on_toggle_toggled(self, checked: bool):
        del checked
        self._update_toggle_visuals()
        self._update_content_visibility()

    def _on_float_clicked(self):
        if self._floating_window is not None:
            self.dock()
        else:
            self.undock()

    def undock(self):
        if self._floating_window is not None:
            try:
                self._floating_window.raise_()
                self._floating_window.activateWindow()
            except Exception:
                pass
            return
        self.btn_toggle.setChecked(True)
        self._update_toggle_visuals()
        self._update_content_visibility()
        self._content_layout.removeWidget(self._content_widget)
        self._content_widget.setParent(None)
        dialog = QtWidgets.QDialog(self.window())
        dialog.setWindowTitle(self._title)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        dlg_layout = QtWidgets.QVBoxLayout(dialog)
        dlg_layout.setContentsMargins(0, 0, 0, 0)
        dlg_layout.setSpacing(0)
        dlg_layout.addWidget(self._content_widget)
        dialog.finished.connect(self._on_floating_closed)
        dialog.resize(self.width(), self.height())
        dialog.show()
        self._floating_window = dialog
        self._update_float_button()
        self._update_content_visibility()

    def dock(self):
        floating = self._floating_window
        if floating is None:
            return
        self._floating_window = None
        try:
            floating.finished.disconnect(self._on_floating_closed)
        except Exception:
            pass
        layout = floating.layout()
        if layout is not None:
            layout.removeWidget(self._content_widget)
        self._content_widget.setParent(self._content_frame)
        self._content_layout.addWidget(self._content_widget)
        self._content_widget.show()
        floating.hide()
        floating.deleteLater()
        self._update_float_button()
        self._update_content_visibility()

    def _on_floating_closed(self, *_):
        if self._floating_window is None:
            return
        self.dock()

    def _update_toggle_visuals(self):
        if self.btn_toggle.isChecked():
            self.btn_toggle.setArrowType(QtCore.Qt.DownArrow)
            self.btn_toggle.setToolTip("Hide processing pane")
        else:
            self.btn_toggle.setArrowType(QtCore.Qt.RightArrow)
            self.btn_toggle.setToolTip("Show processing pane")

    def _update_float_button(self):
        if self._floating_window is not None:
            self.btn_float.setText("Dock")
            self.btn_float.setToolTip("Dock processing pane back to the sidebar")
        else:
            self.btn_float.setText("Float")
            self.btn_float.setToolTip("Undock processing pane to a floating window")

    def _update_content_visibility(self):
        if self._floating_window is not None:
            self._content_frame.hide()
            self._placeholder.show()
            self.btn_toggle.setEnabled(False)
        else:
            self._placeholder.hide()
            self.btn_toggle.setEnabled(True)
            self._content_frame.setVisible(self.btn_toggle.isChecked())


class LoggingDockWidget(QtWidgets.QDockWidget):
    """Dockable pane that displays log entries in real time."""

    def __init__(self, logger: ActionLogger, parent=None):
        super().__init__("Action Log", parent)
        self.setObjectName("actionLogDock")
        self._logger = logger
        self.setFeatures(
            QtWidgets.QDockWidget.DockWidgetMovable
            | QtWidgets.QDockWidget.DockWidgetClosable
        )
        self._collapsed = False

        self._view = QtWidgets.QPlainTextEdit()
        self._view.setReadOnly(True)
        self._view.setLineWrapMode(QtWidgets.QPlainTextEdit.NoWrap)
        self._view.setMaximumBlockCount(2000)
        self._view.setPlaceholderText("User actions will be logged here.")
        self.setWidget(self._view)

        self._build_title_bar()

        clear_action = QtWidgets.QAction("Clear log", self)
        clear_action.triggered.connect(self._clear_log)
        self.addAction(clear_action)
        export_action = QtWidgets.QAction("Export log…", self)
        export_action.triggered.connect(self._export_log)
        self.addAction(export_action)
        self.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)

        for entry in logger.entries():
            self._append(entry)

        logger.log_added.connect(self._append)
        logger.reset.connect(self._reset)

    def _build_title_bar(self):
        title_widget = QtWidgets.QWidget()
        layout = QtWidgets.QHBoxLayout(title_widget)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(4)

        self._toggle_btn = QtWidgets.QToolButton()
        self._toggle_btn.setArrowType(QtCore.Qt.DownArrow)
        self._toggle_btn.setCheckable(True)
        self._toggle_btn.setChecked(True)
        self._toggle_btn.clicked.connect(self._toggle_collapsed)
        layout.addWidget(self._toggle_btn, 0)

        label = QtWidgets.QLabel("Action Log")
        font = label.font()
        font.setBold(True)
        label.setFont(font)
        layout.addWidget(label, 1)

        self._export_btn = QtWidgets.QToolButton()
        self._export_btn.setText("Export…")
        self._export_btn.clicked.connect(self._export_log)
        layout.addWidget(self._export_btn, 0)

        layout.addStretch(0)
        self.setTitleBarWidget(title_widget)

    def _toggle_collapsed(self):
        self._collapsed = not self._collapsed
        self.widget().setVisible(not self._collapsed)
        self._toggle_btn.setArrowType(
            QtCore.Qt.RightArrow if self._collapsed else QtCore.Qt.DownArrow
        )

    def _append(self, entry: str):
        self._view.appendPlainText(entry)
        bar = self._view.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _reset(self):
        self._view.clear()

    def _clear_log(self):
        self._logger.clear()

    def _export_log(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export action log",
            "action-log.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(self._logger.to_text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))


class PreferencesDialog(QtWidgets.QDialog):
    def __init__(self, manager: PreferencesManager, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.resize(520, 420)
        self._manager = manager
        self._data = manager.data()

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self._build_general_tab()
        self._build_colormap_tab()

        controls = QtWidgets.QHBoxLayout()
        self.btn_load_file = QtWidgets.QPushButton("Load from file…")
        self.btn_load_file.clicked.connect(self._on_load_file)
        controls.addWidget(self.btn_load_file)
        self.btn_save_file = QtWidgets.QPushButton("Save to file…")
        self.btn_save_file.clicked.connect(self._on_save_file)
        controls.addWidget(self.btn_save_file)
        controls.addStretch(1)
        layout.addLayout(controls)

        buttons = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _build_general_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(6)

        self.chk_autoscale = QtWidgets.QCheckBox("Autoscale images on load")
        self.chk_autoscale.setChecked(bool(self._data.get("general", {}).get("autoscale_on_load", True)))
        form.addRow(self.chk_autoscale)

        self.txt_layout_label = QtWidgets.QLineEdit(
            str(self._data.get("general", {}).get("default_layout_label", ""))
        )
        form.addRow("Default layout label", self.txt_layout_label)

        export_row = QtWidgets.QHBoxLayout()
        self.txt_export_dir = QtWidgets.QLineEdit(
            str(self._data.get("misc", {}).get("default_export_dir", ""))
        )
        export_row.addWidget(self.txt_export_dir, 1)
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_export_dir)
        export_row.addWidget(btn_browse)
        form.addRow("Default export folder", export_row)

        self.tabs.addTab(tab, "General")

    def _build_colormap_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.cmb_default_cmap = QtWidgets.QComboBox()
        self.cmb_default_cmap.addItem("Use viewer default", "")
        try:
            maps = sorted(pg.colormap.listMaps())
        except Exception:
            maps = ["viridis", "plasma", "magma", "cividis", "gray"]
        for name in maps:
            self.cmb_default_cmap.addItem(name, name)
        current_default = str(self._data.get("colormaps", {}).get("default", ""))
        idx = self.cmb_default_cmap.findData(current_default)
        if idx < 0:
            idx = 0
        self.cmb_default_cmap.setCurrentIndex(idx)
        layout.addWidget(QtWidgets.QLabel("Default colormap"))
        layout.addWidget(self.cmb_default_cmap)

        self.table_colormaps = QtWidgets.QTableWidget(0, 2)
        self.table_colormaps.setHorizontalHeaderLabels(["Variable", "Colormap"])
        self.table_colormaps.horizontalHeader().setStretchLastSection(True)
        self.table_colormaps.verticalHeader().setVisible(False)
        self.table_colormaps.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_colormaps.setEditTriggers(QtWidgets.QAbstractItemView.DoubleClicked | QtWidgets.QAbstractItemView.EditKeyPressed)
        layout.addWidget(QtWidgets.QLabel("Variable-specific colormaps"))
        layout.addWidget(self.table_colormaps, 1)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_colormap = QtWidgets.QPushButton("Add mapping…")
        self.btn_add_colormap.clicked.connect(self._add_colormap_mapping)
        btn_row.addWidget(self.btn_add_colormap)
        self.btn_remove_colormap = QtWidgets.QPushButton("Remove selected")
        self.btn_remove_colormap.clicked.connect(self._remove_colormap_mapping)
        btn_row.addWidget(self.btn_remove_colormap)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        variables = self._data.get("colormaps", {}).get("variables", {})
        if isinstance(variables, dict):
            for var, cmap in sorted(variables.items()):
                row = self.table_colormaps.rowCount()
                self.table_colormaps.insertRow(row)
                self.table_colormaps.setItem(row, 0, QtWidgets.QTableWidgetItem(str(var)))
                self.table_colormaps.setItem(row, 1, QtWidgets.QTableWidgetItem(str(cmap)))

        self.tabs.addTab(tab, "Colormaps")

    def _browse_export_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select default export folder",
            self.txt_export_dir.text(),
        )
        if directory:
            self.txt_export_dir.setText(directory)

    def _add_colormap_mapping(self):
        var, ok = QtWidgets.QInputDialog.getText(self, "Variable name", "Variable name:")
        if not ok or not str(var).strip():
            return
        cmap, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Colormap",
            "Choose a colormap:",
            [self.cmb_default_cmap.itemText(i) for i in range(self.cmb_default_cmap.count()) if self.cmb_default_cmap.itemData(i)],
            editable=True,
        )
        if not ok or not str(cmap).strip():
            return
        row = self.table_colormaps.rowCount()
        self.table_colormaps.insertRow(row)
        self.table_colormaps.setItem(row, 0, QtWidgets.QTableWidgetItem(str(var).strip()))
        self.table_colormaps.setItem(row, 1, QtWidgets.QTableWidgetItem(str(cmap).strip()))

    def _remove_colormap_mapping(self):
        rows = sorted({idx.row() for idx in self.table_colormaps.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table_colormaps.removeRow(row)

    def _on_load_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load preferences",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            temp = PreferencesManager()
            temp.load_from_file(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._data = temp.data()
        self._apply_data()

    def _on_save_file(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save preferences",
            "preferences.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(self._collect_data(), indent=2))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(exc))

    def _collect_data(self) -> Dict[str, object]:
        general = {
            "autoscale_on_load": self.chk_autoscale.isChecked(),
            "default_layout_label": self.txt_layout_label.text(),
        }
        colormaps = {
            "default": self.cmb_default_cmap.currentData() or "",
            "variables": {},
        }
        for row in range(self.table_colormaps.rowCount()):
            var_item = self.table_colormaps.item(row, 0)
            cmap_item = self.table_colormaps.item(row, 1)
            if not var_item or not cmap_item:
                continue
            var = var_item.text().strip()
            cmap = cmap_item.text().strip()
            if var and cmap:
                colormaps["variables"][var] = cmap
        misc = {
            "default_export_dir": self.txt_export_dir.text().strip(),
        }
        return {"general": general, "colormaps": colormaps, "misc": misc}

    def _apply_data(self):
        data = self._data
        self.chk_autoscale.setChecked(bool(data.get("general", {}).get("autoscale_on_load", True)))
        self.txt_layout_label.setText(str(data.get("general", {}).get("default_layout_label", "")))
        self.txt_export_dir.setText(str(data.get("misc", {}).get("default_export_dir", "")))
        default_cmap = str(data.get("colormaps", {}).get("default", ""))
        idx = self.cmb_default_cmap.findData(default_cmap)
        if idx < 0:
            idx = 0
        self.cmb_default_cmap.setCurrentIndex(idx)
        self.table_colormaps.setRowCount(0)
        variables = data.get("colormaps", {}).get("variables", {})
        if isinstance(variables, dict):
            for var, cmap in sorted(variables.items()):
                row = self.table_colormaps.rowCount()
                self.table_colormaps.insertRow(row)
                self.table_colormaps.setItem(row, 0, QtWidgets.QTableWidgetItem(str(var)))
                self.table_colormaps.setItem(row, 1, QtWidgets.QTableWidgetItem(str(cmap)))

    def accept(self):
        self._data = self._collect_data()
        super().accept()

    def result_data(self) -> Dict[str, object]:
        return self._collect_data()
class ProcessingDockWidget(QtWidgets.QWidget):
    def __init__(self, manager: ProcessingManager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.steps: List[ProcessingStep] = []

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(8)

        # --- Step builder ---
        builder = QtWidgets.QGroupBox("Build step")
        builder_layout = QtWidgets.QVBoxLayout(builder)
        builder_layout.setContentsMargins(6, 6, 6, 6)
        builder_layout.setSpacing(6)

        func_row = QtWidgets.QHBoxLayout()
        func_row.addWidget(QtWidgets.QLabel("Function:"))
        self.cmb_function = QtWidgets.QComboBox()
        self.cmb_function.addItem("Select…", "")
        self._stack_indices: Dict[str, int] = {}
        self.param_stack = QtWidgets.QStackedWidget()
        self.param_stack.addWidget(QtWidgets.QWidget())
        for spec in list_processing_functions():
            self.cmb_function.addItem(spec.label, spec.key)
            form = ParameterForm(spec.parameters)
            form.parametersChanged.connect(self._on_function_params_changed)
            idx = self.param_stack.addWidget(form)
            self._stack_indices[spec.key] = idx
        self.cmb_function.currentIndexChanged.connect(self._on_function_changed)
        func_row.addWidget(self.cmb_function, 1)
        builder_layout.addLayout(func_row)
        builder_layout.addWidget(self.param_stack)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add_step = QtWidgets.QPushButton("Add step")
        self.btn_add_step.clicked.connect(self._add_step)
        btn_row.addWidget(self.btn_add_step)
        self.btn_update_step = QtWidgets.QPushButton("Update selected")
        self.btn_update_step.clicked.connect(self._update_step)
        btn_row.addWidget(self.btn_update_step)
        builder_layout.addLayout(btn_row)

        outer.addWidget(builder)

        # --- Steps list ---
        self.list_steps = QtWidgets.QListWidget()
        self.list_steps.currentRowChanged.connect(self._on_step_selected)
        outer.addWidget(self.list_steps, 1)

        step_btns = QtWidgets.QHBoxLayout()
        self.btn_move_up = QtWidgets.QPushButton("Move up")
        self.btn_move_up.clicked.connect(lambda: self._move_step(-1))
        step_btns.addWidget(self.btn_move_up)
        self.btn_move_down = QtWidgets.QPushButton("Move down")
        self.btn_move_down.clicked.connect(lambda: self._move_step(1))
        step_btns.addWidget(self.btn_move_down)
        self.btn_remove_step = QtWidgets.QPushButton("Remove")
        self.btn_remove_step.clicked.connect(self._remove_step)
        step_btns.addWidget(self.btn_remove_step)
        self.btn_clear_steps = QtWidgets.QPushButton("Clear")
        self.btn_clear_steps.clicked.connect(self._clear_steps)
        step_btns.addWidget(self.btn_clear_steps)
        outer.addLayout(step_btns)

        # --- Pipeline info ---
        name_row = QtWidgets.QHBoxLayout()
        name_row.addWidget(QtWidgets.QLabel("Pipeline name:"))
        self.edit_name = QtWidgets.QLineEdit()
        self.edit_name.textChanged.connect(lambda _: self._update_buttons())
        name_row.addWidget(self.edit_name, 1)
        outer.addLayout(name_row)

        save_row = QtWidgets.QHBoxLayout()
        self.btn_save_pipeline = QtWidgets.QPushButton("Save pipeline")
        self.btn_save_pipeline.clicked.connect(self._save_pipeline)
        save_row.addWidget(self.btn_save_pipeline)
        self.btn_interactive = QtWidgets.QPushButton("Interactive edit…")
        self.btn_interactive.clicked.connect(self._interactive_edit)
        save_row.addWidget(self.btn_interactive)
        outer.addLayout(save_row)

        # --- Saved pipelines ---
        saved_box = QtWidgets.QGroupBox("Saved pipelines")
        saved_layout = QtWidgets.QVBoxLayout(saved_box)
        saved_layout.setContentsMargins(6, 6, 6, 6)
        saved_layout.setSpacing(6)
        self.list_saved = QtWidgets.QListWidget()
        self.list_saved.itemSelectionChanged.connect(self._update_buttons)
        self.list_saved.itemDoubleClicked.connect(lambda _: self._load_saved())
        saved_layout.addWidget(self.list_saved)

        saved_btns = QtWidgets.QHBoxLayout()
        self.btn_load_saved = QtWidgets.QPushButton("Load")
        self.btn_load_saved.clicked.connect(self._load_saved)
        saved_btns.addWidget(self.btn_load_saved)
        self.btn_delete_saved = QtWidgets.QPushButton("Delete")
        self.btn_delete_saved.clicked.connect(self._delete_saved)
        saved_btns.addWidget(self.btn_delete_saved)
        self.btn_export_saved = QtWidgets.QPushButton("Export…")
        self.btn_export_saved.clicked.connect(self._export_saved)
        saved_btns.addWidget(self.btn_export_saved)
        self.btn_import_saved = QtWidgets.QPushButton("Import…")
        self.btn_import_saved.clicked.connect(self._import_saved)
        saved_btns.addWidget(self.btn_import_saved)
        saved_layout.addLayout(saved_btns)

        outer.addWidget(saved_box)
        outer.addStretch(1)

        self.manager.pipelines_changed.connect(self._refresh_saved)
        self._refresh_saved()
        self._update_buttons()

    # ----- helpers -----
    def _current_spec(self):
        key = self.cmb_function.currentData()
        if not key:
            return None
        return get_processing_function(str(key))

    def _current_form(self) -> Optional[ParameterForm]:
        key = self.cmb_function.currentData()
        if not key:
            return None
        idx = self._stack_indices.get(str(key))
        if idx is None:
            return None
        widget = self.param_stack.widget(idx)
        return widget if isinstance(widget, ParameterForm) else None

    def _selected_step_index(self) -> int:
        return self.list_steps.currentRow()

    def _selected_saved_name(self) -> Optional[str]:
        items = self.list_saved.selectedItems()
        if not items:
            return None
        return str(items[0].data(QtCore.Qt.UserRole) or items[0].text())

    def _refresh_step_list(self):
        self.list_steps.clear()
        for step in self.steps:
            spec = get_processing_function(step.key)
            label = spec.label if spec else step.key
            summary = summarize_parameters(step.key, step.params)
            text = label
            if summary:
                text += f" ({summary})"
            self.list_steps.addItem(text)
        self._update_buttons()

    def _set_function_selection(self, key: str):
        block = self.cmb_function.blockSignals(True)
        idx = self.cmb_function.findData(key)
        if idx >= 0:
            self.cmb_function.setCurrentIndex(idx)
        self.cmb_function.blockSignals(block)
        self._on_function_changed()

    def _build_pipeline(self) -> ProcessingPipeline:
        name = self.edit_name.text().strip() or "Untitled"
        return ProcessingPipeline(name=name, steps=[ProcessingStep(step.key, dict(step.params)) for step in self.steps])

    # ----- button state -----
    def _update_buttons(self):
        idx = self._selected_step_index()
        has_selection = 0 <= idx < len(self.steps)
        has_steps = bool(self.steps)
        self.btn_update_step.setEnabled(has_selection)
        self.btn_move_up.setEnabled(has_selection and idx > 0)
        self.btn_move_down.setEnabled(has_selection and idx < len(self.steps) - 1)
        self.btn_remove_step.setEnabled(has_selection)
        self.btn_clear_steps.setEnabled(has_steps)
        self.btn_interactive.setEnabled(has_steps)
        self.btn_save_pipeline.setEnabled(has_steps and bool(self.edit_name.text().strip()))
        has_saved = self._selected_saved_name() is not None
        self.btn_load_saved.setEnabled(has_saved)
        self.btn_delete_saved.setEnabled(has_saved)
        self.btn_export_saved.setEnabled(has_saved)

    # ----- slots -----
    def _on_function_changed(self):
        key = self.cmb_function.currentData()
        if key and key in self._stack_indices:
            self.param_stack.setCurrentIndex(self._stack_indices[str(key)])
        else:
            self.param_stack.setCurrentIndex(0)
        self._update_buttons()

    def _on_function_params_changed(self):
        self._update_buttons()

    def _add_step(self):
        spec = self._current_spec()
        form = self._current_form()
        if spec is None or form is None:
            return
        self.steps.append(ProcessingStep(spec.key, form.values()))
        self._refresh_step_list()
        self.list_steps.setCurrentRow(len(self.steps) - 1)

    def _update_step(self):
        idx = self._selected_step_index()
        spec = self._current_spec()
        form = self._current_form()
        if idx < 0 or idx >= len(self.steps) or spec is None or form is None:
            return
        self.steps[idx] = ProcessingStep(spec.key, form.values())
        self._refresh_step_list()
        self.list_steps.setCurrentRow(idx)

    def _on_step_selected(self, index: int):
        if index < 0 or index >= len(self.steps):
            self._update_buttons()
            return
        step = self.steps[index]
        self._set_function_selection(step.key)
        form = self._current_form()
        if form:
            form.set_values(step.params)
        self._update_buttons()

    def _move_step(self, delta: int):
        idx = self._selected_step_index()
        new_idx = idx + delta
        if idx < 0 or new_idx < 0 or new_idx >= len(self.steps):
            return
        self.steps[idx], self.steps[new_idx] = self.steps[new_idx], self.steps[idx]
        self._refresh_step_list()
        self.list_steps.setCurrentRow(new_idx)

    def _remove_step(self):
        idx = self._selected_step_index()
        if idx < 0 or idx >= len(self.steps):
            return
        del self.steps[idx]
        self._refresh_step_list()
        if self.steps:
            self.list_steps.setCurrentRow(min(idx, len(self.steps) - 1))

    def _clear_steps(self):
        if QtWidgets.QMessageBox.question(self, "Clear steps", "Remove all steps from the pipeline?") != QtWidgets.QMessageBox.Yes:
            return
        self.steps.clear()
        self._refresh_step_list()

    def _save_pipeline(self):
        if not self.steps:
            return
        name = self.edit_name.text().strip()
        if not name:
            QtWidgets.QMessageBox.warning(self, "Missing name", "Please enter a name for the pipeline.")
            return
        pipeline = ProcessingPipeline(name=name, steps=[ProcessingStep(step.key, dict(step.params)) for step in self.steps])
        try:
            self.manager.save_pipeline(pipeline)
        except ValueError as e:
            QtWidgets.QMessageBox.warning(self, "Save failed", str(e))
            return
        self._refresh_saved()
        items = self.list_saved.findItems(name, QtCore.Qt.MatchExactly)
        if items:
            self.list_saved.setCurrentItem(items[0])
        QtWidgets.QMessageBox.information(self, "Pipeline saved", f"Pipeline '{name}' saved.")
        log_action(f"Saved processing pipeline '{name}' with {len(self.steps)} step(s)")

    def _interactive_edit(self):
        if not self.steps:
            return
        pipeline = self._build_pipeline()
        dlg = PipelineEditorDialog(self.manager, pipeline, self)
        dlg.exec_()
        updated = dlg.result_pipeline()
        self.steps = [ProcessingStep(step.key, dict(step.params)) for step in updated.steps]
        self._refresh_step_list()

    def _refresh_saved(self):
        selected = self._selected_saved_name()
        self.list_saved.blockSignals(True)
        self.list_saved.clear()
        for name in self.manager.pipeline_names():
            item = QtWidgets.QListWidgetItem(name)
            item.setData(QtCore.Qt.UserRole, name)
            self.list_saved.addItem(item)
            if selected and name == selected:
                item.setSelected(True)
        if selected is None and self.list_saved.count() > 0:
            self.list_saved.setCurrentRow(0)
        self.list_saved.blockSignals(False)
        self._update_buttons()

    def _load_saved(self):
        name = self._selected_saved_name()
        if not name:
            return
        pipeline = self.manager.get_pipeline(name)
        if pipeline is None:
            return
        self.edit_name.setText(pipeline.name)
        self.steps = [ProcessingStep(step.key, dict(step.params)) for step in pipeline.steps]
        self._refresh_step_list()
        if self.steps:
            self.list_steps.setCurrentRow(0)

    def _delete_saved(self):
        name = self._selected_saved_name()
        if not name:
            return
        if QtWidgets.QMessageBox.question(self, "Delete pipeline", f"Delete pipeline '{name}'?") != QtWidgets.QMessageBox.Yes:
            return
        self.manager.delete_pipeline(name)
        self._refresh_saved()
        log_action(f"Deleted processing pipeline '{name}'")

    def _export_saved(self):
        name = self._selected_saved_name()
        if not name:
            return
        pipeline = self.manager.get_pipeline(name)
        if pipeline is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Export pipeline", f"{name}.json", "Pipeline JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(pipeline.to_dict(), fh, indent=2)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(e))

    def _import_saved(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Import pipeline", "", "Pipeline JSON (*.json);;All files (*)")
        if not path:
            return
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            pipeline = ProcessingPipeline.from_dict(data)
            if not pipeline.name:
                pipeline.name = Path(path).stem
            self.manager.save_pipeline(pipeline)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(e))
            return
        self._refresh_saved()
        items = self.list_saved.findItems(pipeline.name, QtCore.Qt.MatchExactly)
        if items:
            self.list_saved.setCurrentItem(items[0])


class ProcessingSelectionDialog(QtWidgets.QDialog):
    def __init__(self, manager: Optional[ProcessingManager], parent=None):
        super().__init__(parent)
        self.manager = manager
        self.setWindowTitle("Apply Processing")
        self.resize(420, 360)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        layout.addWidget(QtWidgets.QLabel("Choose a processing function or pipeline:"))

        self.cmb_mode = QtWidgets.QComboBox()
        layout.addWidget(self.cmb_mode)

        self.stack = QtWidgets.QStackedWidget()
        layout.addWidget(self.stack, 1)

        self._function_forms: Dict[str, ParameterForm] = {}

        none_widget = QtWidgets.QWidget()
        self._none_index = self.stack.addWidget(none_widget)
        self._add_mode_item("No processing", {"type": "none", "stack": self._none_index})

        for spec in list_processing_functions():
            form = ParameterForm(spec.parameters)
            idx = self.stack.addWidget(form)
            self._function_forms[spec.key] = form
            self._add_mode_item(
                f"Function: {spec.label}",
                {"type": "function", "key": spec.key, "stack": idx},
            )

        pipelines = self.manager.list_pipelines() if self.manager else []
        if pipelines:
            self.cmb_mode.insertSeparator(self.cmb_mode.count())
            for pipeline in pipelines:
                summary = QtWidgets.QPlainTextEdit()
                summary.setReadOnly(True)
                summary.setPlainText(self._summarize_pipeline(pipeline))
                summary.setMinimumHeight(160)
                idx = self.stack.addWidget(summary)
                self._add_mode_item(
                    f"Pipeline: {pipeline.name}",
                    {"type": "pipeline", "name": pipeline.name, "stack": idx},
                )

        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self.stack.setCurrentIndex(self._none_index)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _add_mode_item(self, label: str, data: Dict[str, object]):
        self.cmb_mode.addItem(label)
        index = self.cmb_mode.count() - 1
        self.cmb_mode.setItemData(index, data)

    def _on_mode_changed(self, index: int):
        data = self.cmb_mode.itemData(index) or {}
        stack_index = data.get("stack", self._none_index)
        try:
            self.stack.setCurrentIndex(int(stack_index))
        except Exception:
            self.stack.setCurrentIndex(self._none_index)

    def _summarize_pipeline(self, pipeline: ProcessingPipeline) -> str:
        if not pipeline.steps:
            return "(No steps)"
        lines: List[str] = []
        for i, step in enumerate(pipeline.steps, 1):
            spec = get_processing_function(step.key)
            label = spec.label if spec else step.key
            summary = summarize_parameters(step.key, step.params)
            text = f"{i}. {label}"
            if summary:
                text += f" — {summary}"
            lines.append(text)
        return "\n".join(lines)

    def selected_processing(self) -> Tuple[str, Dict[str, object]]:
        index = self.cmb_mode.currentIndex()
        data = self.cmb_mode.itemData(index) or {}
        mode_type = data.get("type")
        if mode_type == "function":
            key = str(data.get("key", ""))
            form = self._function_forms.get(key)
            params = form.values() if form else {}
            return key, params
        if mode_type == "pipeline":
            name = str(data.get("name", ""))
            return f"pipeline:{name}", {}
        return "none", {}

# ---------------------------------------------------------------------------
# Datasets pane (left): load a dataset and list 2D variables (drag from here)
# ---------------------------------------------------------------------------
class DatasetsPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_open_netcdf = QtWidgets.QPushButton("Load NetCDF…")
        self.btn_open_netcdf.clicked.connect(self._open_netcdf)
        btn_row.addWidget(self.btn_open_netcdf)

        self.btn_open_json = QtWidgets.QPushButton("Load JSON…")
        self.btn_open_json.clicked.connect(self._open_json)
        btn_row.addWidget(self.btn_open_json)

        self.btn_open_db = QtWidgets.QPushButton("Load Database…")
        self.btn_open_db.clicked.connect(self._open_database)
        btn_row.addWidget(self.btn_open_db)

        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs, 1)

        self._trees: Dict[str, QtWidgets.QTreeWidget] = {}
        self._roots: Dict[str, Dict[str, QtWidgets.QTreeWidgetItem]] = {
            "2d": {},
            "nd": {},
            "sliced": {},
            "interactive": {},
        }

        for key, title in (
            ("2d", "2D Data"),
            ("nd", ">2D Data"),
            ("sliced", "Sliced Data"),
            ("interactive", "Interactive Data"),
        ):
            tree = self._create_tree()
            self._trees[key] = tree
            self.tabs.addTab(tree, title)

        self._populate_examples()

    def _create_tree(self) -> QtWidgets.QTreeWidget:
        tree = QtWidgets.QTreeWidget()
        tree.setHeaderLabels(["Datasets / Variables"])
        tree.setDragEnabled(True)
        tree.setDefaultDropAction(QtCore.Qt.CopyAction)
        tree.setDropIndicatorShown(False)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)

        def _mime_data(_items, *, _tree=tree):
            md = QtCore.QMimeData()
            sel = _tree.selectedItems()
            if sel:
                payload = sel[0].data(0, QtCore.Qt.UserRole)
                if payload:
                    md.setText(payload)
            return md

        tree.mimeData = _mime_data  # type: ignore[attr-defined]
        return tree

    def _populate_examples(self):
        candidates = [
            Path(__file__).resolve().parent / "example_dataset.nc",
            Path(__file__).resolve().parent / "example_rect_warp.nc",
        ]
        for path in candidates:
            if path.exists():
                self._register_dataset(path, quiet=True)

    def _format_hint(self, da: xr.DataArray) -> str:
        try:
            coords = guess_phys_coords(da)
            if "X" in coords and "Y" in coords:
                return "(X,Y)"
            if "x" in coords and "y" in coords:
                return "(x,y)"
        except Exception:
            pass
        try:
            dims = list(getattr(da, "dims", ())[:2])
            if dims:
                sizes = getattr(da, "sizes", {})
                parts = [f"{dim}[{sizes.get(dim, '?')}]" for dim in dims]
                return "(" + " × ".join(parts) + ")"
        except Exception:
            pass
        return "(data)"

    def _ensure_disk_root(self, category: str, path: Path) -> Tuple[str, QtWidgets.QTreeWidgetItem]:
        key = str(Path(path).resolve())
        roots = self._roots[category]
        if key in roots:
            item = roots[key]
            item.takeChildren()
            return key, item
        item = QtWidgets.QTreeWidgetItem([path.name])
        item.setToolTip(0, str(path))
        item.setExpanded(True)
        item.setData(0, QtCore.Qt.UserRole, DataSetRef(path).to_mime())
        self._trees[category].addTopLevelItem(item)
        roots[key] = item
        return key, item

    def _ensure_memory_root(self, category: str, key: str, label: str, mime: str) -> QtWidgets.QTreeWidgetItem:
        roots = self._roots[category]
        if key in roots:
            item = roots[key]
            item.takeChildren()
            item.setText(0, label)
            item.setData(0, QtCore.Qt.UserRole + 1, category)
            return item
        item = QtWidgets.QTreeWidgetItem([label])
        item.setExpanded(True)
        item.setData(0, QtCore.Qt.UserRole, mime)
        item.setData(0, QtCore.Qt.UserRole + 1, category)
        self._trees[category].addTopLevelItem(item)
        roots[key] = item
        return item

    def _remove_root(self, category: str, key: str):
        item = self._roots[category].pop(key, None)
        if not item:
            return
        tree = self._trees[category]
        index = tree.indexOfTopLevelItem(item)
        if index >= 0:
            tree.takeTopLevelItem(index)

    def _register_dataset(self, path: Path, quiet: bool = False):
        try:
            ds = open_dataset(path)
        except Exception as e:
            if not quiet:
                QtWidgets.QMessageBox.warning(self, "Open failed", str(e))
            return

        try:
            self._populate_disk_category("2d", path, ds, lambda arr: getattr(arr, "ndim", 0) == 2)
            self._populate_disk_category("nd", path, ds, lambda arr: getattr(arr, "ndim", 0) > 2)
            if not quiet:
                log_action(f"Loaded dataset '{path.name}'")
        finally:
            try:
                ds.close()
            except Exception:
                pass

    def _populate_disk_category(
        self,
        category: str,
        path: Path,
        dataset: xr.Dataset,
        predicate,
    ):
        key, item = self._ensure_disk_root(category, path)
        added = False
        for var in dataset.data_vars:
            try:
                da = dataset[var]
            except Exception:
                continue
            if not predicate(da):
                continue
            hint = self._format_hint(da)
            child = QtWidgets.QTreeWidgetItem([f"{var}  {hint}"])
            if category == "2d":
                child.setData(0, QtCore.Qt.UserRole, VarRef(path, var, hint).to_mime())
            else:
                child.setData(0, QtCore.Qt.UserRole, HighDimVarRef(var, hint, path=path).to_mime())
            item.addChild(child)
            added = True
        if not added:
            self._remove_root(category, key)

    def _unique_label(self, category: str, base: str) -> str:
        existing = {itm.text(0) for itm in self._roots[category].values()}
        if base not in existing:
            return base
        idx = 2
        while True:
            candidate = f"{base} ({idx})"
            if candidate not in existing:
                return candidate
            idx += 1

    def register_sliced_dataset(self, label: str, dataset: xr.Dataset):
        name = self._unique_label("sliced", label)
        stored = dataset.copy(deep=True)
        key = MemoryDatasetRegistry.register(stored, name)
        mime = MemoryDatasetRef(key).to_mime()
        item = self._ensure_memory_root("sliced", key, name, mime)
        self._populate_memory_children(item, key, stored)
        if item.childCount() == 0:
            self._remove_root("sliced", key)
        else:
            log_action(f"Registered sliced dataset '{name}' with {stored.dims}")

    def register_interactive_dataset(self, label: str, dataset: xr.Dataset) -> str:
        name = self._unique_label("interactive", label)
        working = dataset
        try:
            working = dataset.load()
        except Exception:
            pass
        stored = working.copy(deep=True)
        key = MemoryDatasetRegistry.register(stored, name)
        mime = MemoryDatasetRef(key).to_mime()
        item = self._ensure_memory_root("interactive", key, name, mime)
        self._populate_memory_children(item, key, stored)
        if item.childCount() == 0:
            self._remove_root("interactive", key)
        try:
            dataset.close()
        except Exception:
            pass
        log_action(f"Registered interactive dataset '{name}'")
        return name

    def register_interactive_dataarray(self, label: str, dataarray: xr.DataArray) -> str:
        var_name = dataarray.name or "variable"
        dataset = xr.Dataset({var_name: dataarray})
        dataset[var_name].attrs.setdefault("_source_var", var_name)
        dataset[var_name].attrs.setdefault("_slice_label", label)
        return self.register_interactive_dataset(label, dataset)

    def _populate_memory_children(self, root: QtWidgets.QTreeWidgetItem, key: str, dataset: xr.Dataset):
        added = False
        category = root.data(0, QtCore.Qt.UserRole + 1) or ""
        for var in dataset.data_vars:
            try:
                da = dataset[var]
            except Exception:
                continue
            ndim = getattr(da, "ndim", 0)
            hint = self._format_hint(da)
            if category == "sliced" and ndim == 2:
                alias = str(da.attrs.get("_source_var", var) or var)
                label = str(da.attrs.get("_slice_label", ""))
                parts = [alias]
                if label:
                    parts.append(f"[{label}]")
                display = " ".join(parts)
                if hint:
                    display = f"{display}  {hint}"
                child = QtWidgets.QTreeWidgetItem([display])
                child.setData(
                    0,
                    QtCore.Qt.UserRole,
                    MemorySliceRef(key, var, alias, label, hint).to_mime(),
                )
                root.addChild(child)
                added = True
                continue

            text = f"{var}  {hint}" if hint else var
            child = QtWidgets.QTreeWidgetItem([text])
            if ndim == 2:
                child.setData(0, QtCore.Qt.UserRole, MemoryVarRef(key, var, hint).to_mime())
                added = True
            elif ndim > 2:
                child.setData(0, QtCore.Qt.UserRole, HighDimVarRef(var, hint, memory_key=key).to_mime())
                added = True
            else:
                continue
            root.addChild(child)
        if not added:
            root.addChild(QtWidgets.QTreeWidgetItem(["(no compatible variables)"]))

    def _open_netcdf(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open xarray Dataset",
            "",
            "NetCDF / Zarr (*.nc *.zarr);;All files (*)",
        )
        if not path:
            return
        self._register_dataset(Path(path))

    def _open_json(self):
        paths, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            "Open JSON Dataset",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not paths:
            return
        for selected in paths:
            log_action(f"Queued JSON dataset '{Path(selected).name}' for conversion")
        # TODO: convert selected JSON files to xarray.Dataset instances.
        pass
        QtWidgets.QMessageBox.information(
            self,
            "Not implemented",
            "JSON dataset conversion will be implemented in a future update.",
        )

    def _open_database(self):
        dialog = DatabaseQueryDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        query = dialog.query()
        if query:
            log_action("Queued database query for dataset import")
        # TODO: execute the database query and convert results to xarray.Dataset.
        pass
        QtWidgets.QMessageBox.information(
            self,
            "Not implemented",
            "Database loading will be implemented in a future update.",
        )


class DatabaseQueryDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Database Query")
        self.resize(420, 320)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        label = QtWidgets.QLabel(
            "Enter a dictionary describing the database query to execute."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        self.text = QtWidgets.QPlainTextEdit()
        self.text.setPlainText("{\n    \"table\": \"example\",\n    \"filters\": {}\n}")
        layout.addWidget(self.text, 1)

        self.lbl_status = QtWidgets.QLabel(" ")
        self.lbl_status.setStyleSheet("color: #888;")
        layout.addWidget(self.lbl_status)

        btns = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def query(self) -> Dict[str, object]:
        text = self.text.toPlainText().strip()
        if not text:
            return {}
        try:
            payload = json.loads(text)
        except Exception:
            self.lbl_status.setText("Invalid JSON; returning empty query.")
            return {}
        if not isinstance(payload, dict):
            self.lbl_status.setText("Query must be a dictionary; returning empty query.")
            return {}
        self.lbl_status.setText(" ")
        return payload


class DimSliceControl(QtWidgets.QWidget):
    def __init__(self, dim: str, size: int, parent=None):
        super().__init__(parent)
        self.dim = dim
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        layout.addWidget(QtWidgets.QLabel("Start"))
        self.spin_start = QtWidgets.QSpinBox()
        self.spin_start.setRange(0, max(size - 1, 0))
        layout.addWidget(self.spin_start)

        layout.addWidget(QtWidgets.QLabel("Stop"))
        self.spin_stop = QtWidgets.QSpinBox()
        self.spin_stop.setRange(0, max(size - 1, 0))
        self.spin_stop.setValue(max(size - 1, 0))
        layout.addWidget(self.spin_stop)

        layout.addWidget(QtWidgets.QLabel("Step"))
        self.spin_step = QtWidgets.QSpinBox()
        self.spin_step.setRange(1, max(size, 1))
        self.spin_step.setValue(1)
        layout.addWidget(self.spin_step)

        if size:
            layout.addWidget(QtWidgets.QLabel(f"(0–{size - 1})"))
        else:
            layout.addWidget(QtWidgets.QLabel("(empty)"))

    def indices(self) -> List[int]:
        start = int(self.spin_start.value())
        stop = int(self.spin_stop.value())
        step = int(self.spin_step.value())
        if step <= 0 or stop < start:
            return []
        return list(range(start, stop + 1, step))


class SliceDataTab(QtWidgets.QWidget):
    def __init__(self, library: DatasetsPane, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.library = library

        self._dataset: Optional[xr.Dataset] = None
        self._dataset_label: str = ""
        self._dataset_path: Optional[Path] = None
        self._dataset_memory_key: Optional[str] = None
        self._current_var: Optional[str] = None
        self._current_da: Optional[xr.DataArray] = None
        self._current_dims: List[str] = []
        self._dim_controls: Dict[str, DimSliceControl] = {}

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        hint = QtWidgets.QLabel(
            "Drop a dataset or high-dimensional variable here to configure slicing."
        )
        hint.setStyleSheet("color: #666;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        top = QtWidgets.QHBoxLayout()
        self.lbl_dataset = QtWidgets.QLabel("No dataset loaded")
        self.lbl_dataset.setStyleSheet("color: #555;")
        top.addWidget(self.lbl_dataset, 1)
        self.btn_load = QtWidgets.QPushButton("Load dataset…")
        self.btn_load.clicked.connect(self._open_dataset)
        top.addWidget(self.btn_load, 0)
        outer.addLayout(top)

        var_row = QtWidgets.QHBoxLayout()
        var_row.addWidget(QtWidgets.QLabel("Variable:"))
        self.cmb_variable = QtWidgets.QComboBox()
        self.cmb_variable.setEnabled(False)
        self.cmb_variable.currentIndexChanged.connect(self._on_variable_changed)
        var_row.addWidget(self.cmb_variable, 1)
        outer.addLayout(var_row)

        axis_group = QtWidgets.QGroupBox("Output plane")
        axis_form = QtWidgets.QFormLayout(axis_group)
        axis_form.setContentsMargins(6, 6, 6, 6)
        axis_form.setSpacing(6)

        self.cmb_row_dim = QtWidgets.QComboBox()
        self.cmb_row_dim.setEnabled(False)
        self.cmb_row_dim.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow("Rows", self.cmb_row_dim)

        self.cmb_col_dim = QtWidgets.QComboBox()
        self.cmb_col_dim.setEnabled(False)
        self.cmb_col_dim.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow("Columns", self.cmb_col_dim)

        self.extra_dims_widget = QtWidgets.QWidget()
        self.extra_dims_layout = QtWidgets.QFormLayout(self.extra_dims_widget)
        self.extra_dims_layout.setContentsMargins(0, 0, 0, 0)
        self.extra_dims_layout.setSpacing(4)
        axis_form.addRow("Additional slices", self.extra_dims_widget)

        outer.addWidget(axis_group)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_generate = QtWidgets.QPushButton("Generate slices")
        self.btn_generate.setEnabled(False)
        self.btn_generate.clicked.connect(self._generate_slices)
        btn_row.addWidget(self.btn_generate)
        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.lbl_status = QtWidgets.QLabel(" ")
        self.lbl_status.setStyleSheet("color: #555;")
        self.lbl_status.setWordWrap(True)
        outer.addWidget(self.lbl_status)

    # ---------- dataset helpers ----------
    def _dispose_dataset(self):
        if self._dataset is not None:
            try:
                close = getattr(self._dataset, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        self._dataset = None
        self._dataset_label = ""
        self._dataset_path = None
        self._dataset_memory_key = None

    def _set_dataset(
        self,
        dataset: xr.Dataset,
        label: str,
        *,
        path: Optional[Path] = None,
        memory_key: Optional[str] = None,
    ):
        self._dispose_dataset()
        self._dataset = dataset
        self._dataset_label = label
        self._dataset_path = path
        self._dataset_memory_key = memory_key
        self.lbl_dataset.setText(label)
        if path is not None:
            self.lbl_dataset.setToolTip(str(path))
        else:
            self.lbl_dataset.setToolTip("")
        self._populate_variable_combo()

    def _populate_variable_combo(self):
        self.cmb_variable.blockSignals(True)
        self.cmb_variable.clear()
        self.cmb_variable.addItem("Select variable…", None)
        if self._dataset is None:
            self.cmb_variable.blockSignals(False)
            self.cmb_variable.setEnabled(False)
            self._clear_current_variable()
            return
        count = 0
        for var in self._dataset.data_vars:
            da = self._dataset[var]
            if getattr(da, "ndim", 0) <= 2:
                continue
            dims = getattr(da, "dims", ())
            shape = " × ".join(str(getattr(da, "sizes", {}).get(dim, "?")) for dim in dims)
            self.cmb_variable.addItem(f"{var} ({shape})", var)
            count += 1
        self.cmb_variable.blockSignals(False)
        self.cmb_variable.setEnabled(count > 0)
        self.btn_generate.setEnabled(False)
        if count == 0:
            self._clear_current_variable()
            self.lbl_status.setText("Dataset does not contain variables with more than two dimensions.")
        else:
            self.lbl_status.setText("Select a variable and configure the slicing parameters.")

    def _clear_current_variable(self):
        self._current_var = None
        self._current_da = None
        self._current_dims = []
        self.cmb_row_dim.blockSignals(True)
        self.cmb_row_dim.clear()
        self.cmb_row_dim.blockSignals(False)
        self.cmb_row_dim.setEnabled(False)
        self.cmb_col_dim.blockSignals(True)
        self.cmb_col_dim.clear()
        self.cmb_col_dim.blockSignals(False)
        self.cmb_col_dim.setEnabled(False)
        self._reset_extra_dim_controls([])

    def _reset_extra_dim_controls(self, dims: List[str]):
        while self.extra_dims_layout.count():
            item = self.extra_dims_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._dim_controls.clear()
        if not dims:
            placeholder = QtWidgets.QLabel("No extra dimensions to iterate.")
            placeholder.setStyleSheet("color: #777;")
            self.extra_dims_layout.addRow(placeholder)
            return
        for dim in dims:
            size = int(getattr(self._current_da, "sizes", {}).get(dim, 0)) if self._current_da is not None else 0
            control = DimSliceControl(dim, size)
            self._dim_controls[dim] = control
            self.extra_dims_layout.addRow(dim, control)

    def _on_variable_changed(self, index: int):
        data = self.cmb_variable.itemData(index)
        if data is None:
            self._clear_current_variable()
            self.btn_generate.setEnabled(False)
            return
        var = str(data)
        if not var or self._dataset is None or var not in self._dataset:
            QtWidgets.QMessageBox.warning(self, "Load failed", "Selected variable is not available.")
            self._clear_current_variable()
            self.btn_generate.setEnabled(False)
            return
        self._current_var = var
        self._current_da = self._dataset[var]
        self._current_dims = list(getattr(self._current_da, "dims", ()))
        self._configure_axis_combos()
        self.btn_generate.setEnabled(True)
        self.lbl_status.setText("Configure axis selections and click Generate slices.")

    def _configure_axis_combos(self):
        dims = list(self._current_dims)
        self.cmb_row_dim.blockSignals(True)
        self.cmb_col_dim.blockSignals(True)
        self.cmb_row_dim.clear()
        self.cmb_col_dim.clear()
        for dim in dims:
            size = getattr(self._current_da, "sizes", {}).get(dim, "?") if self._current_da is not None else "?"
            label = f"{dim} ({size})"
            self.cmb_row_dim.addItem(label, dim)
            self.cmb_col_dim.addItem(label, dim)
        self.cmb_row_dim.blockSignals(False)
        self.cmb_col_dim.blockSignals(False)
        self.cmb_row_dim.setEnabled(bool(dims))
        self.cmb_col_dim.setEnabled(len(dims) >= 2)
        if dims:
            self.cmb_row_dim.setCurrentIndex(0)
        if len(dims) >= 2:
            self.cmb_col_dim.setCurrentIndex(1)
        self._update_extra_dim_controls()

    def _on_axes_changed(self, _index: int):
        self._update_extra_dim_controls()

    def _update_extra_dim_controls(self):
        row_dim = self.cmb_row_dim.currentData()
        col_dim = self.cmb_col_dim.currentData()
        dims = [dim for dim in self._current_dims if dim not in {row_dim, col_dim}]
        self._reset_extra_dim_controls(dims)

    def _generate_slices(self):
        if self._dataset is None or self._current_da is None or self._current_var is None:
            QtWidgets.QMessageBox.warning(self, "Generate slices", "Load a dataset and select a variable first.")
            return
        row_dim = self.cmb_row_dim.currentData()
        col_dim = self.cmb_col_dim.currentData()
        if not row_dim or not col_dim or row_dim == col_dim:
            QtWidgets.QMessageBox.warning(
                self,
                "Generate slices",
                "Please choose two different dimensions for rows and columns.",
            )
            return
        slice_dims = []
        for dim, control in self._dim_controls.items():
            indices = control.indices()
            if not indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Generate slices",
                    f"No indices selected for dimension {dim}.",
                )
                return
            slice_dims.append((dim, indices))

        data_vars: Dict[str, xr.DataArray] = {}

        def _prepare_slice(arr: xr.DataArray) -> xr.DataArray:
            arr = arr.transpose(row_dim, col_dim)
            arr = arr.copy(deep=True)
            try:
                extra_coords = [name for name in arr.coords if name not in arr.dims]
                if extra_coords:
                    arr = arr.drop_vars(extra_coords)
            except Exception:
                # If drop_vars fails (older xarray), fall back to resetting coords
                try:
                    arr = arr.reset_coords(drop=True)
                except Exception:
                    pass
            return arr

        if not slice_dims:
            arr = _prepare_slice(self._current_da)
            arr = arr.copy(deep=True)
            arr.name = self._current_var
            arr.attrs["_source_var"] = self._current_var
            arr.attrs["_slice_label"] = "Full selection"
            data_vars[self._current_var] = arr
        else:
            combos = itertools.product(*[indices for _, indices in slice_dims])
            for combo in combos:
                selectors = {dim: idx for (dim, _), idx in zip(slice_dims, combo)}
                arr = self._current_da.isel(**selectors)
                arr = _prepare_slice(arr)
                arr = arr.copy(deep=True)
                arr.name = self._current_var
                arr.attrs["_source_var"] = self._current_var
                label_parts = [f"{dim}={idx}" for dim, idx in selectors.items()]
                label_text = ", ".join(label_parts)
                if label_text:
                    arr.attrs["_slice_label"] = label_text
                arr.attrs["_slice_selectors"] = selectors
                suffix_parts = [f"{dim}{idx}" for dim, idx in selectors.items()]
                suffix = "_" + "_".join(suffix_parts) if suffix_parts else ""
                name = f"{self._current_var}{suffix}"
                data_vars[name] = arr
        if not data_vars:
            QtWidgets.QMessageBox.information(self, "Generate slices", "No slices were produced.")
            return

        try:
            slice_dataset = xr.Dataset(data_vars)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Generate slices", f"Failed to build dataset: {exc}")
            return

        default_label = f"{self._current_var}_slices"
        label, ok = QtWidgets.QInputDialog.getText(
            self,
            "Name sliced dataset",
            "Dataset name:",
            QtWidgets.QLineEdit.Normal,
            default_label,
        )
        if not ok or not label.strip():
            return
        label = label.strip()
        self.library.register_sliced_dataset(label, slice_dataset)
        self.lbl_status.setText(
            f"Registered {len(data_vars)} slice(s) as '{label}' in the Sliced Data tab."
        )
        log_action(
            f"Generated {len(data_vars)} slice(s) from '{self._current_var}' into dataset '{label}'"
        )

    # ---------- drag & drop ----------
    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent):
        if ev.mimeData().hasText():
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev: QtGui.QDropEvent):
        text = ev.mimeData().text()
        high_ref = HighDimVarRef.from_mime(text)
        if high_ref:
            try:
                dataset = high_ref.load_dataset()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
                ev.ignore()
                return
            label = high_ref.dataset_label()
            path = high_ref.path
            key = high_ref.memory_key
            self._set_dataset(dataset, label, path=path, memory_key=key)
            index = self.cmb_variable.findData(high_ref.var)
            if index >= 0:
                self.cmb_variable.setCurrentIndex(index)
            ev.acceptProposedAction()
            return

        mem_ds = MemoryDatasetRef.from_mime(text)
        if mem_ds:
            try:
                dataset = mem_ds.load()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
                ev.ignore()
                return
            self._set_dataset(dataset, mem_ds.display_name(), memory_key=mem_ds.key)
            ev.acceptProposedAction()
            return

        ds_ref = DataSetRef.from_mime(text)
        if ds_ref:
            try:
                dataset = ds_ref.load()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
                ev.ignore()
                return
            self._set_dataset(dataset, ds_ref.path.name, path=ds_ref.path)
            ev.acceptProposedAction()
            return

        QtWidgets.QMessageBox.information(
            self,
            "Unsupported item",
            "Only datasets or variables with more than two dimensions can be dropped here.",
        )
        ev.ignore()

    def _open_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open dataset",
            "",
            "NetCDF / Zarr (*.nc *.zarr);;All files (*)",
        )
        if not path:
            return
        try:
            dataset = open_dataset(Path(path))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._set_dataset(dataset, Path(path).name, path=Path(path))


class InteractivePreviewWidget(QtWidgets.QWidget):
    """Lightweight preview panel for the interactive Python console."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        self._instructions = QtWidgets.QLabel(
            "Use preview.show(array) to render 1D/2D data from the console, or "
            "preview.plot(x, y) for explicit curves."
        )
        self._instructions.setWordWrap(True)
        self._instructions.setStyleSheet("color: #666;")
        layout.addWidget(self._instructions)

        self._stack = QtWidgets.QStackedWidget()
        self._placeholder = QtWidgets.QLabel("No preview data")
        self._placeholder.setAlignment(QtCore.Qt.AlignCenter)
        self._placeholder.setStyleSheet("color: #777; border: 1px dashed #bbb; padding: 12px;")
        self._stack.addWidget(self._placeholder)

        axis_items = {
            "bottom": ScientificAxisItem("bottom"),
            "left": ScientificAxisItem("left"),
        }
        self._plot_widget = pg.PlotWidget(axisItems=axis_items)
        self._plot_widget.showGrid(x=True, y=True, alpha=0.15)
        self._plot_widget.setBackground("#121212")
        self._plot_widget.setMinimumHeight(160)
        self._stack.addWidget(self._plot_widget)

        self._image_canvas = pg.GraphicsLayoutWidget()
        self._image_view = self._image_canvas.addViewBox()
        self._image_view.setAspectLocked(True)
        self._image_item = pg.ImageItem()
        self._image_view.addItem(self._image_item)
        self._stack.addWidget(self._image_canvas)

        layout.addWidget(self._stack, 1)

        self._status = QtWidgets.QLabel(" ")
        self._status.setWordWrap(True)
        self._status.setStyleSheet("color: #555;")
        layout.addWidget(self._status)

        self._stack.setCurrentWidget(self._placeholder)

    def clear(self):
        self._plot_widget.clear()
        self._image_item.hide()
        self._stack.setCurrentWidget(self._placeholder)
        self._status.setText(" ")

    def plot(
        self,
        x: Iterable[float],
        y: Iterable[float],
        *,
        xlabel: str = "x",
        ylabel: str = "y",
        title: Optional[str] = None,
    ):
        self._plot_widget.clear()
        x_vals = np.asarray(list(x), dtype=float)
        y_vals = np.asarray(list(y), dtype=float)
        self._plot_widget.plot(x_vals, y_vals, pen=pg.mkPen("#1d72b8", width=2))
        self._plot_widget.setLabel("bottom", xlabel)
        self._plot_widget.setLabel("left", ylabel)
        self._plot_widget.setTitle(title or "")
        self._stack.setCurrentWidget(self._plot_widget)
        self._status.setText(f"Displayed curve with {len(x_vals)} samples.")

    def show(self, data: object, *, title: Optional[str] = None):
        if isinstance(data, xr.Dataset):
            if data.data_vars:
                first = next(iter(data.data_vars))
                return self.show(data[first], title=title or first)
            self.clear()
            self._status.setText("Dataset contains no data variables.")
            return

        if isinstance(data, xr.DataArray):
            array = data.values
            dims = list(data.dims)
            coords = [np.asarray(data.coords[d].values) if d in data.coords else None for d in dims]
            name = data.name or title
        else:
            array = np.asarray(data)
            dims = [f"dim_{i}" for i in range(array.ndim)]
            coords = [None for _ in dims]
            name = title

        if array.ndim <= 1:
            x_vals = coords[0] if coords else None
            if x_vals is None:
                size = int(array.size) if hasattr(array, "size") else len(np.asarray(array).ravel())
                x_vals = np.arange(size or 1)
            self.plot(x_vals, np.asarray(array).ravel(), xlabel=dims[0] if dims else "index", title=name)
            return

        data2d = np.asarray(array)
        extra_note = ""
        if data2d.ndim > 2:
            extra_note = " Showing the first slice of higher-dimensional data."
            while data2d.ndim > 2:
                data2d = np.take(data2d, indices=0, axis=-1)

        self._image_item.show()
        self._image_item.setImage(data2d)
        self._image_view.autoRange()
        self._stack.setCurrentWidget(self._image_canvas)
        label = name or "Preview"
        self._status.setText(
            f"Previewing array with shape {tuple(array.shape)}. {extra_note}".strip()
        )
        self._plot_widget.setTitle(label)




class InteractiveConsoleWidget(QtWidgets.QWidget):
    """Embeddable interactive Python console with preview support."""

    datasetRegistered = QtCore.Signal(str)
    dataarrayRegistered = QtCore.Signal(str)
    messageEmitted = QtCore.Signal(str)

    def __init__(self, library: DatasetsPane, parent=None):
        super().__init__(parent)
        self.library = library

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter)

        console_panel = QtWidgets.QWidget()
        console_layout = QtWidgets.QVBoxLayout(console_panel)
        console_layout.setContentsMargins(8, 8, 8, 8)
        console_layout.setSpacing(6)

        instructions = QtWidgets.QLabel(
            "Execute Python code below. Use RegisterDataset(ds, label=...), "
            "RegisterDataArray(da, label=...), or ImportModule('module', alias='m') "
            "to work with data and libraries from this Python environment."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #606060; font-size: 12px;")
        console_layout.addWidget(instructions)

        console_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        console_splitter.setChildrenCollapsible(False)
        console_layout.addWidget(console_splitter, 1)

        output_container = QtWidgets.QWidget()
        output_layout = QtWidgets.QVBoxLayout(output_container)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(4)

        output_label = QtWidgets.QLabel("Console output")
        output_label.setStyleSheet("color: #888; font-weight: 600;")
        output_layout.addWidget(output_label)

        self.output_view = QtWidgets.QTextEdit()
        self.output_view.setReadOnly(True)
        self.output_view.setLineWrapMode(QtWidgets.QTextEdit.NoWrap)
        output_layout.addWidget(self.output_view)

        console_splitter.addWidget(output_container)

        history_container = QtWidgets.QWidget()
        history_layout = QtWidgets.QVBoxLayout(history_container)
        history_layout.setContentsMargins(0, 0, 0, 0)
        history_layout.setSpacing(4)

        history_label = QtWidgets.QLabel("Command history (double-click to reuse)")
        history_label.setStyleSheet("color: #888; font-weight: 600;")
        history_layout.addWidget(history_label)

        self.history_view = QtWidgets.QListWidget()
        self.history_view.setAlternatingRowColors(True)
        self.history_view.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.history_view.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.history_view.customContextMenuRequested.connect(self._on_history_menu)
        self.history_view.itemActivated.connect(self._apply_history_item)
        history_layout.addWidget(self.history_view)

        console_splitter.addWidget(history_container)
        console_splitter.setStretchFactor(0, 3)
        console_splitter.setStretchFactor(1, 1)

        input_container = QtWidgets.QWidget()
        input_layout = QtWidgets.QVBoxLayout(input_container)
        input_layout.setContentsMargins(0, 0, 0, 0)
        input_layout.setSpacing(4)

        input_label = QtWidgets.QLabel("Input (Ctrl+Enter to run, Ctrl+↑/↓ to browse history)")
        input_label.setStyleSheet("color: #888; font-weight: 600;")
        input_layout.addWidget(input_label)

        self.input_edit = QtWidgets.QPlainTextEdit()
        self.input_edit.setPlaceholderText("Type Python code here and click Run (Ctrl+Enter).")
        input_layout.addWidget(self.input_edit)

        console_layout.addWidget(input_container, 0)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_run = QtWidgets.QPushButton("Run")
        self.btn_run.clicked.connect(self._execute_code)
        self.input_edit.installEventFilter(self)
        button_row.addWidget(self.btn_run)
        self.btn_import = QtWidgets.QPushButton("Import module…")
        self.btn_import.clicked.connect(self._open_import_dialog)
        button_row.addWidget(self.btn_import)

        self.btn_clear_in = QtWidgets.QPushButton("Clear input")
        self.btn_clear_in.clicked.connect(self.input_edit.clear)
        button_row.addWidget(self.btn_clear_in)
        self.btn_clear_out = QtWidgets.QPushButton("Clear output")
        self.btn_clear_out.clicked.connect(self.output_view.clear)
        button_row.addWidget(self.btn_clear_out)
        button_row.addStretch(1)
        console_layout.addLayout(button_row)

        splitter.addWidget(console_panel)

        self.preview = InteractivePreviewWidget()
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        base_font = QtGui.QFontDatabase.systemFont(QtGui.QFontDatabase.FixedFont)
        base_font.setPointSize(11)
        self.output_view.setFont(base_font)
        self.input_edit.setFont(base_font)
        self.history_view.setFont(base_font)
        self.output_view.setStyleSheet(
            "QTextEdit { background-color: #1e1e1e; color: #dcdcdc; border: 1px solid #3c3c3c;"
            " border-radius: 4px; padding: 4px; }"
        )
        self.input_edit.setStyleSheet(
            "QPlainTextEdit { background-color: #252526; color: #f3f3f3; border: 1px solid #3c3c3c;"
            " border-radius: 4px; padding: 4px; }"
        )
        self.history_view.setStyleSheet(
            "QListWidget { background-color: #202124; color: #f1f1f1; border: 1px solid #3c3c3c;"
            " border-radius: 4px; padding: 2px; }"
            "QListWidget::item { padding: 4px; }"
        )

        self._globals = {"__builtins__": __builtins__, "__name__": "__console__", "__package__": None}
        self._locals = {
            "np": np,
            "xr": xr,
            "RegisterDataset": self._cmd_register_dataset,
            "RegisterDataArray": self._cmd_register_dataarray,
            "preview": self.preview,
            "ImportModule": self._cmd_import_module,
        }
        self._module_name_cache: Optional[List[str]] = None
        self._history: List[str] = []
        self._history_index: int = -1

    def eventFilter(self, obj, event):
        if obj is self.input_edit and event.type() == QtCore.QEvent.KeyPress:
            key = event.key()
            modifiers = event.modifiers()
            if key in (QtCore.Qt.Key_Enter, QtCore.Qt.Key_Return) and modifiers & QtCore.Qt.ControlModifier:
                self._execute_code()
                return True
            if key == QtCore.Qt.Key_Up and modifiers & QtCore.Qt.ControlModifier:
                self._history_step(-1)
                return True
            if key == QtCore.Qt.Key_Down and modifiers & QtCore.Qt.ControlModifier:
                self._history_step(1)
                return True
        return super().eventFilter(obj, event)

    def _append_output(self, text: str, role: str = "output"):
        if not text:
            return
        cursor = self.output_view.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        fmt = QtGui.QTextCharFormat()
        if role == "input":
            fmt.setForeground(QtGui.QColor("#bbbbbb"))
        elif role == "error":
            fmt.setForeground(QtGui.QColor("#d1495b"))
        else:
            fmt.setForeground(QtGui.QColor("#2a9d8f"))
        cursor.insertText(text, fmt)
        self.output_view.setTextCursor(cursor)
        self.output_view.ensureCursorVisible()

    def _available_modules(self) -> List[str]:
        if self._module_name_cache is not None:
            return list(self._module_name_cache)
        names: Set[str] = set(sys.builtin_module_names)
        names.update(sys.modules.keys())
        try:
            for _, mod_name, _ in pkgutil.iter_modules():
                names.add(mod_name)
        except Exception:
            pass
        self._module_name_cache = sorted(n for n in names if n)
        return list(self._module_name_cache)

    def _open_import_dialog(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle("Import module")
        layout = QtWidgets.QVBoxLayout(dialog)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        label = QtWidgets.QLabel(
            "Enter the module to import. Use syntax 'module.submodule as alias' "
            "to bind the module to a custom name."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

        edit = QtWidgets.QLineEdit()
        edit.setPlaceholderText("e.g. pandas as pd")
        completer = QtWidgets.QCompleter(self._available_modules())
        completer.setCaseSensitivity(QtCore.Qt.CaseInsensitive)
        completer.setFilterMode(QtCore.Qt.MatchContains)
        edit.setCompleter(completer)
        layout.addWidget(edit)

        button_box = QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        layout.addWidget(button_box)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        text = edit.text().strip()
        if not text:
            return

        module_name = text
        alias: Optional[str] = None
        if " as " in text:
            parts = text.split(" as ", 1)
            module_name = parts[0].strip()
            alias = parts[1].strip()

        try:
            self._cmd_import_module(module_name, alias)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(exc))
            return

        bind_name = alias or module_name.split(".")[-1]
        self._append_output(f"Imported module '{module_name}' as '{bind_name}'.\n", role="output")
        self.messageEmitted.emit(f"Imported module '{module_name}'.")

    def _execute_code(self):
        code_text = self.input_edit.toPlainText()
        if not code_text.strip():
            return

        lines = code_text.rstrip().splitlines()
        for idx, line in enumerate(lines):
            prefix = ">>> " if idx == 0 else "... "
            self._append_output(prefix + line + "\n", role="input")

        self._push_history(code_text)
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        try:
            with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
                exec(code_text, self._globals, self._locals)
        except Exception:
            stderr_buffer.write(traceback.format_exc())

        out_text = stdout_buffer.getvalue()
        err_text = stderr_buffer.getvalue()
        if out_text:
            self._append_output(out_text, role="output")
        if err_text:
            self._append_output(err_text, role="error")
            self.messageEmitted.emit("Execution finished with errors.")
        elif not out_text.strip():
            self.messageEmitted.emit("Execution finished.")

    def _push_history(self, code_text: str):
        text = code_text.rstrip()
        if not text:
            return
        if self._history and self._history[-1] == text:
            self._history_index = len(self._history)
            return
        self._history.append(text)
        self._history_index = len(self._history)
        item = QtWidgets.QListWidgetItem(text.splitlines()[0][:120])
        item.setToolTip(text)
        item.setData(QtCore.Qt.UserRole, text)
        self.history_view.addItem(item)
        self.history_view.scrollToBottom()

    def _history_step(self, delta: int):
        if not self._history:
            return
        self._history_index = max(0, min(len(self._history), self._history_index + delta))
        if self._history_index == len(self._history):
            self.input_edit.clear()
            return
        self.input_edit.setPlainText(self._history[self._history_index])
        cursor = self.input_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.input_edit.setTextCursor(cursor)

    def _apply_history_item(self, item: QtWidgets.QListWidgetItem):
        if not item:
            return
        text = item.data(QtCore.Qt.UserRole) or ""
        self.input_edit.setPlainText(str(text))
        self.input_edit.setFocus()
        cursor = self.input_edit.textCursor()
        cursor.movePosition(QtGui.QTextCursor.End)
        self.input_edit.setTextCursor(cursor)

    def _on_history_menu(self, pos: QtCore.QPoint):
        item = self.history_view.itemAt(pos)
        menu = QtWidgets.QMenu(self.history_view)
        if item:
            act_send = menu.addAction("Send to input")
            act_copy = menu.addAction("Copy command text")
        act_clear = menu.addAction("Clear history")
        chosen = menu.exec_(self.history_view.mapToGlobal(pos))
        if chosen is None:
            return
        if chosen == act_send and item:
            self._apply_history_item(item)
        elif chosen == act_copy and item:
            text = item.data(QtCore.Qt.UserRole) or ""
            QtWidgets.QApplication.clipboard().setText(str(text))
        elif chosen == act_clear:
            self._history.clear()
            self._history_index = -1
            self.history_view.clear()

    def _cmd_register_dataset(self, dataset: xr.Dataset, label: Optional[str] = None) -> str:
        if not isinstance(dataset, xr.Dataset):
            raise TypeError("RegisterDataset expects an xarray.Dataset")
        base_label = label or str(dataset.attrs.get("title") or dataset.attrs.get("name") or "interactive_dataset")
        name = self.library.register_interactive_dataset(base_label, dataset)
        print(f"Registered dataset '{name}' in the Interactive Data tab.")
        self.datasetRegistered.emit(name)
        self.messageEmitted.emit(f"Registered dataset '{name}'.")
        return name

    def _cmd_register_dataarray(self, dataarray: xr.DataArray, label: Optional[str] = None) -> str:
        if not isinstance(dataarray, xr.DataArray):
            raise TypeError("RegisterDataArray expects an xarray.DataArray")
        base_label = label or str(dataarray.name or "interactive_array")
        name = self.library.register_interactive_dataarray(base_label, dataarray)
        print(f"Registered data array '{name}' in the Interactive Data tab.")
        self.dataarrayRegistered.emit(name)
        self.messageEmitted.emit(f"Registered data array '{name}'.")
        return name

    def _cmd_import_module(self, module_name: str, alias: Optional[str] = None) -> ModuleType:
        module_name = (module_name or "").strip()
        if not module_name:
            raise ValueError("Module name cannot be empty.")
        try:
            module = importlib.import_module(module_name)
        except Exception as exc:
            raise ImportError(f"Unable to import '{module_name}': {exc}")

        bind_name = (alias or module_name.split(".")[-1]).strip()
        if not bind_name:
            raise ValueError("Alias cannot be empty.")
        if not bind_name.isidentifier():
            raise ValueError("Alias must be a valid Python identifier.")

        self._globals[bind_name] = module
        self._locals[bind_name] = module
        self._module_name_cache = None
        return module


class InteractiveProcessingTab(QtWidgets.QWidget):
    def __init__(
        self,
        library: DatasetsPane,
        bridge: Optional[InteractiveBridgeServer] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.library = library
        self.bridge_server = bridge
        self._active_mode = "jupyter"

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        hint_text = (
            "Choose an interactive environment. When using the Python console, call "
            "<code>RegisterDataset(...)</code> or <code>RegisterDataArray(...)</code> to push results into the "
            "Interactive Data tab."
        )
        if self.bridge_server and self.bridge_server.register_url():
            hint_text += (
                "<br><br>From notebooks in this session you can also run:<br>"
                "<code>from xrdataviewer_bridge import register_dataset, register_dataarray" "<br>"
                "register_dataset(ds, label=\"My result\")</code>"
            )
        hint = QtWidgets.QLabel(hint_text)
        hint.setTextFormat(QtCore.Qt.RichText)
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #666;")
        layout.addWidget(hint)

        mode_row = QtWidgets.QHBoxLayout()
        lbl_mode = QtWidgets.QLabel("Environment:")
        mode_row.addWidget(lbl_mode)
        self.cmb_mode = QtWidgets.QComboBox()
        self._web_engine_available = QtWebEngineWidgets is not None
        if self._web_engine_available:
            self.cmb_mode.addItem("JupyterLab browser", "jupyter")
        else:
            self.cmb_mode.addItem("JupyterLab placeholder", "jupyter")
        self.cmb_mode.addItem("Python console", "python")
        if self._web_engine_available:
            self.cmb_mode.addItem("VS Code (web)", "vscode")
        else:
            self.cmb_mode.addItem("VS Code placeholder", "vscode")
        mode_row.addWidget(self.cmb_mode)
        mode_row.addStretch(1)
        layout.addLayout(mode_row)

        self._jupyter_nav_widget = QtWidgets.QWidget()
        nav_layout = QtWidgets.QHBoxLayout(self._jupyter_nav_widget)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(4)
        if self._web_engine_available:
            self.btn_jupyter_back = QtWidgets.QToolButton()
            self.btn_jupyter_back.setText("◀")
            self.btn_jupyter_back.setToolTip("Go back")
            self.btn_jupyter_back.setAutoRaise(True)
            self.btn_jupyter_back.setEnabled(False)
            self.btn_jupyter_back.clicked.connect(self._on_jupyter_back)
            nav_layout.addWidget(self.btn_jupyter_back)

            self.btn_jupyter_forward = QtWidgets.QToolButton()
            self.btn_jupyter_forward.setText("▶")
            self.btn_jupyter_forward.setToolTip("Go forward")
            self.btn_jupyter_forward.setAutoRaise(True)
            self.btn_jupyter_forward.setEnabled(False)
            self.btn_jupyter_forward.clicked.connect(self._on_jupyter_forward)
            nav_layout.addWidget(self.btn_jupyter_forward)

            self.btn_jupyter_reload = QtWidgets.QToolButton()
            self.btn_jupyter_reload.setText("⟳")
            self.btn_jupyter_reload.setToolTip("Reload current page")
            self.btn_jupyter_reload.setAutoRaise(True)
            self.btn_jupyter_reload.setEnabled(False)
            self.btn_jupyter_reload.clicked.connect(self._on_jupyter_reload)
            nav_layout.addWidget(self.btn_jupyter_reload)

            self.btn_jupyter_external = QtWidgets.QToolButton()
            self.btn_jupyter_external.setText("↗")
            self.btn_jupyter_external.setToolTip("Open in external browser")
            self.btn_jupyter_external.setAutoRaise(True)
            self.btn_jupyter_external.setEnabled(False)
            self.btn_jupyter_external.clicked.connect(self._open_jupyter_external)
            nav_layout.addWidget(self.btn_jupyter_external)

            nav_layout.addStretch(1)
        else:
            nav_layout.addWidget(QtWidgets.QLabel(" "))
            nav_layout.addStretch(1)
        self._jupyter_nav_widget.setVisible(False)
        layout.addWidget(self._jupyter_nav_widget)

        self.stack = QtWidgets.QStackedWidget()
        self._jupyter_view: Optional['QtWebEngineWidgets.QWebEngineView'] = None
        if self._web_engine_available:
            self._jupyter_view = QtWebEngineWidgets.QWebEngineView()
            self._jupyter_page = QuietWebEnginePage(self._jupyter_view)
            self._jupyter_page.consoleMessage.connect(self._on_jupyter_console_message)
            self._jupyter_view.setPage(self._jupyter_page)
            self._jupyter_view.setHtml(
                """
                <html><body style='font-family: sans-serif; color: #333;'>
                <h2>Starting JupyterLab…</h2>
                <p>The embedded server will launch automatically. This placeholder will be replaced once it is ready.</p>
                </body></html>
                """
            )
            self._jupyter_view.urlChanged.connect(self._update_jupyter_nav)
            self._jupyter_view.loadFinished.connect(self._update_jupyter_nav)
            self.stack.addWidget(self._jupyter_view)
        else:
            placeholder = QtWidgets.QTextBrowser()
            placeholder.setHtml(
                "<h2>JupyterLab integration unavailable</h2>"
                "<p>QtWebEngineWidgets is not installed. Launch JupyterLab separately and use the Register buttons below.</p>"
            )
            placeholder.setReadOnly(True)
            placeholder.setOpenExternalLinks(True)
            self.stack.addWidget(placeholder)
        self._jupyter_manager = EmbeddedJupyterManager(self) if self._web_engine_available else None
        if self._jupyter_manager:
            self._jupyter_manager.urlReady.connect(self._on_jupyter_url_ready)
            self._jupyter_manager.failed.connect(self._on_jupyter_failed)
            self._jupyter_manager.message.connect(self._on_jupyter_message)
        self._jupyter_url: Optional[str] = None
        self._jupyter_js_seen: Set[str] = set()
        self._jupyter_js_warned = False

        self.console_widget = InteractiveConsoleWidget(self.library)
        self.console_widget.datasetRegistered.connect(self._on_console_registered)
        self.console_widget.dataarrayRegistered.connect(self._on_console_registered)
        self.console_widget.messageEmitted.connect(self._set_status)
        self.stack.addWidget(self.console_widget)

        if self.bridge_server:
            self.bridge_server.datasetRegistered.connect(self._on_bridge_registered)

        self.vscode_widget = VsCodeWebWidget()
        self.vscode_widget.statusMessage.connect(self._on_vscode_status)
        self.stack.addWidget(self.vscode_widget)

        layout.addWidget(self.stack, 1)

        controls = QtWidgets.QHBoxLayout()
        self.btn_register_dataset = QtWidgets.QPushButton("Register dataset from session…")
        self.btn_register_dataset.clicked.connect(self._register_dataset)
        controls.addWidget(self.btn_register_dataset)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.lbl_status = QtWidgets.QLabel(" ")
        self.lbl_status.setStyleSheet("color: #555;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

        if not self._web_engine_available:
            self.cmb_mode.setCurrentIndex(1)
            self.stack.setCurrentIndex(1)
        else:
            self.stack.setCurrentIndex(0)

        self.cmb_mode.currentIndexChanged.connect(self._on_mode_changed)
        self._on_mode_changed(self.cmb_mode.currentIndex())
        self._shutdown_called = False
        self.destroyed.connect(lambda *_: self.shutdown())

    def _on_mode_changed(self, index: int):
        mode = self.cmb_mode.itemData(index)
        self._active_mode = str(mode)
        show_nav = mode == "jupyter" and self._web_engine_available
        self._jupyter_nav_widget.setVisible(bool(show_nav))
        if self._web_engine_available:
            self._update_jupyter_nav()
        if mode == "python":
            self.stack.setCurrentIndex(1)
            self._set_status(
                "Python console ready. Use RegisterDataset(...) or RegisterDataArray(...) to export results."
            )
            self.console_widget.input_edit.setFocus()
        elif mode == "vscode":
            self.stack.setCurrentIndex(2)
            if QtWebEngineWidgets is None:
                self._set_status(
                    "VS Code web view unavailable. Launch VS Code separately or install QtWebEngineWidgets."
                )
            else:
                self._set_status("VS Code controls ready.")
        else:
            self.stack.setCurrentIndex(0)
            if not self._web_engine_available:
                self._set_status(
                    "QtWebEngineWidgets is not installed. Launch JupyterLab externally and register datasets manually."
                )
            elif self._jupyter_manager:
                if not self._jupyter_manager.is_running():
                    self._set_status("Starting embedded JupyterLab server…")
                    self._jupyter_manager.start()
                elif self._jupyter_manager.url():
                    self._set_status("Embedded JupyterLab ready.")
                else:
                    self._set_status("Preparing embedded JupyterLab…")

    def _set_status(self, message: str):
        if not message:
            message = " "
        self.lbl_status.setText(message)

    def _on_console_registered(self, label: str):
        self._set_status(f"Registered '{label}' in the Interactive Data tab.")

    def _on_bridge_registered(self, label: str):
        self._set_status(f"Registered '{label}' from external session.")

    def _on_jupyter_url_ready(self, url: str):
        self._jupyter_url = url
        if self._jupyter_view is not None:
            self._jupyter_view.setUrl(QtCore.QUrl(url))
            self._update_jupyter_nav()
        if self._active_mode == "jupyter":
            self._set_status("Embedded JupyterLab ready.")
        else:
            self._set_status("Embedded JupyterLab started in background.")

    def _on_jupyter_failed(self, message: str):
        if self._jupyter_view is not None:
            escaped = html.escape(message or "")
            self._jupyter_view.setHtml(
                f"""
                <html><body style='font-family: sans-serif; color: #a33;'>
                <h2>JupyterLab launch failed</h2>
                <p>{escaped or 'Unable to start the embedded server.'}</p>
                </body></html>
                """
            )
        if self._active_mode == "jupyter":
            self._set_status(message)
        else:
            self._set_status("Embedded JupyterLab failed to start.")

    def _on_jupyter_message(self, text: str):
        if self._active_mode == "jupyter" and text:
            self._set_status(text)

    def _update_jupyter_nav(self, *_args):
        if not self._web_engine_available or self._jupyter_view is None:
            return
        history = self._jupyter_view.history()
        current_url = self._jupyter_view.url() if self._jupyter_view else QtCore.QUrl()
        if hasattr(self, "btn_jupyter_back"):
            self.btn_jupyter_back.setEnabled(history.canGoBack())
        if hasattr(self, "btn_jupyter_forward"):
            self.btn_jupyter_forward.setEnabled(history.canGoForward())
        has_url = current_url is not None and not current_url.isEmpty()
        if hasattr(self, "btn_jupyter_reload"):
            self.btn_jupyter_reload.setEnabled(has_url)
        if hasattr(self, "btn_jupyter_external"):
            target = self._jupyter_url or (current_url.toString() if has_url else "")
            self.btn_jupyter_external.setEnabled(bool(target))

    def _on_jupyter_back(self):
        if self._web_engine_available and self._jupyter_view and self._jupyter_view.history().canGoBack():
            self._jupyter_view.back()

    def _on_jupyter_forward(self):
        if self._web_engine_available and self._jupyter_view and self._jupyter_view.history().canGoForward():
            self._jupyter_view.forward()

    def _on_jupyter_reload(self):
        if self._web_engine_available and self._jupyter_view:
            self._jupyter_view.reload()

    def _open_jupyter_external(self):
        if not self._web_engine_available:
            return
        target = self._jupyter_url
        if (not target or not target.strip()) and self._jupyter_view is not None:
            url = self._jupyter_view.url()
            if url and not url.isEmpty():
                target = url.toString()
        if target:
            QtGui.QDesktopServices.openUrl(QtCore.QUrl(target))

    def _on_jupyter_console_message(self, level: int, message: str, line: int, source_id: str):
        if not message:
            return
        key = f"{source_id}:{line}:{level}:{message}"
        if key in self._jupyter_js_seen:
            return
        self._jupyter_js_seen.add(key)
        cleaned = message.strip()
        if not cleaned:
            return
        location = source_id or "unknown source"
        if line:
            location = f"{location}:{line}"
        error_level = int(QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel)
        warning_level = int(QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel.WarningMessageLevel)
        info_level = int(QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel.InfoMessageLevel)
        if level == error_level:
            log_action(f"JupyterLab JS error ({location}): {cleaned}")
            if not self._jupyter_js_warned:
                self._set_status(
                    "Embedded JupyterLab reported browser compatibility warnings. Open in an external browser if features look incorrect."
                )
                self._jupyter_js_warned = True
        elif level == warning_level:
            log_action(f"JupyterLab JS warning ({location}): {cleaned}")
        elif level == info_level:
            log_action(f"JupyterLab JS info ({location}): {cleaned}")
        else:
            log_action(f"JupyterLab JS message ({location}): {cleaned}")

    def _on_vscode_status(self, message: str):
        if self._active_mode == "vscode" and message:
            self._set_status(message)

    def shutdown(self):
        if self._shutdown_called:
            return
        self._shutdown_called = True
        if self._jupyter_manager:
            self._jupyter_manager.stop()
        if self._web_engine_available and self._jupyter_view is not None:
            page = self._jupyter_view.page()
            self._jupyter_view.setPage(None)
            if page is not None:
                try:
                    page.deleteLater()
                except Exception:
                    pass
            try:
                self._jupyter_view.deleteLater()
            except Exception:
                pass
            self._jupyter_view = None
            self._jupyter_page = None
        self.vscode_widget.shutdown()

    def _register_dataset(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Select dataset",
            "",
            "NetCDF / Zarr (*.nc *.zarr);;All files (*)",
        )
        if not path:
            return
        p = Path(path)
        try:
            dataset = open_dataset(p)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        default = p.stem or "interactive_dataset"
        name, ok = QtWidgets.QInputDialog.getText(
            self,
            "Dataset name",
            "Name to display in the Interactive Data tab:",
            QtWidgets.QLineEdit.Normal,
            default,
        )
        if not ok or not name.strip():
            try:
                dataset.close()
            except Exception:
                pass
            return
        label = name.strip()
        final_label = self.library.register_interactive_dataset(label, dataset)
        self._set_status(f"Registered '{final_label}' in the Interactive Data tab.")


# ---------------------------------------------------------------------------
# ViewerFrame: one tile with the image + optional histogram on the right
# ---------------------------------------------------------------------------
class ViewerFrame(QtWidgets.QFrame):
    request_close = QtCore.Signal(object)

    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.setObjectName("viewerFrame")
        self.setProperty("selected", False)
        self.setStyleSheet(
            "QFrame#viewerFrame { border: 1px solid #888; border-radius: 2px; }"
            "QFrame#viewerFrame[selected=\"true\"] { border: 2px solid #1d72b8; "
            "background-color: rgba(29, 114, 184, 40); }"
        )

        self._base_title = title
        self._raw_data: Optional[np.ndarray] = None
        self._processed_data: Optional[np.ndarray] = None
        self._coords: Dict[str, np.ndarray] = {}
        self._display_mode: str = "image"
        self._current_processing: str = "none"
        self._processing_params: Dict[str, object] = {}
        self._selected: bool = False
        self._dataset_path: Optional[Path] = None
        self._dataset: Optional[xr.Dataset] = None
        self._available_variables: List[str] = []
        self._variable_hints: Dict[str, str] = {}
        self._current_variable: Optional[str] = None
        self.preferences: Optional[PreferencesManager] = None

        lay = QtWidgets.QVBoxLayout(self); lay.setContentsMargins(2,2,2,2); lay.setSpacing(2)
        # Header
        hdr = QtWidgets.QFrame(); hl = QtWidgets.QHBoxLayout(hdr); hl.setContentsMargins(6,3,6,3)
        self.lbl = QtWidgets.QLabel(title); hl.addWidget(self.lbl, 1)
        btn_close = QtWidgets.QToolButton(); btn_close.setText("×")
        btn_close.clicked.connect(lambda: self.request_close.emit(self))
        hl.addWidget(btn_close, 0)
        lay.addWidget(hdr, 0)

        # Variable selector
        selector = QtWidgets.QFrame()
        selector_layout = QtWidgets.QHBoxLayout(selector)
        selector_layout.setContentsMargins(6, 0, 6, 0)
        selector_layout.setSpacing(6)
        self._var_label = QtWidgets.QLabel("Variable:")
        self._var_label.setEnabled(False)
        selector_layout.addWidget(self._var_label, 0)
        self.cmb_variable = QtWidgets.QComboBox()
        self.cmb_variable.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cmb_variable.addItem("Select variable…", None)
        self.cmb_variable.setEnabled(False)
        self.cmb_variable.currentIndexChanged.connect(self._on_variable_combo_changed)
        selector_layout.addWidget(self.cmb_variable, 1)
        lay.addWidget(selector, 0)
        self._variable_bar = selector

        # Center: image on left, histogram on right (toggle visibility)
        self.center_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.center_split.setChildrenCollapsible(False)
        self.center_split.setHandleWidth(6)
        lay.addWidget(self.center_split, 1)

        self.viewer = CentralPlotWidget(self)
        self.center_split.addWidget(self.viewer)

        self._hist_master_enabled = True
        self._hist_local_enabled = True
        self._hist_last_split_sizes: Optional[List[int]] = None
        self._hist_widget = self.viewer.histogram_widget()
        if self._hist_widget is not None:
            if self.center_split.indexOf(self._hist_widget) == -1:
                self.center_split.addWidget(self._hist_widget)
            try:
                self.center_split.setStretchFactor(0, 1)
                self.center_split.setStretchFactor(1, 0)
            except Exception:
                pass
        try:
            self.center_split.splitterMoved.connect(self._record_histogram_sizes)
        except Exception:
            pass
        try:
            self.viewer.configure_histogram_toggle(
                getter=self.is_histogram_local_enabled,
                setter=self.set_histogram_local_enabled,
            )
        except Exception:
            pass
        self._update_histogram_visibility()

    def _dataset_display_name(self) -> str:
        if self._dataset_path is not None:
            return self._dataset_path.name
        return self._base_title or "Dataset"

    def _set_header_text(self, variable: Optional[str], *, missing: bool = False, custom: Optional[str] = None):
        if custom is not None:
            self.lbl.setText(custom)
            return
        base = self._dataset_display_name()
        if variable:
            suffix = f"{variable} (missing)" if missing else variable
            self.lbl.setText(f"{base} — {suffix}")
        else:
            self.lbl.setText(base)

    def _update_variable_combo(self):
        block = self.cmb_variable.blockSignals(True)
        current = self._current_variable if self._current_variable in self._available_variables else None
        self.cmb_variable.clear()
        self.cmb_variable.addItem("Select variable…", None)
        for var in self._available_variables:
            label = f"{var}{self._variable_hints.get(var, '')}"
            self.cmb_variable.addItem(label, var)
        self.cmb_variable.blockSignals(block)
        has_vars = bool(self._available_variables)
        self.cmb_variable.setEnabled(has_vars)
        self._var_label.setEnabled(has_vars)
        self._select_combo_value(current)

    def _select_combo_value(self, value: Optional[str]):
        block = self.cmb_variable.blockSignals(True)
        if value:
            idx = self.cmb_variable.findData(value)
            if idx >= 0:
                self.cmb_variable.setCurrentIndex(idx)
            else:
                self.cmb_variable.setCurrentIndex(0)
        else:
            self.cmb_variable.setCurrentIndex(0)
        self.cmb_variable.blockSignals(block)

    def _dispose_dataset(self):
        if self._dataset is not None:
            try:
                close = getattr(self._dataset, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
        self._dataset = None

    def dispose(self):
        self._dispose_dataset()

    def set_dataset(self, dataset: xr.Dataset, path: Path, *, select: Optional[str] = None):
        self._dispose_dataset()
        self._dataset = dataset
        self._dataset_path = Path(path) if path is not None else None
        self._available_variables = []
        self._variable_hints = {}
        if self._dataset_path is not None:
            self.lbl.setToolTip(str(self._dataset_path))
        else:
            self.lbl.setToolTip("")

        try:
            data_vars = getattr(dataset, "data_vars", {})
        except Exception:
            data_vars = {}
        for var in data_vars:
            try:
                da = dataset[var]
            except Exception:
                continue
            if getattr(da, "ndim", 0) != 2:
                continue
            self._available_variables.append(var)
            dims = []
            try:
                dims = [f"{dim}[{getattr(da, 'sizes', {}).get(dim, '?')}]" for dim in da.dims[:2]]
            except Exception:
                dims = []
            if dims:
                self._variable_hints[var] = " (" + " × ".join(dims) + ")"

        self._available_variables.sort()
        self._current_variable = None
        self._update_variable_combo()
        self._clear_display()
        if not self._available_variables:
            self._set_header_text(None)
            self._set_header_text(None, custom=f"{self._dataset_display_name()} — No 2D variables")
            return

        self._set_header_text(None)
        if select and select in self._available_variables:
            self.plot_variable(select)
        else:
            self._select_combo_value(None)

    def available_variables(self) -> List[str]:
        return list(self._available_variables)

    def current_variable(self) -> Optional[str]:
        return self._current_variable

    def _on_variable_combo_changed(self, index: int):
        data = self.cmb_variable.itemData(index)
        if data is None:
            if self._current_variable is not None:
                self._current_variable = None
                self._clear_display()
            else:
                self._clear_display()
            return
        self.plot_variable(str(data))

    def plot_variable(self, var_name: str) -> bool:
        name = str(var_name or "").strip()
        if not name:
            self._current_variable = None
            self._clear_display()
            self._select_combo_value(None)
            return False
        success = self._load_variable(name)
        if success:
            self._select_combo_value(name)
            return True
        self._show_missing_variable(name)
        self._select_combo_value(None)
        return False

    def _show_missing_variable(self, var_name: str):
        self._current_variable = None
        self._clear_display(preserve_header=True)
        self._set_header_text(var_name, missing=True)

    def _load_variable(self, var_name: str) -> bool:
        if self._dataset is None:
            return False
        try:
            da = self._dataset[var_name]
        except Exception:
            return False
        if getattr(da, "ndim", 0) != 2:
            return False
        coords = guess_phys_coords(da)
        self.set_data(da, coords)
        self._current_variable = var_name
        self._set_header_text(var_name)
        return True

    def set_data(self, da, coords):
        Z = np.asarray(getattr(da, "values", da), float)
        self._raw_data = np.asarray(Z, float)
        self._processed_data = np.asarray(Z, float)
        self._coords = {}
        coords = dict(coords or {})
        if "X" in coords and "Y" in coords:
            self._display_mode = "warped"
            self._coords["X"] = np.asarray(coords["X"], float)
            self._coords["Y"] = np.asarray(coords["Y"], float)
        elif "x" in coords and "y" in coords:
            self._display_mode = "rectilinear"
            self._coords["x"] = np.asarray(coords["x"], float)
            self._coords["y"] = np.asarray(coords["y"], float)
        else:
            self._display_mode = "image"
        self._current_processing = "none"
        self._processing_params = {}
        self._display_data(self._processed_data, autorange=True)
        try:
            self.viewer.img_item.setVisible(True)
        except Exception:
            pass
        self._apply_preferences()

    def set_preferences(self, preferences: Optional[PreferencesManager]):
        if self.preferences is preferences:
            return
        if self.preferences:
            try:
                self.preferences.changed.disconnect(self._on_preferences_changed)
            except Exception:
                pass
        self.preferences = preferences
        if preferences is not None:
            try:
                preferences.changed.connect(self._on_preferences_changed)
            except Exception:
                pass
            self._apply_preferences()

    def _on_preferences_changed(self, _data):
        self._apply_preferences()

    def _apply_preferences(self):
        prefs = self.preferences
        if prefs is None:
            return
        cmap_name = prefs.preferred_colormap(self._current_variable)
        if cmap_name:
            try:
                cmap = pg.colormap.get(cmap_name)
                self.viewer.lut.gradient.setColorMap(cmap)
                self.viewer.lut.rehide_stops()
            except Exception:
                pass
        if prefs.autoscale_on_load():
            try:
                self.viewer.autoscale_levels()
            except Exception:
                pass

    def set_selected(self, selected: bool):
        selected = bool(selected)
        if self._selected == selected:
            return
        self._selected = selected
        self.setProperty("selected", selected)
        try:
            self.style().unpolish(self)
            self.style().polish(self)
        except Exception:
            pass
        self.update()

    def is_selected(self) -> bool:
        return bool(self._selected)

    def apply_processing(self, mode: str, params: Dict[str, object], manager: Optional["ProcessingManager"]):
        if self._raw_data is None:
            return
        data = np.asarray(self._raw_data, float)
        mode = mode or "none"
        params = dict(params or {})

        if mode.startswith("pipeline:"):
            if not manager:
                raise RuntimeError("No processing manager is available for pipelines.")
            name = mode.split(":", 1)[1]
            pipeline = manager.get_pipeline(name)
            if pipeline is None:
                raise RuntimeError(f"Pipeline '{name}' is not available.")
            processed = pipeline.apply(data)
            params = {}
        elif mode != "none":
            processed = apply_processing_step(mode, data, params)
        else:
            processed = data

        self._processed_data = np.asarray(processed, float)
        self._current_processing = mode
        self._processing_params = dict(params)
        self._display_data(self._processed_data, autorange=True)

    def reset_processing(self):
        if self._raw_data is None:
            return
        self._processed_data = np.asarray(self._raw_data, float)
        self._current_processing = "none"
        self._processing_params = {}
        self._display_data(self._processed_data, autorange=True)

    def _display_data(self, data: np.ndarray, *, autorange: bool = False):
        if self._display_mode == "warped" and "X" in self._coords and "Y" in self._coords:
            self.viewer.set_warped(self._coords["X"], self._coords["Y"], data, autorange=autorange)
        elif self._display_mode == "rectilinear" and "x" in self._coords and "y" in self._coords:
            self.viewer.set_rectilinear(self._coords["x"], self._coords["y"], data, autorange=autorange)
        else:
            self.viewer.set_image(data, autorange=autorange)
        try:
            self.viewer.img_item.setVisible(True)
        except Exception:
            pass

    def set_histogram_visible(self, on: bool):
        self._hist_master_enabled = bool(on)
        self._update_histogram_visibility()

    def set_histogram_local_enabled(self, on: bool):
        enabled = bool(on)
        if self._hist_local_enabled == enabled:
            return
        self._hist_local_enabled = enabled
        self._update_histogram_visibility()

    def is_histogram_local_enabled(self) -> bool:
        return bool(self._hist_local_enabled)

    def _record_histogram_sizes(self, *_):
        if not self._hist_widget or not self._hist_widget.isVisible():
            return
        try:
            sizes = self.center_split.sizes()
        except Exception:
            return
        if len(sizes) < 2 or sizes[1] <= 0:
            return
        self._hist_last_split_sizes = list(sizes)

    def _update_histogram_visibility(self):
        hist_widget = self._hist_widget or self.viewer.histogram_widget()
        if hist_widget is None:
            return
        if self.center_split.indexOf(hist_widget) == -1:
            self.center_split.addWidget(hist_widget)
        want_visible = self._hist_master_enabled and self._hist_local_enabled
        if want_visible:
            hist_widget.setMinimumWidth(80)
            hist_widget.setMaximumWidth(16777215)
            hist_widget.show()
            sizes = self._hist_last_split_sizes
            if sizes and len(sizes) >= 2 and sizes[1] > 0:
                try:
                    self.center_split.setSizes(list(sizes))
                except Exception:
                    pass
            else:
                try:
                    current = self.center_split.sizes()
                except Exception:
                    current = []
                if len(current) < 2 or current[1] == 0:
                    total = sum(current) if len(current) >= 2 else 0
                    if total <= 0:
                        total = 400
                    hist_width = max(120, total // 4)
                    main_width = max(120, total - hist_width)
                    try:
                        self.center_split.setSizes([int(main_width), int(hist_width)])
                    except Exception:
                        pass
        else:
            if hist_widget.isVisible():
                try:
                    sizes = self.center_split.sizes()
                except Exception:
                    sizes = []
                if len(sizes) >= 2 and sizes[1] > 0:
                    self._hist_last_split_sizes = list(sizes)
            hist_widget.hide()
            hist_widget.setMinimumWidth(0)
            hist_widget.setMaximumWidth(0)

    def _clear_display(self, *, preserve_header: bool = False):
        self._raw_data = None
        self._processed_data = None
        self._coords = {}
        self._display_mode = "image"
        try:
            blank = np.zeros((1, 1), dtype=float)
            self.viewer.img_item.setImage(blank, autoLevels=False)
            try:
                self.viewer.img_item.setLevels((0.0, 1.0))
            except Exception:
                pass
        except Exception:
            pass
        try:
            self.viewer.img_item.setVisible(False)
        except Exception:
            pass
        try:
            self.viewer.hide_crosshair()
            self.viewer.clear_mirrored_crosshair()
        except Exception:
            pass
        if not preserve_header:
            self._set_header_text(None)

    def annotation_defaults(self) -> PlotAnnotationConfig:
        return self.viewer.annotation_defaults()

    def apply_annotation(self, config: PlotAnnotationConfig):
        self.viewer.apply_annotation(config)


# ---------------------------------------------------------------------------
# MultiView grid: drag vars to create tiles; master toggle for histograms
# ---------------------------------------------------------------------------
class MultiViewGrid(QtWidgets.QWidget):
    """
    A splitter-based grid of ViewerFrame tiles.
    - Drag a dataset or variable reference from the DatasetsPane onto this widget to add a tile.
    - 'Columns' spinbox controls how many tiles per row.
    - 'Show histograms' toggles the classic HistogramLUTItem to the right of each tile.
    """
    def __init__(
        self,
        processing_manager: Optional[ProcessingManager] = None,
        preferences: Optional[PreferencesManager] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.frames: List[ViewerFrame] = []
        self.processing_manager = processing_manager
        self.preferences: Optional[PreferencesManager] = None
        self._selected_frames: Set[ViewerFrame] = set()
        self._mouse_down = False
        self._drag_select_active = False
        self._drag_select_add = True

        app = QtWidgets.QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

        v = QtWidgets.QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        # Toolbar
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Columns:"))
        self.col_spin = QtWidgets.QSpinBox()
        self.col_spin.setRange(1, 12)
        self.col_spin.setValue(3)
        self.col_spin.valueChanged.connect(self._reflow)
        bar.addWidget(self.col_spin)

        self.chk_show_hist = QtWidgets.QCheckBox("Show histograms")
        self.chk_show_hist.setChecked(True)        # set False if you prefer off-by-default
        self.chk_show_hist.toggled.connect(self._apply_histogram_visibility)
        bar.addWidget(self.chk_show_hist)

        self.chk_link_levels = QtWidgets.QCheckBox("Lock colorscales")
        self.chk_link_levels.toggled.connect(self._on_link_levels_toggled)
        bar.addWidget(self.chk_link_levels)

        self.chk_link_panzoom = QtWidgets.QCheckBox("Lock pan/zoom")
        self.chk_link_panzoom.toggled.connect(self._on_link_panzoom_toggled)
        bar.addWidget(self.chk_link_panzoom)

        self.chk_cursor_mirror = QtWidgets.QCheckBox("Mirror cursor")
        self.chk_cursor_mirror.setChecked(False)
        self.chk_cursor_mirror.toggled.connect(self._on_link_cursor_toggled)
        bar.addWidget(self.chk_cursor_mirror)

        self.btn_autoscale = QtWidgets.QPushButton("Autoscale colors")
        self.btn_autoscale.clicked.connect(self._autoscale_colors)
        bar.addWidget(self.btn_autoscale)

        self.btn_autopan = QtWidgets.QPushButton("Auto pan/zoom")
        self.btn_autopan.clicked.connect(self._auto_panzoom)
        bar.addWidget(self.btn_autopan)

        self.btn_equalize_rows = QtWidgets.QPushButton("Equalize rows")
        self.btn_equalize_rows.clicked.connect(self.equalize_rows)
        bar.addWidget(self.btn_equalize_rows)

        self.btn_equalize_cols = QtWidgets.QPushButton("Equalize columns")
        self.btn_equalize_cols.clicked.connect(self.equalize_columns)
        bar.addWidget(self.btn_equalize_cols)

        self.btn_select_all = QtWidgets.QPushButton("Select All Plots")
        self.btn_select_all.setEnabled(False)
        self.btn_select_all.clicked.connect(self._select_all_frames)
        bar.addWidget(self.btn_select_all)

        self.btn_apply_processing = QtWidgets.QPushButton("Apply processing…")
        self.btn_apply_processing.setEnabled(False)
        self.btn_apply_processing.clicked.connect(self._on_apply_processing_clicked)
        bar.addWidget(self.btn_apply_processing)

        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_menu = QtWidgets.QMenu(self.btn_export)
        self.act_export_active = export_menu.addAction("Save active plot…")
        self.act_export_active.triggered.connect(self._export_active_plot)
        self.act_export_selected = export_menu.addAction("Save selected plots to folder…")
        self.act_export_selected.triggered.connect(self._export_selected_plots)
        export_menu.addSeparator()
        self.act_export_layout = export_menu.addAction("Save entire layout…")
        self.act_export_layout.triggered.connect(self._export_layout_image)
        self.btn_export.setMenu(export_menu)
        bar.addWidget(self.btn_export)

        self.btn_annotations = QtWidgets.QPushButton("Set annotations…")
        self.btn_annotations.clicked.connect(self._open_annotation_dialog)
        bar.addWidget(self.btn_annotations)

        bar.addStretch(1)
        v.addLayout(bar)

        # A vertical splitter holds "rows" (each row is a horizontal splitter of tiles)
        self.vsplit = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.vsplit.setChildrenCollapsible(False)
        v.addWidget(self.vsplit, 1)

        self._level_handlers = {}
        self._view_handlers = {}
        self._cursor_handlers = {}
        self._syncing_levels = False
        self._syncing_views = False

        self.set_preferences(preferences)

    def set_preferences(self, preferences: Optional[PreferencesManager]):
        if self.preferences is preferences:
            return
        if self.preferences:
            try:
                self.preferences.changed.disconnect(self._on_preferences_changed)
            except Exception:
                pass
        self.preferences = preferences
        for frame in self.frames:
            frame.set_preferences(self.preferences)
        if self.preferences:
            try:
                self.preferences.changed.connect(self._on_preferences_changed)
            except Exception:
                pass
        self._on_preferences_changed(None)

    # ---------- Drag & Drop ----------
    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent):
        ev.acceptProposedAction() if ev.mimeData().hasText() else ev.ignore()

    def dropEvent(self, ev: QtGui.QDropEvent):
        text = ev.mimeData().text()
        high_ref = HighDimVarRef.from_mime(text)
        if high_ref:
            QtWidgets.QMessageBox.information(
                self,
                "Unsupported variable",
                "High-dimensional variables should be sent to the Slice Data tab.",
            )
            ev.ignore()
            return

        ds_ref = DataSetRef.from_mime(text)
        mem_ds = MemoryDatasetRef.from_mime(text) if not ds_ref else None
        vr = None if ds_ref or mem_ds else VarRef.from_mime(text)
        mem_var = None if (ds_ref or mem_ds or vr) else MemoryVarRef.from_mime(text)
        slice_ref = None
        if not ds_ref and not mem_ds and not vr and not mem_var:
            slice_ref = MemorySliceRef.from_mime(text)

        if not ds_ref and not mem_ds and not vr and not mem_var and not slice_ref:
            ev.ignore()
            return

        dataset = None
        frame_title = ""
        try:
            if ds_ref:
                dataset = ds_ref.load()
                frame_title = ds_ref.path.name
            elif mem_ds:
                dataset = mem_ds.load()
                frame_title = mem_ds.display_name()
            elif vr:
                dataset = open_dataset(vr.path)
                frame_title = vr.path.name
            elif mem_var:
                dataset = MemoryDatasetRegistry.get_dataset(mem_var.dataset_key)
                if dataset is None:
                    raise RuntimeError("Dataset is no longer available in memory")
                frame_title = MemoryDatasetRegistry.get_label(mem_var.dataset_key)
            elif slice_ref:
                arr, coords, alias = slice_ref.load()
                dataset = xr.Dataset({alias: arr})
                for key, value in coords.items():
                    try:
                        dataset[alias] = dataset[alias].assign_coords({key: value})
                    except Exception:
                        pass
                frame_title = slice_ref.display_label()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            ev.ignore()
            return

        fr = ViewerFrame(title=frame_title, parent=self)
        fr.set_preferences(self.preferences)
        fr.request_close.connect(self._remove_frame)
        try:
            if ds_ref:
                fr.set_dataset(dataset, ds_ref.path)
            elif mem_ds:
                fr.set_dataset(dataset, None)
            elif vr:
                fr.set_dataset(dataset, vr.path, select=vr.var)
            elif mem_var:
                if mem_var.var not in dataset.data_vars:
                    raise RuntimeError("Selected variable is no longer available")
                fr.set_dataset(dataset, None, select=mem_var.var)
            elif slice_ref:
                fr.set_dataset(dataset, None, select=alias)
        except Exception as exc:
            try:
                close = getattr(dataset, "close", None)
                if callable(close):
                    close()
            except Exception:
                pass
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            ev.ignore()
            return

        self.frames.append(fr)
        self._connect_frame_signals(fr)
        self._sync_new_frame_to_links(fr)
        fr.set_selected(False)

        self._reflow()
        self._update_apply_button_state()
        ev.acceptProposedAction()

    # ---------- Tile management ----------
    def _remove_frame(self, fr: ViewerFrame):
        self._disconnect_frame_signals(fr)
        if fr in self.frames:
            self.frames.remove(fr)
        if fr in self._selected_frames:
            self._selected_frames.remove(fr)
            fr.set_selected(False)
        try:
            fr.setParent(None)
        except Exception:
            pass
        try:
            fr.dispose()
        except Exception:
            pass
        self._reflow()
        self._update_apply_button_state()

    def selected_frames(self) -> List[ViewerFrame]:
        return [fr for fr in self.frames if fr in self._selected_frames]

    def _update_apply_button_state(self):
        has_frames = bool(self.frames)
        if hasattr(self, "btn_select_all"):
            self.btn_select_all.setEnabled(has_frames)
        if hasattr(self, "btn_apply_processing"):
            self.btn_apply_processing.setEnabled(bool(self._selected_frames))

    def _select_all_frames(self):
        if not self.frames:
            self._clear_selection()
            self._update_apply_button_state()
            return
        self._clear_selection()
        for fr in self.frames:
            self._set_frame_selected(fr, True, clear=False)
        self._update_apply_button_state()

    def _set_frame_selected(self, frame: Optional[ViewerFrame], selected: bool, *, clear: bool = False):
        if frame is None or frame not in self.frames:
            if clear:
                self._clear_selection()
            return
        if clear:
            self._clear_selection(exclude=frame if selected else None)
        if selected:
            self._selected_frames.add(frame)
        else:
            self._selected_frames.discard(frame)
        frame.set_selected(selected)
        self._update_apply_button_state()

    def _clear_selection(self, exclude: Optional[ViewerFrame] = None):
        changed = False
        for fr in list(self._selected_frames):
            if exclude is not None and fr is exclude:
                continue
            fr.set_selected(False)
            self._selected_frames.discard(fr)
            changed = True
        if changed:
            self._update_apply_button_state()

    def _frame_at_global_pos(self, global_pos: QtCore.QPoint) -> Optional[ViewerFrame]:
        widget = QtWidgets.QApplication.widgetAt(global_pos)
        while widget is not None:
            if isinstance(widget, ViewerFrame):
                return widget if widget in self.frames else None
            widget = widget.parentWidget()
        return None

    def eventFilter(self, obj, event):
        if isinstance(event, QtGui.QMouseEvent):
            etype = event.type()
            if etype == QtCore.QEvent.MouseButtonPress:
                if event.button() == QtCore.Qt.LeftButton:
                    frame = self._frame_at_global_pos(event.globalPos())
                    if frame is not None:
                        ctrl = bool(event.modifiers() & QtCore.Qt.ControlModifier)
                        if ctrl:
                            target_state = frame not in self._selected_frames
                            self._set_frame_selected(frame, target_state, clear=False)
                            self._drag_select_active = True
                            self._drag_select_add = target_state
                        else:
                            self._set_frame_selected(frame, True, clear=True)
                            self._drag_select_active = False
                        self._mouse_down = True
                elif event.button() == QtCore.Qt.RightButton:
                    frame = self._frame_at_global_pos(event.globalPos())
                    if frame is not None:
                        if frame not in self._selected_frames:
                            self._set_frame_selected(frame, True, clear=True)
                        self._show_plot_context_menu(event.globalPos())
                        return True
            elif etype == QtCore.QEvent.MouseMove:
                if self._mouse_down and self._drag_select_active and event.buttons() & QtCore.Qt.LeftButton:
                    frame = self._frame_at_global_pos(event.globalPos())
                    if frame is not None:
                        self._set_frame_selected(frame, self._drag_select_add, clear=False)
            elif etype == QtCore.QEvent.MouseButtonRelease and event.button() == QtCore.Qt.LeftButton:
                if self._mouse_down:
                    self._mouse_down = False
                    self._drag_select_active = False
        elif isinstance(event, QtGui.QContextMenuEvent):
            frame = self._frame_at_global_pos(event.globalPos())
            if frame is not None:
                if frame not in self._selected_frames:
                    self._set_frame_selected(frame, True, clear=True)
                self._show_plot_context_menu(event.globalPos())
                return True
        return super().eventFilter(obj, event)

    def _show_plot_context_menu(self, global_pos: QtCore.QPoint):
        menu = QtWidgets.QMenu(self)
        act_plot = menu.addAction("Plot Data…")
        if not self.selected_frames():
            act_plot.setEnabled(False)
        act_plot.triggered.connect(self._on_plot_data_requested)
        menu.exec_(global_pos)

    def _on_plot_data_requested(self):
        frames = self.selected_frames()
        if not frames:
            return
        available: Set[str] = set()
        for fr in frames:
            available.update(fr.available_variables())
        available_list = sorted(available)
        default = ""
        for fr in frames:
            cur = fr.current_variable()
            if cur:
                default = cur
                break
        items = list(available_list)
        if default:
            if default not in items:
                items.insert(0, default)
                default_index = 0
            else:
                default_index = items.index(default)
        else:
            default_index = 0
        if not items:
            items = [default] if default else [""]
            default_index = 0
        item, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Plot Data",
            "Data variable:",
            items,
            default_index,
            editable=True,
        )
        if not ok:
            return
            name = str(item).strip()
        for fr in frames:
            fr.plot_variable(name)

    def _open_annotation_dialog(self):
        frames = self.selected_frames()
        if not frames:
            frames = list(self.frames)
        if not frames:
            QtWidgets.QMessageBox.information(
                self,
                "No plots",
                "Add a plot before setting annotations.",
            )
            return
        initial = frames[0].annotation_defaults()
        initial.apply_to_all = False
        dialog = PlotAnnotationDialog(self, initial=initial, allow_apply_all=True)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        config = dialog.annotation_config()
        if config is None:
            return
        targets = self.frames if config.apply_to_all else frames
        base = replace(config, apply_to_all=False)
        for frame in targets:
            frame.apply_annotation(base)

    # ---------- export helpers ----------
    def _on_preferences_changed(self, _data):
        for frame in self.frames:
            frame.set_preferences(self.preferences)

    def _default_layout_label(self) -> str:
        if self.preferences:
            return self.preferences.default_layout_label()
        return ""

    def _default_export_dir(self) -> str:
        if self.preferences:
            return self.preferences.default_export_directory()
        return ""

    def _store_export_dir(self, directory: str):
        if self.preferences and directory:
            data = self.preferences.data()
            misc = data.setdefault("misc", {})
            misc["default_export_dir"] = directory
            self.preferences.update(data)

    def _export_active_plot(self):
        frames = self.selected_frames()
        if len(frames) != 1:
            QtWidgets.QMessageBox.information(
                self,
                "Select a plot",
                "Please select a single plot to export.",
            )
            return
        frame = frames[0]
        suggestion = _sanitize_filename(frame.lbl.text()) + ".png"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save plot",
            suggestion,
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        if not _save_snapshot(frame, target):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to write the selected file.")
            return
        log_action(f"Saved MultiView plot to {target}")

    def _export_selected_plots(self):
        frames = self.selected_frames()
        if not frames:
            QtWidgets.QMessageBox.information(
                self,
                "No plots selected",
                "Select one or more plots to export individually.",
            )
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            self._default_export_dir(),
        )
        if not directory:
            return
        base = Path(directory)
        self._store_export_dir(directory)
        count = 0
        for idx, frame in enumerate(frames, start=1):
            label = frame.lbl.text() or f"plot_{idx}"
            name = _sanitize_filename(label) or f"plot_{idx}"
            target = base / f"{name}_{idx:02d}.png"
            if _save_snapshot(frame, target):
                count += 1
        if count == 0:
            QtWidgets.QMessageBox.warning(self, "Export failed", "No plots were exported.")
            return
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {count} plot(s) to {base}",
        )
        log_action(f"Exported {count} MultiView plots to {base}")

    def _export_layout_image(self):
        if not self.frames:
            QtWidgets.QMessageBox.information(
                self,
                "No plots",
                "Add at least one plot before exporting the layout.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save layout",
            "multiview-layout.png",
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = _ask_layout_label(self, "Layout label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        if not _save_snapshot(self, target, label):
            QtWidgets.QMessageBox.warning(
                self,
                "Save failed",
                "Unable to save the layout image.",
            )
            return
        log_action(f"Saved MultiView layout to {target}")

    def _on_apply_processing_clicked(self):
        frames = self.selected_frames()
        if not frames:
            return
        dialog = ProcessingSelectionDialog(self.processing_manager, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode, params = dialog.selected_processing()
        for frame in frames:
            try:
                frame.apply_processing(mode, params, self.processing_manager)
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Processing failed", str(exc))
                break


    def _reflow(self):
        """Rebuild the splitter grid according to the current column count."""
        # Detach any existing children from row splitters, then clear rows
        for i in range(self.vsplit.count()):
            w = self.vsplit.widget(i)
            if isinstance(w, QtWidgets.QSplitter):
                while w.count():
                    cw = w.widget(0)
                    if cw:
                        cw.setParent(None)
        while self.vsplit.count():
            w = self.vsplit.widget(0)
            w.setParent(None)
            w.deleteLater()

        cols = max(1, self.col_spin.value())
        for r_start in range(0, len(self.frames), cols):
            row_frames = self.frames[r_start:r_start + cols]
            h = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
            h.setChildrenCollapsible(False)
            for fr in row_frames:
                h.addWidget(fr)
            self.vsplit.addWidget(h)

        # Re-apply current histogram visibility to all tiles
        self._apply_histogram_visibility(self.chk_show_hist.isChecked())
        self.equalize_columns()

    def _apply_histogram_visibility(self, on: bool):
        """Show/hide the classic HistogramLUTItem on every tile."""
        on = bool(on)
        for fr in self.frames:
            try:
                fr.set_histogram_visible(on)
            except Exception:
                pass

    def _connect_frame_signals(self, fr: ViewerFrame):
        viewer = getattr(fr, "viewer", None)
        if viewer is None:
            return
        if viewer not in self._level_handlers:
            try:
                handler = partial(self._viewer_levels_changed, viewer)
                viewer.sigLevelsChanged.connect(handler)
                self._level_handlers[viewer] = handler
            except Exception:
                self._level_handlers.pop(viewer, None)
        if viewer not in self._view_handlers:
            try:
                handler = partial(self._viewer_view_changed, viewer)
                viewer.sigViewChanged.connect(handler)
                self._view_handlers[viewer] = handler
            except Exception:
                self._view_handlers.pop(viewer, None)
        if viewer not in self._cursor_handlers:
            try:
                viewer.sigCursorMoved.connect(self._viewer_cursor_moved)
                self._cursor_handlers[viewer] = True
            except Exception:
                self._cursor_handlers.pop(viewer, None)

    def _disconnect_frame_signals(self, fr: ViewerFrame):
        viewer = getattr(fr, "viewer", None)
        if viewer is None:
            return
        handler = self._level_handlers.pop(viewer, None)
        if handler is not None:
            try:
                viewer.sigLevelsChanged.disconnect(handler)
            except Exception:
                pass
        handler = self._view_handlers.pop(viewer, None)
        if handler is not None:
            try:
                viewer.sigViewChanged.disconnect(handler)
            except Exception:
                pass
        if viewer in self._cursor_handlers:
            try:
                viewer.sigCursorMoved.disconnect(self._viewer_cursor_moved)
            except Exception:
                pass
            self._cursor_handlers.pop(viewer, None)
        try:
            viewer.clear_mirrored_crosshair()
        except Exception:
            pass

    def _viewer_levels_changed(self, viewer, levels):
        if not self.chk_link_levels.isChecked() or self._syncing_levels:
            return
        self._syncing_levels = True
        try:
            try:
                lo, hi = (float(levels[0]), float(levels[1]))
            except Exception:
                lo, hi = viewer.get_levels()
            for fr in self.frames:
                if fr.viewer is viewer:
                    continue
                fr.viewer.set_levels(lo, hi)
        finally:
            self._syncing_levels = False

    def _viewer_view_changed(self, viewer, xr, yr):
        if not self.chk_link_panzoom.isChecked() or self._syncing_views:
            return
        self._syncing_views = True
        try:
            xr_vals = tuple(xr) if xr is not None else viewer.get_view_range()[0]
            yr_vals = tuple(yr) if yr is not None else viewer.get_view_range()[1]
            for fr in self.frames:
                if fr.viewer is viewer:
                    continue
                fr.viewer.set_view_range(xr=xr_vals, yr=yr_vals)
        finally:
            self._syncing_views = False

    def _on_link_levels_toggled(self, on: bool):
        if on:
            self._sync_all_levels()

    def _on_link_panzoom_toggled(self, on: bool):
        if on:
            self._sync_all_views()

    def _on_link_cursor_toggled(self, on: bool):
        if not on:
            self._clear_mirrored_crosshairs()

    def _sync_all_levels(self):
        if not self.frames:
            return
        ref = self.frames[0].viewer
        try:
            lo, hi = ref.get_levels()
        except Exception:
            return
        self._syncing_levels = True
        try:
            for fr in self.frames[1:]:
                fr.viewer.set_levels(lo, hi)
        finally:
            self._syncing_levels = False

    def _sync_all_views(self):
        if not self.frames:
            return
        ref = self.frames[0].viewer
        try:
            xr, yr = ref.get_view_range()
        except Exception:
            return
        self._syncing_views = True
        try:
            for fr in self.frames[1:]:
                fr.viewer.set_view_range(xr=xr, yr=yr)
        finally:
            self._syncing_views = False

    def _viewer_cursor_moved(self, viewer, x, y, value, inside, label):
        if not inside or not self.chk_cursor_mirror.isChecked():
            self._clear_mirrored_crosshairs(exclude=viewer)
            return
        for fr in self.frames:
            other = fr.viewer
            if other is viewer:
                continue
            try:
                local_value = other.value_at(x, y) if hasattr(other, "value_at") else None
            except Exception:
                local_value = None
            try:
                other.show_crosshair(x, y, value=local_value, mirrored=True)
            except Exception:
                pass

    def _clear_mirrored_crosshairs(self, exclude=None):
        for fr in self.frames:
            viewer = fr.viewer
            if exclude is not None and viewer is exclude:
                continue
            try:
                viewer.clear_mirrored_crosshair()
            except Exception:
                pass

    def _sync_new_frame_to_links(self, fr: ViewerFrame):
        others = [f for f in self.frames if f is not fr]
        if not others:
            return
        ref = others[0].viewer
        if self.chk_link_levels.isChecked():
            try:
                lo, hi = ref.get_levels()
                fr.viewer.set_levels(lo, hi)
            except Exception:
                pass
        if self.chk_link_panzoom.isChecked():
            try:
                xr, yr = ref.get_view_range()
                fr.viewer.set_view_range(xr=xr, yr=yr)
            except Exception:
                pass

    def _autoscale_colors(self):
        if self.chk_link_levels.isChecked():
            self.chk_link_levels.setChecked(False)
        for fr in self.frames:
            try:
                fr.viewer.autoscale_levels()
            except Exception:
                pass

    def _auto_panzoom(self):
        for fr in self.frames:
            try:
                fr.viewer.auto_view_range()
            except Exception:
                pass

    def equalize_columns(self):
        for splitter in self._iter_row_splitters():
            count = splitter.count()
            if count <= 0:
                continue
            splitter.setSizes([1] * count)

    def equalize_rows(self):
        count = self.vsplit.count()
        if count <= 0:
            return
        self.vsplit.setSizes([1] * count)

    def _iter_row_splitters(self):
        for i in range(self.vsplit.count()):
            w = self.vsplit.widget(i)
            if isinstance(w, QtWidgets.QSplitter):
                yield w

# ---------------------------------------------------------------------------
# Sequential view helpers
# ---------------------------------------------------------------------------


class SequentialRoiWindow(QtWidgets.QWidget):
    axesChanged = QtCore.Signal(tuple)
    reducerChanged = QtCore.Signal(str)
    closed = QtCore.Signal()

    def __init__(self, parent=None):
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Sequential ROI Inspector")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.setMinimumSize(360, 520)
        self.resize(420, 600)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)

        controls.addWidget(QtWidgets.QLabel("Reduce over:"))
        self.cmb_axes = QtWidgets.QComboBox()
        self.cmb_axes.currentIndexChanged.connect(self._emit_axes_changed)
        self.cmb_axes.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        controls.addWidget(self.cmb_axes, 1)

        controls.addWidget(QtWidgets.QLabel("Statistic:"))
        self.cmb_method = QtWidgets.QComboBox()
        self.cmb_method.currentIndexChanged.connect(self._emit_method_changed)
        self.cmb_method.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        controls.addWidget(self.cmb_method, 1)

        layout.addLayout(controls)

        self.lbl_hint = QtWidgets.QLabel()
        self.lbl_hint.setStyleSheet("color: #666;")
        layout.addWidget(self.lbl_hint)

        profile_axes = {
            "bottom": ScientificAxisItem("bottom"),
            "left": ScientificAxisItem("left"),
        }
        self.profile_plot = pg.PlotWidget(axisItems=profile_axes)
        self.profile_plot.setMinimumHeight(140)
        self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self.profile_plot.setLabel("bottom", "Axis")
        self.profile_plot.setLabel("left", "Value")
        self.profile_curve = self.profile_plot.plot([], [], pen=pg.mkPen('#ffaa00', width=2))
        layout.addWidget(self.profile_plot, 1)

        slice_axes = {
            "bottom": ScientificAxisItem("bottom"),
            "left": ScientificAxisItem("left"),
        }
        self.slice_plot = pg.PlotWidget(axisItems=slice_axes)
        self.slice_plot.setMinimumHeight(160)
        self.slice_plot.showGrid(x=True, y=True, alpha=0.3)
        self.slice_plot.setLabel("bottom", "Slice coordinate")
        self.slice_plot.setLabel("left", "ROI statistic")
        self.slice_curve = self.slice_plot.plot([], [], pen=pg.mkPen('#66bbff', width=2))
        layout.addWidget(self.slice_plot, 1)

        self._updating = False

    def set_axis_options(
        self, options: List[Tuple[str, Tuple[int, ...], str, str, Optional[int]]], current_index: int
    ):
        self._updating = True
        self.cmb_axes.clear()
        for entry in options:
            if not entry:
                continue
            label = entry[0]
            axes = entry[1] if len(entry) > 1 else ()
            self.cmb_axes.addItem(label, tuple(int(a) for a in axes))
        self.cmb_axes.setEnabled(bool(options))
        if options:
            self.cmb_axes.setCurrentIndex(max(0, min(current_index, len(options) - 1)))
        self._updating = False

    def set_reducer_options(self, reducers: Dict[str, Tuple[str, object]], current_key: str):
        self._updating = True
        self.cmb_method.clear()
        for key, (label, _fn) in reducers.items():
            self.cmb_method.addItem(label, key)
        idx = max(0, self.cmb_method.findData(current_key))
        self.cmb_method.setCurrentIndex(idx)
        self.cmb_method.setEnabled(self.cmb_method.count() > 0)
        self._updating = False

    def set_hint(self, text: str):
        self.lbl_hint.setText(text)

    def update_profile(self, xs: List[float], ys: List[float], xlabel: str, ylabel: str, visible: bool):
        self.profile_plot.setVisible(visible)
        if not visible:
            self.profile_curve.setData([], [])
            return
        self.profile_plot.setLabel("bottom", xlabel)
        self.profile_plot.setLabel("left", ylabel)
        self.profile_curve.setData(xs, ys)
        self.profile_plot.enableAutoRange()

    def update_slice_curve(self, xs: List[float], ys: List[float], xlabel: str, ylabel: str):
        self.slice_plot.setLabel("bottom", xlabel)
        self.slice_plot.setLabel("left", ylabel)
        self.slice_curve.setData(xs, ys)
        self.slice_plot.enableAutoRange()

    def _emit_axes_changed(self):
        if self._updating:
            return
        data = self.cmb_axes.currentData()
        if data is None:
            return
        try:
            axes = tuple(int(a) for a in data)
        except Exception:
            axes = tuple()
        self.axesChanged.emit(axes)

    def _emit_method_changed(self):
        if self._updating:
            return
        key = self.cmb_method.currentData()
        if key:
            self.reducerChanged.emit(str(key))

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.closed.emit()
        except Exception:
            pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Sequential view: 3D volume visualization helper
# ---------------------------------------------------------------------------


class VolumeAlphaHandle(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, owner: "VolumeAlphaCurveWidget", x_norm: float, y_norm: float):
        radius = 5.0
        super().__init__(-radius, -radius, radius * 2.0, radius * 2.0)
        self._owner = owner
        self.x_norm = float(x_norm)
        self.y_norm = float(y_norm)
        self.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
        pen = QtGui.QPen(QtGui.QColor(20, 20, 20))
        pen.setWidthF(1.2)
        self.setPen(pen)
        self.setZValue(20)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsScenePositionChanges, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemIgnoresTransformations, True)

    def itemChange(self, change: QtWidgets.QGraphicsItem.GraphicsItemChange, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionChange:
            if isinstance(value, QtCore.QPointF):
                point = value
            else:
                point = QtCore.QPointF(value)
            return self._owner.clamp_handle_position(self, point)
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            self._owner.handle_moved(self)
        return super().itemChange(change, value)


class VolumeAlphaCurveWidget(QtWidgets.QWidget):
    curveChanged = QtCore.Signal(list)

    def __init__(self, parent=None, default_value: float = 0.5):
        super().__init__(parent)
        self._scene = QtWidgets.QGraphicsScene(self)
        self._view = QtWidgets.QGraphicsView(self._scene)
        self._view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self._view.setRenderHints(
            QtGui.QPainter.Antialiasing
            | QtGui.QPainter.SmoothPixmapTransform
            | QtGui.QPainter.TextAntialiasing
        )
        self._view.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        self._view.setFrameShape(QtWidgets.QFrame.NoFrame)
        self._view.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        )
        self._view.viewport().installEventFilter(self)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._view)

        self._gradient_item = QtWidgets.QGraphicsPixmapItem()
        self._gradient_item.setZValue(0)
        self._scene.addItem(self._gradient_item)

        border_pen = QtGui.QPen(QtGui.QColor(200, 200, 200))
        border_pen.setWidthF(1.0)
        self._border_item = self._scene.addRect(QtCore.QRectF(0, 0, 1, 1), border_pen)
        self._border_item.setZValue(5)

        curve_pen = QtGui.QPen(QtGui.QColor(245, 245, 245))
        curve_pen.setWidthF(2.0)
        self._curve_item = self._scene.addPath(QtGui.QPainterPath(), curve_pen)
        self._curve_item.setZValue(15)

        self._handles: List[VolumeAlphaHandle] = []
        self._default_positions = [0.0, 0.25, 0.5, 0.75, 1.0]
        self._default_value = max(0.0, min(1.0, float(default_value)))
        self._margin_left = 28.0
        self._margin_right = 16.0
        self._margin_top = 12.0
        self._margin_bottom = 26.0
        self._colormap_name = "viridis"
        self._updating = False

        self.setMinimumHeight(120)
        self.setMaximumHeight(220)
        size_policy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        self.setSizePolicy(size_policy)
        self._update_scene_geometry()
        self.reset_curve()

    # ----- geometry helpers -----
    def showEvent(self, event: QtGui.QShowEvent):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._update_scene_geometry)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self._update_scene_geometry()

    def _effective_rect(self) -> QtCore.QRectF:
        rect = self._scene.sceneRect()
        width = max(1.0, rect.width() - self._margin_left - self._margin_right)
        height = max(1.0, rect.height() - self._margin_top - self._margin_bottom)
        return QtCore.QRectF(
            rect.left() + self._margin_left,
            rect.top() + self._margin_top,
            width,
            height,
        )

    def _update_scene_geometry(self):
        viewport = self._view.viewport()
        width = max(1, viewport.width())
        height = max(1, viewport.height())
        self._scene.setSceneRect(0, 0, float(width), float(height))
        eff = self._effective_rect()
        self._update_gradient_pixmap(int(max(2.0, eff.width())), int(max(2.0, eff.height())))
        self._gradient_item.setPos(eff.left(), eff.top())
        self._border_item.setRect(eff)
        self._position_handles()
        self._update_curve_path()

    # ----- colormap -----
    def set_colormap(self, name: str):
        if not name:
            name = "viridis"
        if self._colormap_name == name:
            return
        self._colormap_name = name
        self._update_gradient_pixmap()
        self._update_curve_path()

    def _update_gradient_pixmap(self, width: Optional[int] = None, height: Optional[int] = None):
        eff = self._effective_rect()
        w = int(width or max(2.0, eff.width()))
        h = int(height or max(2.0, eff.height()))
        try:
            cmap = pg.colormap.get(self._colormap_name)
        except Exception:
            cmap = pg.colormap.get("viridis")
        lut = cmap.map(np.linspace(0.0, 1.0, max(2, w)), mode="byte")
        gradient = np.repeat(lut[np.newaxis, :, :3], max(2, h), axis=0)
        alpha = np.full((gradient.shape[0], gradient.shape[1], 1), 255, dtype=np.uint8)
        rgba = np.concatenate((gradient, alpha), axis=2)
        image = QtGui.QImage(
            rgba.data, rgba.shape[1], rgba.shape[0], int(rgba.strides[0]), QtGui.QImage.Format_RGBA8888
        )
        image = image.copy()
        self._gradient_item.setPixmap(QtGui.QPixmap.fromImage(image))

    # ----- handle interactions -----
    def clamp_handle_position(
        self, handle: VolumeAlphaHandle, value: QtCore.QPointF
    ) -> QtCore.QPointF:
        if self._updating:
            return value
        eff = self._effective_rect()
        width = eff.width()
        height = eff.height()
        x_norm = 0.0
        if width > 0:
            raw_x_norm = (float(value.x()) - eff.left()) / width
            idx = self._handles.index(handle)
            min_norm = 0.0 if idx == 0 else self._handles[idx - 1].x_norm + 1e-4
            max_norm = 1.0 if idx == len(self._handles) - 1 else self._handles[idx + 1].x_norm - 1e-4
            x_norm = max(min_norm, min(max_norm, raw_x_norm))
            x_norm = max(0.0, min(1.0, x_norm))
        x = eff.left() + x_norm * max(width, 1.0)
        y = float(value.y())
        if height > 0:
            y = max(eff.top(), min(eff.bottom(), y))
        return QtCore.QPointF(x, y)

    def handle_moved(self, handle: VolumeAlphaHandle):
        if self._updating:
            return
        eff = self._effective_rect()
        width = eff.width()
        height = eff.height()
        if width <= 0 or height <= 0:
            return
        pos = handle.pos()
        x_norm = (pos.x() - eff.left()) / width
        y_norm = (eff.bottom() - pos.y()) / height
        x_norm = max(0.0, min(1.0, float(x_norm)))
        y_norm = max(0.0, min(1.0, float(y_norm)))
        idx = self._handles.index(handle)
        if idx > 0:
            x_norm = max(x_norm, self._handles[idx - 1].x_norm + 1e-4)
        if idx < len(self._handles) - 1:
            x_norm = min(x_norm, self._handles[idx + 1].x_norm - 1e-4)
        handle.y_norm = y_norm
        handle.x_norm = x_norm
        self._handles.sort(key=lambda item: item.x_norm)
        self._position_handles()
        self._update_curve_path()
        self.curveChanged.emit(self.curve_points())

    def _position_handles(self):
        eff = self._effective_rect()
        if eff.height() <= 0 or eff.width() <= 0:
            return
        self._updating = True
        try:
            for handle in self._handles:
                x = eff.left() + handle.x_norm * eff.width()
                y = eff.bottom() - handle.y_norm * eff.height()
                handle.setPos(QtCore.QPointF(x, y))
        finally:
            self._updating = False

    def _update_curve_path(self):
        if not self._handles:
            self._curve_item.setPath(QtGui.QPainterPath())
            return
        sorted_handles = sorted(self._handles, key=lambda item: item.x_norm)
        path = QtGui.QPainterPath()
        first = sorted_handles[0]
        path.moveTo(first.pos())
        for handle in sorted_handles[1:]:
            path.lineTo(handle.pos())
        self._curve_item.setPath(path)

    # ----- curve helpers -----
    def curve_points(self) -> List[Tuple[float, float]]:
        return [(handle.x_norm, handle.y_norm) for handle in sorted(self._handles, key=lambda h: h.x_norm)]

    def set_curve(self, points: List[Tuple[float, float]]):
        if not points:
            return
        for handle in list(self._handles):
            self._scene.removeItem(handle)
        self._handles.clear()
        for x, y in points:
            self._add_handle(float(x), float(y), emit=False)
        self._update_curve_path()
        self.curveChanged.emit(self.curve_points())

    def reset_curve(self):
        for handle in list(self._handles):
            self._scene.removeItem(handle)
        self._handles.clear()
        for x_norm in self._default_positions:
            self._add_handle(float(x_norm), float(self._default_value), emit=False)
        self._update_curve_path()
        self.curveChanged.emit(self.curve_points())

    def _add_handle(self, x_norm: float, y_norm: float, *, emit: bool = True):
        x_norm = max(0.0, min(1.0, x_norm))
        y_norm = max(0.0, min(1.0, y_norm))
        for existing in self._handles:
            if abs(existing.x_norm - x_norm) < 1e-4:
                existing.x_norm = x_norm
                existing.y_norm = y_norm
                self._handles.sort(key=lambda item: item.x_norm)
                self._position_handles()
                self._update_curve_path()
                if emit:
                    self.curveChanged.emit(self.curve_points())
                return
        handle = VolumeAlphaHandle(self, x_norm, y_norm)
        self._scene.addItem(handle)
        self._handles.append(handle)
        self._handles.sort(key=lambda item: item.x_norm)
        self._position_handles()
        self._update_curve_path()
        if emit:
            self.curveChanged.emit(self.curve_points())

    def eventFilter(self, obj, event):
        if (
            obj is self._view.viewport()
            and event.type() == QtCore.QEvent.MouseButtonDblClick
            and self.isEnabled()
        ):
            if isinstance(event, QtGui.QMouseEvent) and event.button() == QtCore.Qt.LeftButton:
                scene_pos = self._view.mapToScene(event.pos())
                eff = self._effective_rect()
                if eff.contains(scene_pos):
                    width = eff.width() if eff.width() > 0 else 1.0
                    height = eff.height() if eff.height() > 0 else 1.0
                    x_norm = (scene_pos.x() - eff.left()) / width
                    y_norm = (eff.bottom() - scene_pos.y()) / height
                    self._add_handle(x_norm, y_norm)
                    return True
        return super().eventFilter(obj, event)


class SequentialVolumeWindow(QtWidgets.QWidget):
    closed = QtCore.Signal()

    def __init__(self, parent=None, preferences: Optional[PreferencesManager] = None):
        if gl is None:
            raise RuntimeError("pyqtgraph.opengl is not available")
        super().__init__(parent, QtCore.Qt.Window)
        self.setWindowTitle("Sequential Volume Viewer")
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose, False)
        self.resize(680, 520)

        self._data: Optional[np.ndarray] = None
        self._data_min: float = 0.0
        self._data_max: float = 1.0
        self._colormap_name: str = "viridis"
        self.preferences: Optional[PreferencesManager] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        cmap_controls = QtWidgets.QHBoxLayout()
        cmap_controls.setSpacing(6)
        cmap_controls.addWidget(QtWidgets.QLabel("Colormap:"))

        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cmb_colormap.currentIndexChanged.connect(self._on_colormap_combo_changed)
        cmap_controls.addWidget(self.cmb_colormap, 0)

        cmap_controls.addStretch(1)
        layout.addLayout(cmap_controls)

        controls = QtWidgets.QHBoxLayout()
        controls.setSpacing(6)
        controls.addWidget(QtWidgets.QLabel("Opacity curves:"))

        self.btn_reset_curve = QtWidgets.QPushButton("Reset curves")
        self.btn_reset_curve.clicked.connect(self._on_reset_curve)
        controls.addWidget(self.btn_reset_curve)

        self.btn_reset_view = QtWidgets.QPushButton("Reset view")
        self.btn_reset_view.setEnabled(False)
        self.btn_reset_view.clicked.connect(self._on_reset_view)
        controls.addWidget(self.btn_reset_view)

        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_menu = QtWidgets.QMenu(self.btn_export)
        act_snapshot = export_menu.addAction("Save volume snapshot…")
        act_snapshot.triggered.connect(self._export_volume_snapshot)
        act_layout = export_menu.addAction("Save volume layout…")
        act_layout.triggered.connect(self._export_volume_layout)
        self.btn_export.setMenu(export_menu)
        controls.addWidget(self.btn_export)

        controls.addStretch(1)
        layout.addLayout(controls)

        curves_row = QtWidgets.QHBoxLayout()
        curves_row.setSpacing(8)
        layout.addLayout(curves_row)

        self._curve_keys: Tuple[str, ...] = ("value", "slice", "row", "column")
        self._axis_labels: Dict[str, str] = {
            "value": "Value",
            "slice": "Slice axis",
            "row": "Row axis",
            "column": "Column axis",
        }
        self._curve_widgets: Dict[str, VolumeAlphaCurveWidget] = {}
        self._curve_labels: Dict[str, QtWidgets.QLabel] = {}

        for key in self._curve_keys:
            column_layout = QtWidgets.QVBoxLayout()
            column_layout.setSpacing(4)
            label = QtWidgets.QLabel(self._axis_labels[key])
            label.setAlignment(QtCore.Qt.AlignHCenter)
            label.setWordWrap(True)
            column_layout.addWidget(label)

            widget = VolumeAlphaCurveWidget(default_value=0.25)
            widget.setMinimumWidth(150)
            widget.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
            )
            widget.setToolTip(
                "Drag control points to sculpt opacity, double-click to add a new point."
            )
            widget.curveChanged.connect(lambda points, k=key: self._on_alpha_curve_changed(k, points))
            column_layout.addWidget(widget)

            curves_row.addLayout(column_layout, 1)
            self._curve_widgets[key] = widget
            self._curve_labels[key] = label

        self._volume_item: Optional[gl.GLVolumeItem] = None
        self._volume_scalar: Optional[np.ndarray] = None
        self._volume_shape: Tuple[int, int, int] = (1, 1, 1)
        self._curve_points: Dict[str, List[Tuple[float, float]]] = {}
        self._curve_lut_x: Dict[str, np.ndarray] = {}
        self._curve_lut_y: Dict[str, np.ndarray] = {}
        for key, widget in self._curve_widgets.items():
            points = widget.curve_points()
            xs = np.array([max(0.0, min(1.0, float(x))) for x, _ in points], dtype=float)
            ys = np.array([max(0.0, min(1.0, float(y))) for _, y in points], dtype=float)
            self._curve_points[key] = [(float(x), float(y)) for x, y in zip(xs, ys)]
            self._curve_lut_x[key] = xs
            self._curve_lut_y[key] = ys
        self._alpha_scale_base: float = 101.0

        self._populate_colormap_choices()
        self._update_alpha_controls()

        self.view = gl.GLViewWidget()
        self.view.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        )
        self.view.setMinimumHeight(260)
        self.view.opts["distance"] = 400
        self.view.setBackgroundColor(QtGui.QColor(20, 20, 20))
        layout.addWidget(self.view, 1)

    def set_preferences(self, preferences: Optional[PreferencesManager]):
        if self.preferences is preferences:
            return
        if self.preferences:
            try:
                self.preferences.changed.disconnect(self._on_preferences_changed)
            except Exception:
                pass
        self.preferences = preferences
        if preferences is not None:
            try:
                preferences.changed.connect(self._on_preferences_changed)
            except Exception:
                pass
        self._apply_preference_defaults()

    def _on_preferences_changed(self, _data):
        self._apply_preference_defaults()

    def _apply_preference_defaults(self):
        if not self.preferences:
            return
        preferred = self.preferences.preferred_colormap(None)
        if preferred:
            self.set_colormap(preferred)

    def _default_layout_label(self) -> str:
        if self.preferences:
            return self.preferences.default_layout_label()
        return ""

    def _default_export_dir(self) -> str:
        if self.preferences:
            return self.preferences.default_export_directory()
        return ""

    def _store_export_dir(self, directory: str):
        if self.preferences and directory:
            data = self.preferences.data()
            misc = data.setdefault("misc", {})
            misc["default_export_dir"] = directory
            self.preferences.update(data)

    def _initial_path(self, filename: str) -> str:
        base = self._default_export_dir()
        if base:
            return str(Path(base) / filename)
        return filename

        self.set_preferences(preferences)

    # ----- public API -----
    def set_volume(self, data: Optional[np.ndarray]):
        if data is None or data.size == 0:
            self._data = None
            self._remove_volume()
            self._update_alpha_controls()
            return
        arr = np.asarray(data, float)
        finite = np.isfinite(arr)
        if not finite.any():
            arr = np.zeros_like(arr, dtype=float)
            self._data_min = 0.0
            self._data_max = 1.0
        else:
            min_val = float(np.nanmin(arr))
            max_val = float(np.nanmax(arr))
            if min_val == max_val:
                max_val = min_val + 1.0
            arr = np.nan_to_num(arr, nan=min_val)
            self._data_min = min_val
            self._data_max = max_val
        self._data = arr.astype(np.float32, copy=False)
        self._volume_scalar = self._prepare_volume_array(self._data)
        self._volume_shape = self._volume_scalar.shape if self._volume_scalar is not None else (1, 1, 1)
        self._update_alpha_controls()
        self._ensure_volume_item()
        self._center_volume_item()
        self._update_volume_visual()
        self._reset_camera()

    def set_colormap(self, name: Optional[str]):
        if name:
            self._colormap_name = str(name)
        else:
            self._colormap_name = "viridis"
        for widget in self._curve_widgets.values():
            try:
                widget.set_colormap(self._colormap_name)
            except Exception:
                continue
        self._sync_colormap_combo()
        self._update_volume_visual()

    def clear_volume(self):
        self._data = None
        self._volume_scalar = None
        self._volume_shape = (1, 1, 1)
        self._remove_volume()
        self._update_alpha_controls()
        self.btn_reset_view.setEnabled(False)

    # ----- helpers -----
    def _ensure_volume_item(self):
        if self._volume_item is not None:
            return
        data = self._data
        if data is None:
            return
        scalar = self._prepare_volume_array(data)
        self._volume_scalar = scalar
        rgba = self._compute_rgba_volume(scalar)
        self._volume_item = gl.GLVolumeItem(rgba, smooth=False)
        # Use translucent blending so colors remain readable instead of
        # saturating to white as layers accumulate with additive blending.
        self._volume_item.setGLOptions("translucent")
        self._volume_item.resetTransform()
        self._center_volume_item()
        self.view.addItem(self._volume_item)
        if hasattr(self._volume_item, "update"):
            try:
                self._volume_item.update()
            except Exception:
                pass
        self.btn_reset_view.setEnabled(True)

    def _prepare_volume_array(self, data: np.ndarray) -> np.ndarray:
        if data.ndim != 3:
            return np.zeros((1, 1, 1), dtype=np.float32)
        transposed = np.transpose(data, (2, 1, 0))
        return np.ascontiguousarray(transposed, dtype=np.float32)

    def _compute_rgba_volume(self, scalar: np.ndarray) -> np.ndarray:
        if scalar.size == 0:
            return np.zeros(scalar.shape + (4,), dtype=np.ubyte)
        try:
            cmap = pg.colormap.get(self._colormap_name)
        except Exception:
            cmap = pg.colormap.get("viridis")
        data_min = float(self._data_min)
        data_max = float(self._data_max)
        if not np.isfinite(data_min) or not np.isfinite(data_max) or data_min == data_max:
            data_min = float(np.nanmin(scalar))
            data_max = float(np.nanmax(scalar))
            if not np.isfinite(data_min):
                data_min = 0.0
            if not np.isfinite(data_max) or data_max == data_min:
                data_max = data_min + 1.0
        scale = data_max - data_min
        if scale == 0.0:
            scale = 1.0
        norm = (scalar - data_min) / scale
        norm = np.clip(norm, 0.0, 1.0)
        rgba = cmap.map(norm.reshape(-1), mode="byte").reshape(scalar.shape + (4,))

        alpha_value = self._apply_alpha_scale(self._sample_alpha_curve("value", norm))
        alpha_total = alpha_value
        if scalar.ndim >= 3:
            col_len, row_len, slice_len = scalar.shape
            if slice_len > 1:
                slice_positions = np.linspace(0.0, 1.0, slice_len, dtype=float)
                slice_alpha = self._apply_alpha_scale(
                    self._sample_alpha_curve("slice", slice_positions)
                ).reshape(1, 1, slice_len)
                alpha_total = alpha_total * slice_alpha
            if row_len > 1:
                row_positions = np.linspace(0.0, 1.0, row_len, dtype=float)
                row_alpha = self._apply_alpha_scale(
                    self._sample_alpha_curve("row", row_positions)
                ).reshape(1, row_len, 1)
                alpha_total = alpha_total * row_alpha
            if col_len > 1:
                col_positions = np.linspace(0.0, 1.0, col_len, dtype=float)
                col_alpha = self._apply_alpha_scale(
                    self._sample_alpha_curve("column", col_positions)
                ).reshape(col_len, 1, 1)
                alpha_total = alpha_total * col_alpha
        alpha = np.clip(alpha_total * 255.0, 0.0, 255.0)
        rgba = rgba.copy()
        rgba[..., 3] = alpha.astype(np.uint8)
        return np.ascontiguousarray(rgba, dtype=np.ubyte)

    def _update_volume_visual(self):
        if self._volume_item is None or self._volume_scalar is None:
            return
        rgba = self._compute_rgba_volume(self._volume_scalar)
        try:
            self._volume_item.setData(rgba)
        except TypeError:
            # Older pyqtgraph releases expect the array as the first argument.
            self._volume_item.setData(data=rgba)
        self._center_volume_item()
        self.view.update()

    def _center_volume_item(self):
        if self._volume_item is None or self._volume_scalar is None:
            return
        try:
            self._volume_item.resetTransform()
        except Exception:
            pass
        shape = self._volume_scalar.shape
        if len(shape) != 3:
            return
        offset = [-dim / 2.0 for dim in shape]
        try:
            self._volume_item.translate(*offset)
        except Exception:
            pass

    def _reset_camera(self):
        if self._volume_scalar is None or self._volume_scalar.size == 0:
            return
        shape = self._volume_scalar.shape
        max_dim = float(max(shape)) if shape else 1.0
        distance = max(200.0, max_dim * 2.2)
        try:
            self.view.opts["center"] = pg.Vector(0.0, 0.0, 0.0)
        except Exception:
            try:
                self.view.opts["center"] = QtGui.QVector3D(0.0, 0.0, 0.0)
            except Exception:
                pass
        try:
            self.view.setCameraPosition(distance=distance, elevation=26, azimuth=32)
        except Exception:
            self.view.opts["distance"] = distance
        self.view.update()

    def _populate_colormap_choices(self):
        candidates = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "thermal",
            "blues",
            "reds",
        ]
        self.cmb_colormap.blockSignals(True)
        self.cmb_colormap.clear()
        for name in candidates:
            try:
                pg.colormap.get(name)
            except Exception:
                continue
            self.cmb_colormap.addItem(name.title(), name)
        if self.cmb_colormap.count() == 0:
            self.cmb_colormap.addItem("Viridis", "viridis")
        self.cmb_colormap.blockSignals(False)
        self._sync_colormap_combo()

    def _sync_colormap_combo(self):
        if not hasattr(self, "cmb_colormap"):
            return
        name = self._colormap_name or "viridis"
        block = self.cmb_colormap.blockSignals(True)
        idx = self.cmb_colormap.findData(name)
        if idx < 0 and self.cmb_colormap.count():
            idx = 0
            name = self.cmb_colormap.itemData(0)
            self._colormap_name = name
        if idx >= 0:
            self.cmb_colormap.setCurrentIndex(idx)
        self.cmb_colormap.blockSignals(block)

    def _on_colormap_combo_changed(self):
        name = self.cmb_colormap.currentData()
        if not name:
            name = "viridis"
        if name == self._colormap_name:
            return
        self.set_colormap(name)
        if hasattr(self._volume_item, "update"):
            try:
                self._volume_item.update()
            except Exception:
                pass

    def _remove_volume(self):
        if self._volume_item is None:
            return
        try:
            self.view.removeItem(self._volume_item)
        except Exception:
            pass
        self._volume_item = None

    def _update_alpha_controls(self):
        has_data = self._data is not None
        for widget in self._curve_widgets.values():
            widget.setEnabled(has_data)
        self.btn_reset_curve.setEnabled(has_data)
        self.btn_reset_view.setEnabled(has_data and self._volume_item is not None)
        if hasattr(self, "btn_export"):
            self.btn_export.setEnabled(has_data)

    def _export_volume_snapshot(self):
        if self._volume_item is None:
            QtWidgets.QMessageBox.information(
                self,
                "No volume",
                "Load data into the volume view before exporting.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save volume snapshot",
            self._initial_path("volume-snapshot.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = _ask_layout_label(self, "Snapshot label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        try:
            image = self.view.grabFramebuffer()
        except Exception:
            image = QtGui.QImage()
        if image.isNull():
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to capture the 3D view.")
            return
        if label:
            image = _image_with_label(image, label)
        if not image.save(str(target)):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the snapshot.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved volume snapshot to {target}")

    def _export_volume_layout(self):
        if self._volume_item is None:
            QtWidgets.QMessageBox.information(
                self,
                "No volume",
                "Load data into the volume view before exporting.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save volume layout",
            self._initial_path("volume-layout.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = _ask_layout_label(self, "Layout label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        if not _save_snapshot(self, target, label):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the layout image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved volume layout to {target}")

    def set_axis_labels(self, slice_label: str, row_label: str, column_label: str):
        self._axis_labels.update(
            {
                "slice": slice_label or "Slice axis",
                "row": row_label or "Row axis",
                "column": column_label or "Column axis",
                "value": "Value",
            }
        )
        for key, label_widget in self._curve_labels.items():
            label = self._axis_labels.get(key, key.title())
            if key == "value":
                text = label
            else:
                pretty = str(label)
                lower = pretty.lower()
                if lower.endswith("axis"):
                    text = pretty
                else:
                    text = f"{pretty} axis"
            label_widget.setText(text)

    def _store_curve(self, key: str, points: List[Tuple[float, float]]):
        xs = np.array([max(0.0, min(1.0, float(x))) for x, _ in points], dtype=float)
        ys = np.array([max(0.0, min(1.0, float(y))) for _, y in points], dtype=float)
        if xs.size == 0 or ys.size == 0:
            xs = np.array([0.0, 1.0], dtype=float)
            ys = np.array([0.0, 1.0], dtype=float)
        order = np.argsort(xs)
        xs = xs[order]
        ys = ys[order]
        normalized_points = [(float(x), float(y)) for x, y in zip(xs, ys)]
        previous = self._curve_points.get(key)
        self._curve_points[key] = normalized_points
        self._curve_lut_x[key] = xs
        self._curve_lut_y[key] = ys
        changed = previous is None or len(previous) != len(normalized_points) or any(
            abs(px - nx) > 1e-6 or abs(py - ny) > 1e-6
            for (px, py), (nx, ny) in zip(previous or [], normalized_points)
        )
        if changed:
            self._update_volume_visual()

    def _on_alpha_curve_changed(self, key: str, points: List[Tuple[float, float]]):
        if key not in self._curve_keys:
            key = "value"
        if not points:
            default_value = 0.25
            widget = self._curve_widgets.get(key)
            if widget is not None and hasattr(widget, "_default_value"):
                default_value = float(getattr(widget, "_default_value"))
            default_value = max(0.0, min(1.0, default_value))
            points = [(0.0, default_value), (1.0, default_value)]
        self._store_curve(key, points)

    def _on_reset_curve(self):
        for widget in self._curve_widgets.values():
            widget.reset_curve()
        for key, widget in self._curve_widgets.items():
            self._store_curve(key, widget.curve_points())

    def _on_reset_view(self):
        self._center_volume_item()
        self._reset_camera()

    def _sample_alpha_curve(self, key: str, values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, 0.0, 1.0)
        flat = clipped.reshape(-1)
        lut_x = self._curve_lut_x.get(key)
        lut_y = self._curve_lut_y.get(key)
        if lut_x is None or lut_y is None or lut_x.size < 2:
            mapped = flat
        else:
            mapped = np.interp(flat, lut_x, lut_y)
        return mapped.reshape(clipped.shape)

    def _apply_alpha_scale(self, alpha_norm: np.ndarray) -> np.ndarray:
        base = max(2.0, float(self._alpha_scale_base))
        clamped = np.clip(alpha_norm, 0.0, 1.0)
        scaled = (np.power(base, clamped) - 1.0) / (base - 1.0)
        return np.clip(scaled, 0.0, 1.0)

    def closeEvent(self, event: QtGui.QCloseEvent):
        try:
            self.closed.emit()
        except Exception:
            pass
        super().closeEvent(event)


# ---------------------------------------------------------------------------
# Sequential view: explore 2D slices along an arbitrary axis
# ---------------------------------------------------------------------------


class SequentialView(QtWidgets.QWidget):
    def __init__(
        self,
        processing_manager: Optional[ProcessingManager] = None,
        preferences: Optional[PreferencesManager] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.processing_manager = processing_manager
        self.preferences: Optional[PreferencesManager] = None

        self._dataset: Optional[xr.Dataset] = None
        self._dataset_path: Optional[Path] = None
        self._current_variable: Optional[str] = None
        self._current_da: Optional[xr.DataArray] = None
        self._dims: List[str] = []
        self._slice_axis: Optional[str] = None
        self._row_axis: Optional[str] = None
        self._col_axis: Optional[str] = None
        self._fixed_indices: Dict[str, int] = {}
        self._fixed_dim_widgets: Dict[str, QtWidgets.QSpinBox] = {}
        self._processing_mode: str = "none"
        self._processing_params: Dict[str, object] = {}
        self._slice_index: int = 0
        self._slice_count: int = 0
        self._axis_coords: Optional[np.ndarray] = None
        self._current_processed_slice: Optional[np.ndarray] = None
        self._roi_last_shape: Optional[Tuple[int, int]] = None
        self._roi_last_bounds: Optional[Tuple[int, int, int, int]] = None

        self._roi_enabled: bool = False
        self._roi_reducers = {
            "mean": ("Mean", _nan_aware_reducer(lambda arr, axis=None: np.nanmean(arr, axis=axis))),
            "median": ("Median", _nan_aware_reducer(lambda arr, axis=None: np.nanmedian(arr, axis=axis))),
            "min": ("Minimum", _nan_aware_reducer(lambda arr, axis=None: np.nanmin(arr, axis=axis))),
            "max": ("Maximum", _nan_aware_reducer(lambda arr, axis=None: np.nanmax(arr, axis=axis))),
            "std": ("Std. dev", _nan_aware_reducer(lambda arr, axis=None: np.nanstd(arr, axis=axis))),
            "ptp": (
                "Peak-to-peak",
                _nan_aware_reducer(
                    lambda arr, axis=None: np.nanmax(arr, axis=axis) - np.nanmin(arr, axis=axis)
                ),
            ),
        }
        self._roi_method_key: str = "mean"
        self._roi_last_slices: Optional[Tuple[slice, slice]] = None
        self._roi_axis_options: List[Tuple[str, Tuple[int, ...], str, str, Optional[int]]] = []
        self._roi_axes_selection: Tuple[int, ...] = (0, 1)
        self._roi_axis_index: int = 0
        self._current_slice_coords: Dict[str, np.ndarray] = {}
        self._row_coord_1d: Optional[np.ndarray] = None
        self._row_coord_2d: Optional[np.ndarray] = None
        self._col_coord_1d: Optional[np.ndarray] = None
        self._col_coord_2d: Optional[np.ndarray] = None
        self._volume_cache: Optional[np.ndarray] = None
        self._annotation_config: Optional[PlotAnnotationConfig] = None

        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(6)

        hint = QtWidgets.QLabel(
            "Drop a dataset here or use the Load button to explore sequential slices."
        )
        hint.setStyleSheet("color: #666;")
        outer.addWidget(hint)

        top = QtWidgets.QHBoxLayout()
        self.lbl_dataset = QtWidgets.QLabel("No dataset loaded")
        self.lbl_dataset.setStyleSheet("color: #555;")
        top.addWidget(self.lbl_dataset, 1)
        self.btn_load = QtWidgets.QPushButton("Load dataset…")
        self.btn_load.clicked.connect(self._load_dataset_dialog)
        top.addWidget(self.btn_load, 0)
        outer.addLayout(top)

        var_row = QtWidgets.QHBoxLayout()
        var_row.addWidget(QtWidgets.QLabel("Variable:"))
        self.cmb_variable = QtWidgets.QComboBox()
        self.cmb_variable.setEnabled(False)
        self.cmb_variable.currentIndexChanged.connect(self._on_variable_changed)
        var_row.addWidget(self.cmb_variable, 1)
        outer.addLayout(var_row)

        axis_group = QtWidgets.QGroupBox("Slice configuration")
        axis_form = QtWidgets.QFormLayout(axis_group)
        axis_form.setContentsMargins(6, 6, 6, 6)
        axis_form.setSpacing(6)

        self.cmb_slice_axis = QtWidgets.QComboBox()
        self.cmb_slice_axis.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow("Slice axis", self.cmb_slice_axis)

        self.cmb_row_axis = QtWidgets.QComboBox()
        self.cmb_row_axis.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow("Rows", self.cmb_row_axis)

        self.cmb_col_axis = QtWidgets.QComboBox()
        self.cmb_col_axis.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow("Columns", self.cmb_col_axis)

        self.fixed_dims_container = QtWidgets.QWidget()
        self.fixed_dims_layout = QtWidgets.QFormLayout(self.fixed_dims_container)
        self.fixed_dims_layout.setContentsMargins(0, 0, 0, 0)
        self.fixed_dims_layout.setSpacing(4)
        axis_form.addRow("Fixed indices", self.fixed_dims_container)

        outer.addWidget(axis_group)

        slider_row = QtWidgets.QHBoxLayout()
        self.lbl_slice = QtWidgets.QLabel("Slice: –")
        slider_row.addWidget(self.lbl_slice, 0)
        self.sld_slice = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_slice.setEnabled(False)
        self.sld_slice.valueChanged.connect(self._on_slice_changed)
        slider_row.addWidget(self.sld_slice, 1)
        self.spin_slice = QtWidgets.QSpinBox()
        self.spin_slice.setEnabled(False)
        self.spin_slice.valueChanged.connect(self._on_slice_spin_changed)
        slider_row.addWidget(self.spin_slice, 0)
        outer.addLayout(slider_row)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_apply_processing = QtWidgets.QPushButton("Apply processing…")
        self.btn_apply_processing.setEnabled(False)
        self.btn_apply_processing.clicked.connect(self._choose_processing)
        btn_row.addWidget(self.btn_apply_processing)

        self.btn_reset_processing = QtWidgets.QPushButton("Reset processing")
        self.btn_reset_processing.setEnabled(False)
        self.btn_reset_processing.clicked.connect(self._reset_processing)
        btn_row.addWidget(self.btn_reset_processing)

        self.btn_autoscale = QtWidgets.QPushButton("Autoscale colors")
        self.btn_autoscale.setEnabled(False)
        self.btn_autoscale.clicked.connect(self._on_autoscale_clicked)
        btn_row.addWidget(self.btn_autoscale)

        self.btn_autorange = QtWidgets.QPushButton("Auto view")
        self.btn_autorange.setEnabled(False)
        self.btn_autorange.clicked.connect(self._on_autorange_clicked)
        btn_row.addWidget(self.btn_autorange)

        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_menu = QtWidgets.QMenu(self.btn_export)
        act_current = export_menu.addAction("Save current slice…")
        act_current.triggered.connect(self._export_current_slice)
        act_all = export_menu.addAction("Save all slices to folder…")
        act_all.triggered.connect(self._export_all_slices)
        act_movie = export_menu.addAction("Save slice movie…")
        act_movie.triggered.connect(self._export_slice_movie)
        act_grid = export_menu.addAction("Save slices as grid pages…")
        act_grid.triggered.connect(self._export_slice_grid)
        export_menu.addSeparator()
        act_layout = export_menu.addAction("Save view layout…")
        act_layout.triggered.connect(self._export_sequential_layout)
        self.btn_export.setMenu(export_menu)
        btn_row.addWidget(self.btn_export)

        btn_row.addSpacing(12)

        cmap_label = QtWidgets.QLabel("Color map:")
        cmap_label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        btn_row.addWidget(cmap_label, 0)

        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setEnabled(False)
        self.cmb_colormap.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToContents)
        self.cmb_colormap.currentIndexChanged.connect(self._on_colormap_changed)
        btn_row.addWidget(self.cmb_colormap, 0)
        self._populate_colormap_choices()

        btn_row.addStretch(1)
        outer.addLayout(btn_row)

        self.viewer_split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        self.viewer_split.setChildrenCollapsible(False)
        self.viewer_split.setHandleWidth(6)
        outer.addWidget(self.viewer_split, 1)

        self.viewer = CentralPlotWidget(self)
        self.viewer_split.addWidget(self.viewer)
        hist = self.viewer.histogram_widget()
        if hist is not None and self.viewer_split.indexOf(hist) == -1:
            hist.setSizePolicy(
                QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Expanding)
            )
            hist.setMinimumWidth(140)
            hist.setMaximumWidth(180)
            self.viewer_split.addWidget(hist)
            try:
                self.viewer_split.setStretchFactor(0, 1)
                self.viewer_split.setStretchFactor(1, 0)
            except Exception:
                pass
            QtCore.QTimer.singleShot(0, lambda: self.viewer_split.setSizes([600, 150]))

        roi_row = QtWidgets.QHBoxLayout()
        self.btn_toggle_roi = QtWidgets.QPushButton("Enable ROI")
        self.btn_toggle_roi.setCheckable(True)
        self.btn_toggle_roi.setEnabled(False)
        self.btn_toggle_roi.toggled.connect(self._on_roi_toggled)
        roi_row.addWidget(self.btn_toggle_roi)

        self.btn_volume_view = QtWidgets.QPushButton("Open volume view…")
        self.btn_volume_view.setEnabled(False)
        self.btn_volume_view.clicked.connect(self._open_volume_view)
        if gl is None:
            self.btn_volume_view.setToolTip(
                "3D volume rendering requires the optional pyqtgraph.opengl module"
            )
        roi_row.addWidget(self.btn_volume_view)

        self.btn_annotations = QtWidgets.QPushButton("Set annotations…")
        self.btn_annotations.setEnabled(False)
        self.btn_annotations.clicked.connect(self._open_annotation_dialog)
        roi_row.addWidget(self.btn_annotations)

        self.lbl_roi_status = QtWidgets.QLabel("ROI disabled")
        self.lbl_roi_status.setStyleSheet("color: #666;")
        roi_row.addWidget(self.lbl_roi_status, 1)
        outer.addLayout(roi_row)

        self.roi = pg.RectROI([10, 10], [40, 40], pen=pg.mkPen('#ffaa00', width=2))
        self.roi.addScaleHandle((1, 1), (0, 0))
        self.roi.addScaleHandle((0, 0), (1, 1))
        self.roi.hide()
        self.roi.sigRegionChanged.connect(self._on_roi_region_changed)
        try:
            self.roi.sigRegionChangeFinished.connect(self._on_roi_region_changed)
        except Exception:
            pass

        self._roi_window: Optional[SequentialRoiWindow] = None
        self._volume_window: Optional[SequentialVolumeWindow] = None

        self.set_preferences(preferences)

    # ---------- dataset helpers ----------
    def _populate_colormap_choices(self):
        candidates = [
            "gray",
            "viridis",
            "plasma",
            "inferno",
            "magma",
            "cividis",
            "turbo",
            "thermal",
        ]
        self.cmb_colormap.blockSignals(True)
        self.cmb_colormap.clear()
        for name in candidates:
            try:
                pg.colormap.get(name)
            except Exception:
                continue
            self.cmb_colormap.addItem(name.title(), name)
        if self.cmb_colormap.count() == 0:
            self.cmb_colormap.addItem("Default", "default")
        self.cmb_colormap.blockSignals(False)
        if self.cmb_colormap.count():
            self.cmb_colormap.setCurrentIndex(0)
        if self.preferences is not None:
            preferred = self.preferences.preferred_colormap(None)
            if preferred:
                idx = self.cmb_colormap.findData(preferred)
                if idx >= 0:
                    self.cmb_colormap.setCurrentIndex(idx)

    def _on_colormap_changed(self):
        if not hasattr(self, "viewer"):
            return
        self._apply_selected_colormap()
        self._update_volume_window_colormap()

    def _apply_selected_colormap(self):
        if not hasattr(self, "viewer"):
            return
        name = self.cmb_colormap.currentData()
        if not name or name == "default":
            target = "viridis"
        else:
            target = str(name)
        try:
            cmap = pg.colormap.get(target)
        except Exception:
            return
        try:
            self.viewer.lut.gradient.setColorMap(cmap)
        except Exception:
            return
        try:
            self.viewer.lut.rehide_stops()
        except Exception:
            pass
        self._update_volume_window_colormap()

    def set_processing_manager(self, manager: Optional[ProcessingManager]):
        self.processing_manager = manager

    def dragEnterEvent(self, ev: QtGui.QDragEnterEvent):
        if ev.mimeData().hasText():
            ev.acceptProposedAction()
        else:
            ev.ignore()

    def dropEvent(self, ev: QtGui.QDropEvent):
        text = ev.mimeData().text()
        high_ref = HighDimVarRef.from_mime(text)
        if high_ref:
            try:
                dataset = high_ref.load_dataset()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
                ev.ignore()
                return
            label = high_ref.dataset_label()
            self._set_dataset(dataset, high_ref.path, label=label)
            index = self.cmb_variable.findData(high_ref.var)
            if index >= 0:
                self.cmb_variable.setCurrentIndex(index)
            ev.acceptProposedAction()
            return

        mem_ref = MemoryDatasetRef.from_mime(text)
        if mem_ref:
            try:
                dataset = mem_ref.load()
            except Exception as exc:
                QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
                ev.ignore()
                return
            self._set_dataset(dataset, None, label=mem_ref.display_name())
            ev.acceptProposedAction()
            return

        ref = DataSetRef.from_mime(text)
        if not ref:
            ev.ignore()
            return
        try:
            ds = ref.load()
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            ev.ignore()
            return
        self._set_dataset(ds, ref.path)
        ev.acceptProposedAction()

    def _load_dataset_dialog(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open dataset",
            "",
            "NetCDF / Zarr (*.nc *.zarr);;All files (*)",
        )
        if not path:
            return
        p = Path(path)
        try:
            ds = open_dataset(p)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(exc))
            return
        self._set_dataset(ds, p)

    def _clear_view(self):
        self._reset_current_state()
        self._clear_fixed_dim_widgets()
        self.cmb_variable.blockSignals(True)
        self.cmb_variable.clear()
        self.cmb_variable.blockSignals(False)
        for combo in (self.cmb_slice_axis, self.cmb_row_axis, self.cmb_col_axis):
            combo.blockSignals(True)
            combo.clear()
            combo.blockSignals(False)
            combo.setEnabled(False)
        self.cmb_variable.setEnabled(False)
        self._clear_display()

    def _clear_display(self):
        if self.roi.scene() is not None:
            try:
                self.viewer.plot.removeItem(self.roi)
            except Exception:
                pass
        self.roi.hide()
        self.btn_toggle_roi.blockSignals(True)
        self.btn_toggle_roi.setChecked(False)
        self.btn_toggle_roi.blockSignals(False)
        self.btn_toggle_roi.setEnabled(False)
        self.btn_volume_view.setEnabled(False)
        self.btn_annotations.setEnabled(False)
        if self._roi_window is not None:
            try:
                self._roi_window.hide()
                self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
                self._roi_window.update_profile([], [], "", "", False)
            except Exception:
                pass
        if self._volume_window is not None:
            try:
                self._volume_window.clear_volume()
                self._volume_window.hide()
            except Exception:
                pass
        self.lbl_roi_status.setText("ROI disabled")
        self._roi_enabled = False
        self._roi_last_slices = None
        self._roi_last_bounds = None
        self._roi_last_shape = None
        self._cache_slice_coords({})
        self._roi_axis_options = []
        self._roi_axes_selection = (0, 1)
        self._roi_axis_index = 0
        self._volume_cache = None
        self.viewer.set_image(np.zeros((1, 1)), autorange=True)
        self.sld_slice.blockSignals(True)
        self.spin_slice.blockSignals(True)
        self.sld_slice.setRange(0, 0)
        self.spin_slice.setRange(0, 0)
        self.sld_slice.setValue(0)
        self.spin_slice.setValue(0)
        self.sld_slice.blockSignals(False)
        self.spin_slice.blockSignals(False)
        self.sld_slice.setEnabled(False)
        self.spin_slice.setEnabled(False)
        self.btn_apply_processing.setEnabled(False)
        self.btn_reset_processing.setEnabled(False)
        self.btn_autoscale.setEnabled(False)
        self.btn_autorange.setEnabled(False)
        self.cmb_colormap.setEnabled(False)
        self.lbl_slice.setText("Slice: –")

    def _set_dataset(self, ds: xr.Dataset, path: Optional[Path], label: Optional[str] = None):
        if self._dataset is not None and self._dataset is not ds:
            try:
                self._dataset.close()
            except Exception:
                pass
        self._dataset = ds
        self._dataset_path = Path(path) if path else None
        if label:
            self.lbl_dataset.setText(label)
        elif self._dataset_path:
            self.lbl_dataset.setText(self._dataset_path.name)
        else:
            self.lbl_dataset.setText("(in-memory dataset)")
        self.lbl_dataset.setStyleSheet("")
        self._clear_view()
        vars_with_dims = [var for var in ds.data_vars if ds[var].ndim >= 3]
        if not vars_with_dims:
            self.cmb_variable.addItem("No 3D variables available", None)
            return
        self.cmb_variable.blockSignals(True)
        for var in vars_with_dims:
            dims = " × ".join(str(d) for d in ds[var].dims)
            self.cmb_variable.addItem(f"{var}  ({dims})", var)
        self.cmb_variable.setEnabled(True)
        self.cmb_variable.setCurrentIndex(0)
        self.cmb_variable.blockSignals(False)
        self._on_variable_changed()

    def _reset_current_state(self):
        self._current_variable = None
        self._current_da = None
        self._dims = []
        self._slice_axis = None
        self._row_axis = None
        self._col_axis = None
        self._fixed_indices = {}
        self._fixed_dim_widgets = {}
        self._processing_mode = "none"
        self._processing_params = {}
        self._slice_index = 0
        self._slice_count = 0
        self._axis_coords = None
        self._current_processed_slice = None
        self._roi_last_shape = None
        self._roi_last_slices = None
        self._roi_last_bounds = None
        self._roi_axis_options = []
        self._roi_axis_index = 0
        self._roi_axes_selection = (0, 1)
        self._volume_cache = None
        self._update_roi_window_options()

    def _clear_fixed_dim_widgets(self):
        while self.fixed_dims_layout.rowCount():
            self.fixed_dims_layout.removeRow(0)
        self._fixed_dim_widgets.clear()
        self._fixed_indices.clear()

    # ---------- configuration ----------
    def _on_variable_changed(self):
        if self._dataset is None:
            return
        self._reset_current_state()
        self._clear_fixed_dim_widgets()
        index = self.cmb_variable.currentIndex()
        var = self.cmb_variable.itemData(index)
        if not var:
            self._clear_display()
            return
        da = self._dataset[var]
        self._current_variable = var
        self._current_da = da
        self._dims = list(da.dims)
        self._processing_mode = "none"
        self._processing_params = {}
        self._slice_index = 0
        self._slice_count = 0
        self._axis_coords = None
        self._current_processed_slice = None
        self._rebuild_axis_controls()
        self._update_slice_widgets()
        self._update_slice_display(autorange=True)
        self._update_roi_axis_options()
        self.btn_annotations.setEnabled(True)
        self._apply_viewer_annotations()

    def _rebuild_axis_controls(self):
        dims = self._dims
        combos = (self.cmb_slice_axis, self.cmb_row_axis, self.cmb_col_axis)
        for combo in combos:
            combo.blockSignals(True)
            combo.clear()
            for dim in dims:
                combo.addItem(dim, dim)
            combo.blockSignals(False)
            combo.setEnabled(bool(dims))
        if len(dims) >= 3:
            self.cmb_slice_axis.setCurrentIndex(0)
            self.cmb_row_axis.setCurrentIndex(1)
            self.cmb_col_axis.setCurrentIndex(2)
        elif len(dims) >= 2:
            self.cmb_slice_axis.setCurrentIndex(0)
            self.cmb_row_axis.setCurrentIndex(1)
            self.cmb_col_axis.setCurrentIndex(0)
        self._ensure_unique_axes()
        self._rebuild_fixed_indices()

    def _ensure_unique_axes(self):
        dims = self._dims
        combos = (self.cmb_slice_axis, self.cmb_row_axis, self.cmb_col_axis)
        seen: List[str] = []
        for combo in combos:
            idx = combo.currentIndex()
            dim = combo.itemData(idx)
            if dim is None:
                continue
            if dim in seen:
                for alt in dims:
                    if alt not in seen:
                        block = combo.blockSignals(True)
                        combo.setCurrentIndex(combo.findData(alt))
                        combo.blockSignals(block)
                        dim = alt
                        break
            seen.append(dim)
        self._slice_axis = self.cmb_slice_axis.currentData()
        self._row_axis = self.cmb_row_axis.currentData()
        self._col_axis = self.cmb_col_axis.currentData()
        self._update_axis_coords()
        self._update_volume_window_axis_labels()

    def _update_axis_coords(self):
        axis = self._slice_axis
        if self._current_da is None or axis is None:
            self._axis_coords = None
            return
        coord = self._current_da.coords.get(axis)
        if coord is None:
            self._axis_coords = None
            return
        try:
            self._axis_coords = np.asarray(coord.values)
        except Exception:
            self._axis_coords = None

    def _axis_display_name(self, axis: Optional[str]) -> str:
        if not axis:
            return "axis"
        text = str(axis).replace("_", " ").strip()
        return text or "axis"

    def _cache_slice_coords(self, coords: Optional[Dict[str, np.ndarray]]):
        cache = dict(coords or {})
        self._current_slice_coords = cache

        def _extract(key: str, allowed_ndim: Tuple[int, ...]) -> Optional[np.ndarray]:
            arr = cache.get(key)
            if arr is None:
                return None
            try:
                arr = np.asarray(arr, float)
            except Exception:
                return None
            if arr.size == 0:
                return None
            if allowed_ndim and arr.ndim not in allowed_ndim:
                return None
            return arr

        def _first_valid(keys: Iterable[str], allowed_ndim: Tuple[int, ...]) -> Optional[np.ndarray]:
            for key in keys:
                if not key:
                    continue
                arr = _extract(key, allowed_ndim)
                if arr is not None:
                    return arr
            return None

        self._row_coord_1d = _first_valid(
            ("row_values", "y", self._row_axis or ""), (1,)
        )
        self._col_coord_1d = _first_valid(
            ("col_values", "x", self._col_axis or ""), (1,)
        )
        self._row_coord_2d = _first_valid(("row_grid", "Y"), (2,))
        self._col_coord_2d = _first_valid(("col_grid", "X"), (2,))

    def _column_coordinates(self, start: int, stop: int) -> Optional[np.ndarray]:
        if self._col_coord_1d is not None and self._col_coord_1d.size >= stop:
            return np.asarray(self._col_coord_1d[start:stop], float)
        if self._col_coord_2d is not None and self._col_coord_2d.shape[1] >= stop:
            subset = self._col_coord_2d[:, start:stop]
            with np.errstate(all="ignore"):
                vals = np.nanmean(subset, axis=0)
            return np.asarray(vals, float)
        return None

    def _row_coordinates(self, start: int, stop: int) -> Optional[np.ndarray]:
        if self._row_coord_1d is not None and self._row_coord_1d.size >= stop:
            return np.asarray(self._row_coord_1d[start:stop], float)
        if self._row_coord_2d is not None and self._row_coord_2d.shape[0] >= stop:
            subset = self._row_coord_2d[start:stop, :]
            with np.errstate(all="ignore"):
                vals = np.nanmean(subset, axis=1)
            return np.asarray(vals, float)
        return None

    def _roi_profile_coordinates(self, profile_axis: int, length: int) -> List[float]:
        if length <= 0:
            return []
        bounds = self._roi_last_bounds or self._roi_bounds_from_geometry()
        if bounds is None:
            start = 0
        else:
            y0, y1, x0, x1 = bounds
            start = x0 if profile_axis == 1 else y0
        stop = start + length
        coords = (
            self._column_coordinates(start, stop)
            if profile_axis == 1
            else self._row_coordinates(start, stop)
        )
        if coords is None or coords.size != length:
            coords = np.arange(start, start + length, dtype=float)
        return [float(v) if np.isfinite(v) else np.nan for v in np.asarray(coords, float)]

    def _rebuild_fixed_indices(self):
        self._clear_fixed_dim_widgets()
        if self._current_da is None:
            return
        for dim in self._current_da.dims:
            if dim in (self._slice_axis, self._row_axis, self._col_axis):
                continue
            size = int(self._current_da.sizes.get(dim, 1))
            spin = QtWidgets.QSpinBox()
            spin.setRange(0, max(0, size - 1))
            spin.setValue(0)
            spin.valueChanged.connect(partial(self._on_fixed_index_changed, dim))
            self.fixed_dims_layout.addRow(dim, spin)
            self._fixed_dim_widgets[dim] = spin
            self._fixed_indices[dim] = 0

    def _on_axes_changed(self):
        if not self._dims:
            return
        self._ensure_unique_axes()
        self._invalidate_volume_cache()
        self._slice_index = 0
        self._rebuild_fixed_indices()
        self._update_slice_widgets()
        self._update_slice_display(autorange=True)
        self._update_roi_axis_options()

    def _on_fixed_index_changed(self, dim: str, value: int):
        self._fixed_indices[dim] = int(value)
        self._invalidate_volume_cache()
        self._update_slice_display()

    def _update_slice_widgets(self):
        axis = self._slice_axis
        if not axis or self._current_da is None:
            self._clear_display()
            return
        size = int(self._current_da.sizes.get(axis, 0))
        self._slice_count = size
        self._slice_index = min(self._slice_index, max(0, size - 1))
        self.sld_slice.blockSignals(True)
        self.spin_slice.blockSignals(True)
        self.sld_slice.setRange(0, max(0, size - 1))
        self.spin_slice.setRange(0, max(0, size - 1))
        self.sld_slice.setValue(self._slice_index)
        self.spin_slice.setValue(self._slice_index)
        self.sld_slice.blockSignals(False)
        self.spin_slice.blockSignals(False)
        enabled = size > 0
        self.sld_slice.setEnabled(enabled)
        self.spin_slice.setEnabled(enabled)
        self.btn_apply_processing.setEnabled(enabled)
        self.btn_reset_processing.setEnabled(enabled)
        self.btn_autoscale.setEnabled(enabled)
        self.btn_autorange.setEnabled(enabled)
        self.btn_toggle_roi.setEnabled(enabled)
        self._update_volume_button_state()
        self._update_slice_label()

    def _update_slice_label(self):
        if not self._slice_axis:
            self.lbl_slice.setText("Slice: –")
            return
        coord_text = ""
        coords = self._axis_coords
        if coords is not None and 0 <= self._slice_index < coords.size:
            coord = coords[self._slice_index]
            coord_text = f" ({coord})"
        self.lbl_slice.setText(f"Slice: {self._slice_axis} = {self._slice_index}{coord_text}")

    def _update_volume_button_state(self):
        enabled = (
            gl is not None
            and self._current_da is not None
            and self._slice_axis is not None
            and self._row_axis is not None
            and self._col_axis is not None
            and self._slice_count > 0
        )
        self.btn_volume_view.setEnabled(enabled)

    # ---------- slice navigation ----------
    def _on_slice_changed(self, value: int):
        self._slice_index = int(value)
        block = self.spin_slice.blockSignals(True)
        self.spin_slice.setValue(self._slice_index)
        self.spin_slice.blockSignals(block)
        self._update_slice_label()
        self._update_slice_display()

    def _on_slice_spin_changed(self, value: int):
        self._slice_index = int(value)
        block = self.sld_slice.blockSignals(True)
        self.sld_slice.setValue(self._slice_index)
        self.sld_slice.blockSignals(block)
        self._update_slice_label()
        self._update_slice_display()

    def _gather_selection(self, slice_index: Optional[int] = None) -> Dict[str, int]:
        idx = dict(self._fixed_indices)
        axis = self._slice_axis
        if axis:
            idx[axis] = int(self._slice_index if slice_index is None else slice_index)
        return idx

    def _extract_slice(self, slice_index: Optional[int] = None) -> Tuple[Optional[np.ndarray], Dict[str, np.ndarray]]:
        if self._current_da is None or self._row_axis is None or self._col_axis is None:
            return None, {}
        if self._slice_axis is None:
            return None, {}
        select = self._gather_selection(slice_index)
        try:
            slice_da = self._current_da.isel(select)
        except Exception:
            return None, {}
        for dim in (self._row_axis, self._col_axis):
            if dim not in slice_da.dims:
                return None, {}
        try:
            slice_da = slice_da.transpose(self._row_axis, self._col_axis)
        except Exception:
            return None, {}
        data = np.asarray(slice_da.values, float)
        coords = guess_phys_coords(slice_da)
        try:
            row_coord = slice_da.coords.get(self._row_axis)
            if row_coord is not None:
                values = np.asarray(row_coord.values)
                if values.ndim == 1:
                    coords["row_values"] = np.asarray(values, float)
                elif values.ndim >= 2:
                    coords["row_grid"] = np.asarray(values, float)
        except Exception:
            pass
        try:
            col_coord = slice_da.coords.get(self._col_axis)
            if col_coord is not None:
                values = np.asarray(col_coord.values)
                if values.ndim == 1:
                    coords["col_values"] = np.asarray(values, float)
                elif values.ndim >= 2:
                    coords["col_grid"] = np.asarray(values, float)
        except Exception:
            pass
        return data, coords

    def _apply_processing(self, data: np.ndarray) -> np.ndarray:
        mode = self._processing_mode or "none"
        params = dict(self._processing_params or {})
        processed = np.asarray(data, float)
        if mode.startswith("pipeline:"):
            if not self.processing_manager:
                raise RuntimeError("No processing manager is available for pipelines.")
            name = mode.split(":", 1)[1]
            pipeline = self.processing_manager.get_pipeline(name)
            if pipeline is None:
                raise RuntimeError(f"Pipeline '{name}' is not available.")
            processed = pipeline.apply(processed)
        elif mode != "none":
            processed = apply_processing_step(mode, processed, params)
        return np.asarray(processed, float)

    def _update_slice_display(self, *, autorange: bool = False):
        data, coords = self._extract_slice()
        if data is None:
            self._current_processed_slice = None
            self._cache_slice_coords({})
            self.viewer.set_image(np.zeros((1, 1)), autorange=True)
            self._roi_last_shape = None
            if self._roi_enabled:
                self._update_roi_curve()
            self._refresh_volume_window()
            return
        try:
            processed = self._apply_processing(data)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Processing failed", str(exc))
            self._processing_mode = "none"
            self._processing_params = {}
            processed = np.asarray(data, float)
        self._current_processed_slice = np.asarray(processed, float)
        self._cache_slice_coords(coords)
        shape = self._current_processed_slice.shape
        if self._roi_enabled and shape != self._roi_last_shape:
            self._reset_roi_to_image(shape)
        self._roi_last_shape = shape
        if "X" in coords and "Y" in coords:
            self.viewer.set_warped(coords["X"], coords["Y"], processed, autorange=autorange)
        elif "x" in coords and "y" in coords:
            self.viewer.set_rectilinear(coords["x"], coords["y"], processed, autorange=autorange)
        else:
            self.viewer.set_image(processed, autorange=autorange)
        self.cmb_colormap.setEnabled(True)
        self._apply_preference_colormap()
        self._apply_selected_colormap()
        if autorange:
            try:
                self.viewer.autoscale_levels()
            except Exception:
                pass
            try:
                self.viewer.auto_view_range()
            except Exception:
                pass
        self._update_slice_label()
        if self._roi_enabled:
            self._update_roi_slice_reference()
            self._update_roi_curve()
        self._refresh_volume_window()
        self._apply_viewer_annotations()

    def _on_autoscale_clicked(self):
        self.viewer.autoscale_levels()

    def _on_autorange_clicked(self):
        self.viewer.auto_view_range()

    def _annotation_context(self) -> Dict[str, object]:
        idx = int(self._slice_index)
        coords = self._axis_coords
        value = None
        if coords is not None and 0 <= idx < coords.size:
            raw = coords[idx]
            if isinstance(raw, np.ndarray) and raw.size == 1:
                raw = raw.item()
            try:
                value = float(raw)
            except Exception:
                value = raw
        context: Dict[str, object] = {
            "slice_idx": idx,
            "slice_number": idx + 1,
            "n": idx,
            "slice_axis": self._slice_axis or "slice",
        }
        context["slice_val"] = value if value is not None else idx
        return context

    def _apply_viewer_annotations(self):
        if self._annotation_config is None:
            return
        try:
            context = self._annotation_context()
            self.viewer.apply_annotation(self._annotation_config, context=context)
        except Exception:
            pass

    def _open_annotation_dialog(self):
        initial = self.viewer.annotation_defaults()
        if self._annotation_config is not None:
            initial = replace(self._annotation_config, apply_to_all=False)
        hint = (
            "Tip: use Python f-string fields like {slice_idx}, {slice_number}, {slice_val:.3f}, and {slice_axis} "
            "to customise labels per slice."
        )
        dialog = PlotAnnotationDialog(
            self,
            initial=initial,
            allow_apply_all=False,
            template_hint=hint,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        config = dialog.annotation_config()
        if config is None:
            return
        config = replace(config, apply_to_all=False)
        self._annotation_config = config
        self._apply_viewer_annotations()

    # ---------- processing ----------
    def _choose_processing(self):
        dialog = ProcessingSelectionDialog(self.processing_manager, self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        mode, params = dialog.selected_processing()
        self._processing_mode = mode
        self._processing_params = dict(params)
        self._invalidate_volume_cache()
        self._update_slice_display()

    def _reset_processing(self):
        self._processing_mode = "none"
        self._processing_params = {}
        self._invalidate_volume_cache()
        self._update_slice_display(autorange=True)

    def _default_layout_label(self) -> str:
        if self.preferences:
            return self.preferences.default_layout_label()
        return ""

    def _default_export_dir(self) -> str:
        if self.preferences:
            return self.preferences.default_export_directory()
        return ""

    def _store_export_dir(self, directory: str):
        if self.preferences and directory:
            data = self.preferences.data()
            misc = data.setdefault("misc", {})
            misc["default_export_dir"] = directory
            self.preferences.update(data)

    def set_preferences(self, preferences: Optional[PreferencesManager]):
        if self.preferences is preferences:
            return
        if self.preferences:
            try:
                self.preferences.changed.disconnect(self._on_preferences_changed)
            except Exception:
                pass
        self.preferences = preferences
        if preferences is not None:
            try:
                preferences.changed.connect(self._on_preferences_changed)
            except Exception:
                pass
        self._apply_preference_colormap()
        self._apply_selected_colormap()
        if self._volume_window is not None:
            try:
                self._volume_window.set_preferences(preferences)
            except Exception:
                pass

    def _initial_path(self, filename: str) -> str:
        base = self._default_export_dir()
        if base:
            return str(Path(base) / filename)
        return filename

    def _apply_preference_colormap(self):
        if not self.preferences:
            return
        preferred = self.preferences.preferred_colormap(self._current_variable)
        if preferred:
            idx = self.cmb_colormap.findData(preferred)
            if idx >= 0 and idx != self.cmb_colormap.currentIndex():
                block = self.cmb_colormap.blockSignals(True)
                self.cmb_colormap.setCurrentIndex(idx)
                self.cmb_colormap.blockSignals(block)

    def _on_preferences_changed(self, _data):
        self._apply_preference_colormap()
        self._apply_selected_colormap()

    # ---------- export helpers ----------
    def _export_current_slice(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slice",
                "Load a dataset and choose a slice before exporting.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save current slice",
            self._initial_path("sequential-slice.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = _ask_layout_label(self, "Slice label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        _process_events()
        image = self.viewer.grab().toImage()
        if label:
            image = _image_with_label(image, label)
        if not image.save(str(target)):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the slice image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved sequential slice {self._slice_index} to {target}")

    def _export_all_slices(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slices",
                "Load a dataset with at least one slice before exporting.",
            )
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            self._default_export_dir(),
        )
        if not directory:
            return
        base = Path(directory)
        self._store_export_dir(directory)
        base_name = _sanitize_filename(self.lbl_dataset.text()) or "slice"
        original = self._slice_index
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            count = 0
            for idx in range(self._slice_count):
                self._on_slice_changed(idx)
                _process_events()
                image = self.viewer.grab().toImage()
                target = base / f"{base_name}_{idx:03d}.png"
                if image.save(str(target)):
                    count += 1
            if count == 0:
                QtWidgets.QMessageBox.warning(self, "Export failed", "No slices were exported.")
                return
        finally:
            if self._slice_count > 0:
                self._on_slice_changed(original)
            QtWidgets.QApplication.restoreOverrideCursor()
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {count} slice image(s) to {base}",
        )
        log_action(f"Exported {count} sequential slices to {base}")

    def _export_slice_movie(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slices",
                "Load a dataset with at least one slice before exporting a movie.",
            )
            return
        if cv2 is None:
            QtWidgets.QMessageBox.warning(
                self,
                "OpenCV unavailable",
                "Saving movies requires OpenCV (cv2). Install it and try again.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save slice movie",
            self._initial_path("sequential-slices.mp4"),
            "MP4 video (*.mp4);;AVI video (*.avi)",
        )
        if not path:
            return
        suffix = ".avi" if path.lower().endswith(".avi") else ".mp4"
        target = _ensure_extension(path, suffix)
        fps, ok = QtWidgets.QInputDialog.getDouble(
            self,
            "Frames per second",
            "Frames per second:",
            10.0,
            0.1,
            120.0,
            1,
        )
        if not ok:
            return
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if suffix == ".avi" else "mp4v"))
        original = self._slice_index
        writer = None
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for idx in range(self._slice_count):
                self._on_slice_changed(idx)
                _process_events()
                image = self.viewer.grab().toImage()
                frame = _qimage_to_array(image)
                if frame.size == 0:
                    continue
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                height, width = bgr.shape[:2]
                if writer is None:
                    writer = cv2.VideoWriter(str(target), fourcc, float(fps), (width, height))
                    if not writer.isOpened():
                        writer = None
                        raise RuntimeError("Unable to open video writer")
                if writer is not None:
                    writer.write(bgr)
        except Exception as exc:
            if writer is not None:
                writer.release()
            QtWidgets.QApplication.restoreOverrideCursor()
            self._on_slice_changed(original)
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))
            return
        finally:
            if writer is not None:
                writer.release()
            if self._slice_count > 0:
                self._on_slice_changed(original)
            QtWidgets.QApplication.restoreOverrideCursor()
        self._store_export_dir(str(Path(target).parent))
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved movie to {target}",
        )
        log_action(f"Saved sequential slice movie to {target}")

    def _export_slice_grid(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slices",
                "Load a dataset with at least one slice before exporting grids.",
            )
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            self._default_export_dir(),
        )
        if not directory:
            return
        rows, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Grid rows",
            "Number of rows per page:",
            2,
            1,
            10,
        )
        if not ok:
            return
        cols, ok = QtWidgets.QInputDialog.getInt(
            self,
            "Grid columns",
            "Number of columns per page:",
            2,
            1,
            10,
        )
        if not ok:
            return
        ok, label_prefix = _ask_layout_label(self, "Grid label")
        if not ok:
            return
        base = Path(directory)
        self._store_export_dir(directory)
        images: List[QtGui.QImage] = []
        original = self._slice_index
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            for idx in range(self._slice_count):
                self._on_slice_changed(idx)
                _process_events()
                images.append(self.viewer.grab().toImage())
        finally:
            if self._slice_count > 0:
                self._on_slice_changed(original)
            QtWidgets.QApplication.restoreOverrideCursor()
        if not images:
            QtWidgets.QMessageBox.warning(self, "Export failed", "No slice images were captured.")
            return
        tile_width = images[0].width()
        tile_height = images[0].height()
        per_page = max(1, rows * cols)
        count = 0
        for page_idx in range(0, len(images), per_page):
            subset = images[page_idx : page_idx + per_page]
            if not subset:
                continue
            page_image = QtGui.QImage(
                cols * tile_width,
                rows * tile_height,
                QtGui.QImage.Format_ARGB32,
            )
            page_image.fill(QtGui.QColor("white"))
            painter = QtGui.QPainter(page_image)
            for idx, img in enumerate(subset):
                r = idx // cols
                c = idx % cols
                painter.drawImage(c * tile_width, r * tile_height, img)
            painter.end()
            label = label_prefix
            if label_prefix:
                label = f"{label_prefix} – page {count + 1}"
            if label:
                page_image = _image_with_label(page_image, label)
            target = base / f"slice-grid_{count + 1:02d}.png"
            page_image.save(str(target))
            count += 1
        if count == 0:
            QtWidgets.QMessageBox.warning(self, "Export failed", "No grids were produced.")
            return
        QtWidgets.QMessageBox.information(
            self,
            "Export complete",
            f"Saved {count} grid page(s) to {base}",
        )
        log_action(f"Exported {count} sequential slice grids to {base}")

    def _export_sequential_layout(self):
        if self._slice_count <= 0:
            QtWidgets.QMessageBox.information(
                self,
                "No slices",
                "Load a dataset before exporting the layout.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save sequential view layout",
            self._initial_path("sequential-layout.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = _ask_layout_label(self, "Layout label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        if not _save_snapshot(self, target, label):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the layout image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved sequential view layout to {target}")

    # ---------- volume viewer ----------
    def _invalidate_volume_cache(self):
        self._volume_cache = None

    def _ensure_volume_window(self) -> SequentialVolumeWindow:
        if gl is None:
            raise RuntimeError("pyqtgraph.opengl is not available")
        window = self._volume_window
        if window is None:
            window = SequentialVolumeWindow(self, self.preferences)
            window.closed.connect(self._on_volume_window_closed)
            self._volume_window = window
        else:
            try:
                window.set_preferences(self.preferences)
            except Exception:
                pass
        return window

    def _on_volume_window_closed(self):
        self._volume_window = None

    def _open_volume_view(self):
        if gl is None:
            QtWidgets.QMessageBox.information(
                self,
                "Volume rendering unavailable",
                "3D volume rendering requires the optional pyqtgraph.opengl package.",
            )
            return
        volume = self._collect_volume_data()
        if volume is None:
            QtWidgets.QMessageBox.information(
                self,
                "Volume unavailable",
                "Unable to assemble a 3D volume with the current axis and index selection.",
            )
            return
        try:
            window = self._ensure_volume_window()
        except RuntimeError as exc:
            QtWidgets.QMessageBox.warning(self, "Volume rendering error", str(exc))
            return
        slice_label = self._axis_display_name(self._slice_axis)
        row_label = self._axis_display_name(self._row_axis)
        col_label = self._axis_display_name(self._col_axis)
        window.set_axis_labels(slice_label, row_label, col_label)
        window.set_volume(volume)
        cmap_name = self.cmb_colormap.currentData() or "viridis"
        window.set_colormap(cmap_name)
        window.show()
        window.raise_()
        window.activateWindow()

    def _collect_volume_data(self) -> Optional[np.ndarray]:
        if self._volume_cache is not None:
            return self._volume_cache
        if (
            self._current_da is None
            or self._slice_axis is None
            or self._row_axis is None
            or self._col_axis is None
        ):
            return None
        for axis in (self._slice_axis, self._row_axis, self._col_axis):
            if axis not in self._current_da.dims:
                return None
        select: Dict[str, int] = {}
        for dim in self._current_da.dims:
            if dim in (self._slice_axis, self._row_axis, self._col_axis):
                continue
            select[dim] = int(self._fixed_indices.get(dim, 0))
        try:
            subset = self._current_da.isel(select)
        except Exception:
            return None
        try:
            subset = subset.transpose(self._slice_axis, self._row_axis, self._col_axis)
        except Exception:
            return None
        data = np.asarray(subset.values, float)
        if data.ndim != 3:
            return None
        frames: List[np.ndarray] = []
        for idx in range(data.shape[0]):
            frame = np.asarray(data[idx], float)
            try:
                processed = self._apply_processing(frame)
            except Exception:
                processed = frame
            frames.append(np.asarray(processed, float))
        try:
            volume = np.stack(frames, axis=0)
        except Exception:
            return None
        self._volume_cache = volume
        return volume

    def _refresh_volume_window(self):
        if self._volume_window is None or not self._volume_window.isVisible():
            return
        volume = self._collect_volume_data()
        if volume is None:
            self._volume_window.clear_volume()
            return
        self._volume_window.set_volume(volume)
        cmap_name = self.cmb_colormap.currentData() or "viridis"
        self._volume_window.set_colormap(cmap_name)

    def _update_volume_window_colormap(self):
        if self._volume_window is None or not self._volume_window.isVisible():
            return
        cmap_name = self.cmb_colormap.currentData() or "viridis"
        self._volume_window.set_colormap(cmap_name)

    def _update_volume_window_axis_labels(self):
        if self._volume_window is None:
            return
        try:
            slice_label = self._axis_display_name(self._slice_axis)
            row_label = self._axis_display_name(self._row_axis)
            col_label = self._axis_display_name(self._col_axis)
            self._volume_window.set_axis_labels(slice_label, row_label, col_label)
        except Exception:
            pass

    # ---------- ROI controls ----------
    def _ensure_roi_window(self) -> SequentialRoiWindow:
        window = self._roi_window
        if window is None:
            window = SequentialRoiWindow(self)
            window.axesChanged.connect(self._on_roi_axes_changed_from_window)
            window.reducerChanged.connect(self._set_roi_method)
            window.closed.connect(self._on_roi_window_closed)
            self._roi_window = window
        self._update_roi_window_options()
        return window

    def _on_roi_window_closed(self):
        if self._roi_enabled:
            self.btn_toggle_roi.blockSignals(True)
            self.btn_toggle_roi.setChecked(False)
            self.btn_toggle_roi.blockSignals(False)
            self._on_roi_toggled(False)

    def _update_roi_axis_options(self):
        options: List[Tuple[str, Tuple[int, ...], str, str, Optional[int]]] = []
        if self._row_axis and self._col_axis:
            row_label = self._axis_display_name(self._row_axis)
            col_label = self._axis_display_name(self._col_axis)
            slice_label = self._axis_display_name(self._slice_axis)
            options.append(
                (
                    f"Reduce {row_label} & {col_label} → curve along {slice_label}",
                    (0, 1),
                    f"{row_label} & {col_label}",
                    slice_label,
                    None,
                )
            )
            options.append(
                (
                    f"Reduce {row_label} → profile across {col_label}",
                    (0,),
                    row_label,
                    col_label,
                    1,
                )
            )
            options.append(
                (
                    f"Reduce {col_label} → profile across {row_label}",
                    (1,),
                    col_label,
                    row_label,
                    0,
                )
            )
        self._roi_axis_options = options
        if not options:
            self._roi_axis_index = 0
            self._roi_axes_selection = (0, 1)
        else:
            self._roi_axis_index = max(0, min(self._roi_axis_index, len(options) - 1))
            self._roi_axes_selection = tuple(options[self._roi_axis_index][1])
        self._update_roi_window_options()

    def _on_roi_axes_changed_from_window(self, axes: Tuple[int, ...]):
        axes = tuple(int(a) for a in axes) if axes else (0, 1)
        matched = False
        for idx, option in enumerate(self._roi_axis_options):
            if tuple(option[1]) == axes:
                self._roi_axis_index = idx
                matched = True
                break
        if not matched and self._roi_axis_options:
            self._roi_axis_index = 0
            axes = tuple(self._roi_axis_options[0][1])
        self._roi_axes_selection = axes if axes else (0, 1)
        self._update_roi_window_hint()
        if self._roi_enabled:
            self._update_roi_curve()

    def _current_roi_option(self) -> Optional[Tuple[str, Tuple[int, ...], str, str, Optional[int]]]:
        if not self._roi_axis_options:
            return None
        idx = max(0, min(self._roi_axis_index, len(self._roi_axis_options) - 1))
        return self._roi_axis_options[idx]

    def _update_roi_window_options(self):
        if self._roi_window is None:
            return
        self._roi_window.set_axis_options(self._roi_axis_options, self._roi_axis_index)
        self._roi_window.set_reducer_options(self._roi_reducers, self._roi_method_key)
        self._update_roi_window_hint()

    def _update_roi_window_hint(self):
        if self._roi_window is None:
            return
        option = self._current_roi_option()
        if option is None:
            self._roi_window.set_hint("")
            return
        axes = option[1] if len(option) > 1 else ()
        collapsed = option[2] if len(option) > 2 else "region"
        remaining = option[3] if len(option) > 3 else ""
        axes = tuple(int(a) for a in axes)
        if set(axes) == {0, 1}:
            slice_label = remaining or self._axis_display_name(self._slice_axis)
            hint = f"Collapsing {collapsed} to track statistics along {slice_label}."
        elif axes == (0,):
            target = remaining or self._axis_display_name(self._col_axis)
            hint = f"Collapsing {collapsed} to profile across {target} for the active slice."
        elif axes == (1,):
            target = remaining or self._axis_display_name(self._row_axis)
            hint = f"Collapsing {collapsed} to profile across {target} for the active slice."
        else:
            hint = ""
        self._roi_window.set_hint(hint)

    def _current_roi_array(self) -> Optional[np.ndarray]:
        if self._current_processed_slice is None:
            return None
        return self._roi_extract_region(self._current_processed_slice)

    def _on_roi_toggled(self, checked: bool):
        checked = bool(checked)
        view = getattr(self.viewer, "plot", None)
        if checked and self._current_processed_slice is not None:
            if view is not None and self.roi.scene() is None:
                view.addItem(self.roi)
            self.roi.show()
            self._roi_enabled = True
            self.lbl_roi_status.setText(self._describe_roi())
            self._reset_roi_to_image(self._current_processed_slice.shape)
            self._update_roi_slice_reference()
            self._update_roi_axis_options()
            window = self._ensure_roi_window()
            window.show()
            window.raise_()
            self._update_roi_curve()
        else:
            if view is not None and self.roi.scene() is not None:
                try:
                    view.removeItem(self.roi)
                except Exception:
                    pass
            self.roi.hide()
            self._roi_enabled = False
            self._roi_last_slices = None
            self._roi_last_bounds = None
            self._roi_last_shape = None
            if self._roi_window is not None:
                try:
                    self._roi_window.hide()
                    self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
                    self._roi_window.update_profile([], [], "", "", False)
                except Exception:
                    pass
            self.lbl_roi_status.setText("ROI disabled")

    def _reset_roi_to_image(self, shape: Optional[Tuple[int, int]] = None):
        if not self._roi_enabled:
            return
        self._roi_last_bounds = None
        if shape is None:
            if self._current_processed_slice is None:
                return
            shape = self._current_processed_slice.shape
        if not shape or len(shape) < 2:
            return
        height, width = int(shape[0]), int(shape[1])
        if height <= 0 or width <= 0:
            return
        rect_w = max(2, width // 2)
        rect_h = max(2, height // 2)
        pos_x = max(0, (width - rect_w) // 2)
        pos_y = max(0, (height - rect_h) // 2)
        try:
            self.roi.blockSignals(True)
            self.roi.setPos((pos_x, pos_y))
            self.roi.setSize((rect_w, rect_h))
        finally:
            try:
                self.roi.blockSignals(False)
            except Exception:
                pass
        self._update_roi_slice_reference()

    def _on_roi_region_changed(self, *_args):
        if not self._roi_enabled:
            return
        self._update_roi_slice_reference()
        self._update_roi_curve()

    def _update_roi_slice_reference(self):
        if not self._roi_enabled or self._current_processed_slice is None:
            self._roi_last_slices = None
            self._roi_last_bounds = None
            return
        img_item = getattr(self.viewer, "img_item", None)
        if img_item is None:
            self._roi_last_slices = None
            self._roi_last_bounds = None
            return
        slices = None
        try:
            try:
                _, slc = self.roi.getArraySlice(
                    self._current_processed_slice,
                    img_item,
                    returnSlice=True,
                )
            except TypeError:
                _, slc = self.roi.getArraySlice(self._current_processed_slice, img_item)
            if isinstance(slc, tuple):
                slices = slc
        except Exception:
            slices = None

        if (
            isinstance(slices, tuple)
            and len(slices) >= 2
            and all(isinstance(s, slice) for s in slices[:2])
        ):
            sy, sx = slices[0], slices[1]
            self._roi_last_slices = (sy, sx)
            self._roi_last_bounds = self._normalize_roi_bounds(sy, sx)
        else:
            self._roi_last_slices = None
            self._roi_last_bounds = self._roi_bounds_from_geometry()

    def _normalize_roi_bounds(self, sy: slice, sx: slice) -> Optional[Tuple[int, int, int, int]]:
        if self._current_processed_slice is None:
            return None
        height, width = self._current_processed_slice.shape[:2]

        def _bounds(sl: slice, limit: int) -> Tuple[int, int]:
            start = float(sl.start) if sl.start is not None else 0.0
            stop = float(sl.stop) if sl.stop is not None else float(limit)
            step = sl.step
            if step is not None and step < 0:
                start, stop = stop, start
            a = int(np.floor(start))
            b = int(np.ceil(stop))
            a = max(0, min(limit, a))
            b = max(a, min(limit, b))
            return a, b

        y0, y1 = _bounds(sy, height)
        x0, x1 = _bounds(sx, width)
        if y1 <= y0 or x1 <= x0:
            return None
        return (y0, y1, x0, x1)

    def _roi_bounds_from_geometry(self) -> Optional[Tuple[int, int, int, int]]:
        if self._current_processed_slice is None:
            return None
        img_item = getattr(self.viewer, "img_item", None)
        if img_item is None:
            return None
        try:
            rect = self.roi.boundingRect()
            top_left_scene = self.roi.mapToScene(rect.topLeft())
            bottom_right_scene = self.roi.mapToScene(rect.bottomRight())
            top_left_item = img_item.mapFromScene(top_left_scene)
            bottom_right_item = img_item.mapFromScene(bottom_right_scene)
        except Exception:
            return None

        xs = [float(top_left_item.x()), float(bottom_right_item.x())]
        ys = [float(top_left_item.y()), float(bottom_right_item.y())]
        x0 = int(np.floor(min(xs)))
        x1 = int(np.ceil(max(xs)))
        y0 = int(np.floor(min(ys)))
        y1 = int(np.ceil(max(ys)))
        height, width = self._current_processed_slice.shape[:2]
        x0 = max(0, min(width, x0))
        x1 = max(x0, min(width, x1))
        y0 = max(0, min(height, y0))
        y1 = max(y0, min(height, y1))
        if y1 <= y0 or x1 <= x0:
            return None
        return (y0, y1, x0, x1)

    def _roi_extract_region(self, data: np.ndarray) -> Optional[np.ndarray]:
        arr = np.asarray(data, float)
        bounds = self._roi_last_bounds
        if bounds is None:
            bounds = self._roi_bounds_from_geometry()
            if bounds is None:
                return None
            self._roi_last_bounds = bounds
        if arr.ndim < 2:
            return arr
        height, width = arr.shape[:2]
        y0, y1, x0, x1 = bounds
        y0 = max(0, min(height, y0))
        y1 = max(y0, min(height, y1))
        x0 = max(0, min(width, x0))
        x1 = max(x0, min(width, x1))
        if y1 <= y0 or x1 <= x0:
            return None
        region = arr[y0:y1, x0:x1]
        if region.size == 0:
            return None
        return np.asarray(region, float)

    def _compute_roi_value(self, data: np.ndarray) -> float:
        reducer_entry = self._roi_reducers.get(self._roi_method_key)
        if reducer_entry is None:
            return float("nan")
        _, reducer = reducer_entry
        roi_data = self._roi_extract_region(data)
        if roi_data is None:
            roi_data = np.asarray(data, float)
        axes = tuple(int(a) for a in self._roi_axes_selection) or (0, 1)
        axes = tuple(sorted(set(axes)))
        with np.errstate(all="ignore"):
            if not axes:
                result = reducer(roi_data, axis=None)
            else:
                axis_param = axes[0] if len(axes) == 1 else axes
                result = reducer(roi_data, axis=axis_param)
            while isinstance(result, np.ndarray) and result.ndim > 0:
                if result.ndim == 1:
                    result = reducer(result, axis=0)
                else:
                    result = reducer(result, axis=tuple(range(result.ndim)))
        try:
            return float(np.asarray(result).item())
        except Exception:
            try:
                return float(result)
            except Exception:
                return float("nan")

    def _update_roi_curve(self):
        if self._roi_window is not None and not self._roi_window.isVisible():
            self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
            self._roi_window.update_profile([], [], "", "", False)
        if not self._roi_enabled or self._current_da is None:
            if self._roi_window is not None:
                self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
                self._roi_window.update_profile([], [], "", "", False)
            return
        count = max(0, self._slice_count)
        if count == 0:
            if self._roi_window is not None:
                self._roi_window.update_slice_curve([], [], "Slice coordinate", "ROI statistic")
                self._roi_window.update_profile([], [], "", "", False)
            return
        self._update_roi_slice_reference()
        values: List[float] = []
        xs: List[float] = []
        coords = self._axis_coords
        for idx in range(count):
            data, _ = self._extract_slice(slice_index=idx)
            if data is None:
                values.append(np.nan)
            else:
                try:
                    processed = self._apply_processing(data)
                except Exception:
                    processed = np.asarray(data, float)
                values.append(self._compute_roi_value(processed))
            if coords is not None and idx < coords.size:
                xs.append(float(coords[idx]))
            else:
                xs.append(float(idx))
        name = self._roi_reducers.get(self._roi_method_key, ("ROI statistic",))[0]
        axis_label = self._axis_display_name(self._slice_axis)
        if self._roi_window is not None:
            self._roi_window.update_slice_curve(xs, values, axis_label, name)
            self._update_roi_profile_plot()
        self.lbl_roi_status.setText(self._describe_roi())

    def _update_roi_profile_plot(self):
        if self._roi_window is None or not self._roi_enabled:
            return
        option = self._current_roi_option()
        if option is None:
            self._roi_window.update_profile([], [], "", "", False)
            return
        axes = option[1] if len(option) > 1 else ()
        remaining = option[3] if len(option) > 3 else ""
        profile_axis = option[4] if len(option) > 4 else None
        axes = tuple(int(a) for a in axes)
        if set(axes) == {0, 1} or profile_axis is None:
            self._roi_window.update_profile([], [], "", "", False)
            return
        data = self._current_roi_array()
        reducer_entry = self._roi_reducers.get(self._roi_method_key)
        if data is None or reducer_entry is None:
            self._roi_window.update_profile([], [], "", "", False)
            return
        _, reducer = reducer_entry
        axis = axes[0] if axes else 0
        with np.errstate(all="ignore"):
            profile = reducer(data, axis=axis)
        try:
            prof_arr = np.asarray(profile, float)
        except Exception:
            self._roi_window.update_profile([], [], "", "", False)
            return
        if prof_arr.ndim > 1:
            prof_arr = np.asarray(prof_arr).ravel()
        xs = list(range(int(prof_arr.size)))
        ys = [float(val) if np.isfinite(val) else np.nan for val in prof_arr]
        coords = self._roi_profile_coordinates(int(profile_axis), len(xs))
        if len(coords) == len(xs):
            xs = coords
        xlabel = remaining or (
            self._axis_display_name(self._col_axis) if profile_axis == 1 else self._axis_display_name(self._row_axis)
        )
        ylabel = self._roi_reducers.get(self._roi_method_key, ("Value",))[0]
        self._roi_window.update_profile(xs, ys, xlabel, ylabel, True)

    def _describe_roi(self) -> str:
        name = self._roi_reducers.get(self._roi_method_key, ("statistic",))[0]
        option = self._current_roi_option()
        collapsed = option[2] if option else "region"
        axis = option[3] if option and len(option) > 3 else self._axis_display_name(self._slice_axis)
        axis = axis or self._axis_display_name(self._slice_axis)
        return f"ROI {name.lower()} of {collapsed} across {axis}"

    def _set_roi_method(self, key: str):
        if key not in self._roi_reducers or key == self._roi_method_key:
            return
        self._roi_method_key = key
        self._update_roi_window_options()
        if self._roi_enabled:
            self._update_roi_curve()
        self.lbl_roi_status.setText(self._describe_roi())

# ---------------------------------------------------------------------------
# Overlay view: stack multiple layers with per-layer controls
# ---------------------------------------------------------------------------
class OverlayLayer(QtCore.QObject):
    def __init__(self, view: "OverlayView", title: str, data: np.ndarray, rect: QtCore.QRectF | None):
        super().__init__(view)
        self.view = view
        self.title = title
        self.base_data = np.asarray(data, float)
        self.processed_data = np.array(self.base_data, copy=True)
        self.rect = rect
        self.image_item = pg.ImageItem()
        self.image_item.setImage(self.processed_data, autoLevels=False)
        if rect is not None:
            try:
                self.image_item.setRect(rect)
            except Exception:
                pass
        self.image_item.setOpacity(1.0)
        self.image_item.setVisible(True)
        self._levels = self._compute_levels(self.processed_data)
        try:
            self.image_item.setLevels(self._levels)
        except Exception:
            pass
        self.colormap_name = "viridis"
        self.visible = True
        self.opacity = 1.0
        self.current_processing = "none"
        self.processing_params: dict = {}
        self.widget: Optional["OverlayLayerWidget"] = None
        self.set_colormap(self.colormap_name)

    # ---------- helpers ----------
    def _compute_levels(self, data: np.ndarray) -> Tuple[float, float]:
        data = np.asarray(data, float)
        finite = np.isfinite(data)
        if not finite.any():
            return (0.0, 1.0)
        vals = data[finite]
        try:
            lo = float(np.nanmin(vals))
            hi = float(np.nanmax(vals))
        except Exception:
            lo, hi = 0.0, 1.0
        if not np.isfinite(lo):
            lo = 0.0
        if not np.isfinite(hi):
            hi = 1.0
        if hi == lo:
            hi = lo + 1.0
        return (lo, hi)

    def set_widget(self, widget: "OverlayLayerWidget"):
        self.widget = widget
        widget.update_from_layer()

    # ---------- layer controls ----------
    def set_visible(self, on: bool):
        self.visible = bool(on)
        try:
            self.image_item.setVisible(self.visible)
        except Exception:
            pass

    def set_opacity(self, alpha: float):
        alpha = float(np.clip(alpha, 0.0, 1.0))
        self.opacity = alpha
        try:
            self.image_item.setOpacity(alpha)
        except Exception:
            pass
        if self.widget:
            self.widget.update_opacity_label(alpha)

    def set_colormap(self, name: str):
        self.colormap_name = name or "viridis"
        try:
            cmap = pg.colormap.get(self.colormap_name)
            if hasattr(cmap, "getLookupTable"):
                lut = cmap.getLookupTable(0.0, 1.0, 256)
                self.image_item.setLookupTable(lut)
        except Exception:
            pass

    def set_levels(self, lo: float, hi: float, *, update_widget: bool = True):
        if not np.isfinite(lo) or not np.isfinite(hi):
            return
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-6)
        self._levels = (float(lo), float(hi))
        try:
            self.image_item.setLevels(self._levels)
        except Exception:
            pass
        if update_widget and self.widget:
            self.widget.update_level_spins(self._levels[0], self._levels[1])

    def auto_levels(self):
        lo, hi = self._compute_levels(self.processed_data)
        self.set_levels(lo, hi, update_widget=True)

    def apply_processing(self, mode: str, params: dict):
        data = np.asarray(self.base_data, float)
        mode = mode or "none"
        params = dict(params or {})

        if mode.startswith("pipeline:"):
            name = mode.split(":", 1)[1]
            manager = getattr(self.view, "processing_manager", None)
            pipeline = manager.get_pipeline(name) if manager else None
            if pipeline is None:
                QtWidgets.QMessageBox.warning(self.view, "Processing failed", f"Pipeline '{name}' is not available.")
                return
            try:
                data = pipeline.apply(data)
            except Exception as e:
                QtWidgets.QMessageBox.warning(self.view, "Processing failed", str(e))
                return
            params = {}
        elif mode != "none":
            try:
                data = apply_processing_step(mode, data, params)
            except KeyError:
                QtWidgets.QMessageBox.warning(self.view, "Processing failed", f"Unknown processing mode: {mode}")
                return
            except Exception as e:
                QtWidgets.QMessageBox.warning(self.view, "Processing failed", str(e))
                return

        self.current_processing = mode
        self.processing_params = dict(params)
        self.processed_data = np.asarray(data, float)
        try:
            self.image_item.setImage(self.processed_data, autoLevels=False)
        except Exception:
            pass
        self.auto_levels()

    def get_display_rect(self):
        rect = self.rect
        if rect is None:
            try:
                rect = self.image_item.mapRectToParent(self.image_item.boundingRect())
            except Exception:
                rect = None
        return rect


class OverlayLayerWidget(QtWidgets.QGroupBox):
    def __init__(self, view: "OverlayView", layer: OverlayLayer):
        super().__init__(layer.title)
        self.view = view
        self.layer = layer
        self._ready = False

        self.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        )

        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(8, 6, 8, 6)
        lay.setSpacing(6)

        # Visibility / remove
        header = QtWidgets.QHBoxLayout()
        self.chk_visible = QtWidgets.QCheckBox("Visible")
        self.chk_visible.setChecked(True)
        self.chk_visible.toggled.connect(self._on_visibility)
        header.addWidget(self.chk_visible)
        header.addStretch(1)
        self.btn_remove = QtWidgets.QToolButton()
        self.btn_remove.setText("✕")
        self.btn_remove.setToolTip("Remove layer")
        self.btn_remove.clicked.connect(self._on_remove)
        header.addWidget(self.btn_remove)
        lay.addLayout(header)

        # Colormap selection
        cmap_row = QtWidgets.QHBoxLayout()
        cmap_row.addWidget(QtWidgets.QLabel("Colormap:"))
        self.cmb_colormap = QtWidgets.QComboBox()
        self.cmb_colormap.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_colormap.setMinimumContentsLength(12)
        self.cmb_colormap.setMinimumWidth(200)
        try:
            cmaps = sorted(pg.colormap.listMaps())
        except Exception:
            cmaps = ["viridis", "plasma", "magma", "cividis", "gray"]
        for name in cmaps:
            self.cmb_colormap.addItem(name)
        self.cmb_colormap.currentTextChanged.connect(self._on_colormap)
        cmap_row.addWidget(self.cmb_colormap, 1)
        lay.addLayout(cmap_row)

        # Levels controls
        lvl_row = QtWidgets.QHBoxLayout()
        lvl_row.addWidget(QtWidgets.QLabel("Levels:"))
        self.spin_min = QtWidgets.QDoubleSpinBox()
        self.spin_min.setDecimals(6)
        self.spin_min.setRange(-1e12, 1e12)
        self.spin_min.valueChanged.connect(self._on_levels_changed)
        lvl_row.addWidget(self.spin_min)
        lvl_row.addWidget(QtWidgets.QLabel("→"))
        self.spin_max = QtWidgets.QDoubleSpinBox()
        self.spin_max.setDecimals(6)
        self.spin_max.setRange(-1e12, 1e12)
        self.spin_max.valueChanged.connect(self._on_levels_changed)
        lvl_row.addWidget(self.spin_max)
        self.btn_autoscale = QtWidgets.QPushButton("Auto")
        self.btn_autoscale.clicked.connect(self._on_autoscale)
        lvl_row.addWidget(self.btn_autoscale)
        lay.addLayout(lvl_row)

        # Opacity slider
        opacity_row = QtWidgets.QHBoxLayout()
        opacity_row.addWidget(QtWidgets.QLabel("Opacity:"))
        self.sld_opacity = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.sld_opacity.setRange(0, 100)
        self.sld_opacity.setValue(100)
        self.sld_opacity.valueChanged.connect(self._on_opacity)
        opacity_row.addWidget(self.sld_opacity, 1)
        self.lbl_opacity = QtWidgets.QLabel("100%")
        opacity_row.addWidget(self.lbl_opacity)
        lay.addLayout(opacity_row)

        # Processing controls
        manager_ref = getattr(view, "processing_manager", None)
        self.manager: Optional[ProcessingManager] = None
        proc_box = QtWidgets.QGroupBox("Processing")
        proc_box.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        )
        proc_layout = QtWidgets.QVBoxLayout(proc_box)
        proc_layout.setContentsMargins(6, 6, 6, 6)
        proc_layout.setSpacing(4)

        proc_row = QtWidgets.QHBoxLayout()
        proc_row.addWidget(QtWidgets.QLabel("Operation:"))
        self.cmb_processing = QtWidgets.QComboBox()
        self.cmb_processing.setSizeAdjustPolicy(QtWidgets.QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.cmb_processing.setMinimumContentsLength(14)
        self.cmb_processing.setMinimumWidth(220)
        self.cmb_processing.currentIndexChanged.connect(self._on_processing_mode_changed)
        proc_row.addWidget(self.cmb_processing, 1)
        proc_layout.addLayout(proc_row)

        self.param_stack = QtWidgets.QStackedWidget()
        self.param_stack.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        )
        self.param_stack.addWidget(QtWidgets.QWidget())  # None placeholder
        self._function_forms: Dict[str, ParameterForm] = {}
        self._function_indices: Dict[str, int] = {}
        for spec in list_processing_functions():
            form = ParameterForm(spec.parameters)
            form.parametersChanged.connect(self._on_processing_params_changed)
            idx = self.param_stack.addWidget(form)
            self._function_forms[spec.key] = form
            self._function_indices[spec.key] = idx
        summary_container = QtWidgets.QWidget()
        summary_layout = QtWidgets.QVBoxLayout(summary_container)
        summary_layout.setContentsMargins(0, 0, 0, 0)
        summary_layout.setSpacing(0)
        self.lbl_pipeline_summary = QtWidgets.QLabel("Select a pipeline to view steps.")
        self.lbl_pipeline_summary.setWordWrap(True)
        self.lbl_pipeline_summary.setStyleSheet("color: #666;")
        summary_layout.addWidget(self.lbl_pipeline_summary)
        self._pipeline_summary_index = self.param_stack.addWidget(summary_container)

        proc_layout.addWidget(self.param_stack)
        self.btn_apply = QtWidgets.QPushButton("Apply")
        self.btn_apply.clicked.connect(self._apply_processing)
        proc_layout.addWidget(self.btn_apply, alignment=QtCore.Qt.AlignRight)
        lay.addWidget(proc_box)
        self.set_processing_manager(manager_ref)

        self._ready = True
        self._on_processing_mode_changed()

    # ---------- UI helpers ----------
    def update_from_layer(self):
        self._ready = False
        self.setTitle(self.layer.title)
        self.chk_visible.setChecked(self.layer.visible)
        self._set_colormap_selection(self.layer.colormap_name)
        lo, hi = getattr(self.layer, "_levels", (0.0, 1.0))
        self.update_level_spins(lo, hi)
        self.update_opacity_label(self.layer.opacity)
        self.sld_opacity.setValue(int(round(self.layer.opacity * 100)))
        current_mode = self.layer.current_processing or "none"
        self._refresh_processing_options()
        self._select_processing_mode(current_mode)
        if current_mode.startswith("pipeline:"):
            name = current_mode.split(":", 1)[1]
            self._update_pipeline_summary(name)
        else:
            form = self._function_forms.get(current_mode)
            if form:
                form.set_values(self.layer.processing_params)
        self._ready = True
        self._apply_processing()

    def _set_colormap_selection(self, name: str):
        idx = self.cmb_colormap.findText(name, QtCore.Qt.MatchFixedString)
        if idx < 0:
            idx = self.cmb_colormap.findText("viridis", QtCore.Qt.MatchFixedString)
        if idx >= 0:
            block = self.cmb_colormap.blockSignals(True)
            self.cmb_colormap.setCurrentIndex(idx)
            self.cmb_colormap.blockSignals(block)

    def update_level_spins(self, lo: float, hi: float):
        block = self.spin_min.blockSignals(True)
        self.spin_min.setValue(float(lo))
        self.spin_min.blockSignals(block)
        block = self.spin_max.blockSignals(True)
        self.spin_max.setValue(float(hi))
        self.spin_max.blockSignals(block)

    def update_opacity_label(self, alpha: float):
        pct = int(round(float(alpha) * 100))
        self.lbl_opacity.setText(f"{pct}%")

    def processing_parameters(self) -> dict:
        data = self._current_processing_data()
        if data.get("type") == "function":
            form = self._function_forms.get(data.get("key", ""))
            if form:
                return form.values()
        return {}

    def current_processing(self) -> str:
        data = self._current_processing_data()
        if data.get("type") == "function":
            return data.get("key", "none")
        if data.get("type") == "pipeline":
            return f"pipeline:{data.get('name', '')}"
        return "none"

    def _current_processing_data(self) -> Dict[str, object]:
        data = self.cmb_processing.currentData()
        if isinstance(data, dict):
            return data
        return {"type": "none"}

    def _find_data_index(self, target: Dict[str, object]) -> int:
        for idx in range(self.cmb_processing.count()):
            data = self.cmb_processing.itemData(idx)
            if isinstance(data, dict) and data.get("type") == target.get("type"):
                if data.get("type") == "function" and data.get("key") == target.get("key"):
                    return idx
                if data.get("type") == "pipeline" and data.get("name") == target.get("name"):
                    return idx
                if data.get("type") == "none":
                    return idx
        return -1

    def _refresh_processing_options(self):
        current = self._current_processing_data()
        block = self.cmb_processing.blockSignals(True)
        self.cmb_processing.clear()
        self.cmb_processing.addItem("None", {"type": "none"})
        for spec in list_processing_functions():
            self.cmb_processing.addItem(spec.label, {"type": "function", "key": spec.key})
        if self.manager:
            for name in self.manager.pipeline_names():
                self.cmb_processing.addItem(f"Pipeline: {name}", {"type": "pipeline", "name": name})
        idx = self._find_data_index(current)
        if idx < 0:
            idx = 0
        self.cmb_processing.setCurrentIndex(idx)
        self.cmb_processing.blockSignals(block)
        self._on_processing_mode_changed()

    def _update_pipeline_summary(self, name: str):
        if not self.manager:
            self.lbl_pipeline_summary.setText("No processing manager available.")
            return
        pipeline = self.manager.get_pipeline(name)
        if pipeline is None:
            self.lbl_pipeline_summary.setText(f"Pipeline '{name}' not found.")
            return
        lines = []
        for i, step in enumerate(pipeline.steps, start=1):
            spec = get_processing_function(step.key)
            label = spec.label if spec else step.key
            params = summarize_parameters(step.key, step.params)
            if params:
                lines.append(f"{i}. {label} ({params})")
            else:
                lines.append(f"{i}. {label}")
        self.lbl_pipeline_summary.setText("\n".join(lines))

    def set_processing_manager(self, manager: Optional[ProcessingManager]):
        if getattr(self, "manager", None) is manager:
            return
        if getattr(self, "manager", None):
            try:
                self.manager.pipelines_changed.disconnect(self._on_pipelines_changed)
            except Exception:
                pass
        self.manager = manager
        if manager:
            manager.pipelines_changed.connect(self._on_pipelines_changed)
        self._refresh_processing_options()

    def _on_pipelines_changed(self):
        self._refresh_processing_options()

    # ---------- Slots ----------
    def _on_visibility(self, on: bool):
        self.layer.set_visible(on)
        if on:
            self.view.auto_view_range()

    def _on_remove(self):
        self.view.remove_layer(self.layer)

    def _on_colormap(self, name: str):
        if not self._ready:
            return
        self.layer.set_colormap(name)

    def _on_levels_changed(self):
        if not self._ready:
            return
        lo = self.spin_min.value()
        hi = self.spin_max.value()
        if hi <= lo:
            hi = lo + max(abs(lo) * 1e-6, 1e-6)
            block = self.spin_max.blockSignals(True)
            self.spin_max.setValue(hi)
            self.spin_max.blockSignals(block)
        self.layer.set_levels(lo, hi, update_widget=False)

    def _on_autoscale(self):
        self.layer.auto_levels()

    def _on_opacity(self, value: int):
        alpha = float(value) / 100.0
        self.layer.set_opacity(alpha)

    def _on_processing_mode_changed(self):
        data = self._current_processing_data()
        if data.get("type") == "function":
            idx = self._function_indices.get(data.get("key", ""), 0)
            try:
                self.param_stack.setCurrentIndex(idx)
            except Exception:
                pass
        elif data.get("type") == "pipeline":
            self._update_pipeline_summary(str(data.get("name", "")))
            try:
                self.param_stack.setCurrentIndex(self._pipeline_summary_index)
            except Exception:
                pass
        else:
            try:
                self.param_stack.setCurrentIndex(0)
            except Exception:
                pass
        self._apply_processing()

    def _on_processing_params_changed(self, *_):
        self._apply_processing()

    def _apply_processing(self):
        if not self._ready:
            return
        mode = self.current_processing()
        params = self.processing_parameters()
        self.layer.apply_processing(mode, params)

    def _select_processing_mode(self, mode: str):
        if mode.startswith("pipeline:"):
            name = mode.split(":", 1)[1]
            target = {"type": "pipeline", "name": name}
        elif mode and mode != "none":
            target = {"type": "function", "key": mode}
        else:
            target = {"type": "none"}
        idx = self._find_data_index(target)
        if idx < 0:
            idx = 0
        block = self.cmb_processing.blockSignals(True)
        self.cmb_processing.setCurrentIndex(idx)
        self.cmb_processing.blockSignals(block)
        self._on_processing_mode_changed()


class OverlayView(QtWidgets.QWidget):
    def __init__(
        self,
        processing_manager: Optional[ProcessingManager] = None,
        preferences: Optional[PreferencesManager] = None,
        parent=None,
    ):
        super().__init__(parent)
        self.setAcceptDrops(True)
        self.layers: List[OverlayLayer] = []
        self.processing_manager: Optional[ProcessingManager] = processing_manager
        self.preferences: Optional[PreferencesManager] = None
        self._annotation_config: Optional[PlotAnnotationConfig] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        toolbar = QtWidgets.QHBoxLayout()
        self.btn_auto_view = QtWidgets.QPushButton("Auto view")
        self.btn_auto_view.clicked.connect(self.auto_view_range)
        toolbar.addWidget(self.btn_auto_view)
        self.btn_clear = QtWidgets.QPushButton("Clear layers")
        self.btn_clear.clicked.connect(self.clear_layers)
        toolbar.addWidget(self.btn_clear)
        self.btn_export = QtWidgets.QToolButton()
        self.btn_export.setText("Export")
        self.btn_export.setPopupMode(QtWidgets.QToolButton.InstantPopup)
        export_menu = QtWidgets.QMenu(self.btn_export)
        act_composite = export_menu.addAction("Save composite image…")
        act_composite.triggered.connect(self._export_composite)
        act_layers = export_menu.addAction("Save layers to folder…")
        act_layers.triggered.connect(self._export_individual_layers)
        export_menu.addSeparator()
        act_layout = export_menu.addAction("Save overlay layout…")
        act_layout.triggered.connect(self._export_full_layout)
        self.btn_export.setMenu(export_menu)
        toolbar.addWidget(self.btn_export)
        self.btn_annotations = QtWidgets.QPushButton("Set annotations…")
        self.btn_annotations.clicked.connect(self._open_annotation_dialog)
        toolbar.addWidget(self.btn_annotations)
        toolbar.addStretch(1)
        layout.addLayout(toolbar)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        layout.addWidget(splitter, 1)

        # Layer controls panel
        panel = QtWidgets.QWidget()
        panel.setMinimumWidth(360)
        panel.setSizePolicy(
            QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        )
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(6)

        self.layer_scroll = QtWidgets.QScrollArea()
        self.layer_scroll.setWidgetResizable(True)
        self.layer_list_widget = QtWidgets.QWidget()
        self.layer_list_layout = QtWidgets.QVBoxLayout(self.layer_list_widget)
        self.layer_list_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_list_layout.setSpacing(6)
        self.layer_list_layout.addStretch(1)
        self.layer_scroll.setWidget(self.layer_list_widget)
        panel_layout.addWidget(self.layer_scroll, 1)

        self.lbl_hint = QtWidgets.QLabel("Drag datasets here to overlay them.")
        self.lbl_hint.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_hint.setWordWrap(True)
        panel_layout.addWidget(self.lbl_hint)

        splitter.addWidget(panel)

        # Plot area
        self.glw = pg.GraphicsLayoutWidget()
        self.plot = self.glw.addPlot(row=0, col=0)
        self.plot.invertY(False)
        self.plot.showGrid(x=False, y=False)
        self.plot.setLabel("left", "Y")
        self.plot.setLabel("bottom", "X")
        splitter.addWidget(self.glw)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        QtCore.QTimer.singleShot(0, lambda: splitter.setSizes([420, 780]))

        self.set_preferences(preferences)

    # ---------- drag & drop ----------
    def dragEnterEvent(self, ev):
        ev.acceptProposedAction() if ev.mimeData().hasText() else ev.ignore()

    def dropEvent(self, ev):
        payload = ev.mimeData().text()
        vr = VarRef.from_mime(payload)
        mem_var = None if vr else MemoryVarRef.from_mime(payload)
        slice_ref = None
        if not vr and not mem_var:
            slice_ref = MemorySliceRef.from_mime(payload)
        if not vr and not mem_var and not slice_ref:
            ev.ignore()
            return
        try:
            if vr:
                da, coords = vr.load()
                label = f"{vr.path.name}:{vr.var}"
            elif mem_var:
                da, coords = mem_var.load()
                label = f"{MemoryDatasetRegistry.get_label(mem_var.dataset_key)}:{mem_var.var}"
            else:
                da, coords, alias = slice_ref.load()
                label = f"{slice_ref.display_label()}:{alias}"
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Load failed", str(e))
            return
        data, rect = self._prepare_image(da, coords)
        layer = OverlayLayer(self, label, data, rect)
        self.plot.addItem(layer.image_item)
        widget = OverlayLayerWidget(self, layer)
        widget.set_processing_manager(self.processing_manager)
        layer.set_widget(widget)
        self.layers.append(layer)
        self._insert_layer_widget(widget)
        self._apply_preferences_to_layers()
        self._update_hint()
        self.auto_view_range()
        ev.acceptProposedAction()

    # ---------- layer management ----------
    def _insert_layer_widget(self, widget: OverlayLayerWidget):
        stretch = self.layer_list_layout.itemAt(self.layer_list_layout.count() - 1)
        if stretch and stretch.spacerItem():
            self.layer_list_layout.insertWidget(self.layer_list_layout.count() - 1, widget)
        else:
            self.layer_list_layout.addWidget(widget)

    def remove_layer(self, layer: OverlayLayer):
        if layer in self.layers:
            self.layers.remove(layer)
        try:
            self.plot.removeItem(layer.image_item)
        except Exception:
            pass
        if layer.widget:
            w = layer.widget
            layer.widget = None
            w.setParent(None)
            w.deleteLater()
        self._update_hint()
        self.auto_view_range()

    def clear_layers(self):
        for layer in list(self.layers):
            self.remove_layer(layer)

    def _open_annotation_dialog(self):
        initial = plotitem_annotation_state(self.plot)
        if self._annotation_config is not None:
            initial = replace(self._annotation_config, apply_to_all=False)
        dialog = PlotAnnotationDialog(self, initial=initial, allow_apply_all=False)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        config = dialog.annotation_config()
        if config is None:
            return
        config = replace(config, apply_to_all=False)
        self._annotation_config = config
        apply_plotitem_annotation(
            self.plot,
            config,
            background_widget=self.glw,
        )

    def set_preferences(self, preferences: Optional[PreferencesManager]):
        if self.preferences is preferences:
            return
        if self.preferences:
            try:
                self.preferences.changed.disconnect(self._on_preferences_changed)
            except Exception:
                pass
        self.preferences = preferences
        if preferences is not None:
            try:
                preferences.changed.connect(self._on_preferences_changed)
            except Exception:
                pass
        self._apply_preferences_to_layers()

    def _on_preferences_changed(self, _data):
        self._apply_preferences_to_layers()

    def _apply_preferences_to_layers(self):
        if not self.preferences:
            return
        preferred = self.preferences.preferred_colormap(None)
        if not preferred:
            return
        fallback_names = {"", "default", "viridis"}
        for layer in self.layers:
            if layer.colormap_name not in fallback_names:
                continue
            if preferred != layer.colormap_name:
                layer.set_colormap(preferred)
                if layer.widget is not None:
                    layer.widget._set_colormap_selection(preferred)

    def _default_layout_label(self) -> str:
        if self.preferences:
            return self.preferences.default_layout_label()
        return ""

    def _default_export_dir(self) -> str:
        if self.preferences:
            return self.preferences.default_export_directory()
        return ""

    def _store_export_dir(self, directory: str):
        if self.preferences and directory:
            data = self.preferences.data()
            misc = data.setdefault("misc", {})
            misc["default_export_dir"] = directory
            self.preferences.update(data)

    def _initial_path(self, filename: str) -> str:
        base = self._default_export_dir()
        if base:
            return str(Path(base) / filename)
        return filename

    # ---------- export helpers ----------
    def _export_composite(self):
        if not self.layers:
            QtWidgets.QMessageBox.information(
                self,
                "No layers",
                "Add a layer before exporting the composite image.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save composite image",
            self._initial_path("overlay-composite.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        if not _save_snapshot(self.glw, target):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the composite image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved overlay composite to {target}")

    def _export_individual_layers(self):
        if not self.layers:
            QtWidgets.QMessageBox.information(
                self,
                "No layers",
                "Add a layer before exporting individual images.",
            )
            return
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select export folder",
            self._default_export_dir(),
        )
        if not directory:
            return
        base = Path(directory)
        self._store_export_dir(directory)
        original_states = [(layer, bool(layer.visible)) for layer in self.layers]
        count = 0
        try:
            for idx, layer in enumerate(self.layers, start=1):
                for other in self.layers:
                    other.set_visible(other is layer)
                _process_events()
                name = _sanitize_filename(layer.title) or f"layer_{idx}"
                target = base / f"{name}_{idx:02d}.png"
                if _save_snapshot(self.glw, target):
                    count += 1
        finally:
            for layer, state in original_states:
                layer.set_visible(state)
            _process_events()
            self.auto_view_range()
        if count == 0:
            QtWidgets.QMessageBox.warning(self, "Export failed", "No layers were exported.")
            return
            QtWidgets.QMessageBox.information(
                self,
                "Export complete",
                f"Saved {count} layer image(s) to {base}",
            )
        log_action(f"Exported {count} overlay layers to {base}")

    def _export_full_layout(self):
        if not self.layers:
            QtWidgets.QMessageBox.information(
                self,
                "No layers",
                "Add a layer before exporting the layout.",
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save overlay layout",
            self._initial_path("overlay-layout.png"),
            "PNG image (*.png);;JPEG image (*.jpg *.jpeg);;All files (*)",
        )
        if not path:
            return
        ok, label = _ask_layout_label(self, "Layout label", self._default_layout_label())
        if not ok:
            return
        suffix = ".jpg" if path.lower().endswith((".jpg", ".jpeg")) else ".png"
        target = _ensure_extension(path, suffix)
        if not _save_snapshot(self, target, label):
            QtWidgets.QMessageBox.warning(self, "Save failed", "Unable to save the layout image.")
            return
        self._store_export_dir(str(Path(target).parent))
        log_action(f"Saved overlay layout to {target}")

    def set_processing_manager(self, manager: Optional[ProcessingManager]):
        self.processing_manager = manager
        for layer in self.layers:
            if layer.widget:
                layer.widget.set_processing_manager(manager)

    def auto_view_range(self):
        rects = []
        for layer in self.layers:
            if not layer.visible:
                continue
            rect = layer.get_display_rect()
            if rect is None or rect.isNull():
                continue
            rects.append(rect)
        if not rects:
            return
        union = rects[0]
        for r in rects[1:]:
            try:
                union = union.united(r)
            except Exception:
                pass
        try:
            self.plot.vb.setRange(rect=union, padding=0.0)
        except Exception:
            pass

    def _update_hint(self):
        self.lbl_hint.setVisible(not self.layers)

    # ---------- data prep ----------
    def _prepare_image(self, da, coords):
        Z = np.asarray(da.values, float)
        if "X" in coords and "Y" in coords:
            return self._resample_warped(coords["X"], coords["Y"], Z)
        if "x" in coords and "y" in coords:
            return self._resample_rectilinear(coords["x"], coords["y"], Z)
        Ny, Nx = Z.shape
        rect = self._rect_to_qrectf(0.0, float(Nx), 0.0, float(Ny))
        return np.asarray(Z, float), rect

    def _rect_to_qrectf(self, x0, x1, y0, y1):
        from PySide2.QtCore import QRectF
        return QRectF(float(x0), float(y0), float(x1 - x0), float(y1 - y0))

    def _resample_rectilinear(self, x1, y1, Z):
        x1 = np.asarray(x1, float)
        y1 = np.asarray(y1, float)
        Z = np.asarray(Z, float)
        Ny, Nx = Z.shape
        xs = np.argsort(x1)
        ys = np.argsort(y1)
        x_sorted = x1[xs]
        y_sorted = y1[ys]
        Zs = Z[np.ix_(ys, xs)]
        x_uni = np.linspace(x_sorted[0], x_sorted[-1], Nx)
        y_uni = np.linspace(y_sorted[0], y_sorted[-1], Ny)
        Zx = np.empty((Ny, Nx), float)
        for i in range(Ny):
            Zx[i, :] = np.interp(x_uni, x_sorted, Zs[i, :], left=np.nan, right=np.nan)
        Zu = np.empty((Ny, Nx), float)
        for j in range(Nx):
            col = Zx[:, j]
            m = np.isfinite(col)
            if m.sum() >= 2:
                Zu[:, j] = np.interp(y_uni, y_sorted[m], col[m], left=np.nan, right=np.nan)
            else:
                Zu[:, j] = np.nan
        rect = self._rect_to_qrectf(x_uni[0], x_uni[-1], y_uni[0], y_uni[-1])
        return Zu, rect

    def _resample_warped(self, X, Y, Z):
        try:
            from scipy.interpolate import griddata
        except Exception:
            rect = self._rect_to_qrectf(0.0, float(Z.shape[1]), 0.0, float(Z.shape[0]))
            return np.asarray(Z, float), rect
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        Z = np.asarray(Z, float)
        Ny, Nx = Z.shape
        xmin, xmax = np.nanmin(X), np.nanmax(X)
        ymin, ymax = np.nanmin(Y), np.nanmax(Y)
        x_t = np.linspace(xmin, xmax, Nx)
        y_t = np.linspace(ymin, ymax, Ny)
        XX, YY = np.meshgrid(x_t, y_t)
        pts = np.column_stack([X.ravel(), Y.ravel()])
        vals = Z.ravel()
        Zu = griddata(pts, vals, (XX, YY), method="linear")
        if np.isnan(Zu).any():
            Zun = griddata(pts, vals, (XX, YY), method="nearest")
            mask = np.isnan(Zu)
            Zu[mask] = Zun[mask]
        rect = self._rect_to_qrectf(x_t[0], x_t[-1], y_t[0], y_t[-1])
        return np.asarray(Zu, float), rect


# ---------------------------------------------------------------------------
# Main window with tabs
# ---------------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Dataset Multi-Viewer")

        main = QtWidgets.QSplitter()
        self.setCentralWidget(main)

        self.preferences = PreferencesManager()
        self.processing_manager = ProcessingManager()

        self._build_menus()

        left_splitter = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)
        left_splitter.setHandleWidth(8)
        self.datasets = DatasetsPane()
        self.bridge_server = InteractiveBridgeServer(self.datasets, self)
        self.bridge_server.start()
        left_splitter.addWidget(self.datasets)
        self.processing_dock = ProcessingDockWidget(self.processing_manager)
        self.processing_panel = ProcessingDockContainer("Processing Pipelines", self.processing_dock)
        left_splitter.addWidget(self.processing_panel)
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
        main.addWidget(left_splitter)

        QtCore.QTimer.singleShot(0, lambda: left_splitter.setSizes([1, 1]))

        self.tabs = QtWidgets.QTabWidget()
        main.addWidget(self.tabs)
        main.setStretchFactor(1, 1)

        self.tab_multiview = MultiViewGrid(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_multiview, "MultiView")
        self.tab_sequential = SequentialView(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_sequential, "Sequential View")
        self.tab_slice = SliceDataTab(self.datasets)
        self.tabs.addTab(self.tab_slice, "Slice Data")
        self.tab_overlay = OverlayView(self.processing_manager, self.preferences)
        self.tabs.addTab(self.tab_overlay, "Overlay")
        self.tab_overlay.set_processing_manager(self.processing_manager)
        self.tab_interactive = InteractiveProcessingTab(self.datasets, self.bridge_server)
        self.tabs.addTab(self.tab_interactive, "Interactive Processing")

        self.log_dock = LoggingDockWidget(ACTION_LOGGER, self)
        self.addDockWidget(QtCore.Qt.BottomDockWidgetArea, self.log_dock)
        self.log_dock.setFloating(False)
        self.log_dock.resize(800, 200)

        self.resize(1500, 900)

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
        prefs_menu = menubar.addMenu("Preferences")

        act_edit = prefs_menu.addAction("Edit preferences…")
        act_edit.triggered.connect(self._edit_preferences)

        prefs_menu.addSeparator()

        act_load = prefs_menu.addAction("Load preferences…")
        act_load.triggered.connect(self._load_preferences)

        act_save = prefs_menu.addAction("Save preferences…")
        act_save.triggered.connect(self._save_preferences)

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


def main():
    app = QtWidgets.QApplication([])
    pg.setConfigOptions(imageAxisOrder='row-major')
    win = MainWindow(); win.show()
    app.exec_()


if __name__ == "__main__":
    main()

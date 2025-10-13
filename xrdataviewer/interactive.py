from __future__ import annotations

import base64
import html
import importlib
import io
import json
import os
import pkgutil
import re
import shutil
import socket
import subprocess
import tempfile
import textwrap
import threading
import time
import traceback
import urllib.request
import warnings
from contextlib import redirect_stderr, redirect_stdout
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Dict, Iterable, List, Optional, Set, Tuple, TYPE_CHECKING

import sys

import numpy as np
import pyqtgraph as pg
import xarray as xr
from PySide2 import QtCore, QtGui, QtWidgets

try:  # Optional dependency for embedded browsers
    from PySide2 import QtWebEngineWidgets  # type: ignore
except Exception:  # pragma: no cover - QtWebEngine may be unavailable
    QtWebEngineWidgets = None

from app_logging import log_action
from xr_plot_widget import ScientificAxisItem

if TYPE_CHECKING:  # pragma: no cover - typing helpers
    from .datasets import DatasetsPane


if QtWebEngineWidgets is not None:  # pragma: no cover - requires QtWebEngine

    class _ConsoleAwarePage(QtWebEngineWidgets.QWebEnginePage):
        """QWebEnginePage variant that re-emits console output as a Qt signal."""

        consoleMessage = QtCore.Signal(int, str, int, str)

        def javaScriptConsoleMessage(self, level, message, lineNumber, sourceID):  # type: ignore[override]
            # Emit the console message so the embedding widget can surface it in the
            # logging pane or display compatibility hints when the bundled
            # Chromium version cannot render a page.
            self.consoleMessage.emit(level, message, lineNumber, sourceID)
            super().javaScriptConsoleMessage(level, message, lineNumber, sourceID)

else:

    class _ConsoleAwarePage:  # type: ignore[too-many-ancestors]
        """Fallback shim when QtWebEngine is unavailable."""

        def __init__(self, *args, **kwargs):  # pragma: no cover - placeholder
            raise RuntimeError("QtWebEngineWidgets is not available")


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
        self._startup_script: Optional[str] = None
        self._kernelspec_dir: Optional[str] = None
        self._kernel_name = "xrdataviewer"

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
                f"--ServerApp.default_kernel_name={self._kernel_name}",
            ]
        )

        env = os.environ.copy()
        env.setdefault("PYTHONUNBUFFERED", "1")
        env["JUPYTER_DEFAULT_KERNEL_NAME"] = self._kernel_name

        self._kernelspec_dir = None
        try:
            self._kernelspec_dir = tempfile.mkdtemp(prefix="xrdataviewer_kernel_")
            kernels_dir = Path(self._kernelspec_dir) / "kernels" / self._kernel_name
            kernels_dir.mkdir(parents=True, exist_ok=True)
            kernel_spec = {
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                "display_name": "Python (XRDataViewer)",
                "language": "python",
                "env": {},
            }
            with open(kernels_dir / "kernel.json", "w", encoding="utf-8") as handle:
                json.dump(kernel_spec, handle)
            existing_path = env.get("JUPYTER_PATH")
            if existing_path:
                env["JUPYTER_PATH"] = os.pathsep.join([self._kernelspec_dir, existing_path])
            else:
                env["JUPYTER_PATH"] = self._kernelspec_dir
        except Exception as exc:
            if self._kernelspec_dir:
                shutil.rmtree(self._kernelspec_dir, ignore_errors=True)
                self._kernelspec_dir = None
            self.message.emit(
                "Warning: failed to provision dedicated Jupyter kernel; the embedded server may use a different environment."
            )
            self.message.emit(str(exc))

        startup_code = textwrap.dedent(
            """
            try:
                import xrdataviewer_bridge  # noqa: F401
                xrdataviewer_bridge.enable_auto_sync()
            except Exception:
                pass
            """
        )
        try:
            fd, script_path = tempfile.mkstemp(prefix="xrdataviewer_startup_", suffix=".py")
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(startup_code)
            env["IPYTHONSTARTUP"] = script_path
            self._startup_script = script_path
        except Exception as exc:
            self._startup_script = None
            warnings.warn(f"Unable to configure XRDataViewer auto-sync startup script: {exc}")

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
            if self._startup_script:
                try:
                    os.remove(self._startup_script)
                except Exception:
                    pass
                self._startup_script = None
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
        if self._startup_script:
            try:
                os.remove(self._startup_script)
            except Exception:
                pass
            self._startup_script = None
        if self._kernelspec_dir:
            shutil.rmtree(self._kernelspec_dir, ignore_errors=True)
            self._kernelspec_dir = None

    def url(self) -> Optional[str]:
        return self._url

    def _resolve_command(self) -> Optional[List[str]]:
        python_exe = sys.executable
        if python_exe and Path(python_exe).exists():
            return [python_exe, "-m", "jupyterlab"]
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
        if self._startup_script:
            try:
                os.remove(self._startup_script)
            except Exception:
                pass
            self._startup_script = None
        if self._kernelspec_dir:
            shutil.rmtree(self._kernelspec_dir, ignore_errors=True)
            self._kernelspec_dir = None

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
        """QWebEnginePage that relays console output through Qt signals."""

        consoleMessage = QtCore.Signal(int, str, int, str)

        def javaScriptConsoleMessage(
            self,
            level: QtWebEngineWidgets.QWebEnginePage.JavaScriptConsoleMessageLevel,
            message: str,
            line_number: int,
            source_id: str,
        ) -> None:  # type: ignore[override]
            self.consoleMessage.emit(int(level), message, int(line_number), source_id)


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
            "Choose an interactive environment. The embedded JupyterLab session automatically mirrors any "
            "<code>xarray.Dataset</code> or <code>xarray.DataArray</code> defined in your notebook into the Interactive "
            "Data tab after each cell runs."
        )
        hint_text += (
            "<br><br>The Python console exposes <code>RegisterDataset(...)</code> and <code>RegisterDataArray(...)</code> "
            "helpers if you prefer to push results manually."
        )
        if self.bridge_server and self.bridge_server.register_url():
            hint_text += (
                "<br><br>External notebooks can <code>import xrdataviewer_bridge</code> and call "
                "<code>enable_auto_sync()</code> to keep their namespace in sync, or use the provided register helpers."
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
            friendly = None
            if "Invalid selector" in cleaned:
                friendly = (
                    "The embedded browser does not support some of JupyterLab's styles. "
                    "Layout may look different; open in an external browser for full support."
                )
            elif "Cannot read property 'id' of null" in cleaned:
                friendly = (
                    "JupyterLab hit a browser compatibility issue. Reload or open the session in an external browser "
                    "if parts of the UI fail to appear."
                )
            if friendly:
                self._set_status(friendly)
            elif not self._jupyter_js_warned:
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

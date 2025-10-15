from __future__ import annotations

import copy
import itertools
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import xarray as xr
from PySide2 import QtCore, QtGui, QtWidgets

from app_logging import log_action
from xr_coords import guess_phys_coords

from .utils import open_dataset

_BATCH_PREFIX = "BatchRef:"


def encode_mime_payloads(payloads: Iterable[str]) -> str:
    values = [p for p in payloads if p]
    if not values:
        return ""
    if len(values) == 1:
        return values[0]
    try:
        return _BATCH_PREFIX + json.dumps(values)
    except Exception:
        # Fallback: return the first payload if encoding fails
        return values[0]


def decode_mime_payloads(text: str) -> List[str]:
    if not text:
        return []
    if text.startswith(_BATCH_PREFIX):
        try:
            raw = json.loads(text.split(":", 1)[1])
        except Exception:
            return []
        if isinstance(raw, list):
            return [str(item) for item in raw if isinstance(item, str) and item]
        return []
    return [text]


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
        if self.var not in ds.data_vars:
            raise RuntimeError(f"Variable {self.var!r} is not present in {self.path}")
        da = ds[self.var]
        ndim = getattr(da, "ndim", 0)
        if ndim <= 0:
            raise RuntimeError("Variable has no dimensions to plot")
        if ndim > 2:
            raise RuntimeError(f"{self.var!r} is higher than 2D in {self.path}")
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
        ndim = getattr(da, "ndim", 0)
        if ndim <= 0:
            raise RuntimeError("Variable has no dimensions to plot")
        if ndim > 2:
            raise RuntimeError("Variable is higher than two-dimensional")
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

class _DatasetsTree(QtWidgets.QTreeWidget):
    def startDrag(self, supported_actions: QtCore.Qt.DropActions) -> None:
        # Allow click-and-drag row selection without immediately starting a drag
        if self.state() == QtWidgets.QAbstractItemView.DragSelectingState:
            return
        super().startDrag(supported_actions)


class DatasetsPane(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("datasetsPane")

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(12)

        btn_frame = QtWidgets.QFrame()
        btn_frame.setProperty("modernSection", True)
        btn_layout = QtWidgets.QHBoxLayout(btn_frame)
        btn_layout.setContentsMargins(12, 8, 12, 8)
        btn_layout.setSpacing(8)

        self.btn_open_netcdf = QtWidgets.QPushButton("Load NetCDF…")
        self.btn_open_netcdf.clicked.connect(self._open_netcdf)
        btn_layout.addWidget(self.btn_open_netcdf)

        self.btn_open_json = QtWidgets.QPushButton("Load JSON…")
        self.btn_open_json.clicked.connect(self._open_json)
        btn_layout.addWidget(self.btn_open_json)

        self.btn_open_db = QtWidgets.QPushButton("Load Database…")
        self.btn_open_db.clicked.connect(self._open_database)
        btn_layout.addWidget(self.btn_open_db)

        btn_layout.addStretch(1)
        layout.addWidget(btn_frame)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.setProperty("compactTabs", True)
        layout.addWidget(self.tabs, 1)

        self._trees: Dict[str, QtWidgets.QTreeWidget] = {}
        self._roots: Dict[str, Dict[str, QtWidgets.QTreeWidgetItem]] = {
            "datasets": {},
            "sliced": {},
            "interactive": {},
        }

        for key, title in (
            ("datasets", "Datasets"),
            ("sliced", "Sliced Data"),
            ("interactive", "Interactive Data"),
        ):
            tree = self._create_tree()
            self._trees[key] = tree
            self.tabs.addTab(tree, title)

        self._populate_examples()

    def _create_tree(self) -> QtWidgets.QTreeWidget:
        tree = _DatasetsTree()
        tree.setHeaderLabels(["Datasets / Variables"])
        tree.setDragEnabled(True)
        tree.setDefaultDropAction(QtCore.Qt.CopyAction)
        tree.setDropIndicatorShown(False)
        tree.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        tree.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        tree.setUniformRowHeights(True)

        def _mime_data(_items, *, _tree=tree):
            md = QtCore.QMimeData()
            sel = _tree.selectedItems()
            payloads = [item.data(0, QtCore.Qt.UserRole) for item in sel if item is not None]
            payloads = [p for p in payloads if isinstance(p, str) and p]
            text = encode_mime_payloads(payloads)
            if text:
                md.setText(text)
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
        item.setData(0, QtCore.Qt.UserRole + 1, category)
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
            self._populate_dataset_tree(path, ds)
            if not quiet:
                log_action(f"Loaded dataset '{path.name}'")
        finally:
            try:
                ds.close()
            except Exception:
                pass

    def _populate_dataset_tree(self, path: Path, dataset: xr.Dataset):
        key, item = self._ensure_disk_root("datasets", path)
        added = False
        for var in dataset.data_vars:
            try:
                da = dataset[var]
            except Exception:
                continue
            ndim = getattr(da, "ndim", 0)
            if ndim <= 0:
                continue
            hint = self._format_hint(da)
            text = f"{var}  {hint}" if hint else var
            child = QtWidgets.QTreeWidgetItem([text])
            if ndim <= 2:
                child.setData(0, QtCore.Qt.UserRole, VarRef(path, var, hint).to_mime())
            else:
                child.setData(0, QtCore.Qt.UserRole, HighDimVarRef(var, hint, path=path).to_mime())
            item.addChild(child)
            added = True
        if not added:
            self._remove_root("datasets", key)

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
            if 0 < ndim <= 2:
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
        self._output_mode: str = "2d"

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

        mode_row = QtWidgets.QHBoxLayout()
        mode_row.addWidget(QtWidgets.QLabel("Slice output:"))
        self.cmb_output_mode = QtWidgets.QComboBox()
        self.cmb_output_mode.addItem("2D slices (images)", "2d")
        self.cmb_output_mode.addItem("1D slices (profiles)", "1d")
        self.cmb_output_mode.currentIndexChanged.connect(self._on_output_mode_changed)
        self.cmb_output_mode.setEnabled(False)
        mode_row.addWidget(self.cmb_output_mode, 1)
        mode_row.addStretch(1)
        outer.addLayout(mode_row)

        axis_group = QtWidgets.QGroupBox("Slice configuration")
        self.axis_group = axis_group
        axis_form = QtWidgets.QFormLayout(axis_group)
        axis_form.setContentsMargins(6, 6, 6, 6)
        axis_form.setSpacing(6)

        self.lbl_row_dim = QtWidgets.QLabel("Rows")
        self.cmb_row_dim = QtWidgets.QComboBox()
        self.cmb_row_dim.setEnabled(False)
        self.cmb_row_dim.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow(self.lbl_row_dim, self.cmb_row_dim)

        self.lbl_col_dim = QtWidgets.QLabel("Columns")
        self.cmb_col_dim = QtWidgets.QComboBox()
        self.cmb_col_dim.setEnabled(False)
        self.cmb_col_dim.currentIndexChanged.connect(self._on_axes_changed)
        axis_form.addRow(self.lbl_col_dim, self.cmb_col_dim)

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
            if getattr(da, "ndim", 0) < 2:
                continue
            dims = getattr(da, "dims", ())
            shape = " × ".join(str(getattr(da, "sizes", {}).get(dim, "?")) for dim in dims)
            self.cmb_variable.addItem(f"{var} ({shape})", var)
            count += 1
        self.cmb_variable.blockSignals(False)
        self.cmb_variable.setEnabled(count > 0)
        self.cmb_output_mode.setEnabled(count > 0)
        self.btn_generate.setEnabled(False)
        if count == 0:
            self._clear_current_variable()
            self.lbl_status.setText("Dataset does not contain variables with at least two dimensions.")
        else:
            self.lbl_status.setText("Select a variable and configure the slicing parameters.")

    def _clear_current_variable(self):
        self._current_var = None
        self._current_da = None
        self._current_dims = []
        self._set_output_mode("2d", force=True)
        self.cmb_row_dim.blockSignals(True)
        self.cmb_row_dim.clear()
        self.cmb_row_dim.blockSignals(False)
        self.cmb_row_dim.setEnabled(False)
        self.cmb_col_dim.blockSignals(True)
        self.cmb_col_dim.clear()
        self.cmb_col_dim.blockSignals(False)
        self.cmb_col_dim.setEnabled(False)
        self.cmb_output_mode.setEnabled(False)
        self.axis_group.setEnabled(False)
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
        has_dims = bool(dims)
        if dims:
            self.cmb_row_dim.setCurrentIndex(0)
        if len(dims) >= 2:
            self.cmb_col_dim.setCurrentIndex(1)
        default_mode = "2d" if len(dims) >= 3 else "1d"
        self.axis_group.setEnabled(has_dims)
        self.cmb_output_mode.setEnabled(has_dims)
        self._set_output_mode(default_mode)

    def _set_output_mode(self, mode: str, *, force: bool = False):
        if mode not in {"1d", "2d"}:
            return
        if not force and mode == self._output_mode:
            self._apply_output_mode()
            return
        self._output_mode = mode
        idx = self.cmb_output_mode.findData(mode)
        if idx >= 0:
            block = self.cmb_output_mode.blockSignals(True)
            self.cmb_output_mode.setCurrentIndex(idx)
            self.cmb_output_mode.blockSignals(block)
        self._apply_output_mode()

    def _on_output_mode_changed(self, _index: int):
        mode = self.cmb_output_mode.currentData()
        if not mode:
            return
        self._set_output_mode(str(mode))

    def _apply_output_mode(self):
        mode = self._output_mode
        dims = list(self._current_dims)
        has_dims = bool(dims)
        self.axis_group.setEnabled(has_dims)
        self.cmb_row_dim.setEnabled(has_dims)
        if mode == "1d":
            self.lbl_row_dim.setText("Profile axis")
            self.lbl_col_dim.setVisible(False)
            self.cmb_col_dim.setVisible(False)
            self.cmb_col_dim.setEnabled(False)
        else:
            self.lbl_row_dim.setText("Rows")
            self.lbl_col_dim.setVisible(True)
            self.cmb_col_dim.setVisible(True)
            self.cmb_col_dim.setEnabled(len(dims) >= 2)
        self._update_extra_dim_controls()

    def _on_axes_changed(self, _index: int):
        self._update_extra_dim_controls()

    def _update_extra_dim_controls(self):
        row_dim = self.cmb_row_dim.currentData()
        col_dim = self.cmb_col_dim.currentData() if self._output_mode == "2d" else None
        excluded = {dim for dim in (row_dim, col_dim) if dim}
        dims = [dim for dim in self._current_dims if dim not in excluded]
        self._reset_extra_dim_controls(dims)

    def _generate_slices(self):
        if self._dataset is None or self._current_da is None or self._current_var is None:
            QtWidgets.QMessageBox.warning(self, "Generate slices", "Load a dataset and select a variable first.")
            return
        if self._output_mode == "1d":
            data_vars = self._build_1d_slices()
        else:
            data_vars = self._build_2d_slices()
        if not data_vars:
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

    def _build_2d_slices(self) -> Optional[Dict[str, xr.DataArray]]:
        row_dim = self.cmb_row_dim.currentData()
        col_dim = self.cmb_col_dim.currentData()
        if not row_dim or not col_dim or row_dim == col_dim:
            QtWidgets.QMessageBox.warning(
                self,
                "Generate slices",
                "Please choose two different dimensions for rows and columns.",
            )
            return None
        slice_dims = []
        for dim, control in self._dim_controls.items():
            indices = control.indices()
            if not indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Generate slices",
                    f"No indices selected for dimension {dim}.",
                )
                return None
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
            return None
        return data_vars

    def _build_1d_slices(self) -> Optional[Dict[str, xr.DataArray]]:
        line_dim = self.cmb_row_dim.currentData()
        if not line_dim:
            QtWidgets.QMessageBox.warning(
                self,
                "Generate slices",
                "Select a dimension to use for the 1D profile.",
            )
            return None
        slice_dims = []
        for dim, control in self._dim_controls.items():
            indices = control.indices()
            if not indices:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Generate slices",
                    f"No indices selected for dimension {dim}.",
                )
                return None
            slice_dims.append((dim, indices))

        data_vars: Dict[str, xr.DataArray] = {}

        def _prepare_line(arr: xr.DataArray) -> xr.DataArray:
            try:
                arr = arr.transpose(line_dim)
            except Exception:
                pass
            arr = arr.copy(deep=True)
            try:
                extra_coords = [name for name in arr.coords if name not in arr.dims]
                if extra_coords:
                    arr = arr.drop_vars(extra_coords)
            except Exception:
                try:
                    arr = arr.reset_coords(drop=True)
                except Exception:
                    pass
            return arr

        if not slice_dims:
            arr = _prepare_line(self._current_da)
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
                arr = _prepare_line(arr)
                if arr.ndim != 1:
                    QtWidgets.QMessageBox.warning(
                        self,
                        "Generate slices",
                        "Unable to reduce selection to a 1D profile; adjust the slice ranges.",
                    )
                    return None
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
            return None
        return data_vars

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
            "Only datasets or variables with at least two dimensions can be dropped here.",
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

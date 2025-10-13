from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, Optional

from PySide2 import QtCore, QtWidgets


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

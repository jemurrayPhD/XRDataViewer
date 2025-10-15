from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import fnmatch
from PySide2 import QtCore, QtWidgets

from .appearance import (
    ACCENT_OPTIONS,
    BACKGROUND_OPTIONS,
    BUTTON_SHAPE_OPTIONS,
    BUILTIN_PROFILES,
    FONT_OPTIONS,
    default_appearance,
    sanitize_appearance,
    sanitize_profile_values,
)
from .colormaps import available_colormap_names, is_scientific_colormap
from .utils import default_documents_directory


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
                "value_precision": 6,
            },
            "colormaps": {
                "default": "viridis",
                "variables": [],
            },
            "misc": {
                "default_export_dir": "",
            },
            "interactive": {
                "jupyter_root_dir": str(default_documents_directory()),
            },
            "appearance": default_appearance(),
        }

    def data(self) -> Dict[str, object]:
        return copy.deepcopy(self._data)

    def update(self, data: Dict[str, object]):
        normalized = self._default_prefs()
        general = data.get("general", {}) if isinstance(data, dict) else {}
        if isinstance(general, dict):
            normalized["general"].update(general)
            digits = general.get("value_precision")
            try:
                digits = int(digits)
            except Exception:
                digits = normalized["general"].get("value_precision", 6)
            digits = max(0, min(8, digits))
            normalized["general"]["value_precision"] = digits
        colormaps = data.get("colormaps", {}) if isinstance(data, dict) else {}
        normalized_variables: List[Dict[str, str]] = []
        if isinstance(colormaps, dict):
            default_map = colormaps.get("default")
            if default_map is not None:
                normalized["colormaps"]["default"] = str(default_map)
            raw_variables = colormaps.get("variables", [])
        else:
            raw_variables = []

        items: List[Tuple[object, object]] = []
        if isinstance(raw_variables, dict):
            items.extend(raw_variables.items())
        elif isinstance(raw_variables, Iterable):
            for entry in raw_variables:
                if isinstance(entry, dict):
                    pattern = entry.get("pattern") or entry.get("variable")
                    cmap = entry.get("colormap") or entry.get("map")
                    items.append((pattern, cmap))

        for pattern, cmap in items:
            if pattern is None or cmap is None:
                continue
            pat_str = str(pattern).strip()
            cmap_str = str(cmap).strip()
            if not pat_str or not cmap_str:
                continue
            normalized_variables.append({"pattern": pat_str, "colormap": cmap_str})

        normalized["colormaps"]["variables"] = normalized_variables
        misc = data.get("misc", {}) if isinstance(data, dict) else {}
        if isinstance(misc, dict):
            normalized["misc"].update(misc)
        interactive = data.get("interactive", {}) if isinstance(data, dict) else {}
        if isinstance(interactive, dict):
            root = str(interactive.get("jupyter_root_dir", "")).strip()
            if root:
                expanded = os.path.expandvars(root)
                normalized["interactive"]["jupyter_root_dir"] = str(Path(expanded).expanduser())
        appearance = data.get("appearance") if isinstance(data, dict) else None
        normalized["appearance"] = sanitize_appearance(appearance)
        self._data = normalized
        self.changed.emit(self.data())

    def autoscale_on_load(self) -> bool:
        return bool(self._data.get("general", {}).get("autoscale_on_load", True))

    def default_layout_label(self) -> str:
        return str(self._data.get("general", {}).get("default_layout_label", ""))

    def default_export_directory(self) -> str:
        return str(self._data.get("misc", {}).get("default_export_dir", ""))

    def value_precision(self) -> int:
        try:
            return int(self._data.get("general", {}).get("value_precision", 6))
        except Exception:
            return 6

    def jupyter_root_directory(self) -> str:
        path = str(self._data.get("interactive", {}).get("jupyter_root_dir", "")).strip()
        if not path:
            return str(default_documents_directory())
        return path

    def preferred_colormap(self, variable: Optional[str]) -> Optional[str]:
        colormap_data = self._data.get("colormaps", {})
        variables = colormap_data.get("variables", [])

        entries: List[Tuple[str, str]] = []
        if isinstance(variables, dict):
            for key, value in variables.items():
                if key is None or value is None:
                    continue
                entries.append((str(key), str(value)))
        elif isinstance(variables, Iterable):
            for entry in variables:
                if not isinstance(entry, dict):
                    continue
                pattern = entry.get("pattern") or entry.get("variable")
                cmap = entry.get("colormap") or entry.get("map")
                if pattern is None or cmap is None:
                    continue
                entries.append((str(pattern), str(cmap)))

        value = str(variable) if variable is not None else ""
        if variable:
            for pattern, cmap in entries:
                if pattern == value or pattern.lower() == value.lower():
                    return cmap

        if variable:
            lowered = value.lower()
            for pattern, cmap in entries:
                pat = pattern or ""
                try:
                    if fnmatch.fnmatchcase(value, pat):
                        return cmap
                except Exception:
                    pass
                try:
                    if fnmatch.fnmatchcase(lowered, pat.lower()):
                        return cmap
                except Exception:
                    pass
                if pat and lowered and pat.lower() in lowered:
                    return cmap
        else:
            for pattern, cmap in entries:
                pat = pattern.strip()
                if pat in ("*", ""):
                    return cmap

        default = colormap_data.get("default")
        if default:
            return str(default)
        return None

    def appearance(self) -> Dict[str, object]:
        return copy.deepcopy(self._data.get("appearance", default_appearance()))

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
        self.resize(600, 460)
        self.setMinimumSize(640, 520)
        self._manager = manager
        self._data = manager.data()
        self._appearance_profiles = copy.deepcopy(
            self._data.get("appearance", {}).get("profiles", {})
        )
        self._appearance_active_profile = str(
            self._data.get("appearance", {}).get("active_profile", "")
        )
        self._apply_button: Optional[QtWidgets.QPushButton] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)

        self._build_general_tab()
        self._build_appearance_tab()
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

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok
            | QtWidgets.QDialogButtonBox.Cancel
            | QtWidgets.QDialogButtonBox.Apply
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        self._apply_button = buttons.button(QtWidgets.QDialogButtonBox.Apply)
        if self._apply_button is not None:
            self._apply_button.clicked.connect(self._apply_preferences)
            self._apply_button.setEnabled(False)
        layout.addWidget(buttons)

        self._update_apply_button_state()

    def _build_general_tab(self):
        tab = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(tab)
        form.setContentsMargins(6, 6, 6, 6)
        form.setSpacing(6)

        self.chk_autoscale = QtWidgets.QCheckBox("Autoscale images on load")
        self.chk_autoscale.setChecked(bool(self._data.get("general", {}).get("autoscale_on_load", True)))
        self.chk_autoscale.toggled.connect(self._on_field_modified)
        form.addRow(self.chk_autoscale)

        self.txt_layout_label = QtWidgets.QLineEdit(
            str(self._data.get("general", {}).get("default_layout_label", ""))
        )
        self.txt_layout_label.textChanged.connect(self._on_field_modified)
        form.addRow("Default layout label", self.txt_layout_label)

        self.spn_value_precision = QtWidgets.QSpinBox()
        self.spn_value_precision.setRange(0, 8)
        self.spn_value_precision.setValue(
            int(self._data.get("general", {}).get("value_precision", 6))
        )
        self.spn_value_precision.setSuffix(" decimals")
        self.spn_value_precision.setToolTip(
            "Number of decimal places used for value readouts and controls."
        )
        self.spn_value_precision.valueChanged.connect(self._on_field_modified)
        form.addRow("Value precision", self.spn_value_precision)

        export_row = QtWidgets.QHBoxLayout()
        self.txt_export_dir = QtWidgets.QLineEdit(
            str(self._data.get("misc", {}).get("default_export_dir", ""))
        )
        self.txt_export_dir.textChanged.connect(self._on_field_modified)
        export_row.addWidget(self.txt_export_dir, 1)
        btn_browse = QtWidgets.QPushButton("Browse…")
        btn_browse.clicked.connect(self._browse_export_dir)
        export_row.addWidget(btn_browse)
        form.addRow("Default export folder", export_row)

        jupyter_row = QtWidgets.QHBoxLayout()
        self.txt_jupyter_root = QtWidgets.QLineEdit(
            str(self._data.get("interactive", {}).get("jupyter_root_dir", ""))
        )
        self.txt_jupyter_root.textChanged.connect(self._on_field_modified)
        jupyter_row.addWidget(self.txt_jupyter_root, 1)
        btn_jupyter = QtWidgets.QPushButton("Browse…")
        btn_jupyter.clicked.connect(self._browse_jupyter_root)
        jupyter_row.addWidget(btn_jupyter)
        form.addRow("Embedded Jupyter directory", jupyter_row)

        self.tabs.addTab(tab, "General")

    def _build_appearance_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        self.cmb_profiles = QtWidgets.QComboBox()
        self.cmb_profiles.currentIndexChanged.connect(self._on_profile_selected)
        layout.addWidget(QtWidgets.QLabel("Profiles"))
        layout.addWidget(self.cmb_profiles)

        form = QtWidgets.QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setSpacing(6)

        self.cmb_font_family = QtWidgets.QComboBox()
        for name in FONT_OPTIONS:
            self.cmb_font_family.addItem(name, name)
        self.cmb_font_family.currentIndexChanged.connect(self._on_appearance_modified)
        form.addRow("Font family", self.cmb_font_family)

        self.spn_font_size = QtWidgets.QDoubleSpinBox()
        self.spn_font_size.setDecimals(1)
        self.spn_font_size.setRange(8.0, 14.0)
        self.spn_font_size.setSingleStep(0.5)
        self.spn_font_size.valueChanged.connect(self._on_appearance_modified)
        form.addRow("Font size", self.spn_font_size)

        self.cmb_accent = QtWidgets.QComboBox()
        for name in ACCENT_OPTIONS:
            self.cmb_accent.addItem(name, name)
        self.cmb_accent.currentIndexChanged.connect(self._on_appearance_modified)
        form.addRow("Accent color", self.cmb_accent)

        self.cmb_background = QtWidgets.QComboBox()
        for name in BACKGROUND_OPTIONS:
            self.cmb_background.addItem(name, name)
        self.cmb_background.currentIndexChanged.connect(self._on_appearance_modified)
        form.addRow("Background", self.cmb_background)

        self.cmb_button_shape = QtWidgets.QComboBox()
        for name in BUTTON_SHAPE_OPTIONS:
            self.cmb_button_shape.addItem(name, name)
        self.cmb_button_shape.currentIndexChanged.connect(self._on_appearance_modified)
        form.addRow("Button shape", self.cmb_button_shape)

        layout.addLayout(form)

        button_row = QtWidgets.QHBoxLayout()
        self.btn_save_profile = QtWidgets.QPushButton("Save profile…")
        self.btn_save_profile.clicked.connect(self._save_profile)
        button_row.addWidget(self.btn_save_profile)

        self.btn_delete_profile = QtWidgets.QPushButton("Delete profile")
        self.btn_delete_profile.clicked.connect(self._delete_profile)
        button_row.addWidget(self.btn_delete_profile)

        self.btn_export_profile = QtWidgets.QPushButton("Export selected…")
        self.btn_export_profile.clicked.connect(self._export_profile)
        button_row.addWidget(self.btn_export_profile)

        self.btn_import_profile = QtWidgets.QPushButton("Import profile…")
        self.btn_import_profile.clicked.connect(self._import_profile)
        button_row.addWidget(self.btn_import_profile)

        button_row.addStretch(1)
        layout.addLayout(button_row)

        self.tabs.addTab(tab, "Appearance")

        self._apply_appearance_tab()

    def _build_colormap_tab(self):
        tab = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.cmb_default_cmap = QtWidgets.QComboBox()
        self.cmb_default_cmap.addItem("Use viewer default", "")
        for name in available_colormap_names():
            label = name
            if is_scientific_colormap(name):
                label = f"{name} (Scientific)"
            self.cmb_default_cmap.addItem(label, name)
        current_default = str(self._data.get("colormaps", {}).get("default", ""))
        idx = self.cmb_default_cmap.findData(current_default)
        if idx < 0:
            idx = 0
        block = self.cmb_default_cmap.blockSignals(True)
        self.cmb_default_cmap.setCurrentIndex(idx)
        self.cmb_default_cmap.blockSignals(block)
        self.cmb_default_cmap.currentIndexChanged.connect(self._on_field_modified)
        layout.addWidget(QtWidgets.QLabel("Default colormap"))
        layout.addWidget(self.cmb_default_cmap)

        self.table_colormaps = QtWidgets.QTableWidget(0, 2)
        self.table_colormaps.setHorizontalHeaderLabels(["Variable", "Colormap"])
        header = self.table_colormaps.horizontalHeader()
        try:
            if isinstance(header, QtWidgets.QHeaderView):
                header.setStretchLastSection(True)
            elif hasattr(header, "setStretchLastSection"):
                header.setStretchLastSection(True)
        except Exception:
            # Some PySide2 builds have been observed to hand back placeholder
            # QWidget instances here; fall back silently when the API is
            # missing so the dialog still loads.
            pass
        self.table_colormaps.verticalHeader().setVisible(False)
        self.table_colormaps.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table_colormaps.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked
            | QtWidgets.QAbstractItemView.EditKeyPressed
        )
        self.table_colormaps.itemChanged.connect(self._on_field_modified)
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

        variables = self._data.get("colormaps", {}).get("variables", [])
        entries: List[Tuple[str, str]] = []
        if isinstance(variables, dict):
            entries.extend((str(k), str(v)) for k, v in variables.items())
        elif isinstance(variables, Iterable):
            for entry in variables:
                if not isinstance(entry, dict):
                    continue
                pattern = entry.get("pattern") or entry.get("variable")
                cmap = entry.get("colormap") or entry.get("map")
                if pattern is None or cmap is None:
                    continue
                entries.append((str(pattern), str(cmap)))
        for pattern, cmap in entries:
            row = self.table_colormaps.rowCount()
            self.table_colormaps.insertRow(row)
            self.table_colormaps.setItem(row, 0, QtWidgets.QTableWidgetItem(pattern))
            self.table_colormaps.setItem(row, 1, QtWidgets.QTableWidgetItem(cmap))

        self.tabs.addTab(tab, "Colormaps")

    def _browse_export_dir(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select default export folder",
            self.txt_export_dir.text(),
        )
        if directory:
            self.txt_export_dir.setText(directory)

    def _browse_jupyter_root(self):
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select embedded Jupyter directory",
            self.txt_jupyter_root.text() or self.txt_export_dir.text(),
        )
        if directory:
            self.txt_jupyter_root.setText(directory)

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
        self._update_apply_button_state()

    def _remove_colormap_mapping(self):
        rows = sorted({idx.row() for idx in self.table_colormaps.selectedIndexes()}, reverse=True)
        for row in rows:
            self.table_colormaps.removeRow(row)
        if rows:
            self._update_apply_button_state()

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
            "default_layout_label": self.txt_layout_label.text().strip(),
            "value_precision": int(self.spn_value_precision.value()),
        }
        colormaps = {
            "default": self.cmb_default_cmap.currentData() or "",
            "variables": [],
        }
        for row in range(self.table_colormaps.rowCount()):
            var_item = self.table_colormaps.item(row, 0)
            cmap_item = self.table_colormaps.item(row, 1)
            if not var_item or not cmap_item:
                continue
            var = var_item.text().strip()
            cmap = cmap_item.text().strip()
            if var and cmap:
                colormaps["variables"].append({"pattern": var, "colormap": cmap})
        misc = {
            "default_export_dir": self.txt_export_dir.text().strip(),
        }
        interactive = {
            "jupyter_root_dir": self.txt_jupyter_root.text().strip(),
        }
        appearance = {
            "font_family": self.cmb_font_family.currentData(),
            "font_size": float(self.spn_font_size.value()),
            "accent": self.cmb_accent.currentData(),
            "background": self.cmb_background.currentData(),
            "button_shape": self.cmb_button_shape.currentData(),
            "active_profile": self._appearance_active_profile,
            "profiles": copy.deepcopy(self._appearance_profiles),
        }
        return {
            "general": general,
            "colormaps": colormaps,
            "misc": misc,
            "interactive": interactive,
            "appearance": appearance,
        }

    def _apply_data(self):
        data = self._data
        self.chk_autoscale.setChecked(bool(data.get("general", {}).get("autoscale_on_load", True)))
        self.txt_layout_label.setText(str(data.get("general", {}).get("default_layout_label", "")))
        self.spn_value_precision.setValue(int(data.get("general", {}).get("value_precision", 6)))
        self.txt_export_dir.setText(str(data.get("misc", {}).get("default_export_dir", "")))
        root_dir = str(data.get("interactive", {}).get("jupyter_root_dir", "")).strip()
        if not root_dir:
            try:
                root_dir = self._manager.jupyter_root_directory()
            except Exception:
                root_dir = ""
        self.txt_jupyter_root.setText(root_dir)
        default_cmap = str(data.get("colormaps", {}).get("default", ""))
        idx = self.cmb_default_cmap.findData(default_cmap)
        if idx < 0:
            idx = 0
        block = self.cmb_default_cmap.blockSignals(True)
        self.cmb_default_cmap.setCurrentIndex(idx)
        self.cmb_default_cmap.blockSignals(block)
        block_table = self.table_colormaps.blockSignals(True)
        self.table_colormaps.setRowCount(0)
        variables = data.get("colormaps", {}).get("variables", {})
        if isinstance(variables, dict):
            for var, cmap in sorted(variables.items()):
                row = self.table_colormaps.rowCount()
                self.table_colormaps.insertRow(row)
                self.table_colormaps.setItem(row, 0, QtWidgets.QTableWidgetItem(str(var)))
                self.table_colormaps.setItem(row, 1, QtWidgets.QTableWidgetItem(str(cmap)))
        self.table_colormaps.blockSignals(block_table)
        self._appearance_profiles = copy.deepcopy(
            data.get("appearance", {}).get("profiles", {})
        )
        self._appearance_active_profile = str(
            data.get("appearance", {}).get("active_profile", "")
        )
        self._apply_appearance_tab()
        self._update_apply_button_state()

    # ---------- appearance helpers ----------

    def _apply_appearance_tab(self):
        appearance = sanitize_appearance(self._data.get("appearance", {}))
        self._appearance_profiles = copy.deepcopy(appearance.get("profiles", {}))
        self._appearance_active_profile = str(appearance.get("active_profile", ""))
        self._set_combobox_value(self.cmb_font_family, appearance.get("font_family"))
        self._set_font_size(float(appearance.get("font_size", 10.5)))
        self._set_combobox_value(self.cmb_accent, appearance.get("accent"))
        self._set_combobox_value(self.cmb_background, appearance.get("background"))
        self._set_combobox_value(self.cmb_button_shape, appearance.get("button_shape"))
        self._refresh_profile_combo()
        self._update_apply_button_state()

    def _set_combobox_value(self, combo: QtWidgets.QComboBox, value):
        idx = combo.findData(value)
        if idx < 0:
            idx = 0
        block = combo.blockSignals(True)
        combo.setCurrentIndex(idx)
        combo.blockSignals(block)

    def _set_font_size(self, value: float):
        block = self.spn_font_size.blockSignals(True)
        self.spn_font_size.setValue(max(8.0, min(14.0, float(value))))
        self.spn_font_size.blockSignals(block)

    def _refresh_profile_combo(self):
        current = self._appearance_active_profile
        self.cmb_profiles.blockSignals(True)
        self.cmb_profiles.clear()
        self.cmb_profiles.addItem("Custom (unsaved)", {"kind": "custom", "name": ""})
        for name in BUILTIN_PROFILES:
            self.cmb_profiles.addItem(f"{name} (built-in)", {"kind": "builtin", "name": name})
        for name in sorted(self._appearance_profiles):
            self.cmb_profiles.addItem(name, {"kind": "stored", "name": name})
        index = 0
        if current:
            for i in range(self.cmb_profiles.count()):
                info = self.cmb_profiles.itemData(i)
                if isinstance(info, dict) and info.get("name") == current:
                    index = i
                    break
        self.cmb_profiles.setCurrentIndex(index)
        self.cmb_profiles.blockSignals(False)
        self._update_profile_buttons()

    def _update_profile_buttons(self):
        info = self.cmb_profiles.currentData()
        is_stored = isinstance(info, dict) and info.get("kind") == "stored"
        self.btn_delete_profile.setEnabled(is_stored)
        self.btn_export_profile.setEnabled(info is not None)

    def _on_profile_selected(self, index: int):
        info = self.cmb_profiles.itemData(index)
        if not isinstance(info, dict):
            return
        kind = info.get("kind")
        name = info.get("name", "")
        if kind == "custom":
            self._appearance_active_profile = ""
            self._update_profile_buttons()
            self._update_apply_button_state()
            return
        if kind == "builtin":
            values = BUILTIN_PROFILES.get(name, {})
        else:
            values = self._appearance_profiles.get(name, {})
        if not values:
            return
        self._set_combobox_value(self.cmb_font_family, values.get("font_family"))
        self._set_font_size(float(values.get("font_size", 10.5)))
        self._set_combobox_value(self.cmb_accent, values.get("accent"))
        self._set_combobox_value(self.cmb_background, values.get("background"))
        self._set_combobox_value(self.cmb_button_shape, values.get("button_shape"))
        self._appearance_active_profile = name
        self._update_profile_buttons()
        self._update_apply_button_state()

    def _on_appearance_modified(self):
        self._appearance_active_profile = ""
        block = self.cmb_profiles.blockSignals(True)
        self.cmb_profiles.setCurrentIndex(0)
        self.cmb_profiles.blockSignals(block)
        self._update_profile_buttons()
        self._update_apply_button_state()

    def _current_appearance(self) -> Dict[str, object]:
        return {
            "font_family": self.cmb_font_family.currentData(),
            "font_size": float(self.spn_font_size.value()),
            "accent": self.cmb_accent.currentData(),
            "background": self.cmb_background.currentData(),
            "button_shape": self.cmb_button_shape.currentData(),
        }

    def _save_profile(self):
        name, ok = QtWidgets.QInputDialog.getText(self, "Save profile", "Profile name:")
        if not ok:
            return
        key = str(name).strip()
        if not key:
            return
        values = sanitize_profile_values(self._current_appearance())
        self._appearance_profiles[key] = values
        self._appearance_active_profile = key
        self._refresh_profile_combo()
        self._update_apply_button_state()

    def _delete_profile(self):
        info = self.cmb_profiles.currentData()
        if not isinstance(info, dict) or info.get("kind") != "stored":
            return
        name = info.get("name", "")
        if name in self._appearance_profiles:
            del self._appearance_profiles[name]
        self._appearance_active_profile = ""
        self._refresh_profile_combo()
        self._update_apply_button_state()

    def _export_profile(self):
        info = self.cmb_profiles.currentData()
        name = "Custom"
        if isinstance(info, dict):
            kind = info.get("kind")
            name = info.get("name", "Custom")
            if kind == "builtin":
                values = BUILTIN_PROFILES.get(name, {})
            elif kind == "stored":
                values = self._appearance_profiles.get(name, {})
            else:
                values = self._current_appearance()
        else:
            values = self._current_appearance()
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Export aesthetic profile",
            f"{name or 'profile'}.json",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            payload = {
                "name": name or "Custom",
                "profile": sanitize_profile_values(values),
            }
            Path(path).write_text(json.dumps(payload, indent=2))
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Export failed", str(exc))

    def _import_profile(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Import aesthetic profile",
            "",
            "JSON files (*.json);;All files (*)",
        )
        if not path:
            return
        try:
            data = json.loads(Path(path).read_text())
        except Exception as exc:
            QtWidgets.QMessageBox.warning(self, "Import failed", str(exc))
            return
        name = "Imported"
        if isinstance(data, dict):
            if isinstance(data.get("profile"), dict):
                values = sanitize_profile_values(data["profile"])
                name = str(data.get("name", name))
            else:
                values = sanitize_profile_values(data)
                if "name" in data:
                    name = str(data.get("name", name))
        else:
            QtWidgets.QMessageBox.warning(self, "Import failed", "File format not recognized.")
            return
        key = name.strip() or "Imported"
        base_key = key
        counter = 1
        while key in self._appearance_profiles or key in BUILTIN_PROFILES:
            counter += 1
            key = f"{base_key} {counter}"
        self._appearance_profiles[key] = values
        self._appearance_active_profile = key
        self._set_combobox_value(self.cmb_font_family, values.get("font_family"))
        self._set_font_size(float(values.get("font_size", 10.5)))
        self._set_combobox_value(self.cmb_accent, values.get("accent"))
        self._set_combobox_value(self.cmb_background, values.get("background"))
        self._set_combobox_value(self.cmb_button_shape, values.get("button_shape"))
        self._refresh_profile_combo()
        self._update_apply_button_state()


    def accept(self):
        self._data = copy.deepcopy(self._collect_data())
        self._update_apply_button_state()
        super().accept()

    def result_data(self) -> Dict[str, object]:
        return self._collect_data()

    def _apply_preferences(self):
        new_data = self._collect_data()
        if new_data == self._data:
            self._update_apply_button_state()
            return
        self._data = copy.deepcopy(new_data)
        self._manager.update(new_data)
        self._update_apply_button_state()

    def _on_field_modified(self, *_args):
        self._update_apply_button_state()

    def _update_apply_button_state(self):
        if self._apply_button is None:
            return
        if not hasattr(self, "cmb_default_cmap") or not hasattr(self, "table_colormaps"):
            self._apply_button.setEnabled(False)
            return
        try:
            dirty = self._collect_data() != self._data
        except Exception:
            dirty = True
        self._apply_button.setEnabled(dirty)

from __future__ import annotations

"""Helpers for translating appearance preferences into Qt stylesheets."""

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Tuple


@dataclass(frozen=True)
class FontOption:
    name: str
    css_stack: str


@dataclass(frozen=True)
class AccentOption:
    name: str
    base: str
    hover: str
    pressed: str
    text: str


@dataclass(frozen=True)
class BackgroundOption:
    name: str
    window: str
    panel: str
    surface_alt: str
    border: str
    divider: str
    text: str
    muted_text: str
    menu: str
    menu_text: str
    menu_border: str
    dock_title: str
    status: str
    input_bg: str
    input_border: str
    input_focus: str
    tab_bg: str
    tab_selected_bg: str
    tab_text: str
    tab_selected_text: str
    scrollbar_groove: str
    scrollbar_handle: str
    splitter: str
    header_bg: str


FONT_OPTIONS: Dict[str, FontOption] = {
    "Segoe UI": FontOption("Segoe UI", "'Segoe UI', 'Roboto', 'Noto Sans', sans-serif"),
    "Roboto": FontOption("Roboto", "'Roboto', 'Segoe UI', 'Noto Sans', sans-serif"),
    "Source Sans Pro": FontOption(
        "Source Sans Pro", "'Source Sans Pro', 'Segoe UI', 'Noto Sans', sans-serif"
    ),
    "Open Sans": FontOption("Open Sans", "'Open Sans', 'Segoe UI', 'Noto Sans', sans-serif"),
}


ACCENT_OPTIONS: Dict[str, AccentOption] = {
    "Azure": AccentOption("Azure", "#1a73e8", "#1967d2", "#174ea6", "#ffffff"),
    "Emerald": AccentOption("Emerald", "#0f9d58", "#0b8043", "#0a6631", "#ffffff"),
    "Coral": AccentOption("Coral", "#ff7043", "#ff5c29", "#f4511e", "#ffffff"),
    "Violet": AccentOption("Violet", "#8e63ce", "#7c52bd", "#6a46a6", "#ffffff"),
}


BACKGROUND_OPTIONS: Dict[str, BackgroundOption] = {
    "Frosted Light": BackgroundOption(
        "Frosted Light",
        window="#f1f3f4",
        panel="#ffffff",
        surface_alt="#eef1f6",
        border="#dadce0",
        divider="#c7cbd1",
        text="#202124",
        muted_text="#5f6368",
        menu="#ffffff",
        menu_text="#202124",
        menu_border="#dadce0",
        dock_title="#eef1f6",
        status="#eef1f6",
        input_bg="#ffffff",
        input_border="#d2d6dc",
        input_focus="#1a73e8",
        tab_bg="#eef1f6",
        tab_selected_bg="#ffffff",
        tab_text="#5f6368",
        tab_selected_text="#202124",
        scrollbar_groove="#e0e3eb",
        scrollbar_handle="#b9bec7",
        splitter="#dadce0",
        header_bg="#eef1f6",
    ),
    "Nightfall": BackgroundOption(
        "Nightfall",
        window="#1f2227",
        panel="#2b3038",
        surface_alt="#262a32",
        border="#3a404a",
        divider="#353b45",
        text="#e8eaed",
        muted_text="#b0b6c0",
        menu="#2e333b",
        menu_text="#e8eaed",
        menu_border="#3a404a",
        dock_title="#2e333b",
        status="#262a32",
        input_bg="#343a44",
        input_border="#474f5c",
        input_focus="#8ab4f8",
        tab_bg="#2a2f37",
        tab_selected_bg="#343a44",
        tab_text="#b0b6c0",
        tab_selected_text="#e8eaed",
        scrollbar_groove="#2b3038",
        scrollbar_handle="#525a66",
        splitter="#3a404a",
        header_bg="#343a44",
    ),
    "Sunrise": BackgroundOption(
        "Sunrise",
        window="#f7f3ef",
        panel="#ffffff",
        surface_alt="#f0e6de",
        border="#d9cfc3",
        divider="#cbbfb0",
        text="#3d3027",
        muted_text="#6a5a50",
        menu="#ffffff",
        menu_text="#3d3027",
        menu_border="#d9cfc3",
        dock_title="#f0e6de",
        status="#f2eae2",
        input_bg="#ffffff",
        input_border="#d4c9bb",
        input_focus="#f18f43",
        tab_bg="#efe2d6",
        tab_selected_bg="#ffffff",
        tab_text="#6a5a50",
        tab_selected_text="#3d3027",
        scrollbar_groove="#e4d7cb",
        scrollbar_handle="#c8b8a9",
        splitter="#d9cfc3",
        header_bg="#efe2d6",
    ),
}


BUTTON_SHAPE_OPTIONS: Dict[str, int] = {
    "Rounded": 8,
    "Pill": 18,
    "Square": 2,
}


BUILTIN_PROFILES: Dict[str, Dict[str, object]] = {
    "Serene Light": {
        "font_family": "Segoe UI",
        "font_size": 10.5,
        "accent": "Azure",
        "background": "Frosted Light",
        "button_shape": "Rounded",
    },
    "Midnight Dusk": {
        "font_family": "Source Sans Pro",
        "font_size": 11.0,
        "accent": "Violet",
        "background": "Nightfall",
        "button_shape": "Rounded",
    },
    "Citrus Glow": {
        "font_family": "Roboto",
        "font_size": 10.0,
        "accent": "Coral",
        "background": "Sunrise",
        "button_shape": "Pill",
    },
}


DEFAULT_PROFILE_NAME = "Serene Light"


def default_appearance() -> Dict[str, object]:
    base = BUILTIN_PROFILES[DEFAULT_PROFILE_NAME]
    return {
        "font_family": base["font_family"],
        "font_size": float(base["font_size"]),
        "accent": base["accent"],
        "background": base["background"],
        "button_shape": base["button_shape"],
        "active_profile": DEFAULT_PROFILE_NAME,
        "profiles": {},
    }


def sanitize_profile_values(
    values: Mapping[str, object], fallback: Mapping[str, object] | None = None
) -> Dict[str, object]:
    base = default_appearance() if fallback is None else dict(fallback)
    result = dict(base)

    font = str(values.get("font_family", result["font_family"])) if isinstance(values, Mapping) else result["font_family"]
    if font in FONT_OPTIONS:
        result["font_family"] = font

    size = values.get("font_size") if isinstance(values, Mapping) else None
    try:
        size = float(size)
    except Exception:
        size = result["font_size"]
    result["font_size"] = max(8.0, min(14.0, float(size)))

    accent = str(values.get("accent", result["accent"])) if isinstance(values, Mapping) else result["accent"]
    if accent in ACCENT_OPTIONS:
        result["accent"] = accent

    background = (
        str(values.get("background", result["background"]))
        if isinstance(values, Mapping)
        else result["background"]
    )
    if background in BACKGROUND_OPTIONS:
        result["background"] = background

    shape = str(values.get("button_shape", result["button_shape"])) if isinstance(values, Mapping) else result["button_shape"]
    if shape in BUTTON_SHAPE_OPTIONS:
        result["button_shape"] = shape

    return {
        "font_family": result["font_family"],
        "font_size": result["font_size"],
        "accent": result["accent"],
        "background": result["background"],
        "button_shape": result["button_shape"],
    }


def sanitize_appearance(data: Mapping[str, object] | None) -> Dict[str, object]:
    sanitized = default_appearance()
    if not isinstance(data, Mapping):
        return sanitized

    sanitized.update(sanitize_profile_values(data, sanitized))

    profiles: Dict[str, Dict[str, object]] = {}
    raw_profiles = data.get("profiles")
    if isinstance(raw_profiles, Mapping):
        for name, profile in raw_profiles.items():
            if not isinstance(name, str):
                continue
            key = name.strip()
            if not key:
                continue
            if not isinstance(profile, Mapping):
                continue
            profiles[key] = sanitize_profile_values(profile, sanitized)

    sanitized["profiles"] = profiles

    active = str(data.get("active_profile", sanitized.get("active_profile", "")))
    active = active.strip()
    if active in BUILTIN_PROFILES or active in profiles:
        sanitized["active_profile"] = active
    else:
        sanitized["active_profile"] = ""

    return sanitized


def available_profiles(custom_profiles: Mapping[str, Mapping[str, object]] | None = None) -> Dict[str, Dict[str, object]]:
    profiles = {name: dict(values) for name, values in BUILTIN_PROFILES.items()}
    if isinstance(custom_profiles, Mapping):
        for name, values in custom_profiles.items():
            profiles[name] = dict(values)
    return profiles


def _hex_to_rgb(color: str) -> Tuple[int, int, int]:
    value = color.lstrip("#")
    if len(value) != 6:
        raise ValueError(f"Invalid color {color!r}")
    return tuple(int(value[i : i + 2], 16) for i in range(0, 6, 2))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Iterable[int]) -> str:
    values = [max(0, min(255, int(round(channel)))) for channel in rgb]
    return "#" + "".join(f"{channel:02x}" for channel in values)


def _mix(color_a: str, color_b: str, ratio: float) -> str:
    ratio = max(0.0, min(1.0, float(ratio)))
    rgb_a = _hex_to_rgb(color_a)
    rgb_b = _hex_to_rgb(color_b)
    blended = [
        rgb_a[i] * (1.0 - ratio) + rgb_b[i] * ratio
        for i in range(3)
    ]
    return _rgb_to_hex(blended)


def build_stylesheet(appearance: Mapping[str, object] | None) -> str:
    settings = sanitize_profile_values(appearance or {}, default_appearance())
    font = FONT_OPTIONS[settings["font_family"]]
    accent = ACCENT_OPTIONS[settings["accent"]]
    background = BACKGROUND_OPTIONS[settings["background"]]
    button_radius = BUTTON_SHAPE_OPTIONS[settings["button_shape"]]
    tab_radius = max(button_radius + 6, 10)
    control_radius = max(button_radius, 4)

    tab_min_width = max(int(settings["font_size"] * 9.5), 140)
    button_min_width = max(int(settings["font_size"] * 7.5), 120)
    button_min_height = max(int(settings["font_size"] * 2.8), 34)
    checkbox_min_height = max(int(settings["font_size"] * 2.2), 26)
    label_min_height = max(int(settings["font_size"] * 1.8), 20)
    label_min_width = max(int(settings["font_size"] * 6.0), 96)

    accent_disabled = _mix(accent.base, background.panel, 0.55)
    accent_border = _mix(accent.base, background.border, 0.35)
    focus_outline = background.input_focus if background.input_focus else accent.base
    hover_surface = _mix(accent.base, background.panel, 0.9)

    return f"""
QWidget {{
    font-family: {font.css_stack};
    font-size: {settings['font_size']:.1f}pt;
    color: {background.text};
}}
QMainWindow, QDialog {{
    background-color: {background.window};
}}
QMenuBar {{
    background: {background.surface_alt};
    color: {background.text};
    padding: 4px 8px;
    border-bottom: 1px solid {background.border};
}}
QMenuBar::item {{
    background: transparent;
    padding: 4px 12px;
    border-radius: {control_radius + 2}px;
}}
QMenuBar::item:selected {{
    background: {_mix(accent.base, background.panel, 0.85)};
    color: {accent.base};
}}
QMenu {{
    background: {background.menu};
    color: {background.menu_text};
    border: 1px solid {background.menu_border};
    border-radius: {control_radius + 4}px;
    padding: 6px 0;
}}
QMenu::item {{
    padding: 6px 20px;
    border-radius: {control_radius}px;
}}
QMenu::item:selected {{
    background: {_mix(accent.base, background.menu, 0.85)};
    color: {accent.base};
}}
QSplitter::handle:horizontal, QSplitter::handle:vertical {{
    background: {background.splitter};
    border-radius: {control_radius}px;
}}
QSplitter::handle:horizontal:hover, QSplitter::handle:vertical:hover {{
    background: {_mix(background.splitter, accent.base, 0.3)};
}}
QTabWidget::pane {{
    border: 1px solid {background.border};
    border-radius: {tab_radius}px;
    padding: 8px;
    background: {background.panel};
}}
QTabBar::tab {{
    background: {background.tab_bg};
    border: 1px solid {background.border};
    border-bottom: none;
    border-top-left-radius: {tab_radius}px;
    border-top-right-radius: {tab_radius}px;
    padding: 12px 32px;
    margin: 0 8px;
    min-width: {tab_min_width}px;
    color: {background.tab_text};
    font-weight: 500;
}}
QTabBar::tab:selected {{
    background: {background.tab_selected_bg};
    color: {background.tab_selected_text};
    border-color: {accent_border};
    font-weight: 600;
}}
QTabBar::tab:hover {{
    background: {hover_surface};
}}
QFrame[modernSection="true"] {{
    background: {background.panel};
    border: 1px solid {background.border};
    border-radius: {tab_radius}px;
    padding: 10px 14px;
}}
QLabel[modernSectionTitle="true"] {{
    font-size: {max(settings['font_size'] - 0.5, 8.5):.1f}pt;
    font-weight: 600;
    color: {background.muted_text};
    padding-bottom: 4px;
    qproperty-wordWrap: false;
    min-width: 0;
}}
QGroupBox {{
    border: 1px solid {background.border};
    border-radius: {tab_radius - 2}px;
    margin-top: 12px;
    padding: 12px;
    background: {background.panel};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 18px;
    padding: 0 6px;
    color: {background.muted_text};
    font-weight: 600;
}}
QTreeWidget, QListWidget, QTextEdit, QPlainTextEdit, QTableView, QLineEdit,
QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {background.input_bg};
    border: 1px solid {background.input_border};
    border-radius: {control_radius + 2}px;
    padding: 4px 6px;
}}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background: {_mix(accent.base, background.panel, 0.8)};
    color: {background.text};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {focus_outline};
}}
QPushButton, QToolButton {{
    background-color: {accent.base};
    color: {accent.text};
    border: none;
    border-radius: {button_radius}px;
    padding: 8px 20px;
    font-weight: 600;
    min-width: {button_min_width}px;
    min-height: {button_min_height}px;
}}
QPushButton:disabled, QToolButton:disabled {{
    background-color: {accent_disabled};
    color: {_mix(background.text, background.panel, 0.55)};
}}
QPushButton:hover, QToolButton:hover {{
    background-color: {accent.hover};
}}
QPushButton:pressed, QToolButton:pressed {{
    background-color: {accent.pressed};
}}
QToolButton[headerAction="true"] {{
    background: transparent;
    color: {background.muted_text};
    padding: 4px 10px;
    border-radius: {control_radius + 2}px;
    font-weight: 500;
    min-width: 0;
    min-height: 0;
}}
QToolButton[headerAction="true"]:hover {{
    background: {_mix(accent.base, background.panel, 0.85)};
    color: {accent.base};
}}
QDockWidget {{
    titlebar-close-icon: url();
    titlebar-normal-icon: url();
    font-size: {settings['font_size']:.1f}pt;
}}
QDockWidget::title {{
    text-align: center;
    background: {background.dock_title};
    border: 1px solid {background.border};
    border-radius: {tab_radius}px {tab_radius}px 0 0;
    padding: 6px;
    color: {background.muted_text};
    font-weight: 600;
}}
QHeaderView::section {{
    background: {background.header_bg};
    border: none;
    padding: 6px 10px;
    font-weight: 600;
    color: {background.muted_text};
}}
QStatusBar {{
    background: {background.status};
    border-top: 1px solid {background.border};
}}
QWidget#datasetsPane {{
    background: {background.panel};
    border: 1px solid {background.border};
    border-radius: {tab_radius + 6}px;
}}
QWidget#processingContainer {{
    background: {background.panel};
    border: 1px solid {background.border};
    border-radius: {tab_radius + 6}px;
}}
QWidget#processingHeader {{
    border-bottom: 1px solid {background.divider};
    background: {_mix(background.surface_alt, background.panel, 0.4)};
    border-top-left-radius: {tab_radius + 6}px;
    border-top-right-radius: {tab_radius + 6}px;
}}
QWidget#processingContent {{
    border-bottom-left-radius: {tab_radius + 6}px;
    border-bottom-right-radius: {tab_radius + 6}px;
}}
QLabel#processingPlaceholder {{
    color: {background.muted_text};
    padding: 16px;
}}
QScrollBar:vertical {{
    width: 12px;
    background: transparent;
}}
QScrollBar::handle:vertical {{
    background: {background.scrollbar_handle};
    border-radius: {control_radius}px;
}}
QScrollBar::handle:vertical:hover {{
    background: {_mix(background.scrollbar_handle, accent.base, 0.35)};
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar:horizontal {{
    height: 12px;
    background: transparent;
}}
QScrollBar::handle:horizontal {{
    background: {background.scrollbar_handle};
    border-radius: {control_radius}px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {_mix(background.scrollbar_handle, accent.base, 0.35)};
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}
QToolTip {{
    background: {background.panel};
    color: {background.text};
    border: 1px solid {background.border};
    padding: 6px 8px;
}}
QCheckBox, QRadioButton {{
    spacing: 6px;
    min-width: {label_min_width}px;
    min-height: {checkbox_min_height}px;
    qproperty-wordWrap: true;
}}
QLabel {{
    min-width: {label_min_width}px;
    min-height: {label_min_height}px;
    qproperty-wordWrap: true;
}}
""".strip()



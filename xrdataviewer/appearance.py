from __future__ import annotations

"""Helpers for translating appearance preferences into a restrained Qt stylesheet."""

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
    "Segoe UI": FontOption("Segoe UI", "'Segoe UI', 'Noto Sans', 'Roboto', sans-serif"),
    "Roboto": FontOption("Roboto", "'Roboto', 'Segoe UI', 'Noto Sans', sans-serif"),
    "Noto Sans": FontOption("Noto Sans", "'Noto Sans', 'Segoe UI', 'Roboto', sans-serif"),
    "Source Sans Pro": FontOption(
        "Source Sans Pro", "'Source Sans Pro', 'Segoe UI', 'Noto Sans', sans-serif"
    ),
}


ACCENT_OPTIONS: Dict[str, AccentOption] = {
    "Blue": AccentOption("Blue", "#3a6db3", "#315f9b", "#285282", "#ffffff"),
    "Green": AccentOption("Green", "#4a8f49", "#3f7a3e", "#356535", "#ffffff"),
    "Orange": AccentOption("Orange", "#d98032", "#c46f25", "#a95c1e", "#ffffff"),
    "Plum": AccentOption("Plum", "#8060a5", "#6f5290", "#5e4578", "#ffffff"),
}


BACKGROUND_OPTIONS: Dict[str, BackgroundOption] = {
    "Classic Light": BackgroundOption(
        "Classic Light",
        window="#f2f2f3",
        panel="#ffffff",
        surface_alt="#e6e7e9",
        border="#c1c4ca",
        divider="#d2d5da",
        text="#202124",
        muted_text="#5f6368",
        menu="#ffffff",
        menu_text="#202124",
        menu_border="#c1c4ca",
        dock_title="#eceef1",
        status="#e7e9ec",
        input_bg="#ffffff",
        input_border="#b7bbc1",
        input_focus="#3a6db3",
        tab_bg="#e7e9ec",
        tab_selected_bg="#ffffff",
        tab_text="#4d5156",
        tab_selected_text="#202124",
        scrollbar_groove="#e0e3e8",
        scrollbar_handle="#c7cbd1",
        splitter="#c7cbd1",
        header_bg="#edeff2",
    ),
    "Slate Dark": BackgroundOption(
        "Slate Dark",
        window="#1f2227",
        panel="#2b3037",
        surface_alt="#262a31",
        border="#3a414a",
        divider="#343a44",
        text="#eceef1",
        muted_text="#a9afb7",
        menu="#2e333b",
        menu_text="#eceef1",
        menu_border="#3a414a",
        dock_title="#2e333b",
        status="#242930",
        input_bg="#323841",
        input_border="#444b55",
        input_focus="#648ad1",
        tab_bg="#2a2f37",
        tab_selected_bg="#323841",
        tab_text="#b0b6c0",
        tab_selected_text="#eceef1",
        scrollbar_groove="#2a2f37",
        scrollbar_handle="#4a525e",
        splitter="#3a414a",
        header_bg="#30363f",
    ),
    "Warm Gray": BackgroundOption(
        "Warm Gray",
        window="#eeedeb",
        panel="#f8f7f5",
        surface_alt="#e3e1df",
        border="#c5c1bc",
        divider="#d3cfcb",
        text="#2f302c",
        muted_text="#6c6964",
        menu="#f8f7f5",
        menu_text="#2f302c",
        menu_border="#c5c1bc",
        dock_title="#e8e6e3",
        status="#e2dfdb",
        input_bg="#ffffff",
        input_border="#c1bcb6",
        input_focus="#d98032",
        tab_bg="#e8e6e3",
        tab_selected_bg="#fdfcfa",
        tab_text="#615f5a",
        tab_selected_text="#2f302c",
        scrollbar_groove="#dedbd7",
        scrollbar_handle="#beb9b2",
        splitter="#c5c1bc",
        header_bg="#eae7e3",
    ),
}


BACKGROUND_ALIASES = {
    "Frosted Light": "Classic Light",
    "Nightfall": "Slate Dark",
    "Sunrise": "Warm Gray",
}


BUTTON_SHAPE_OPTIONS: Dict[str, int] = {
    "Square": 2,
    "Soft": 4,
    "Rounded": 6,
}


BUTTON_SHAPE_ALIASES = {
    "Pill": "Rounded",
}


BUILTIN_PROFILES: Dict[str, Dict[str, object]] = {
    "Classic Light": {
        "font_family": "Segoe UI",
        "font_size": 10.0,
        "accent": "Blue",
        "background": "Classic Light",
        "button_shape": "Square",
    },
    "Slate Dark": {
        "font_family": "Noto Sans",
        "font_size": 10.0,
        "accent": "Plum",
        "background": "Slate Dark",
        "button_shape": "Soft",
    },
    "Warm Gray": {
        "font_family": "Roboto",
        "font_size": 10.0,
        "accent": "Orange",
        "background": "Warm Gray",
        "button_shape": "Soft",
    },
}


DEFAULT_PROFILE_NAME = "Classic Light"


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
    values: Mapping[str, object] | None, base: Mapping[str, object]
) -> Dict[str, object]:
    result = dict(base)

    if isinstance(values, Mapping):
        font = str(values.get("font_family", result["font_family"]))
        if font in FONT_OPTIONS:
            result["font_family"] = font

        size = values.get("font_size")
        try:
            size = float(size)
        except Exception:
            size = result["font_size"]
        result["font_size"] = max(8.0, min(14.0, float(size)))

        accent = str(values.get("accent", result["accent"]))
        if accent in ACCENT_OPTIONS:
            result["accent"] = accent

        background = str(values.get("background", result["background"]))
        background = BACKGROUND_ALIASES.get(background, background)
        if background in BACKGROUND_OPTIONS:
            result["background"] = background

        shape = str(values.get("button_shape", result["button_shape"]))
        shape = BUTTON_SHAPE_ALIASES.get(shape, shape)
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
            if not isinstance(name, str) or not isinstance(profile, Mapping):
                continue
            key = name.strip()
            if not key:
                continue
            profiles[key] = sanitize_profile_values(profile, sanitized)
    sanitized["profiles"] = profiles

    active = str(data.get("active_profile", sanitized.get("active_profile", ""))).strip()
    if active in BUILTIN_PROFILES or active in profiles:
        sanitized["active_profile"] = active
    else:
        sanitized["active_profile"] = ""

    return sanitized


def available_profiles(
    custom_profiles: Mapping[str, Mapping[str, object]] | None = None
) -> Dict[str, Dict[str, object]]:
    profiles = {name: dict(values) for name, values in BUILTIN_PROFILES.items()}
    if isinstance(custom_profiles, Mapping):
        for name, values in custom_profiles.items():
            if isinstance(name, str) and isinstance(values, Mapping):
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


def build_stylesheet(
    appearance: Mapping[str, object] | None,
    *,
    support_checkable_wordwrap: bool = True,
) -> str:
    defaults = default_appearance()
    settings = sanitize_profile_values(appearance or {}, defaults)

    font = FONT_OPTIONS.get(settings["font_family"], FONT_OPTIONS[defaults["font_family"]])
    accent = ACCENT_OPTIONS.get(settings["accent"], ACCENT_OPTIONS[defaults["accent"]])
    background = BACKGROUND_OPTIONS.get(
        settings["background"], BACKGROUND_OPTIONS[defaults["background"]]
    )
    button_radius = max(0, min(8, BUTTON_SHAPE_OPTIONS.get(settings["button_shape"], 2)))
    tab_radius = max(button_radius, 2)

    font_size = float(settings["font_size"])
    tab_min_width = max(int(font_size * 6.8), 112)
    compact_tab_min_width = max(int(font_size * 5.2), 92)
    button_min_width = max(int(font_size * 4.0), 72)
    button_min_height = max(int(font_size * 1.9), 24)
    compact_button_min_width = max(int(font_size * 3.2), 60)
    compact_button_min_height = max(int(font_size * 1.7), 22)
    button_padding_y = max(int(font_size * 0.35), 3)
    button_padding_x = max(int(font_size * 0.9), 8)
    compact_button_padding_y = max(button_padding_y - 1, 2)
    compact_button_padding_x = max(button_padding_x - 2, 6)
    checkbox_min_height = max(int(font_size * 2.1), 24)
    label_min_height = max(int(font_size * 1.9), 20)
    label_min_width = max(int(font_size * 5.6), 96)

    hover_surface = _mix(accent.base, background.panel, 0.85)
    pressed_surface = _mix(accent.base, background.panel, 0.7)
    disabled_text = _mix(background.text, background.panel, 0.6)
    focus_outline = background.input_focus or accent.base
    menubar_bg = _mix(background.surface_alt, background.panel, 0.5)
    subtle_border = _mix(background.border, background.panel, 0.7)

    checkable_props = [
        f"    min-height: {checkbox_min_height}px;",
        "    spacing: 6px;",
    ]
    if support_checkable_wordwrap:
        checkable_props.append("    qproperty-wordWrap: true;")
    checkable_block = "\n".join(checkable_props)

    return f"""
QWidget {{
    font-family: {font.css_stack};
    font-size: {font_size:.1f}pt;
    color: {background.text};
}}
QMainWindow, QDialog {{
    background: {background.window};
}}
QTabWidget::pane {{
    border: 1px solid {background.border};
    padding: 6px;
    background: {background.panel};
}}
QTabWidget[compactTabs="true"]::pane {{
    padding: 4px;
}}
QTabBar::tab {{
    padding: 4px 12px;
    margin: 0 2px;
    min-width: {tab_min_width}px;
    border: 1px solid {background.border};
    border-bottom: none;
    border-top-left-radius: {tab_radius}px;
    border-top-right-radius: {tab_radius}px;
    background: {background.tab_bg};
    color: {background.tab_text};
}}
QTabBar::tab:selected {{
    background: {background.tab_selected_bg};
    color: {background.tab_selected_text};
    border-color: {accent.base};
    font-weight: 600;
}}
QTabBar::tab:hover {{
    background: {hover_surface};
}}
QTabWidget[compactTabs="true"] QTabBar::tab {{
    min-width: {compact_tab_min_width}px;
    padding: 3px 10px;
}}
QPushButton, QToolButton {{
    padding: {button_padding_y}px {button_padding_x}px;
    min-width: {button_min_width}px;
    min-height: {button_min_height}px;
    border: 1px solid {background.border};
    border-radius: {button_radius}px;
    background: {background.panel};
    color: {background.text};
}}
QPushButton[sizeVariant="compact"], QToolButton[sizeVariant="compact"] {{
    padding: {compact_button_padding_y}px {compact_button_padding_x}px;
    min-width: {compact_button_min_width}px;
    min-height: {compact_button_min_height}px;
}}
QPushButton:disabled, QToolButton:disabled {{
    color: {disabled_text};
}}
QPushButton:hover, QToolButton:hover {{
    background: {hover_surface};
}}
QPushButton:pressed, QToolButton:pressed {{
    background: {pressed_surface};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit, QListWidget, QTreeWidget, QTableView {{
    background: {background.input_bg};
    border: 1px solid {background.input_border};
    border-radius: {max(button_radius, 2)}px;
    padding: 2px 4px;
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus, QTextEdit:focus, QPlainTextEdit:focus {{
    border-color: {focus_outline};
}}
QGroupBox {{
    border: 1px solid {background.border};
    border-radius: {max(button_radius, 2)}px;
    margin-top: 10px;
    padding: 8px;
    background: {background.panel};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 4px;
    color: {background.muted_text};
}}
QMenuBar {{
    background: {menubar_bg};
    padding: 2px 6px;
}}
QMenuBar::item {{
    padding: 4px 10px;
    border-radius: {max(button_radius, 2)}px;
}}
QMenuBar::item:selected {{
    background: {hover_surface};
}}
QMenu {{
    background: {background.menu};
    color: {background.menu_text};
    border: 1px solid {background.menu_border};
    padding: 4px 0;
}}
QMenu::item {{
    padding: 4px 14px;
}}
QMenu::item:selected {{
    background: {hover_surface};
    color: {accent.base};
}}
QSplitter::handle {{
    background: {background.splitter};
}}
QStatusBar {{
    background: {background.status};
    border-top: 1px solid {subtle_border};
}}
QLabel {{
    min-width: {label_min_width}px;
    min-height: {label_min_height}px;
    qproperty-wordWrap: true;
}}
QCheckBox, QRadioButton {{
{checkable_block}
}}
""".strip()

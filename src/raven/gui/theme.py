"""
Light and dark themes: a Fusion palette plus one app-wide stylesheet, and theme-colored icons.
"""
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer

ASSETS_DIR = Path(__file__).with_name("assets")

# the fill colors the icons in gui/assets were drawn with, swapped for the theme's text color
ICON_SOURCE_COLORS = ("#1C274C", "#000000")

THEMES = {
    "light": {
        "window": "#f3f4f6", "base": "#ffffff", "alt": "#eceef1", "text": "#1c1f24", "muted": "#6b7280",
        "border": "#d3d7dd", "button": "#e8eaee", "highlight": "#2563eb", "header": "#ffffff",
    },
    "dark": {
        "window": "#1d2025", "base": "#25292f", "alt": "#2d3239", "text": "#e6e8eb", "muted": "#99a1ac",
        "border": "#3a4048", "button": "#30353d", "highlight": "#4c8dff", "header": "#16181c",
    },
}

# accent colors for Settings > Appearance: name -> (light theme shade, dark theme shade)
ACCENTS = {
    "blue": ("#2563eb", "#4c8dff"),
    "violet": ("#7c3aed", "#9466ff"),
    "teal": ("#0d8a7e", "#14a394"),
    "orange": ("#d9620f", "#ef7a2a"),
    "rose": ("#db2763", "#ee4f86"),
}
DEFAULT_ACCENT = "blue"
_accent = DEFAULT_ACCENT


def set_accent(name: str):
    """
    Pick the accent color; it takes effect the next time apply_theme runs.
    """
    global _accent
    _accent = name if name in ACCENTS else DEFAULT_ACCENT


def accent_color(accent: str, theme: str) -> str:
    return ACCENTS[accent][0 if theme == "light" else 1]


STYLESHEET = """
#header {{ background: {header}; border-bottom: 1px solid {border}; }}
#brand {{
    color: {text}; font-family: Georgia, Cambria, "Times New Roman", serif;
    font-size: 30px; font-weight: 600; letter-spacing: 4px; padding: 0 18px 0 6px;
}}
QPushButton#navTab {{
    background: transparent; color: {muted}; border: none; border-bottom: 3px solid transparent;
    padding: 12px 24px 9px 24px; font-size: 16px; font-weight: 600;
}}
QPushButton#navTab:hover {{ color: {text}; background: {alt}; }}
QPushButton#navTab:checked {{ color: {text}; border-bottom: 3px solid {highlight}; }}
QToolButton#themeButton {{ border: 1px solid transparent; border-radius: 8px; padding: 6px; }}
QToolButton#themeButton:hover {{ background: {alt}; border-color: {border}; }}
QLabel#demoPill {{
    background: #f5a524; color: #2b1d00; border-radius: 11px; padding: 3px 12px; font-weight: 600;
}}
QLabel#hint {{ color: {muted}; }}
QToolButton#advancedToggle {{ border: none; font-weight: bold; }}
QFrame#videoBar {{ background: {window}; border: none; }}
QWidget#alertsStrip {{ background: {window}; }}
QFrame#alertInfo {{ background: {base}; border: 1px solid {border}; border-radius: 8px; }}
QPushButton#subTab {{
    background: {button}; color: {text}; border: 1px solid {border}; padding: 6px 18px; font-weight: 600;
}}
QPushButton#subTab[position="first"] {{ border-top-left-radius: 6px; border-bottom-left-radius: 6px; }}
QPushButton#subTab[position="last"] {{ border-top-right-radius: 6px; border-bottom-right-radius: 6px; border-left: none; }}
QPushButton#subTab:hover {{ background: {alt}; }}
QPushButton#subTab:checked {{ background: {highlight}; color: #ffffff; border-color: {highlight}; }}
QFrame#feedCard {{ background: {alt}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#barSeparator {{ color: {border}; }}
QFrame#logPanel {{ background: {window}; }}
QFrame#linkBar {{ background: {window}; border-top: 1px solid {border}; }}
QFrame#profilePanel {{ background: {base}; border: 1px solid {border}; border-radius: 10px; }}
QLabel#profileAvatar {{ background: {alt}; border: 1px solid {border}; border-radius: 28px; }}
QLabel#profileName {{ font-size: 17px; font-weight: 600; }}
QLabel#profileHeading {{ color: {muted}; font-weight: 600; letter-spacing: 1px; }}
QToolButton#eyeButton {{ border: 1px solid transparent; border-radius: 4px; padding: 0; background: transparent; }}
QToolButton#eyeButton:hover {{ background: {alt}; border-color: {border}; }}
QPushButton#logoutButton {{
    background: transparent; color: {text}; border: 1px solid {border}; border-radius: 6px; padding: 4px 12px;
}}
QPushButton#logoutButton:hover {{ background: {alt}; }}
QFrame#profileDrone {{ background: {alt}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#profileDrone QLabel {{ background: transparent; }}
QFrame#profileDrone QProgressBar {{ background: {base}; border: 1px solid {border}; border-radius: 4px; }}
QSplitter::handle {{ background: {border}; }}
QSplitter::handle:horizontal {{ width: 3px; }}
QSplitter::handle:vertical {{ height: 3px; }}
QSplitter::handle:hover, QSplitter::handle:pressed {{ background: {highlight}; }}
QToolButton#accentSwatch {{ border: 2px solid transparent; border-radius: 15px; padding: 2px; }}
QToolButton#accentSwatch:hover {{ border-color: {border}; }}
QToolButton#accentSwatch:checked {{ border-color: {text}; }}
QPushButton#backToMap {{
    background: {highlight}; color: #ffffff; border: none; border-radius: 6px;
    padding: 8px 20px; font-size: 14px; font-weight: 600;
}}
QPushButton#backToMap:hover {{ background: {text}; color: {base}; }}
"""


def system_theme() -> str:
    """
    The operating system's light/dark preference, used until the operator picks one.
    """
    scheme = QGuiApplication.styleHints().colorScheme()
    return "dark" if scheme == Qt.ColorScheme.Dark else "light"


def colors(name: str) -> dict:
    """
    The theme's colors, with the chosen accent as its highlight.
    """
    return {**THEMES[name], "highlight": accent_color(_accent, name)}


def apply_theme(app, name: str):
    """
    Switch the whole application to the light or dark theme.
    """
    c = colors(name)
    app.setStyle("Fusion")

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(c["window"]))
    palette.setColor(QPalette.WindowText, QColor(c["text"]))
    palette.setColor(QPalette.Base, QColor(c["base"]))
    palette.setColor(QPalette.AlternateBase, QColor(c["alt"]))
    palette.setColor(QPalette.Text, QColor(c["text"]))
    palette.setColor(QPalette.PlaceholderText, QColor(c["muted"]))
    palette.setColor(QPalette.Button, QColor(c["button"]))
    palette.setColor(QPalette.ButtonText, QColor(c["text"]))
    palette.setColor(QPalette.Highlight, QColor(c["highlight"]))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ToolTipBase, QColor(c["base"]))
    palette.setColor(QPalette.ToolTipText, QColor(c["text"]))
    palette.setColor(QPalette.Link, QColor(c["highlight"]))
    palette.setColor(QPalette.Mid, QColor(c["border"]))
    palette.setColor(QPalette.Midlight, QColor(c["alt"]))
    palette.setColor(QPalette.Light, QColor(c["alt"]))
    palette.setColor(QPalette.Dark, QColor(c["border"]))
    for role in (QPalette.Text, QPalette.WindowText, QPalette.ButtonText):
        palette.setColor(QPalette.Disabled, role, QColor(c["muted"]))
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET.format(**c))


def themed_icon(file_name: str, name: str, size: int = 48) -> QIcon:
    """
    Load an SVG from gui/assets and recolor it to the theme's text color, so it reads on light and dark.
    """
    svg = (ASSETS_DIR / file_name).read_text(encoding="utf-8")
    for color in ICON_SOURCE_COLORS:
        svg = svg.replace(color, THEMES[name]["text"]).replace(color.lower(), THEMES[name]["text"])
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    renderer.render(painter, QRectF(0, 0, size, size))
    painter.end()
    return QIcon(pixmap)


def media_icon(kind: str, name: str, size: int = 48) -> QIcon:
    """
    Playback icons (play, pause, stop, step_back, step_forward, prev_alert, next_alert) drawn in the theme's
    text color, since the style's built-in ones are too dark to read in dark mode.
    """
    from PySide6.QtCore import QPointF
    from PySide6.QtGui import QPolygonF

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(THEMES[name]["text"]))
    scale = size / 48

    def triangle(x_from, x_to, top=10, bottom=38):
        # pointing right when x_to > x_from, left otherwise
        middle = (top + bottom) / 2
        painter.drawPolygon(QPolygonF([QPointF(x_from * scale, top * scale), QPointF(x_to * scale, middle * scale),
                                       QPointF(x_from * scale, bottom * scale)]))

    def bar(x, width=5, top=10, bottom=38):
        painter.drawRect(QRectF(x * scale, top * scale, width * scale, (bottom - top) * scale))

    if kind == "play":
        triangle(14, 38)
    elif kind == "pause":
        bar(13, 8)
        bar(27, 8)
    elif kind == "stop":
        painter.drawRect(QRectF(12 * scale, 12 * scale, 24 * scale, 24 * scale))
    elif kind == "step_forward":
        triangle(10, 31)
        bar(33)
    elif kind == "step_back":
        bar(10)
        triangle(38, 17)
    elif kind == "next_alert":
        triangle(5, 21, 12, 36)
        triangle(20, 36, 12, 36)
        bar(37, 5, 12, 36)
    elif kind == "prev_alert":
        bar(6, 5, 12, 36)
        triangle(27, 11, 12, 36)
        triangle(43, 27, 12, 36)
    painter.end()
    return QIcon(pixmap)

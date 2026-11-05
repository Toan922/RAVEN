"""
Light and dark themes: a Fusion palette plus one app-wide stylesheet, and theme-colored icons.
"""
from pathlib import Path

from PySide6.QtCore import QByteArray, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer

ASSETS_DIR = Path(__file__).with_name("assets")

# the fill color the icons in gui/assets were drawn with, swapped for the theme's text color
ICON_SOURCE_COLOR = "#1C274C"

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
QFrame#videoBar {{ background: {window}; border: none; border-top: 1px solid {border}; }}
QWidget#alertsStrip {{ background: {window}; border-left: 1px solid {border}; }}
QFrame#alertInfo {{ background: {base}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#feedCard {{ background: {alt}; border: 1px solid {border}; border-radius: 8px; }}
QFrame#barSeparator {{ color: {border}; }}
QFrame#logPanel {{ background: {window}; border-left: 1px solid {border}; }}
QFrame#linkBar {{ background: {window}; border-top: 1px solid {border}; }}
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
    return THEMES[name]


def apply_theme(app, name: str):
    """
    Switch the whole application to the light or dark theme.
    """
    c = THEMES[name]
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
    svg = svg.replace(ICON_SOURCE_COLOR, THEMES[name]["text"]).replace(ICON_SOURCE_COLOR.lower(), THEMES[name]["text"])
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

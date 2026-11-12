"""
A one-line label that cuts its text short with "…" when it doesn't fit, instead of growing.

Labels whose text changes as alerts and frames come in (counts, times, positions) would otherwise raise the
minimum width of their panel each time, and the splitter would take that space from the panel next to it.
"""
from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QLabel, QSizePolicy, QStyle, QStyleOption


class ElidedLabel(QLabel):
    def __init__(self, text: str = "", parent=None, min_width: int = 40, auto_tooltip: bool = True):
        super().__init__(text, parent)
        self.setTextFormat(Qt.PlainText)  # it draws its text itself, as plain text; put markup in a separate QLabel
        self._min_width = min_width
        self._auto_tooltip = auto_tooltip  # off when the caller sets its own tooltip
        # takes its full width when there is room, and gives it up down to min_width when there is not
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)

    def minimumSizeHint(self) -> QSize:
        return QSize(self._min_width, super().minimumSizeHint().height())

    def sizeHint(self) -> QSize:
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + 4, super().sizeHint().height())

    def setText(self, text: str):
        super().setText(text)
        self._update_tooltip()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_tooltip()

    def _update_tooltip(self):
        # the whole text on hover, but only when some of it is cut off
        if self._auto_tooltip:
            cut = self.fontMetrics().horizontalAdvance(self.text()) > self.contentsRect().width()
            self.setToolTip(self.text() if cut else "")

    def paintEvent(self, _event):
        painter = QPainter(self)
        option = QStyleOption()
        option.initFrom(self)
        self.style().drawPrimitive(QStyle.PE_Widget, option, painter, self)
        rect = self.contentsRect()
        text = self.fontMetrics().elidedText(self.text(), Qt.ElideRight, rect.width())
        self.style().drawItemText(painter, rect, int(self.alignment()) | Qt.AlignVCenter, self.palette(),
                                  self.isEnabled(), text, self.foregroundRole())

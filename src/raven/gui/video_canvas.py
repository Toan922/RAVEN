"""
The video area shared by the Feed tab's Live and Review views: a frame scaled to fit, with detection boxes.
"""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget

BOX_COLOR = QColor("#e5484d")


class VideoCanvas(QWidget):
    """
    Draws the current frame scaled to fit, with detection boxes (solid on a checked frame, dashed while held),
    or a message when there is nothing to show.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(320, 240)
        self._image: QImage | None = None
        self._boxes: list = []
        self._held = False
        self._centered = False
        self._message = "Double-click an alert, or pick one from the list on the right, to review its footage."

    def show_frame(self, image: QImage | None, boxes=(), held=False, message="", centered=False):
        """
        With a frame, a message shows in a band along the bottom, or in a box in the middle when centered.
        """
        self._image = image
        self._boxes = list(boxes)
        self._held = held
        self._message = message
        self._centered = centered
        self.update()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#000000"))
        if self._image is None or self._image.isNull():
            painter.setPen(QColor("#c8ccd2"))
            painter.drawText(self.rect().adjusted(40, 40, -40, -40), Qt.AlignCenter | Qt.TextWordWrap, self._message)
            return

        # fit the frame in the widget, keeping its aspect ratio
        size = self._image.size().scaled(self.size(), Qt.KeepAspectRatio)
        x = (self.width() - size.width()) / 2
        y = (self.height() - size.height()) / 2
        target = QRectF(x, y, size.width(), size.height())
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.drawImage(target, self._image)

        scale = size.width() / self._image.width()
        pen = QPen(BOX_COLOR)
        pen.setWidthF(2.5)
        if self._held:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setRenderHint(QPainter.Antialiasing)
        for box in self._boxes:
            x1, y1, x2, y2 = box["box_xyxy"]
            painter.drawRect(QRectF(QPointF(x + x1 * scale, y + y1 * scale), QPointF(x + x2 * scale, y + y2 * scale)))

        if self._message:
            painter.setPen(QColor("#ffffff"))
            if self._centered:
                bounds = painter.fontMetrics().boundingRect(
                    self.rect().adjusted(60, 0, -60, 0), Qt.AlignCenter | Qt.TextWordWrap, self._message)
                box = QRectF(bounds).adjusted(-18, -12, 18, 12)
                painter.setPen(Qt.NoPen)
                painter.setBrush(QColor(0, 0, 0, 170))
                painter.drawRoundedRect(box, 8, 8)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(box, Qt.AlignCenter | Qt.TextWordWrap, self._message)
            else:
                band = QRectF(0, self.height() - 34, self.width(), 34)
                painter.fillRect(band, QColor(0, 0, 0, 160))
                painter.drawText(band, Qt.AlignCenter, self._message)

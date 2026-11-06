"""
Alerts sidebar: a card per detection, status actions, and a strip that collapses the panel into the right edge.
"""
from PySide6.QtCore import QItemSelectionModel, QModelIndex, QRect, QSize, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont, QIcon, QPainter, QPainterPath, QPalette, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView, QComboBox, QHBoxLayout, QLabel, QListView, QMenu, QPushButton, QSizePolicy,
    QStyle, QStyledItemDelegate, QToolButton, QVBoxLayout, QWidget,
)

from raven.gui.alerts import ALERT_ROLE, STATUS_COLORS, Alert, AlertFilter, AlertModel, ThumbnailCache, drone_colors
from raven.gui.shortcuts import registry

ALL_DRONES = "All drones"


def drone_icon(color: str, size: int = 12) -> QIcon:
    """
    A filled dot in a drone's color, for the drone filter and legends.
    """
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.Antialiasing)
    painter.setPen(Qt.NoPen)
    painter.setBrush(QColor(color))
    painter.drawEllipse(1, 1, size - 2, size - 2)
    painter.end()
    return QIcon(pixmap)

CARD_HEIGHT = 96
STRIP_WIDTH = 30


class AlertDelegate(QStyledItemDelegate):
    """
    Draws each alert as a card: status stripe, thumbnail with boxes, title, times, coordinates, status pill.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.thumbnails = ThumbnailCache()

    def sizeHint(self, option, index):
        return QSize(option.rect.width(), CARD_HEIGHT)

    def paint(self, painter: QPainter, option, index):
        alert: Alert = index.data(ALERT_ROLE)
        palette = option.palette
        selected = bool(option.state & QStyle.State_Selected)

        painter.save()
        painter.setRenderHint(QPainter.Antialiasing)
        if alert.status == "dismissed":
            painter.setOpacity(0.55)

        # card background and border
        card = option.rect.adjusted(6, 3, -6, -3)
        path = QPainterPath()
        path.addRoundedRect(card, 6, 6)
        background = palette.color(QPalette.Base)
        if selected:
            highlight = palette.color(QPalette.Highlight)
            background = QColor(highlight.red(), highlight.green(), highlight.blue(), 60)
        painter.fillPath(path, background)
        painter.setPen(palette.color(QPalette.Highlight) if selected else palette.color(QPalette.Mid))
        painter.drawPath(path)

        # status stripe on the left edge
        stripe = QPainterPath()
        stripe.addRoundedRect(card.left(), card.top(), 5, card.height(), 2, 2)
        painter.fillPath(stripe, QColor(STATUS_COLORS[alert.status]))

        # thumbnail
        thumb_w, thumb_h = self.thumbnails.size
        thumb_rect = QRect(card.left() + 12, card.top() + (card.height() - thumb_h) // 2, thumb_w, thumb_h)
        pixmap = self.thumbnails.get(alert)
        if pixmap is not None:
            target = pixmap.size().scaled(thumb_rect.size(), Qt.KeepAspectRatio)
            x = thumb_rect.left() + (thumb_w - target.width()) // 2
            y = thumb_rect.top() + (thumb_h - target.height()) // 2
            painter.drawPixmap(QRect(x, y, target.width(), target.height()), pixmap)
        else:
            painter.fillRect(thumb_rect, palette.color(QPalette.AlternateBase))
            painter.setPen(palette.color(QPalette.PlaceholderText))
            painter.drawText(thumb_rect, Qt.AlignCenter, "no frame")

        # text column
        text_left = thumb_rect.right() + 10
        text_rect = QRect(text_left, card.top() + 8, card.right() - text_left - 8, card.height() - 16)

        pill_text = alert.status.upper()
        small = QFont(option.font)
        small.setPointSizeF(option.font.pointSizeF() * 0.8)
        small.setBold(True)
        painter.setFont(small)
        pill_w = painter.fontMetrics().horizontalAdvance(pill_text) + 12
        pill = QRect(text_rect.right() - pill_w, text_rect.top(), pill_w, painter.fontMetrics().height() + 2)
        pill_path = QPainterPath()
        pill_path.addRoundedRect(pill, pill.height() / 2, pill.height() / 2)
        painter.fillPath(pill_path, QColor(STATUS_COLORS[alert.status]))
        painter.setPen(QColor("white"))
        painter.drawText(pill, Qt.AlignCenter, pill_text)

        title_font = QFont(option.font)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(palette.color(QPalette.Text))
        line_h = painter.fontMetrics().height()
        title_rect = QRect(text_rect.left(), text_rect.top(), text_rect.width() - pill_w - 6, line_h)
        painter.drawText(title_rect, Qt.AlignLeft | Qt.AlignVCenter,
                         painter.fontMetrics().elidedText(alert.title, Qt.ElideRight, title_rect.width()))

        painter.setFont(option.font)
        painter.setPen(palette.color(QPalette.PlaceholderText))
        lines = [
            f"Live · {alert.received:%H:%M:%S}" if alert.is_live else f"Video {alert.video_time} · {alert.received:%H:%M:%S}",
            alert.where if alert.has_location else "No location",
        ]
        y = text_rect.top() + line_h + 4
        metrics = painter.fontMetrics()
        for line in lines:
            painter.drawText(QRect(text_rect.left(), y, text_rect.width(), metrics.height()),
                             Qt.AlignLeft | Qt.AlignVCenter, metrics.elidedText(line, Qt.ElideRight, text_rect.width()))
            y += metrics.height()

        # drone line: a dot in the drone's color (same as its map pins), then its name and video
        dot = 8
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(alert.drone_color))
        painter.drawEllipse(text_rect.left(), y + (metrics.height() - dot) // 2, dot, dot)
        drone_line = f"{alert.drone} · {alert.video_name}" if alert.video_name else alert.drone
        painter.setPen(palette.color(QPalette.Text))
        painter.drawText(QRect(text_rect.left() + dot + 5, y, text_rect.width() - dot - 5, metrics.height()),
                         Qt.AlignLeft | Qt.AlignVCenter,
                         metrics.elidedText(drone_line, Qt.ElideRight, text_rect.width() - dot - 5))

        painter.restore()


class VerticalLabel(QWidget):
    """
    Text drawn bottom-to-top, for the collapsed strip.
    """

    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        self._text = text
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)

    def sizeHint(self):
        metrics = self.fontMetrics()
        return QSize(metrics.height() + 4, metrics.horizontalAdvance(self._text) + 8)

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, _):
        painter = QPainter(self)
        painter.setPen(self.palette().color(QPalette.WindowText))
        painter.translate(0, self.sizeHint().height())
        painter.rotate(-90)
        painter.drawText(QRect(0, 0, self.sizeHint().height(), self.width()), Qt.AlignCenter, self._text)


class AlertsSidebar(QWidget):
    alert_selected = Signal(str)           # alert id, single click / keyboard
    alert_activated = Signal(str)          # alert id, double click / Enter
    status_changed = Signal(str, str)      # alert id, new status
    clear_requested = Signal()
    collapsed_changed = Signal(bool)

    def __init__(self, model: AlertModel, parent=None):
        super().__init__(parent)
        self.model = model
        self._collapsed = False
        self._syncing = False  # True while selecting from code, so it isn't reported back as a user click

        # ---- collapse strip, always visible on the right edge
        self.toggle_button = QToolButton()
        self.toggle_button.setArrowType(Qt.RightArrow)
        self.toggle_button.setToolTip("Collapse the alerts panel")
        self.toggle_button.setAutoRaise(True)
        self.toggle_button.setFixedSize(26, 30)
        self.toggle_button.setIconSize(QSize(16, 16))
        self.toggle_button.setCursor(Qt.PointingHandCursor)
        self.toggle_button.clicked.connect(lambda: self.set_collapsed(not self._collapsed))
        self.badge = QLabel()
        self.badge.setAlignment(Qt.AlignCenter)
        self.badge.setStyleSheet(f"background:{STATUS_COLORS['new']}; color:white; border-radius:9px;"
                                 "font-weight:bold; padding:1px 4px;")
        self.badge.setToolTip("New alerts")
        self.strip_label = VerticalLabel("ALERTS")
        self.strip_label.setVisible(False)
        self.setMinimumWidth(260)

        self.strip = QWidget()
        self.strip.setObjectName("alertsStrip")
        self.strip.setAttribute(Qt.WA_StyledBackground, True)
        self.strip.setFixedWidth(STRIP_WIDTH)
        strip_layout = QVBoxLayout(self.strip)
        strip_layout.setContentsMargins(2, 4, 2, 4)
        strip_layout.addWidget(self.toggle_button, 0, Qt.AlignHCenter)
        strip_layout.addWidget(self.badge, 0, Qt.AlignHCenter)
        strip_layout.addWidget(self.strip_label, 0, Qt.AlignHCenter)
        strip_layout.addStretch(1)

        # ---- panel content
        header = QLabel("Alerts")
        header_font = header.font()
        header_font.setPointSizeF(header_font.pointSizeF() * 1.25)
        header_font.setBold(True)
        header.setFont(header_font)
        self.summary = QLabel()
        self.summary.setObjectName("hint")

        self.filter_combo = QComboBox()
        self.filter_combo.addItems(list(AlertFilter.MODES))
        self.filter_combo.setToolTip("Active hides dismissed alerts")

        self.proxy = AlertFilter(self)
        self.proxy.setSourceModel(model)
        self.filter_combo.currentTextChanged.connect(self.proxy.set_mode)

        # drone filter, shown once alerts from more than one drone are in the list
        self.drone_combo = QComboBox()
        self.drone_combo.setToolTip("Show alerts from one drone only")
        self.drone_combo.addItem(ALL_DRONES)
        self.drone_combo.setVisible(False)
        self.drone_combo.currentIndexChanged.connect(self._drone_filter_changed)

        self.list = QListView()
        self.list.setModel(self.proxy)
        self.list.setItemDelegate(AlertDelegate(self.list))
        self.list.setUniformItemSizes(True)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        # cards always fit the panel width, so a horizontal scrollbar is never useful
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.list.setFrameShape(QListView.NoFrame)
        self.list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list.customContextMenuRequested.connect(self._context_menu)
        self.list.selectionModel().currentChanged.connect(self._current_changed)
        self.list.doubleClicked.connect(lambda index: self.alert_activated.emit(index.data(ALERT_ROLE).id))
        self.list.activated.connect(lambda index: self.alert_activated.emit(index.data(ALERT_ROLE).id))

        self.reviewed_button = QPushButton("Reviewed")
        self.reviewed_button.clicked.connect(lambda: self._set_current_status("reviewed"))
        self.dismiss_button = QPushButton("Dismiss")
        self.dismiss_button.clicked.connect(lambda: self._set_current_status("dismissed"))
        clear = QPushButton("Clear list")
        clear.setToolTip("Remove all alerts from the list and map. Restarting the console shows them again from the detection log.")
        clear.clicked.connect(self.clear_requested)
        self.clear_button = clear
        # Reviewer and Viewer roles can't change some of this (set_review_allowed)
        self._can_review = True

        # keys come from the shortcut settings and work while the list has focus
        for action_id, status in (("alert_reviewed", "reviewed"), ("alert_dismissed", "dismissed"), ("alert_new", "new")):
            registry().bind(action_id, self.list, lambda s=status: self._set_current_status(s), Qt.WidgetShortcut)
        registry().changed.connect(self._update_tooltips)
        self._update_tooltips()

        top = QHBoxLayout()
        top.addWidget(header)
        top.addStretch(1)
        top.addWidget(self.drone_combo)
        top.addWidget(self.filter_combo)

        actions = QHBoxLayout()
        actions.addWidget(self.reviewed_button)
        actions.addWidget(self.dismiss_button)
        actions.addStretch(1)
        actions.addWidget(clear)

        self.content = QWidget()
        content_layout = QVBoxLayout(self.content)
        content_layout.setContentsMargins(4, 6, 6, 6)
        content_layout.addLayout(top)
        content_layout.addWidget(self.summary)
        content_layout.addWidget(self.list, 1)
        content_layout.addLayout(actions)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.strip)
        layout.addWidget(self.content, 1)

        for signal in (model.rowsInserted, model.dataChanged, model.modelReset):
            signal.connect(self._update_counts)
        for signal in (model.rowsInserted, model.modelReset):
            signal.connect(self._update_drones)
        self._update_counts()

    def _update_tooltips(self, *_):
        keys = {action_id: registry().text(action_id) for action_id in ("alert_reviewed", "alert_dismissed", "alert_new")}

        def key(action_id):
            return f" ({keys[action_id]})" if keys[action_id] else ""

        self.reviewed_button.setToolTip("Mark the selected alert reviewed" + key("alert_reviewed"))
        self.dismiss_button.setToolTip("Dismiss the selected alert" + key("alert_dismissed"))
        parts = ["Click: select", "Double-click: review footage"]
        for action_id, label in (("alert_reviewed", "reviewed"), ("alert_dismissed", "dismiss"), ("alert_new", "new")):
            if keys[action_id]:
                parts.append(f"{keys[action_id]}: {label}")
        self.list.setToolTip(" · ".join(parts))

    # ------------------------------------------------------------------ drone filter

    def _update_drones(self, *_):
        """
        Keep the drone filter's choices in step with the drones that have alerts in the list.
        """
        drones = self.model.drones()
        current = self.proxy.drone()
        if current is not None and current not in drones:
            drones.append(current)  # keep the operator's choice until they change it
        listed = [self.drone_combo.itemData(i) for i in range(1, self.drone_combo.count())]
        if listed != drones:
            self.drone_combo.blockSignals(True)
            self.drone_combo.clear()
            self.drone_combo.addItem(ALL_DRONES, None)
            for name in drones:
                self.drone_combo.addItem(drone_icon(drone_colors.color(name)), name, name)
            self.drone_combo.setCurrentIndex(max(0, self.drone_combo.findData(current)) if current else 0)
            self.drone_combo.blockSignals(False)
        self.drone_combo.setVisible(len(drones) > 1 or current is not None)

    def _drone_filter_changed(self, index: int):
        self.proxy.set_drone(self.drone_combo.itemData(index) if index > 0 else None)
        self._update_counts()

    def set_drone_filter(self, drone: str | None):
        index = self.drone_combo.findData(drone) if drone else 0
        self.drone_combo.setCurrentIndex(max(0, index))

    # ------------------------------------------------------------------ collapse

    def is_collapsed(self) -> bool:
        return self._collapsed

    def set_collapsed(self, collapsed: bool):
        if collapsed == self._collapsed:
            return
        self._collapsed = collapsed
        self.content.setVisible(not collapsed)
        self.strip_label.setVisible(collapsed)
        self.toggle_button.setArrowType(Qt.LeftArrow if collapsed else Qt.RightArrow)
        self.toggle_button.setToolTip("Show the alerts panel" if collapsed else "Collapse the alerts panel")
        if collapsed:
            self.setFixedWidth(STRIP_WIDTH)
        else:
            self.setMinimumWidth(260)
            self.setMaximumWidth(16777215)
        self.collapsed_changed.emit(collapsed)

    # ------------------------------------------------------------------ selection / status

    def select_alert(self, alert_id: str):
        """
        Select an alert (e.g. its map pin was clicked), showing it even if the filter hides it.
        """
        row = self.model.row_of(alert_id)
        if row < 0:
            return
        if self._collapsed:
            self.set_collapsed(False)
        source_index = self.model.index(row)
        index = self.proxy.mapFromSource(source_index)
        if not index.isValid():
            # hidden by a filter: widen the filters so the alert can be shown
            if self.proxy.drone() not in (None, self.model.get(alert_id).drone):
                self.set_drone_filter(None)
            index = self.proxy.mapFromSource(source_index)
        if not index.isValid():
            self.filter_combo.setCurrentText("All")
            index = self.proxy.mapFromSource(source_index)
        # move the highlight and the current item together, without echoing alert_selected back to the caller
        self._syncing = True
        try:
            self.list.selectionModel().setCurrentIndex(index, QItemSelectionModel.ClearAndSelect)
        finally:
            self._syncing = False
        # only scroll when the card is off-screen: scrolling a card the operator just clicked would move another
        # card under the cursor before the mouse button is released
        self.list.scrollTo(index, QAbstractItemView.EnsureVisible)

    def clear_selection(self):
        """
        Unselect the current alert (e.g. its map pin was clicked again), without telling the map back.
        """
        self._syncing = True
        try:
            self.list.selectionModel().clearCurrentIndex()
            self.list.selectionModel().clearSelection()
        finally:
            self._syncing = False

    def current_alert(self) -> Alert | None:
        index = self.list.currentIndex()
        return index.data(ALERT_ROLE) if index.isValid() else None

    def _current_changed(self, current: QModelIndex, _previous):
        if current.isValid() and not self._syncing:
            self.alert_selected.emit(current.data(ALERT_ROLE).id)

    def set_review_allowed(self, can_review: bool, can_clear: bool):
        """
        Grey out the status buttons and Clear list for roles that can't use them.
        """
        self._can_review = can_review
        for button in (self.reviewed_button, self.dismiss_button):
            button.setEnabled(can_review)
            button.setToolTip("" if can_review else "Your role can't review or dismiss alerts")
        self.clear_button.setEnabled(can_clear)

    def _set_current_status(self, status: str):
        alert = self.current_alert()
        if alert is not None and self._can_review:
            row = self.list.currentIndex().row()
            self.status_changed.emit(alert.id, status)
            # keep a sensible selection if the alert just left the filtered list
            if not self.list.currentIndex().isValid() and self.proxy.rowCount():
                self.list.setCurrentIndex(self.proxy.index(min(row, self.proxy.rowCount() - 1), 0))

    def _context_menu(self, pos):
        index = self.list.indexAt(pos)
        if not index.isValid():
            return
        alert = index.data(ALERT_ROLE)
        self.list.setCurrentIndex(index)
        menu = QMenu(self)
        open_action = QAction("Review footage", menu)
        open_action.triggered.connect(lambda: self.alert_activated.emit(alert.id))
        menu.addAction(open_action)
        menu.addSeparator()
        for status, label in (("new", "Mark new"), ("reviewed", "Mark reviewed"), ("dismissed", "Dismiss")):
            action = QAction(label, menu)
            action.setEnabled(alert.status != status and self._can_review)
            action.triggered.connect(lambda _=False, s=status: self.status_changed.emit(alert.id, s))
            menu.addAction(action)
        menu.exec(self.list.viewport().mapToGlobal(pos))

    def _update_counts(self, *_):
        new = self.model.count("new")
        total = self.model.rowCount()
        self.badge.setText(str(new))
        self.badge.setVisible(new > 0)
        text = (f"{total} alert(s) · {new} new · {self.model.count('reviewed')} reviewed · "
                f"{self.model.count('dismissed')} dismissed")
        if self.proxy.drone() is not None:
            text += f" · showing {self.proxy.rowCount()} from {self.proxy.drone()}"
        self.summary.setText(text)

    def clear_thumbnails(self):
        self.list.itemDelegate().thumbnails.clear()

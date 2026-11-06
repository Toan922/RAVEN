"""
Top row of the window: RAVEN title, big Map / Feed / Settings tab buttons, the search session's name and state,
then the light/dark toggle and the operator's profile button on the far right.
"""
from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import QButtonGroup, QFrame, QHBoxLayout, QLabel, QPushButton, QToolButton

from raven.gui.shortcuts import registry
from raven.gui.theme import themed_icon


class HeaderBar(QFrame):
    page_selected = Signal(int)
    theme_toggle_requested = Signal()

    def __init__(self, page_names: list[str], page_actions: list[str], demo: bool = False, parent=None):
        super().__init__(parent)
        self.setObjectName("header")

        brand = QLabel("RAVEN")
        brand.setObjectName("brand")

        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._buttons = []
        for i, name in enumerate(page_names):
            button = QPushButton(name)
            button.setObjectName("navTab")
            button.setCheckable(True)
            button.setCursor(Qt.PointingHandCursor)
            self._group.addButton(button, i)
            self._buttons.append(button)
        self._buttons[0].setChecked(True)
        self._page_actions = page_actions
        self._theme = "light"
        self._group.idClicked.connect(self.page_selected)

        self.theme_button = QToolButton()
        self.theme_button.setObjectName("themeButton")
        self.theme_button.setIconSize(QSize(24, 24))
        self.theme_button.setCursor(Qt.PointingHandCursor)
        self.theme_button.clicked.connect(self.theme_toggle_requested)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 0, 12, 0)
        layout.setSpacing(0)
        layout.addWidget(brand, 0, Qt.AlignVCenter)
        for button in self._buttons:
            layout.addWidget(button, 0, Qt.AlignBottom)
        layout.addStretch(1)

        # the search session's name and whether the console can start (Not ready / Ready for footage / Running)
        self.session_label = QLabel()
        self.session_label.setObjectName("sessionName")
        self.state_label = QLabel()
        self.state_label.setObjectName("sessionState")
        layout.addWidget(self.session_label, 0, Qt.AlignVCenter)
        layout.addSpacing(8)
        layout.addWidget(self.state_label, 0, Qt.AlignVCenter)
        layout.addSpacing(16)
        self.set_session("", "not_ready")

        if demo:
            pill = QLabel("DEMO · simulated detector")
            pill.setObjectName("demoPill")
            pill.setToolTip("Started with --demo: alerts come from a simulated detector on generated "
                            "drone flights. Nothing here is real.")
            layout.addWidget(pill, 0, Qt.AlignVCenter)
            layout.addSpacing(12)

        layout.addWidget(self.theme_button, 0, Qt.AlignVCenter)
        self._layout = layout

        registry().changed.connect(self._update_tooltips)
        self._update_tooltips()

    def _update_tooltips(self, *_):
        for button, action_id in zip(self._buttons, self._page_actions):
            key = registry().text(action_id)
            button.setToolTip(f"{button.text()} ({key})" if key else button.text())
        self.set_theme(self._theme)

    def add_trailing(self, widget):
        """
        Put a widget at the right end of the header, after the light/dark toggle.
        """
        self._layout.addSpacing(6)
        self._layout.addWidget(widget, 0, Qt.AlignVCenter)

    def set_session(self, name: str, state: str, detail: str = ""):
        """
        state is "not_ready", "ready" or "running"; detail (what's missing, for Not ready) goes in the tooltip.
        """
        self.session_label.setText(name or "Unnamed session")
        self.session_label.setToolTip("Search session (Manage > Search Sessions)")
        text = {"not_ready": "Not ready", "ready": "Ready for footage", "running": "Running"}[state]
        self.state_label.setText(text)
        self.state_label.setProperty("state", state)
        self.state_label.setToolTip(detail or text)
        # re-apply the stylesheet so the [state=...] rule follows the new value
        self.state_label.style().unpolish(self.state_label)
        self.state_label.style().polish(self.state_label)

    def set_page_visible(self, index: int, visible: bool):
        """
        Show or hide a page's tab button (pages a role can't use are hidden).
        """
        self._buttons[index].setVisible(visible)

    def is_page_visible(self, index: int) -> bool:
        return not self._buttons[index].isHidden()

    def set_page(self, index: int):
        self._buttons[index].setChecked(True)

    def set_theme(self, name: str):
        """
        Show the icon for the theme a click switches to: the moon in light mode, the sun in dark mode.
        """
        self._theme = name
        key = registry().text("toggle_theme")
        key = f" ({key})" if key else ""
        if name == "light":
            self.theme_button.setIcon(themed_icon("moon.svg", name))
            self.theme_button.setToolTip("Switch to dark mode" + key)
        else:
            self.theme_button.setIcon(themed_icon("sun.svg", name))
            self.theme_button.setToolTip("Switch to light mode" + key)

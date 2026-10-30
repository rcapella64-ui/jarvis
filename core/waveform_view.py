"""
The HUD's Three.js waveform — core/waveform3d/index.html in a QWebEngineView.

HudCanvas keeps the audio level and the state, as it does for the head and the
core, and pushes them here on its own timer. The scene runs at the browser's
frame rate on the GPU, so it costs the Qt thread nothing but that small call.

Anything that goes wrong (PyQt6-WebEngine not installed, no WebGL) leaves
`failed` set and the HUD paints its reactor core instead: the panel is never
empty because of a graphics driver.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

# Must be imported before the QApplication exists, which is why ui.py imports
# this module at the top. The flags must be set before the import.
os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--ignore-gpu-blocklist")
try:
    from PyQt6.QtWebEngineCore import QWebEnginePage, QWebEngineSettings
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    AVAILABLE = True
except Exception as _e:                 # noqa: BLE001
    print(f"[Waveform] QtWebEngine unavailable, using the painted HUD: {_e}")
    QWebEnginePage = QWebEngineSettings = QWebEngineView = None
    AVAILABLE = False

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QColor

SCENE = Path(__file__).resolve().parent / "waveform3d" / "index.html"


if AVAILABLE:
    class _Page(QWebEnginePage):
        def javaScriptConsoleMessage(self, level, message, line, source):
            if level == QWebEnginePage.JavaScriptConsoleMessageLevel.ErrorMessageLevel:
                print(f"[Waveform] JS error: {message} (line {line})")


    class WaveformView(QWebEngineView):
        def __init__(self, bg: str, parent=None):
            super().__init__(parent)
            self.ready = False
            self.failed = False
            self.on_failed = None               # callable(), set by HudCanvas
            self._theme: dict | None = None
            self._last: tuple | None = None

            self.setPage(_Page(self))
            self.page().setBackgroundColor(QColor(bg))
            # A status strip over the scene, not a browser: no menu, no focus,
            # and clicks go through to the HUD underneath.
            self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
            self.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            s = self.settings()
            s.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
            s.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)
            s.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
            self.titleChanged.connect(self._on_title)
            self.load(QUrl.fromLocalFile(str(SCENE)))

        def _on_title(self, title: str) -> None:
            # The page's only way of talking back: see index.html.
            if title == "waveform-ok":
                self.ready = True
                self._last = None
                if self._theme:
                    self.set_theme(self._theme)
            elif title == "waveform-fail":
                self._fail("WebGL is not available")

        def _fail(self, why: str) -> None:
            print(f"[Waveform] {why}: using the painted HUD")
            self.failed = True
            self.ready = False
            self.hide()
            if self.on_failed:
                self.on_failed()

        def set_theme(self, colors: dict) -> None:
            self._theme = dict(colors)
            self.page().setBackgroundColor(QColor(colors.get("bg", "#000308")))
            if self.ready:
                self.page().runJavaScript(f"window.jarvis.theme({json.dumps(colors)})")

        def push(self, amp: float, state: str, speaking: bool, muted: bool) -> None:
            """Hand the current level and state to the scene. Called from the HUD's
            timer; skipped while nothing changed so silence costs nothing."""
            if not self.ready:
                return
            cur = (round(amp, 3), state, speaking, muted)
            if cur == self._last:
                return
            self._last = cur
            self.page().runJavaScript(
                f"window.jarvis.update({cur[0]},{json.dumps(state)},"
                f"{'true' if speaking else 'false'},{'true' if muted else 'false'})")
else:
    WaveformView = None

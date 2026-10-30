"""
HoloDeck — floating "hologram" cards drawn over the avatar.

When JARVIS brings information (a briefing, a revenue figure, who is waiting
in support, the agenda) the data is shown as a card beside the face, one at a
time, in the order he talks about it. A card slides in, holds while he speaks
about it, and slides out when he moves on to the next subject; a number on a
card counts up from zero to its value, the way a film HUD does.

How "moves on" is detected: every card carries trigger words. The live
transcript of what JARVIS is saying is fed in (`on_transcript`); when a word
of a LATER card shows up, the deck advances to it. A card also advances on
its own after `hold` seconds, and the whole deck clears a few seconds after
JARVIS stops speaking.

Pure QPainter, no child widgets; mouse events pass through to the HUD.

Card dict keys (all optional but `title`):
    title   : str                  header line
    value   : float | int | None   big number (animated from 0)
    fmt     : "money" | "int" | "pct" | "text"   how to format `value`
    unit    : str                  small text right after the number ("vendas")
    lines   : list[str]            detail lines under the number
    keys    : list[str]            transcript words that mean "now talking about this"
    hold    : float                seconds before auto-advance (default 16)
    accent  : "ok" | "warn" | None  tint the number green / orange
"""
from __future__ import annotations

import math
import time
import unicodedata

from PyQt6.QtCore import QRectF, Qt, QTimer
from PyQt6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import QWidget

_IN_S, _OUT_S, _COUNT_S = 0.38, 0.26, 1.5
_DEFAULT_HOLD = 16.0
_CLEAR_AFTER_SILENCE_S = 7.0


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFKD", (s or "").lower())
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _ease_out(t: float, p: float = 3.0) -> float:
    t = max(0.0, min(1.0, t))
    return 1.0 - (1.0 - t) ** p


def fmt_value(v, fmt: str) -> str:
    if v is None:
        return ""
    if fmt == "money":
        s = f"{float(v):,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
        return f"R$ {s}"
    if fmt == "pct":
        return f"{float(v):.0f}%"
    if fmt == "int":
        return f"{int(round(float(v))):,}".replace(",", ".")
    return str(v)


class HoloDeck(QWidget):
    def __init__(self, parent=None, colors: dict | None = None):
        super().__init__(parent)
        c = colors or {}
        self._pri = QColor(c.get("pri", "#00e5ff"))
        self._pri_dim = QColor(c.get("pri_dim", "#0a7f8c"))
        self._text = QColor(c.get("text", "#d8f6ff"))
        self._text_dim = QColor(c.get("text_dim", "#6c9aa6"))
        self._ok = QColor(c.get("ok", "#39ff88"))
        self._warn = QColor(c.get("warn", "#ffb347"))
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._cards: list[dict] = []
        self._idx = -1
        self._phase = "idle"            # idle | in | hold | out
        self._t0 = 0.0
        self._next_idx: int | None = None
        self._speaking = False
        self._silence_since = 0.0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._tick)
        self.hide()

    # ── public (Qt thread) ───────────────────────────────────────────────────

    def show_cards(self, cards: list[dict]) -> None:
        cards = [dict(c) for c in (cards or []) if isinstance(c, dict) and c.get("title")]
        if not cards:
            self.clear()
            return
        self._cards = cards
        self._idx = 0
        self._next_idx = None
        self._phase = "in"
        self._t0 = time.monotonic()
        self._silence_since = 0.0
        self.show()
        self.raise_()
        if not self._timer.isActive():
            self._timer.start()
        self.update()

    def add_cards(self, cards: list[dict]) -> None:
        """Append to the current sequence (or start one)."""
        if self._phase == "idle" or not self._cards:
            self.show_cards(cards)
            return
        self._cards.extend(dict(c) for c in (cards or []) if isinstance(c, dict) and c.get("title"))

    def advance(self) -> None:
        if self._phase in ("idle", "out") or not self._cards:
            return
        if self._idx + 1 < len(self._cards):
            self._go_to(self._idx + 1)
        else:
            self.clear()

    def clear(self) -> None:
        if self._phase in ("idle",) or not self._cards:
            self._cards, self._idx, self._phase = [], -1, "idle"
            self.hide()
            return
        self._next_idx = None
        self._phase = "out"
        self._t0 = time.monotonic()

    def on_transcript(self, text: str) -> None:
        """A chunk of what JARVIS is saying right now."""
        if self._phase == "idle" or not self._cards or not text:
            return
        hay = _fold(text)
        for j in range(self._idx + 1, len(self._cards)):
            keys = self._cards[j].get("keys") or []
            if any(_fold(k) in hay for k in keys if k):
                self._go_to(j)
                return

    def set_speaking(self, speaking: bool) -> None:
        self._speaking = bool(speaking)
        self._silence_since = 0.0 if speaking else time.monotonic()

    # ── internals ────────────────────────────────────────────────────────────

    def _go_to(self, j: int) -> None:
        if j == self._idx and self._phase != "out":
            return
        self._next_idx = j
        if self._phase == "in":
            self._phase = "out"
            self._t0 = time.monotonic()
        elif self._phase == "hold":
            self._phase = "out"
            self._t0 = time.monotonic()
        # already "out": the pending target is simply updated

    def _tick(self) -> None:
        if self._phase == "idle":
            self._timer.stop()
            self.hide()
            return
        now = time.monotonic()
        dt = now - self._t0
        if self._phase == "in" and dt >= _IN_S:
            self._phase, self._t0 = "hold", now
        elif self._phase == "hold":
            hold = float(self._card().get("hold") or _DEFAULT_HOLD)
            if dt >= hold and self._idx + 1 < len(self._cards):
                self._go_to(self._idx + 1)
            elif (not self._speaking and self._silence_since
                  and now - self._silence_since >= _CLEAR_AFTER_SILENCE_S):
                self.clear()
        elif self._phase == "out" and dt >= _OUT_S:
            if self._next_idx is not None and self._next_idx < len(self._cards):
                self._idx, self._next_idx = self._next_idx, None
                self._phase, self._t0 = "in", now
            else:
                self._cards, self._idx, self._phase = [], -1, "idle"
                self.hide()
                self._timer.stop()
                return
        self.update()

    def _card(self) -> dict:
        return self._cards[self._idx] if 0 <= self._idx < len(self._cards) else {}

    # ── painting ─────────────────────────────────────────────────────────────

    def paintEvent(self, _):
        card = self._card()
        if not card or self._phase == "idle":
            return
        now = time.monotonic()
        dt = now - self._t0
        if self._phase == "in":
            k = _ease_out(dt / _IN_S)
            alpha, dx, dy = k, (1.0 - k) * 46.0, 0.0
            count_t = 0.0
        elif self._phase == "hold":
            alpha, dx, dy = 1.0, 0.0, 0.0
            count_t = _ease_out(dt / _COUNT_S, 4.0)
        else:                                   # out
            k = _ease_out(dt / _OUT_S, 2.0)
            alpha, dx, dy = 1.0 - k, 0.0, -18.0 * k
            count_t = 1.0

        W, H = self.width(), self.height()
        w = int(max(250, min(440, W * 0.36)))
        title = str(card.get("title") or "").upper()
        value = card.get("value")
        fmt = str(card.get("fmt") or ("text" if value is None else "int"))
        unit = str(card.get("unit") or "")
        lines = [str(x) for x in (card.get("lines") or [])][:7]
        pad = 16
        h = pad + 18 + (58 if value is not None else 6) + len(lines) * 19 + pad
        x = W - w - 28 + dx
        y = (H - h) / 2.0 + dy
        rect = QRectF(x, y, w, h)

        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        p.setOpacity(alpha)

        # panel
        fill = QColor(0, 18, 28, 178)
        path = QPainterPath()
        path.addRoundedRect(rect, 4, 4)
        p.fillPath(path, fill)
        border = QColor(self._pri)
        border.setAlpha(140)
        p.setPen(QPen(border, 1))
        p.drawPath(path)
        # scanlines
        scan = QColor(self._pri)
        scan.setAlpha(14)
        p.setPen(QPen(scan, 1))
        yy = int(y) + 2
        while yy < y + h - 2:
            p.drawLine(int(x) + 2, yy, int(x + w) - 2, yy)
            yy += 3
        # corner brackets
        p.setPen(QPen(self._pri, 2))
        L = 14
        for cx, sx in ((x, 1), (x + w, -1)):
            for cy, sy in ((y, 1), (y + h, -1)):
                p.drawLine(int(cx), int(cy), int(cx + sx * L), int(cy))
                p.drawLine(int(cx), int(cy), int(cx), int(cy + sy * L))
        # connector toward the face
        glow = QColor(self._pri)
        glow.setAlpha(90)
        p.setPen(QPen(glow, 1, Qt.PenStyle.DashLine))
        p.drawLine(int(x), int(y + h / 2), int(x - 34), int(y + h / 2))
        p.setBrush(self._pri)
        p.setPen(Qt.PenStyle.NoPen)
        p.drawEllipse(QRectF(x - 37, y + h / 2 - 3, 6, 6))

        # title
        cy = y + pad
        p.setFont(QFont("Bahnschrift", 8, QFont.Weight.Bold))
        p.setPen(self._pri_dim)
        p.drawText(QRectF(x + pad, cy, w - 2 * pad, 16),
                   Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, "◈ " + title)
        cy += 18

        # value (counted up)
        if value is not None:
            try:
                target = float(value)
            except Exception:
                target = 0.0
            shown = target * count_t if fmt in ("money", "int", "pct") else target
            txt = fmt_value(shown, fmt) if fmt != "text" else str(value)
            col = {"ok": self._ok, "warn": self._warn}.get(card.get("accent"), self._pri)
            f = QFont("Bahnschrift", 24, QFont.Weight.Bold)
            p.setFont(f)
            halo = QColor(col)
            halo.setAlpha(70)
            r = QRectF(x + pad, cy, w - 2 * pad, 48)
            p.setPen(halo)
            for ox, oy in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
                p.drawText(r.translated(ox, oy), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)
            p.setPen(col)
            p.drawText(r, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, txt)
            if unit:
                tw = p.fontMetrics().horizontalAdvance(txt)
                p.setFont(QFont("Bahnschrift", 9))
                p.setPen(self._text_dim)
                p.drawText(QRectF(x + pad + tw + 10, cy + 6, w - 2 * pad - tw - 10, 40),
                           Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, unit)
            cy += 58
        else:
            cy += 6

        # detail lines, revealed one by one during the count-up
        p.setFont(QFont("Bahnschrift", 9))
        n_show = len(lines) if self._phase != "hold" else max(1, int(math.ceil(len(lines) * min(1.0, dt / 0.9))))
        for i, line in enumerate(lines[:n_show]):
            p.setPen(self._text if i == 0 or not line.startswith("  ") else self._text_dim)
            fm = p.fontMetrics()
            p.drawText(QRectF(x + pad, cy, w - 2 * pad, 18),
                       Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                       fm.elidedText(line, Qt.TextElideMode.ElideRight, int(w - 2 * pad)))
            cy += 19

        # progress dots: which card of how many
        n = len(self._cards)
        if n > 1:
            p.setPen(Qt.PenStyle.NoPen)
            for i in range(n):
                d = QColor(self._pri if i == self._idx else self._pri_dim)
                d.setAlpha(255 if i == self._idx else 120)
                p.setBrush(d)
                p.drawEllipse(QRectF(x + w - pad - (n - i) * 9, y + h - 9, 4, 4))
        p.end()

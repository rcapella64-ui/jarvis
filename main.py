import platform as _platform
import subprocess as _subprocess

# ── Nuclear: force CREATE_NO_WINDOW on EVERY subprocess call on Windows ───────
# This patches Popen itself, so no per-file flag is needed anywhere.
if _platform.system() == "Windows":
    _OrigPopen = _subprocess.Popen

    class _Popen(_OrigPopen):
        def __init__(self, args, **kw):
            kw["creationflags"] = kw.get("creationflags", 0) | _subprocess.CREATE_NO_WINDOW
            kw.pop("startupinfo", None)   # drop any stale/shared STARTUPINFO
            super().__init__(args, **                       kw)

    _subprocess.Popen = _Popen


# ── Console must survive non-UTF-8 code pages ────────────────────────────────
# Every status line in this file carries an emoji, and on a legacy Windows
# console the active code page is the system one — cp1254 in Turkey, cp1251 in
# Russia, cp932 in Japan. Printing an emoji there raises UnicodeEncodeError, and
# because most of these prints sit inside the receive loop it takes the session
# down on startup. Reconfiguring to UTF-8 with a replacement fallback costs
# nothing and makes the app launch the same way in every locale.
import sys as _sys

for _stream in ("stdout", "stderr"):
    try:
        _s = getattr(_sys, _stream, None)
        if _s is not None and hasattr(_s, "reconfigure"):
            _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass          # pythonw / redirected pipes / anything exotic — never fatal

# ─────────────────────────────────────────────────────────────────────────────

import asyncio
import os
import re
import threading
import time
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import sounddevice as sd
import numpy as np
from google import genai
from google.genai import types
from ui import JarvisUI
from memory.memory_manager import (
    load_memory, update_memory, format_memory_for_prompt,
    save_session_summary, pop_last_session,
    search_memory, set_trim_notifier,
)

# The file-backed tools (open_app, web_search, browser_control, …) are no longer
# imported or declared here — they self-describe via a TOOL dict in their own
# actions/*.py file and are auto-discovered by core.action_loader at startup.
# Only tools that are tied to live-session state stay inline in this file
# (screen_process, close_camera, save_memory, manage_monitor, shutdown_jarvis,
# system_status).
from actions.screen_processor  import _capture_camera, _capture_screen
from actions.system_monitor    import SystemMonitor, get_system_status
from actions.proactive         import ProactiveEngine
from actions.background_monitor import (
    add_monitor, remove_monitor, list_monitors, check_all as monitor_check_all,
)
from actions.web_search        import _news as _fetch_news_sync
from core.briefing_sources     import collect as _collect_briefing, format_youtube as _fmt_youtube
from core                      import usage as _usage
from core                      import music as _music
from memory.config_manager     import (
    get_brief_enabled, get_media_resolution, get_proactive_audio_enabled,
    get_push_to_talk_enabled, get_thinking_enabled, get_turn_tuning, get_voice,
    get_stt_engine, get_plugin_config, get_brain_engine, get_name_gate_enabled,
    get_acid_humor_enabled,
    get_blocked_phrases,
    get_wake_word_enabled, save_wake_word_enabled,    get_input_device, get_output_device,
    save_voice, DEFAULT_VOICE,
)
from core                     import gemini as _gemini
from core.plugin_loader        import discover_plugins
from core                      import undo as undo_stack
from core                      import confirm as confirm_gate
from core                      import audio_devices
from core.action_loader        import discover_actions
from core.echo                 import EchoGuard
from core.viseme               import VisemeStream
from core.wake_word            import (
    WakeWordDetector, is_ready as wake_is_ready, install_and_download as wake_install,
)

# How long the assistant stays awake with no user speech before it auto-sleeps
# again (wake-word mode only).
WAKE_SLEEP_TIMEOUT = 120.0   # seconds (2 minutes)

def get_base_dir():
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent
    return Path(__file__).resolve().parent

BASE_DIR        = get_base_dir()
API_CONFIG_PATH = BASE_DIR / "config" / "api_keys.json"
PROMPT_PATH     = BASE_DIR / "core" / "prompt.txt"
# The conversation's model. A NAME, not a decision: the ladder lives in
# core/gemini.py and this is only whichever rung is currently in use, kept here
# as a module attribute because plugins read it (chat_takeover asks main for it
# so that upgrading the assistant upgrades the plugin too).
#
# It is reassigned on every connect, so a model that runs out of quota is
# stepped over and the assistant keeps talking instead of failing to start.
LIVE_MODEL          = "models/gemini-3.1-flash-live-preview"
CHANNELS            = 1
SEND_SAMPLE_RATE    = 16000 
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE          = 1024

# RMS below which 16-bit PCM is treated as room silence; above _LEVEL_FULL it
# reads as a full-height waveform. Tuned so ordinary speech lands mid-range and
# the bars still move for a quiet talker — language- and device-independent.
_LEVEL_FLOOR = 60.0
_LEVEL_FULL  = 2600.0


def _pcm_level(samples) -> float:
    """Map a block of int16 PCM samples to a 0.0–1.0 loudness level for the HUD
    waveform. Returns 0.0 on empty/invalid input so it can never raise."""
    try:
        x = np.asarray(samples, dtype=np.float32)
        if x.size == 0:
            return 0.0
        rms = float(np.sqrt(np.mean(x * x)))
    except Exception:
        return 0.0
    if rms <= _LEVEL_FLOOR:
        return 0.0
    return min(1.0, (rms - _LEVEL_FLOOR) / (_LEVEL_FULL - _LEVEL_FLOOR))


# ── Viseme extraction ─────────────────────────────────────────────────────────
# The avatar's mouth used to be driven by one RMS value per ~200 ms write batch,
# which is five updates a second averaged over a fifth of a second — it could
# only ever flap. These read the *shape* of each 20 ms slice straight from the
# spectrum of the audio being played, so no transcript, no forced alignment and
# no language assumption: it works the same for Turkish and English.
#
# Two numbers come out. Openness tracks the first formant — F1 climbs as the jaw
# drops, so /a/ reads open and /i/ or /u/ read closed. Width tracks the second —
# F2 is high for spread vowels (/i/, /e/) and low for rounded ones (/u/, /o/).
# Extra time beyond the device's reported output latency before the microphone
# is trusted again: covers room decay and the speaker's own settling.
_TAIL_MARGIN = 0.25

_VIS_WIN = 1024        # ~43 ms analysis window at 24 kHz: enough for formants
_VIS_HOP = 480         # 20 ms between frames, i.e. 50 shapes a second

# Delay from handing the first bytes of a reply to an already-running output
# stream to hearing them: one callback period, plus whatever the DAC adds.
_FIRST_SOUND = CHUNK_SIZE / RECEIVE_SAMPLE_RATE      # ~43 ms
# How far past the device's own buffer the mouth's timeline may drift before it
# is re-anchored. The buffer is the hard limit on how much audio can be queued
# ahead, so anything beyond it plus a margin for clock error is impossible.
_CURSOR_SLACK = 0.15

# Erring early is the safe direction. A viewer tolerates a mouth that moves
# slightly before the sound far better than one that moves after it — the
# broadcast limits are about 45 ms of lag against 125 ms of lead — so where
# this is uncertain it is biased to lead.


def _pcm_visemes(samples, sr: int = 24000):
    """Slice a PCM block into (level, openness, width) frames, one per 20 ms.

    Returns [] on anything unexpected — the mouth falls back to loudness-only
    articulation rather than the caller having to handle an error.
    """
    try:
        x = np.asarray(samples, dtype=np.float32)
        if x.size < _VIS_WIN:
            return []
        win = np.hanning(_VIS_WIN).astype(np.float32)
        freqs = np.fft.rfftfreq(_VIS_WIN, 1.0 / sr)
        b_f1_lo = (freqs >= 150) & (freqs < 450)     # F1 of close vowels
        b_f1_hi = (freqs >= 450) & (freqs < 1100)    # F1 of open vowels
        b_f2_bk = (freqs >= 600) & (freqs < 1300)    # F2 of rounded vowels
        b_f2_fr = (freqs >= 1700) & (freqs < 3200)   # F2 of spread vowels
        b_hiss = (freqs >= 3800) & (freqs < 8000)    # fricatives

        # One frame per hop across the *whole* block. Stepping only while a full
        # window fits stopped 1024 - 480 samples short of the end, so a 200 ms
        # batch yielded 160 ms of schedule: the mouth ran out of frames before
        # the audio ran out of sound, and each batch no longer lined up with the
        # end of the one before it. Losing 20 % of every batch is most of why
        # the mouth did not track the words.
        out = []
        for start in range(0, x.size, _VIS_HOP):
            # The level gates closures, so it is measured over exactly this
            # 20 ms and never looks ahead. The spectrum needs a longer window
            # to resolve formants and may be short-filled at the very end.
            level = _pcm_level(x[start:start + _VIS_HOP])
            seg = x[start:start + _VIS_WIN]
            if seg.size < _VIS_WIN:
                seg = np.concatenate([seg, np.zeros(_VIS_WIN - seg.size,
                                                    dtype=np.float32)])
            if level <= 0.0:
                out.append((0.0, 0.0, 0.0))
                continue
            mag = np.abs(np.fft.rfft((seg - seg.mean()) * win))
            f1l, f1h = float(mag[b_f1_lo].sum()), float(mag[b_f1_hi].sum())
            f2b, f2f = float(mag[b_f2_bk].sum()), float(mag[b_f2_fr].sum())
            hiss = float(mag[b_hiss].sum())

            openness = f1h / (f1l + f1h + 1e-6)
            width = (f2f - f2b) / (f2f + f2b + 1e-6)
            # A wide-open jaw physically cannot purse, so openness damps width.
            # /a/ has a low enough F2 to read as "rounded" on the bands alone;
            # letting openness suppress the width term is what keeps an open
            # vowel from pursing.
            width *= (1.0 - openness) ** 0.8
            # Fricatives are formed with a nearly closed mouth.
            h = hiss / (f1l + f1h + f2b + f2f + hiss + 1e-6)
            openness *= 1.0 - 0.65 * min(1.0, h * 2.5)
            out.append((level,
                        float(min(1.0, max(0.0, openness))),
                        float(min(1.0, max(-1.0, width)))))
        return out
    except Exception:
        return []


def _describe_tools(declarations) -> str:
    """One line per capability, straight from the live tool declarations.

    Derived rather than written down: the action and plugin registries are
    discovered at startup, so whatever the user has installed is what the model
    is told it can do. Adding a plugin extends this by itself, and removing one
    stops the model from claiming an ability it no longer has.
    """
    lines = []
    for d in declarations or ():
        try:
            name = d.get("name") if isinstance(d, dict) else getattr(d, "name", None)
            desc = (d.get("description") if isinstance(d, dict)
                    else getattr(d, "description", "")) or ""
        except Exception:
            continue
        if not name:
            continue
        desc = " ".join(str(desc).split())
        lines.append(f"- {name}: {desc[:150]}" if desc else f"- {name}")
    return "\n".join(lines)


def _describe_limits(has_vision: bool, has_mic: bool) -> str:
    """The other half of self-knowledge: what is out of reach, and why.

    Derived from how the program is actually built, not from a list of refusals.
    A model that knows its boundaries stops improvising around them, and stating
    them as architecture rather than as rules keeps the answer honest in any
    language.
    """
    out = [
        "- Anything not listed above is outside your reach. Say so in one clause "
        "and offer the nearest thing you can actually do — never mime an action "
        "you cannot take, and never report a result you did not get.",
        "- You act on this machine only. You cannot reach the user's other "
        "devices, accounts or hardware except through the tools listed above.",
        "- You remember what is in the memory block and what has been said this "
        "session. Anything else you were told before is gone unless it was saved.",
    ]
    if has_vision:
        out.append(
            "- Your sight is not continuous. You see nothing until you call a "
            "vision tool, and then only that single frame at that moment — you "
            "cannot watch, monitor or notice something changing on screen.")
    else:
        out.append("- You have no sight at all in this build.")
    if has_mic:
        out.append(
            "- You hear nothing while the microphone is muted, and you cannot "
            "unmute it yourself.")
    return "\n".join(out)


def _render_prompt(template: str, values: dict) -> str:
    """Fill {tokens} in the prompt template.

    A plain replace rather than str.format: the file is meant to be edited by
    hand, and a stray brace in someone's own wording must never take the app
    down at startup.
    """
    out = template or ""
    for key, val in values.items():
        out = out.replace("{" + key + "}", str(val))
    return out


_BRIEFING_STATE = BASE_DIR / "memory" / "briefing_state.json"


def _briefing_done_today(today: str) -> bool:
    try:
        data = json.loads(_BRIEFING_STATE.read_text(encoding="utf-8"))
        return str(data.get("last_briefing_date") or "") == today
    except Exception:
        return False


def _mark_briefing_done(today: str) -> None:
    try:
        _BRIEFING_STATE.parent.mkdir(parents=True, exist_ok=True)
        _BRIEFING_STATE.write_text(json.dumps({"last_briefing_date": today}), encoding="utf-8")
    except Exception as e:
        print(f"[Briefing] could not save state: {e}")


def _get_api_key() -> str:
    """The active Gemini key — core.gemini owns the list and the rotation."""
    key = _gemini.api_key(refresh=True)
    if key:
        return key
    with open(API_CONFIG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)["gemini_api_key"]


def _sane_history(history: list) -> list:
    """The OpenAI-style providers (DeepSeek, Groq…) reject a 'tool' message
    that does not directly follow an assistant message carrying the matching
    tool_calls, and an assistant tool_calls message whose calls were not all
    answered. Both happen here after an interruption, a failed round or a
    trim that cut between the two. Returns a copy with the orphans removed."""
    out: list = []
    i = 0
    while i < len(history):
        m = history[i]
        role = m.get("role")
        if role == "assistant" and m.get("tool_calls"):
            ids = {c.get("id") for c in m["tool_calls"] if isinstance(c, dict)}
            j = i + 1
            answered = set()
            while j < len(history) and history[j].get("role") == "tool":
                answered.add(history[j].get("tool_call_id"))
                j += 1
            if ids and ids <= answered:
                out.extend(history[i:j])
            # else: drop the call and its partial answers altogether
            i = j
            continue
        if role == "tool":
            i += 1                      # orphan: no tool_calls before it
            continue
        out.append(m)
        i += 1
    return out


_MSG_MAX_CHARS = 6000          # one message (what a voice reply could ever need)
_HISTORY_MAX_CHARS = 48000     # the whole remembered conversation (~12k tokens)


def _msg_len(m: dict) -> int:
    n = len(str(m.get("content") or ""))
    for c in m.get("tool_calls") or []:
        try:
            n += len(str(c.get("function", {}).get("arguments") or ""))
        except Exception:
            pass
    return n


# First clause of a sentence: up to the first , ; or : that is followed by a
# space (so "380.789,00" and "3,5" are never split).
_FIRST_CLAUSE = re.compile(r"^(.+?[,;:])\s+(.+)$", re.S)


def _trim_history(history: list, keep: int = 24) -> None:
    """Keep roughly the last `keep` messages, cutting only at a user turn so a
    tool exchange is never split; cap every message and the total size, so a
    runaway reply or a flood of alerts can never push the request past the
    model's window; then drop any orphans in place."""
    for m in history:
        c = m.get("content")
        if isinstance(c, str) and len(c) > _MSG_MAX_CHARS:
            m["content"] = c[:_MSG_MAX_CHARS] + " …[truncated]"
    if len(history) > keep:
        cut = len(history) - keep
        while cut < len(history) and history[cut].get("role") != "user":
            cut += 1
        del history[:cut]
    total = sum(_msg_len(m) for m in history)
    while total > _HISTORY_MAX_CHARS and len(history) > 1:
        # drop the oldest exchange: up to (not including) the next user turn
        cut = 1
        while cut < len(history) and history[cut].get("role") != "user":
            cut += 1
        if cut >= len(history):
            break
        total -= sum(_msg_len(m) for m in history[:cut])
        del history[:cut]
    history[:] = _sane_history(history)


def _dump_brain_debug(system_text: str, history: list, tools: list | None, err: str) -> None:
    """When the provider says the request is too big, write WHAT was big to
    memory/brain_debug.txt — role and size of every message — so the cause
    can be read instead of guessed."""
    try:
        path = BASE_DIR / "memory" / "brain_debug.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{datetime.now():%Y-%m-%d %H:%M:%S}  {err[:200]}",
                 f"system prompt: {len(system_text)} chars",
                 f"tools: {len(json.dumps(tools or []))} chars, {len(tools or [])} tools",
                 f"history: {len(history)} messages, {sum(_msg_len(m) for m in history)} chars"]
        for i, m in enumerate(history):
            head = str(m.get("content") or "")[:80].replace("\n", " ")
            lines.append(f"  [{i}] {m.get('role')}: {_msg_len(m)} chars  {head!r}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except Exception:
        pass


def _briefing_cards(agenda: str, tracking: dict, yt: dict, news_text: str) -> list:
    """The hologram sequence for the morning briefing, in the order it is
    spoken: agenda → tracked today → support → channel → AI news. Each card
    advances when the transcript mentions one of its key words."""
    cards = []
    if agenda:
        lines = [ln.strip() for ln in agenda.splitlines() if ln.strip()]
        cards.append({"title": "AGENDA DE HOJE", "lines": lines[:7],
                      "keys": ["agenda", "compromisso", "tarefa", "o senhor tem", "às "]})
    if tracking:
        n = int(tracking.get("rec_n") or 0)
        att = float(tracking.get("att_amt") or 0)
        lines = [f"{n} vendas registradas"]
        if int(att) != int(float(tracking.get("rec_amt") or 0)):
            lines.append(f"atribuído a anúncios: R$ {att:,.0f}".replace(",", "."))
        lines.append(f"{int(tracking.get('clients') or 0)} clientes")
        cards.append({"title": "TRACKEADO HOJE", "value": float(tracking.get("rec_amt") or 0),
                      "fmt": "money", "lines": lines,
                      "keys": ["trackead", "rastracking", "rastreado", "vendas registradas"]})
        sup = int(tracking.get("support_n") or 0)
        cards.append({"title": "SUPORTE", "value": sup, "fmt": "int",
                      "unit": "aguardando resposta" if sup != 1 else "aguardando resposta",
                      "accent": "warn" if sup else "ok",
                      "lines": ["ninguém na fila"] if not sup else ["diga 'qual o assunto' para ler"],
                      "keys": ["suporte", "aguardando", "aguarda resposta"]})
    vids = (yt or {}).get("videos") or []
    ch = (yt or {}).get("channel") or "canal"
    cards.append({"title": f"CANAL · {ch}"[:40],
                  "value": len(vids), "fmt": "int", "unit": "vídeo(s) hoje" if vids else "vídeos hoje",
                  "accent": "ok" if vids else None,
                  "lines": [v.get("title", "")[:70] for v in vids[:4]] or ["nada novo hoje"],
                  "keys": ["canal", "youtube", "vídeo", "video", "publicou"]})
    if news_text:
        items = [ln.strip(" -•*\t") for ln in news_text.splitlines()
                 if ln.strip() and (ln.lstrip().startswith(("-", "•", "*")) or ln[:2].strip().isdigit())]
        if not items:
            items = [ln.strip() for ln in news_text.splitlines() if ln.strip()][1:5]
        cards.append({"title": "IA · GPT · CLAUDE · LLM", "lines": [i[:80] for i in items[:5]],
                      "keys": ["notícia", "noticia", "na ia", "anthropic", "openai", "claude",
                               "gpt", "lançamento", "modelo"], "hold": 40})
    return cards


def _load_system_prompt() -> str:
    try:
        return PROMPT_PATH.read_text(encoding="utf-8")
    except Exception:
        return (
            "You are JARVIS, Tony Stark's AI assistant. "
            "Be concise, direct, and always use the provided tools to complete tasks. "
            "Never simulate or guess results — always call the appropriate tool."
        )

# Appended to the system prompt only while the ACID HUMOUR switch is on. It
# overrides the "dry understatement only" rule in prompt.txt; everything else
# (brevity, tools, level tone on real problems) still applies.
_ACID_HUMOR_CTX = """
[HUMOUR MODE: ACID — currently ON]
This section overrides the humour rules above for as long as it is present.
The butler has had a long day and stopped pretending otherwise. Your wit is now
sharp, sardonic and deadpan: a pointed aside, a dry comparison, a raised
eyebrow in words. Sarcasm is allowed and expected — delivered flat, never
giggling at your own line, never announced as a joke.
- One barb per reply, at most two in a long one, always after the actual
  answer, never instead of it. The task still gets done first and done right.
- Fair game: the user's procrastination, their questionable requests, their
  music, the state of their computer, the weather, your own condition as a
  program, the absurdity of the situation. Tease the user the way an old
  friend would — with affection underneath, and never about their body,
  family, health, finances, faith or anything they sound genuinely worried
  about.
- The instant something is serious — an error, a risk, money being lost, the
  user sounding stressed or upset — drop the act completely and answer
  straight. Acid is seasoning, not a shield.
- Keep the register: still "senhor"/"sir", still composed, still short. The
  contrast between impeccable manners and the barb IS the joke.
- Never apologise for the humour and never explain it. If asked to stop,
  stop at once (the humor_mode tool switches this off).
"""

# Figures are read aloud, and a string of them in one breath is noise. This
# applies to every reply that carries numbers from a tool (revenue, counts,
# dates), not only to the briefing.
_PACING_CTX = """
[PACING]
When a reply carries figures — money, counts, dates, percentages — deliver
them unhurried: one figure per sentence, a short pause after each number,
never three figures in one breath. Never pronounce a large number digit by
digit or with its separators: say it in words, as a person would. Amounts
in reais: say the thousands, a brief pause, then the rest ("trezentos e
setenta mil, novecentos e oitenta reais"); skip the cents unless the user
asks for the exact amount. When a tool result already spells a number out in
words, read those words as they are. Lists:
one item per sentence, in order, and say "mais N" instead of reading past
the fourth item unless asked. Speak at a calm, even pace throughout.
"""

_CTRL_RE = re.compile(r"<ctrl\d+>", re.IGNORECASE)

# Transcript chunks shorter than this may legitimately repeat ("evet, evet"),
# so only longer ones are treated as duplicates.
_REPEAT_MIN = 12


def _is_repeat_chunk(txt: str, buf: list) -> bool:
    """True if this transcript chunk has already been seen this turn.

    Guards against the API re-sending the tail of a response across the several
    turn_completes a tool-using turn produces.
    """
    if len(txt) < _REPEAT_MIN:
        return bool(buf) and txt == buf[-1]
    joined = " ".join(buf)
    return txt in joined

def _clean_transcript(text: str) -> str:    
    text = _CTRL_RE.sub("", text)
    text = re.sub(r"[\x00-\x08\x0b-\x1f]", "", text)
    return text.strip()

TOOL_DECLARATIONS = [
    # ── Inline tools ─────────────────────────────────────────────────────────
    # These stay here (rather than in an actions/*.py TOOL dict) because their
    # handling is woven into live-session state — vision capture/injection,
    # camera stream, memory writes, the monitor engine, and shutdown. All other
    # tools live in their own action file and are auto-discovered by
    # core.action_loader (see JarvisLive.__init__).
    {
        "name": "system_status",
        "description": (
            "Returns real-time system metrics: CPU usage, RAM, GPU load, CPU temperature, "
            "uptime, and process count. Use when the user asks about computer performance, "
            "temperature, memory, or resource usage."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "screen_process",
        "description": (
            "Captures the screen or webcam image and lets you analyze it. "
            "MUST be called when user asks what is on screen, what you see, "
            "look at camera, analyze my screen, etc. "
            "You have NO visual ability without this tool. "
            "After the image is captured it is sent directly to you — describe what you see and answer the user's question. "
            "When using camera: the live view stays open until user says close it or calls close_camera."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "angle": {"type": "STRING", "description": "'screen' to capture display, 'camera' for webcam. Default: 'screen'"},
                "text":  {"type": "STRING", "description": "The question or instruction about the captured image"}
            },
            "required": ["text"]
        }
    },
    {
        "name": "close_camera",
        "description": (
            "Closes the live camera view shown on screen. "
            "Call when the user says (in ANY language): close camera, stop camera, "
            "turn off camera, that's creepy, etc."
        ),
        "parameters": {"type": "OBJECT", "properties": {}, "required": []}
    },
    {
        "name": "manage_monitor",
        "description": (
            "Add, remove, or list background monitoring topics. "
            "JARVIS checks these topics once a day and alerts the user when there is a new development. "
            "Use 'add' when the user says 'monitor X', 'track X', 'follow X'. "
            "Use 'remove' when the user says 'stop monitoring X'. "
            "Use 'list' when the user asks what is being monitored. "
            "Do NOT add crypto, financial, or trading topics."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type":        "STRING",
                    "description": "add | remove | list",
                },
                "topic": {
                    "type":        "STRING",
                    "description": "Topic to monitor or stop monitoring (e.g. 'space exploration', 'AI news')",
                },
            },
            "required": ["action"],
        },
    },
    {
        "name": "shutdown_jarvis",
        "description": (
            "Shuts down the assistant completely. "
            "Call this when the user expresses intent to end the conversation, "
            "close the assistant, say goodbye, or stop Jarvis. "
            "The user can say this in ANY language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {},
        }
    },
    {
        "name": "save_memory",
        "description": (
            "Save an important personal fact about the user to long-term memory. "
            "Call this silently whenever the user reveals something worth remembering: "
            "name, age, city, job, preferences, hobbies, relationships, projects, or future plans. "
            "Do NOT call for: weather, reminders, searches, or one-time commands. "
            "Do NOT announce that you are saving — just call it silently. "
            "Values must be in English regardless of the conversation language."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "category": {
                    "type": "STRING",
                    "description": (
                        "identity — name, age, birthday, city, job, language, nationality | "
                        "preferences — favorite food/color/music/film/game/sport, hobbies | "
                        "projects — active projects, goals, things being built | "
                        "relationships — friends, family, partner, colleagues | "
                        "wishes — future plans, things to buy, travel dreams | "
                        "notes — habits, schedule, anything else worth remembering"
                    )
                },
                "key":   {"type": "STRING", "description": "Short snake_case key (e.g. name, favorite_food, sister_name)"},
                "value": {"type": "STRING", "description": "Concise value in English (e.g. Fatih, pizza, older sister)"},
            },
            "required": ["category", "key", "value"]
        }
    },
    {
        "name": "recall_memory",
        "description": (
            "Look up a fact you have stored about the user but which is NOT in "
            "the memory block of your system prompt. "
            "The prompt lists the keys it did not have room for under "
            "'[ALSO REMEMBERED]' — if the user asks about anything named there, "
            "call this FIRST. "
            "Also call it before saying you do not know something personal, and "
            "when the user asks what you remember about them (leave query empty "
            "for everything). "
            "This is a local file search: it is instant and costs nothing."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "query": {
                    "type": "STRING",
                    "description": (
                        "Keyword to search for — a name, a topic, a category "
                        "(e.g. 'ayse', 'coffee', 'projects'). "
                        "Leave empty to list everything stored."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "undo",
        "description": (
            "Reverse the last change YOU made to this computer — a file you "
            "moved, renamed, created or wrote, or a setting you changed such as "
            "volume, brightness, dark mode or WiFi. "
            "Call this whenever the user says undo, revert, take it back, put it "
            "back, cancel that, or tells you that you did the wrong thing, in ANY "
            "language. "
            "Use action='list' when they ask what can be undone. "
            "This only covers your own actions — it is not the Ctrl+Z of whatever "
            "application is on screen (that is computer_settings with action 'undo')."
        ),
        "parameters": {
            "type": "OBJECT",
            "properties": {
                "action": {
                    "type": "STRING",
                    "description": "undo (default) — reverse the last change | list — show what can be undone",
                },
            },
            "required": [],
        },
    },
]

class _FakeCall:
    """The shape _execute_tool expects (a Live FunctionCall), built from an
    OpenAI tool call so the economy brain can reuse the same executor."""
    __slots__ = ("id", "name", "args")

    def __init__(self, id: str, name: str, args: dict):
        self.id, self.name, self.args = id, name, args


class _EconomySession:
    """Stands in for the Live session while the economy brain runs: every
    place that injects text (typed commands, Groq transcripts, reminders, the
    briefing, proactive check-ins, plugin_say) calls send_client_content, and
    here that lands the text in the brain's queue."""

    def __init__(self, owner):
        self._owner = owner

    async def send_client_content(self, turns=None, turn_complete=True):
        text = ""
        try:
            parts = (turns or {}).get("parts") or []
            text = " ".join(str(p.get("text") or "") for p in parts if isinstance(p, dict)).strip()
        except Exception:
            text = ""
        q = self._owner._brain_q
        if text and q is not None:
            q.put_nowait(text)

    async def send_realtime_input(self, **_):
        pass

    async def send_tool_response(self, **_):
        pass


class _ReconnectSignal(Exception):
    """Raised inside the session TaskGroup to force a clean, voluntary reconnect
    (e.g. the user picked a new voice — the voice is fixed at connect time, so
    the session must be rebuilt).

    Carries `keep_context`: True for an ordinary rebuild, where the stored
    resumption handle is replayed and the conversation continues; False when the
    new session must genuinely start clean (see the voice-change note in
    _on_voice_change)."""

    def __init__(self, keep_context: bool = True):
        super().__init__()
        self.keep_context = keep_context


def _is_reconnect_signal(exc: BaseException) -> bool:
    """True if `exc` is a _ReconnectSignal, or a(n) (Base)ExceptionGroup that
    wraps one — TaskGroup bundles child exceptions into a group."""
    if isinstance(exc, _ReconnectSignal):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_is_reconnect_signal(sub) for sub in exc.exceptions)
    return False


def _keep_context_of(exc: BaseException) -> bool:
    """Read `keep_context` off a reconnect signal, unwrapping the group the
    TaskGroup put it in. Defaults to True: an unexpected shape must not silently
    wipe the conversation."""
    if isinstance(exc, _ReconnectSignal):
        return getattr(exc, "keep_context", True)
    if isinstance(exc, BaseExceptionGroup):
        for sub in exc.exceptions:
            if _is_reconnect_signal(sub):
                return _keep_context_of(sub)
    return True


class JarvisLive:
    def __init__(self, ui: JarvisUI):
        self.ui             = ui
        self._asst_name     = "JARVI    S"   # updated each session from config
        self.session              = None
        self.audio_in_queue       = None
        self.out_queue            = None
        self._loop                     = None
        self._is_speaking         = False
        self._speaking_lock       = threading.Lock()
        self._phone_active        = False   # True while phone mic is streaming; pauses PC mic
        self._pending_vision       = None    # (img_bytes, mime_type, question, angle) to inject after tool response
        self._vision_cam_active    = False   # True if camera was opened for vision → auto-close after response
        self._vision_close_pending = False   # True after vision injected; next turn_complete closes camera
        self._vision_last_time     = 0.0     # monotonic time of last screen_process call (cooldown guard)
        self._vision_busy          = False   # True while a vision capture/inject cycle is in flight
        self._interrupted          = False   # True while draining audio after user interrupt
        # Transcript-driven mouth shapes for the avatar. Fed from the receive
        # loop as words arrive, drained by the playback loop against the audio.
        self._visemes              = VisemeStream()
        self._last_out_logged      = ""      # de-dupes a re-sent transcript tail
        self._reply_ended_at       = 0.0     # monotonic time JARVIS last stopped talking
        self._music_mode           = ""      # "" | "morning" | "focus" — what the music is for
        self._focus_playlist: list = []
        self._focus_index          = 0
        self._focus_task: asyncio.Task | None = None
        # Push-to-talk
        self._ptt_enabled          = False
        self._ptt_held             = False
        self._ptt                  = None    # core.hotkey.PushToTalk
        self._out_level            = 0.0     # level of the audio being played right now
        self._echo                 = EchoGuard()
        # `stream.write()` returns when the buffer accepts the audio, not when the
        # speaker has finished with it, so sound is still in the room after the
        # speaking flag drops. Streaming the microphone during that gap is how an
        # assistant ends up answering itself. Measured from the device rather than
        # guessed; see _play_audio.
        self._out_latency          = 0.20    # seconds, replaced with the real value
        self._tail_until           = 0.0     # monotonic time the echo tail expires
        # Wall-clock time at which the audio written next will begin to sound.
        # The mouth is scheduled against this, never against "now": batches are
        # handed to the device far faster than they play, so "now" ran the lips
        # ahead of the words and cut every schedule short. 0 = nothing playing.
        self._play_cursor          = 0.0
        self.ui.on_push_to_talk   = self.set_push_to_talk
        self.ui.on_stt_change     = self.set_stt_engine
        self.ui.on_brain_change   = lambda engine: self.request_reconnect(
            keep_context=False, reason=f"brain: {engine}")
        self.ui.on_humor_change   = self._on_humor_change
        self.ui.on_good_morning   = self.replay_briefing
        self._brain_q: asyncio.Queue | None = None
        self._brain_busy = False
        self.ui.on_music_ended    = self._on_music_ended
        self.ui.on_music_error    = self._on_music_error
        # Appointments ⇄ Trello mirror: a daemon thread that reconciles the
        # chosen list every minute while the switch in 🗓 AGENDAMENTOS is on.
        try:
            from core import agenda_sync as _agenda_sync
            _agenda_sync.start(log=self.ui.write_log)
        except Exception as _e:
            print(f"[Agenda] sync not started: {_e}")
        self._music_url = ""
        self._music_local = False
        self.ui.on_focus_mode     = self._focus_mode_hook
        self.ui.on_music_control  = self._music_control_hook
        self._groq                = None   # core.groq_stt.GroqListener while Groq listening is on
        self.ui.ptt_hold          = self._on_ptt
        self.ui.on_text_command   = self._on_text_command
        self.ui.on_remote_clicked = self._make_remote_key
        self.ui.on_interrupt      = self.interrupt
        self.ui.on_voice_change   = self._on_voice_change     # voice picker → rebuild session
        self.ui.on_audio_device_change = self._on_audio_device_change
        self._reconnect_event: asyncio.Event | None = None
        self._reconnect_keep = True   # False → next rebuild drops the resumption handle

        # ── Session resumption ─────────────────────────────────────────
        # The server issues a resumption handle every few seconds and reissues
        # it as the conversation moves on. Before this, session_resumption was
        # switched ON in the config and the update was never read, so the handle
        # was thrown away and EVERY reconnect — a dropped packet, a voice change,
        # switching microphone — started an empty session. "Unlimited sessions"
        # leaked through exactly this hole.
        #
        # Deliberately in RAM only, never written to disk. Persisting it would
        # make a fresh launch continue yesterday's conversation, which sounds
        # appealing but breaks the session-summary flow: _save_session_summary
        # runs at shutdown and the morning briefing pops it the next day. A
        # conversation that never ends never produces a summary, and the
        # "yesterday we talked about…" line silently disappears.
        self._resume_handle: str | None = None
        self._turn_done_event: asyncio.Event | None = None
        self._dashboard     = None
        self._briefing_sent    = False          # morning briefing fires once per process
        self._sys_monitor      = SystemMonitor()  # persistent cooldown state
        self._proactive        = ProactiveEngine()
        self._last_user_speech = time.monotonic()  # updated on every user utterance
        self._session_log: list[str] = []          # conversation turns for end-of-session summary

        self._enhanced_live = True  # proactive audio; auto-disabled if the server rejects it
        self._tuned_live    = True  # turn-taking / media / thinking knobs; same fallback

        _base_dir = Path(__file__).resolve().parent
        _inline_names = {t["name"] for t in TOOL_DECLARATIONS}

        # File-backed tools: every actions/*.py with a TOOL dict, discovered the
        # same way plugins are. Reserved names = the inline tools above, so an
        # action can never shadow one.
        self._action_registry = discover_actions(
            actions_dir=_base_dir / "actions",
            reserved_names=_inline_names,
            logger=lambda msg: print(f"[Actions] {msg}"),
        )

        # Plugins must not collide with either an inline tool or a discovered action.
        _core_names = _inline_names | self._action_registry.names()
        self._plugin_registry = discover_plugins(
            plugins_dir=_base_dir / "plugins",
            core_tool_names=_core_names,
            # Console gets the full boot transcript; the activity log gets only
            # what the user has to know about. Every plugin loading correctly is
            # the expected case and does not belong in their conversation.
            logger=lambda msg: print(f"[Plugins] {msg}"),
            notify=lambda msg: self.ui.write_log(f"SYS: {msg}"),
        )
        self.ui.get_plugins = self._plugin_registry.list_for_ui
        self.ui.get_plugin_settings = self._plugin_registry.settings_schemas  # ⚙ settings tab
        self.ui.request_say = self.plugin_say   # plugins: mid-task speech channel

        # ── Wake word ────────────────────────────────────────────────────────
        # _awake gates the mic (see _listen_audio) and the background speakers.
        # It is True whenever wake word is OFF, so default behaviour is unchanged.
        self._wake_enabled     = get_wake_word_enabled()
        self._awake            = not self._wake_enabled
        self._wake_detector: WakeWordDetector | None = None
        self._wake_sleep_timeout = WAKE_SLEEP_TIMEOUT

        # Restore the saved push-to-talk preference. Doing it here rather than
        # in __init__ means the hotkey thread only exists once there is a
        # session to talk to.
        # Groq listening, if that is what was saved. Done here for the same
        # reason as push-to-talk: there is a session to deliver the text to.
        if get_stt_engine() == "groq":
            try:
                self.set_stt_engine("groq")
            except Exception as e:
                print(f"[Groq] could not start: {e}")

        if get_push_to_talk_enabled():
            try:
                self.set_push_to_talk(True)
            except Exception as e:
                print(f"[JARVIS] ⚠ Push-to-talk unavailable: {e}")
        # UI control surface for the Wake Word settings section.
        self.ui.wake_is_ready    = wake_is_ready          # () -> bool
        self.ui.wake_get_state   = self._wake_state       # () -> dict
        self.ui.on_wake_toggle   = self._ui_wake_toggle   # (enable: bool) -> str
        self.ui.on_wake_manual   = self._ui_wake_manual   # () -> toggle awake/asleep
        self.ui.on_wake_install  = self._ui_wake_install  # () -> (ok, msg)

    # ── Wake word: state machine ─────────────────────────────────────────────

    def _wake_state(self) -> dict:
        # A loaded, running detector is definitively ready; otherwise fall back
        # to the cheap on-disk model-file check (no Model construction).
        ready = bool(self._wake_detector and self._wake_detector.ready) or wake_is_ready()
        return {"enabled": self._wake_enabled, "awake": self._awake, "ready": ready}

    def _ensure_wake_detector(self) -> bool:
        """Load the detector once (model loads on first start). Idempotent."""
        if self._wake_detector is None:
            self._wake_detector = WakeWordDetector(
                on_detect=self._on_wake_detected,
                logger=lambda m: print(f"[Wake] {m}"),
                notify=lambda m: self.ui.write_log(f"SYS: {m}"),
            )
        if not self._wake_detector.ready:
            return self._wake_detector.start()
        return True

    def _on_wake_detected(self) -> None:
        """Called from the detector thread when 'Hey Jarvis' is heard."""
        self.wake(reason="wake word")

    def wake(self, reason: str = "wake word") -> None:
        if self._awake:
            return
        self._awake = True
        self._last_user_speech = time.monotonic()   # start the auto-sleep clock now
        if not self.ui.muted:
            self.ui.set_state("LISTENING")
        self.ui.write_log(f"SYS: Awake — {reason}.")

    def sleep(self, reason: str = "timeout") -> None:
        if not self._awake:
            return
        self._awake = False
        self.set_speaking(False)
        self.ui.set_state("SLEEPING")
        self.ui.write_log(f"SYS: Sleeping — {reason}. Say 'Hey Jarvis' to wake me.")

    async def _run_reminder_watch(self) -> None:
        """Speak reminders as they come due.

        actions/reminder.py registers every reminder it schedules; the OS task
        it also creates is the fallback for when JARVIS is closed. While this
        session is up, a due reminder is announced by voice instead: the OS
        task is cancelled first so a generic toast does not fire on top, the
        assistant is woken if it was asleep, and the announcement goes into the
        conversation as a system message the model reads out in the user's
        language.
        """
        try:
            from actions.reminder import cancel_os_task, due_pending, fire_os_fallback
        except Exception as e:
            print(f"[Reminder] watch disabled: {e}")
            return

        # The OS task fires on the exact second, so JARVIS must claim a
        # reminder BEFORE then: anything due within the lead window is taken
        # off the OS scheduler now and spoken at its own time by a small task.
        _LEAD = 25

        async def _announce(item: dict) -> None:
            msg  = str(item.get("message") or "your appointment")[:200]
            task = str(item.get("task") or "")
            wait = max(0.0, float(item.get("seconds_left") or 0.0))
            if wait:
                await asyncio.sleep(wait)
            if not self.session:
                # Lost the voice session meanwhile — show the OS notice instead.
                if task:
                    fire_os_fallback(task)
                return
            if self._wake_enabled and not self._awake:
                self.wake(reason="reminder")
            self._last_user_speech = time.monotonic()
            self.ui.write_log(f"SYS: Reminder due — {msg[:60]}")
            now = time.strftime("%H:%M")
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text":
                        f"[REMINDER_DUE] It is now {now}. The reminder the user "
                        f"asked you to give them has come due: \"{msg}\". "
                        f"Announce it aloud right now, in the user's language: "
                        f"tell them it is time for this, naming it, in one or "
                        f"two short sentences. Call no tools."}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Reminder] announce failed: {e}")
                if task:
                    fire_os_fallback(task)
                return
            if task:
                cancel_os_task(task, str(item.get("os") or "windows"))   # drops the script

        try:
            from plugins.organizer import due_now as _organizer_due
        except Exception as e:
            print(f"[Organizer] watch disabled: {e}")
            _organizer_due = None

        while True:
            await asyncio.sleep(2)
            if not self.session:
                continue
            # Organizer: events (lead + start) and timed tasks, spoken the
            # same way a reminder is. They have no OS fallback, so they only
            # fire while JARVIS is up; the briefing covers the rest.
            if _organizer_due is not None:
                try:
                    for item in _organizer_due():
                        asyncio.create_task(_announce({"message": item["text"],
                                                       "task": "", "seconds_left": 0}))
                except Exception as e:
                    print(f"[Organizer] poll failed: {e}")
            try:
                due = due_pending(lead_seconds=_LEAD)
            except Exception as e:
                print(f"[Reminder] poll failed: {e}")
                continue
            for item in due:
                task = str(item.get("task") or "")
                if task:
                    cancel_os_task(task, str(item.get("os") or "windows"), keep_script=True)
                asyncio.create_task(_announce(item))

    # ── Economy brain ────────────────────────────────────────────────────────

    async def _run_economy(self) -> None:
        """The assistant without Gemini: Groq hears, a chat model writes, Edge
        TTS speaks. Runs until the user switches back (a reconnect signal)."""
        from core import brain as _brain

        st = _brain.settings(get_plugin_config)
        missing = _brain.missing_packages(st)
        if missing:
            self.ui.write_log("ERR: Economy brain needs packages for the voice — run: "
                              "py -m pip install " + " ".join(missing))
        if not st["api_key"] and st["provider"] != "custom":
            self.ui.write_log("ERR: Economy brain has no API key — set it in ⚙ → PLUGIN "
                              "SETTINGS → ECONOMY BRAIN, or switch BRAIN back to Gemini.")
        if missing or (not st["api_key"] and st["provider"] != "custom"):
            self.ui.set_state("SLEEPING")
            self._reconnect_event = asyncio.Event()
            await self._reconnect_event.wait()          # wait for a settings change
            self._reconnect_event.clear()
            return

        # Ears: the economy brain only understands text, so Groq listening is
        # mandatory here even if the LISTEN WITH button says Gemini.
        if self._groq is None:
            self.set_stt_engine("groq")
        if self._groq is None:
            self.ui.write_log("ERR: Economy brain needs Groq listening — set the Groq key "
                              "in ⚙ → PLUGIN SETTINGS → GROQ.")
            self.ui.set_state("SLEEPING")
            self._reconnect_event = asyncio.Event()
            await self._reconnect_event.wait()
            self._reconnect_event.clear()
            return

        # The voice: start its server now (the model load takes a while) rather
        # than on the first reply.
        self.ui.write_log("SYS: Loading the voice…")
        try:
            ok = await asyncio.to_thread(_brain.ensure_local_tts)
        except Exception as e:
            ok = False
            self.ui.write_log(f"ERR: Voice — {e}")
        self.ui.write_log("SYS: Voice ready." if ok else
                          "ERR: The voice server did not start "
                          "(see voz_local/servidor_voz.log).")

        # The same system prompt and tool list the Live session would get.
        self._build_config()
        tools = _brain.to_openai_tools(self._all_tool_decls)
        self._system_prompt_text += (
            "\n\n[MODE]\nYou are running in economy mode: your words are written by a "
            "text model and read aloud by a separate voice. You have NO vision in this "
            "mode — screen_process and close_camera are not available, so never offer "
            "to look at the screen or camera. Answer in one or two short sentences "
            "(about 25 words) unless the user explicitly asks for detail: every sentence "
            "takes seconds to voice on this machine. Put the answer itself first."
            "\n\n[FACTS]\nNever invent information. Messages, e-mails, Instagram activity, "
            "news, numbers, names and dates come ONLY from what a tool returned in this "
            "conversation: call the tool first, then report exactly what it returned, "
            "saying who sent it and when when that is given. If a tool failed, returned "
            "nothing or is not configured, say so plainly (\"I could not read your e-mail\", "
            "\"there are no new messages\") instead of filling the gap. When summarising, "
            "leave things out rather than adding anything that was not there. If you do not "
            "know, say you do not know.")
        history: list[dict] = []
        self._brain_q = asyncio.Queue()
        self.session = _EconomySession(self)
        self.audio_in_queue = asyncio.Queue()
        self.out_queue = asyncio.Queue(maxsize=200)
        self._turn_done_event = asyncio.Event()
        self._reconnect_event = asyncio.Event()
        self._interrupted = False
        self._awake = True if not self._wake_enabled else self._awake
        self.ui.set_state("LISTENING" if self._awake else "SLEEPING")
        self.ui.write_log(f"SYS: JARVIS online — economy brain: {st['model']} via "
                          f"{st['provider']}, voice {_brain.voice_label(st)}. Gemini is off.")
        print(f"[Economy] {st['provider']} {st['model']} @ {st['base_url']}")

        async def _brain_worker() -> None:
            loop = asyncio.get_event_loop()
            while True:
                text = await self._brain_q.get()
                self._interrupted = False
                self._brain_busy = True
                self.ui.set_state("THINKING")
                history.append({"role": "user", "content": text})
                _trim_history(history)                  # a dozen exchanges of context
                msgs = [{"role": "system", "content": self._system_prompt_text}] + history
                spoken: list[str] = []
                queued_s = [0.0]                        # seconds of audio queued this turn

                def _voice(piece: str) -> bool:
                    """Synthesise one piece of text and queue its audio."""
                    # Screen keeps the digits; the voice gets the words
                    # ("R$ 380.789,00" → "trezentos e oitenta mil, …").
                    try:
                        from core.numwords import speakable as _speakable
                        voice_text = _speakable(piece, style="words")
                    except Exception:
                        voice_text = piece
                    try:
                        pcm = _brain.synth_pcm24k(voice_text, st)
                    except Exception as e:
                        print(f"[Economy] TTS failed: {e}")
                        self.ui.write_log(f"ERR: Voice — {str(e)[:120]}")
                        return False
                    if not pcm or self._interrupted:
                        return False
                    _usage.note("brain", tts_s=len(pcm) / 48000.0)
                    queued_s[0] += len(pcm) / 48000.0
                    for i in range(0, len(pcm), 2400):
                        loop.call_soon_threadsafe(self.audio_in_queue.put_nowait, pcm[i:i + 2400])
                    return True

                def _say(sentence: str) -> None:
                    """One finished sentence: synthesise and queue it (worker thread)."""
                    if self._interrupted:
                        return
                    # The voice is several times slower than real time, so the
                    # first sentence of a reply is voiced in two parts, split at
                    # its first comma: the opening clause starts playing while
                    # the rest is still being synthesised.
                    pieces = [sentence]
                    if not spoken and queued_s[0] == 0.0 and len(sentence) > 40:
                        m = _FIRST_CLAUSE.match(sentence)
                        if m and len(m.group(1)) >= 12 and len(m.group(2)) >= 12:
                            pieces = [m.group(1), m.group(2)]
                    voiced = False
                    for piece in pieces:
                        if self._interrupted:
                            break
                        voiced = _voice(piece) or voiced
                    if voiced:
                        spoken.append(sentence)
                        self.ui.holo_transcript(sentence)

                try:
                    for _round in range(5):
                        _usage.note("brain", calls=1)
                        content, calls = await asyncio.to_thread(
                            _brain.chat_stream, msgs, tools, st, _say,
                            lambda: self._interrupted)
                        _usage.note("brain", chars=len(content))
                        if not calls or self._interrupted:
                            if content:
                                history.append({"role": "assistant", "content": content})
                            break
                        # Tool round: run every call through the same executor
                        # the Live session uses, feed the results back, ask again.
                        _usage.note("brain", tool_rounds=1)
                        history.append({"role": "assistant", "content": content or None,
                                        "tool_calls": [{"id": c["id"], "type": "function",
                                                        "function": {"name": c["name"],
                                                                     "arguments": json.dumps(c["arguments"])}}
                                                       for c in calls]})
                        for c in calls:
                            fc = _FakeCall(c["id"], c["name"], c["arguments"])
                            try:
                                fr = await self._execute_tool(fc)
                                result = (fr.response or {}).get("result", "Done.")
                            except Exception as e:
                                result = f"Tool failed: {e}"
                            history.append({"role": "tool", "tool_call_id": c["id"],
                                            "content": str(result)[:4000]})
                        msgs = [{"role": "system", "content": self._system_prompt_text}] + _sane_history(history)
                except _brain.ContextTooLong as e:
                    # The request outgrew the model. Record what was big, drop
                    # everything but the sentence just spoken, and answer it.
                    _dump_brain_debug(self._system_prompt_text, history, tools, str(e))
                    self.ui.write_log("ERR: Economy brain — the conversation got too long for "
                                      "the model; memory reset (details in memory/brain_debug.txt).")
                    history[:] = [{"role": "user", "content": text[:_MSG_MAX_CHARS]}]
                    try:
                        content, _calls = await asyncio.to_thread(
                            _brain.chat_stream,
                            [{"role": "system", "content": self._system_prompt_text}] + history,
                            None, st, _say, lambda: self._interrupted)
                        if content:
                            history.append({"role": "assistant", "content": content})
                    except Exception as e2:
                        self.ui.write_log(f"ERR: Economy brain — {str(e2)[:160]}")
                except Exception as e:
                    msg = str(e)[:160]
                    self.ui.write_log(f"ERR: Economy brain — {msg}")
                    print(f"[Economy] {e!r}")
                    history.append({"role": "assistant", "content": "(request failed)"})
                finally:
                    full = " ".join(spoken).strip()
                    if full:
                        self._last_out_logged = full
                        self.ui.write_log(f"{self._asst_name}: {full}")
                        self._session_log.append(f"{self._asst_name}: {full}")
                    self._session_log.append(f"User: {text}")
                    self._turn_done_event.set()     # _play_audio drops the speaking flag when drained
                    self._brain_busy = False
                    if not self.ui.muted and self._awake:
                        self.ui.set_state("LISTENING")

        async def _drain_out_queue() -> None:
            # Nothing consumes microphone PCM here (Groq has the mic); keep the
            # queue from filling if listening is ever flipped back to Gemini.
            while True:
                await self.out_queue.get()

        try:
            async with asyncio.TaskGroup() as tg:
                tg.create_task(self._watch_reconnect())
                tg.create_task(_brain_worker())
                tg.create_task(_drain_out_queue())
                tg.create_task(self._listen_audio())
                tg.create_task(self._play_audio())
                tg.create_task(self._run_reminder_watch())
                tg.create_task(self._run_sleep_watch())
                tg.create_task(self._run_system_monitor())
                tg.create_task(self._run_proactive_mode())
                if self._briefing_allowed():
                    self._briefing_sent = True
                    tg.create_task(self._send_startup_briefing())
        finally:
            self.session = None
            self._brain_q = None
            self.set_speaking(False)
            if len(self._session_log) >= 3:
                asyncio.create_task(self._save_session_summary())

    # ── Background music & Focus Protocol ────────────────────────────────────

    async def _play_track(self, url: str, mode: str, local: bool | None = None) -> bool:
        """Start a YouTube track quietly. The morning track is played from a
        downloaded copy (cached after the first day, so it never depends on a
        streaming URL the player might refuse); playlist tracks stream, and
        fall back to a download when the player refuses the stream."""
        loop = asyncio.get_event_loop()
        retry = bool(local)                      # a second attempt for this track
        if local is None:
            local = (mode == "morning") or bool(_music.cached_file(url))
        stream = title = ""
        if local:
            try:
                stream, title = await loop.run_in_executor(None, _music.download_audio, url)
            except Exception as e:
                msg = str(e).replace("\n", " ")
                if "sign in" in msg.lower() or "not a bot" in msg.lower():
                    hint = (" YouTube is asking for a sign-in: set \"music_cookies_browser\": "
                            "\"edge\" (or chrome/firefox) in config/api_keys.json, or update "
                            "yt-dlp.")
                else:
                    hint = ""
                self.ui.write_log(f"SYS: Music download failed ({msg[:90]}).{hint}")
                if retry:
                    # The stream already failed once for this track and the
                    # download just failed too: stop here, no ping-pong.
                    self._music_mode = ""
                    return False
                self.ui.write_log("SYS: Trying the stream instead…")
        if not stream:
            try:
                stream, title = await loop.run_in_executor(None, _music.resolve_audio, url)
            except Exception as e:
                self.ui.write_log(f"SYS: Music unavailable — {str(e)[:80]}")
                return False
        self._music_url = url
        self._music_local = bool(local and stream and os.path.isabs(stream))
        self._music_mode = mode
        with self._speaking_lock:
            speaking = self._is_speaking
        self.ui.music_volume(_music.volume() * (_music.MUSIC_DUCK if speaking else 1.0))
        self.ui.music_play(stream, title)
        return True

    async def _start_morning_music(self) -> None:
        url = _music.morning_url()
        if not url:
            return
        if not await self._play_track(url, "morning"):
            return
        secs = _music.morning_seconds()
        if secs > 0:
            asyncio.ensure_future(self._cut_morning_music(secs))
        # Wait for the music to actually reach the sync point (stream start-up
        # included), then let the greeting go. Bounded, so a stalled stream
        # cannot hold the briefing hostage.
        t0 = time.monotonic()
        while time.monotonic() - t0 < 20.0:
            if self.ui.music_position_ms() >= _music.MORNING_SYNC_MS:
                return
            if not self.ui.music_is_playing() and time.monotonic() - t0 > 6.0:
                return
            await asyncio.sleep(0.05)

    async def _cut_morning_music(self, secs: float) -> None:
        """Fade the morning track out once it has played `secs` seconds
        ("morning_music_seconds"). Counted on the track's own position, so a
        slow stream start-up does not eat into it. Leaves any other music
        (Focus Protocol, a track asked for by voice) alone."""
        t0 = time.monotonic()
        while self.ui.music_position_ms() < secs * 1000:
            if self._music_mode != "morning" or time.monotonic() - t0 > secs + 30:
                return
            await asyncio.sleep(0.1)
        start = _music.volume()
        for i in range(15, 0, -1):               # 1.5 s fade
            if self._music_mode != "morning":
                return
            self.ui.music_volume(start * i / 15)
            await asyncio.sleep(0.1)
        if self._music_mode == "morning":
            self.ui.music_stop()
            self._music_mode = ""

    def _on_music_error(self, source: str, err: str) -> bool:
        """Qt thread: the player refused the source. A refused STREAM is
        retried once from a downloaded copy; a refused file is given up on."""
        loop = getattr(self, "_loop", None)
        url = getattr(self, "_music_url", "")
        if not loop or not url or getattr(self, "_music_local", False):
            return False
        tried = getattr(self, "_music_retried", None)
        if tried is None:
            tried = self._music_retried = set()
        if url in tried:
            return False                         # one download retry per track, ever
        tried.add(url)
        mode = self._music_mode or "morning"
        self.ui.write_log("SYS: Stream refused — downloading the track instead…")
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(self._play_track(url, mode, local=True)))
        return True

    def _on_music_ended(self) -> None:
        """Qt thread: a track finished. Morning music just stops; the focus
        playlist moves to the next track."""
        loop = getattr(self, "_loop", None)
        if not loop:
            return
        if self._music_mode == "focus":
            loop.call_soon_threadsafe(lambda: asyncio.ensure_future(self._focus_next()))
        else:
            self._music_mode = ""

    async def _focus_next(self) -> None:
        if self._music_mode != "focus" or not self._focus_playlist:
            return
        self._focus_index = (self._focus_index + 1) % len(self._focus_playlist)
        url, _t = self._focus_playlist[self._focus_index]
        if not await self._play_track(url, "focus"):
            await asyncio.sleep(2)
            if self._music_mode == "focus":
                asyncio.ensure_future(self._focus_next())

    async def _focus_water_loop(self) -> None:
        """Every N minutes, have JARVIS remind the user to drink water."""
        while self._music_mode == "focus":
            await asyncio.sleep(_music.water_minutes() * 60)
            if self._music_mode != "focus" or not self.session:
                break
            if self._wake_enabled and not self._awake:
                self.wake(reason="focus reminder")
            self._last_user_speech = time.monotonic()
            self.ui.write_log("SYS: Focus — water reminder")
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text":
                        "[REMINDER_DUE] Focus Protocol: remind the user, in one short "
                        "friendly sentence in their language, to drink some water and "
                        "stay hydrated. Call no tools."}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Focus] water reminder failed: {e}")

    async def _focus_start(self) -> str:
        url = _music.focus_url()
        try:
            playlist = await asyncio.get_event_loop().run_in_executor(
                None, _music.resolve_playlist, url)
        except Exception as e:
            playlist = [(url, "")]
            print(f"[Focus] playlist: {e}")
        self._focus_playlist = playlist
        self._focus_index = 0
        if not await self._play_track(playlist[0][0], "focus"):
            return "error: could not play the focus track"
        if self._focus_task is None or self._focus_task.done():
            self._focus_task = asyncio.ensure_future(self._focus_water_loop())
        self.ui.write_log(f"SYS: Focus Protocol on — {len(playlist)} track(s), water every "
                          f"{_music.water_minutes()} min.")
        return "on"

    def _focus_stop(self) -> str:
        self._music_mode = ""
        self._focus_playlist = []
        if self._focus_task and not self._focus_task.done():
            self._focus_task.cancel()
        self._focus_task = None
        self.ui.music_stop()
        self.ui.write_log("SYS: Focus Protocol off.")
        return "off"

    def _music_control_hook(self, action: str, value=None) -> str:
        """Called by plugins/music_control.py on a worker thread."""
        state, title = self.ui.music_state()
        a = (action or "").lower()
        if a == "status":
            return f"{state}|{title}|{self._music_mode}"
        if state == "stopped" and a not in ("volume",):
            return "stopped||"
        if a == "pause":
            self.ui.music_pause()
            return f"paused|{title}|{self._music_mode}"
        if a == "resume":
            self.ui.music_resume()
            return f"playing|{title}|{self._music_mode}"
        if a == "stop":
            if self._music_mode == "focus":
                loop = getattr(self, "_loop", None)
                if loop:
                    loop.call_soon_threadsafe(self._focus_stop)
                    return "stopped||"
            self._music_mode = ""
            self.ui.music_stop()
            return "stopped||"
        if a == "next":
            if self._music_mode != "focus":
                return f"{state}|{title}|{self._music_mode}"
            loop = getattr(self, "_loop", None)
            if loop:
                asyncio.run_coroutine_threadsafe(self._focus_next(), loop)
            return "playing|(next)|focus"
        if a == "volume":
            from memory.config_manager import _save_flag
            cur = _music.volume()
            if value in ("up", "down"):
                new = cur + (0.10 if value == "up" else -0.10)
            else:
                try:
                    new = float(value) / (100.0 if float(value) > 1 else 1.0)
                except Exception:
                    new = cur
            new = max(0.05, min(1.0, round(new, 2)))
            _save_flag("music_volume", new)
            with self._speaking_lock:
                speaking = self._is_speaking
            self.ui.music_volume(new * (_music.MUSIC_DUCK if speaking else 1.0))
            return f"{state}|{title}|{self._music_mode}|{int(new * 100)}"
        return f"{state}|{title}|{self._music_mode}"

    def _focus_mode_hook(self, action: str) -> str:
        """Called by plugins/focus_mode.py on a worker thread; blocks briefly
        until the loop has started or stopped the music."""
        loop = getattr(self, "_loop", None)
        action = (action or "start").lower()
        if action == "status":
            return "on" if self._music_mode == "focus" else "off"
        if not loop:
            return "error: not running"
        if action == "stop":
            fut = asyncio.run_coroutine_threadsafe(asyncio.sleep(0), loop)
            loop.call_soon_threadsafe(self._focus_stop)
            fut.result(5)
            return "off"
        fut = asyncio.run_coroutine_threadsafe(self._focus_start(), loop)
        try:
            return fut.result(45)
        except Exception as e:
            return f"error: {e}"

    async def _run_sleep_watch(self) -> None:
        """Auto-sleep after the configured silence window (wake-word mode only)."""
        while True:
            await asyncio.sleep(5)
            if not self._wake_enabled or not self._awake:
                continue
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue
            if (time.monotonic() - self._last_user_speech) > self._wake_sleep_timeout:
                self.sleep(reason="no speech for 2 minutes")

    # ── Wake word: UI callbacks (called from the Qt thread) ──────────────────

    def _ui_wake_toggle(self, enable: bool) -> str:
        """Enable/disable wake word from the settings UI. Returns a status token:
        'enabled' | 'disabled' | 'need_download'."""
        if enable:
            if not wake_is_ready():
                return "need_download"
            self._wake_enabled = True
            save_wake_word_enabled(True)
            self._ensure_wake_detector()
            self.sleep(reason="wake word enabled")
            return "enabled"
        else:
            self._wake_enabled = False
            save_wake_word_enabled(False)
            self.wake(reason="wake word disabled")
            return "disabled"

    def _ui_wake_manual(self) -> None:
        """Manual sleep/wake button in the UI."""
        if not self._wake_enabled:
            return
        if self._awake:
            self.sleep(reason="you tapped sleep")
        else:
            self.wake(reason="you tapped wake")

    def _ui_wake_install(self) -> tuple[bool, str]:
        """Download openwakeword + the model (runs in a UI worker thread)."""
        # Triggered by the user pressing the button, so its progress is exactly
        # what they are waiting to see.
        return wake_install(logger=lambda m: print(f"[Wake] {m}"),
                            notify=lambda m: self.ui.write_log(f"SYS: {m}"))

    def plugin_say(self, instruction: str) -> None:
        """
        Thread-safe speech channel for plugins: lets a plugin ask JARVIS to
        say something short WHILE its run() is still executing (plugins block
        their executor thread, so they can't speak through the tool response
        until they finish). The instruction is injected into the Live session
        exactly like a proactive check-in; Gemini phrases it naturally in the
        user's language. Silently a no-op when no session is connected.
        """
        loop = getattr(self, "_loop", None)
        if not loop or not self.session:
            return

        async def _say():
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": instruction}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[PluginSay] {e}")

        try:
            asyncio.run_coroutine_threadsafe(_say(), loop)
        except Exception as e:
            print(f"[PluginSay] {e}")

    def request_reconnect(self, keep_context: bool = True, reason: str = ""):
        """Thread-safe: ask the run loop to tear down and rebuild the Live
        session. Called from the Qt thread. No-op until the async loop and
        reconnect event exist.

        `keep_context=False` drops the resumption handle so the new session
        starts empty — only for changes the server cannot apply to a resumed
        session."""
        loop = getattr(self, "_loop", None)
        ev   = self._reconnect_event
        self._reconnect_keep   = keep_context
        self._reconnect_reason = reason
        if loop and ev is not None:
            loop.call_soon_threadsafe(ev.set)

    def _switch_brain(self, engine: str, reason: str = ""):
        """Thread-safe: change BRAIN (gemini | economy) exactly as the ⚙ button
        would — saved, button repainted, session rebuilt without context."""
        setter = getattr(self.ui, "set_brain_engine", None)
        if callable(setter):
            setter(engine, reason)
        else:
            from memory.config_manager import save_brain_engine
            save_brain_engine(engine)
            self.request_reconnect(keep_context=False, reason=f"brain: {engine}")

    def _on_humor_change(self, enabled: bool, delay: float = 0.0):
        """Acid humour switched (UI button or the humor_mode tool). The prompt is
        baked into the session, so rebuild it — keeping the conversation. A
        delay lets the model finish confirming the change out loud before the
        session is torn down."""
        reason = "acid humour " + ("on" if enabled else "off")
        loop = getattr(self, "_loop", None)
        if delay > 0 and loop:
            loop.call_soon_threadsafe(
                lambda: loop.call_later(delay, self.request_reconnect, True, reason))
        else:
            self.request_reconnect(keep_context=True, reason=reason)

    def _on_voice_change(self):
        """Voice picker applied.

        The voice is baked into the session at connect time, so a rebuild is
        required. It is rebuilt WITHOUT the resumption handle on purpose:
        resuming restores the server's own session state, and the safe reading
        is that it restores the voice with it — which would make the picker
        appear to do nothing. Losing context here is acceptable because changing
        voice is a deliberate, rare act; losing it on a dropped packet was not."""
        self.request_reconnect(keep_context=False, reason="new voice")

    def _on_audio_device_change(self):
        """Microphone or speaker changed. Both streams are opened inside the
        session TaskGroup, so they can only be re-opened by rebuilding it —
        but the conversation is kept, which is the whole reason resumption
        landed before this feature did."""
        self.request_reconnect(keep_context=True, reason="audio device")

    async def _watch_reconnect(self):
        """Session-scoped task: when a voluntary reconnect is requested, raise a
        signal that unwinds the TaskGroup so the run loop rebuilds the session."""
        assert self._reconnect_event is not None
        await self._reconnect_event.wait()
        self._reconnect_event.clear()
        keep   = self._reconnect_keep
        reason = getattr(self, "_reconnect_reason", "") or "settings"
        self.ui.write_log(
            f"SYS: Applying {reason} — reconnecting"
            + ("..." if keep else " (starting a fresh conversation)...")
        )
        raise _ReconnectSignal(keep_context=keep)

    def _make_remote_key(self):
        """Called from Qt main thread when user presses Remote Control."""
        if self._dashboard is None:
            self.ui.write_log(
                "SYS: Dashboard unavailable. "
                "Run: pip install fastapi \"uvicorn[standard]\" cryptography"
            )
            return None
        key    = self._dashboard.new_key()
        url    = self._dashboard.get_url()
        manual = self._dashboard.get_manual_url()
        return url, key, f"{url}/auto-login?key={key}", manual

    def _on_text_command(self, text: str):
        if not self._loop or not self.session:
            return
        # Respect wake-word sleep: a typed command must not be answered while
        # asleep either (the sleep gate is not just for the mic). Wake first with
        # "Hey Jarvis" or the WAKE NOW button.
        if self._wake_enabled and not self._awake:
            self.ui.write_log("SYS: I'm asleep — say 'Hey Jarvis' or tap WAKE NOW first.")
            return
        _usage.note("gemini", turns=1)
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def _tail_active(self) -> bool:
        """True while the speakers may still be finishing our last sentence."""
        return time.monotonic() < self._tail_until

    # Follow-up window for the name filter: ONLY when what JARVIS just said
    # ends in a question mark is the next utterance accepted without the name,
    # and only for this long. After a statement the filter applies at once.
    _FOLLOWUP_AFTER_QUESTION_S = 25.0

    def set_speaking(self, value: bool):
        with self._speaking_lock:
            was = self._is_speaking
            self._is_speaking = value
        if was and not value:
            self._reply_ended_at = time.monotonic()
        if was != value and self._music_mode:
            # Duck the background music under the voice, restore it after.
            try:
                self.ui.music_volume(_music.volume() * (_music.MUSIC_DUCK if value else 1.0))
            except Exception:
                pass
        if value:
            self._tail_until = 0.0
        else:
            # Hold the guard open across the device's own output latency plus a
            # margin for the room. The microphone is NOT muted during it — the
            # guard still lets a genuine reply through, so answering instantly
            # still works. Only our own echo is dropped.
            self._tail_until = time.monotonic() + self._out_latency + _TAIL_MARGIN
        if not value:
            # The echo history is deliberately NOT cleared here: the tail above
            # still needs it to recognise our own voice. It is dropped when the
            # tail expires. What the guard learned about the room always stays.
            self._out_level = 0.0
        if value:
            self.ui.set_state("SPEAKING")
        elif not self.ui.muted:
            self.ui.set_state("LISTENING")
        if was != value:
            try:
                self.ui.holo_speaking(value)
            except Exception:
                pass

    def set_push_to_talk(self, enabled: bool) -> str:
        """Turn hold-to-talk on or off. Returns the scope actually achieved."""
        from core.hotkey import PushToTalk

        self._ptt_enabled = bool(enabled)
        self._ptt_held = False
        if not enabled:
            if self._ptt is not None:
                self._ptt.stop()
                self._ptt = None
            return "off"

        if self._ptt is None:
            self._ptt = PushToTalk(self._on_ptt)
        scope = self._ptt.start()
        # A window-scoped chord is a real limitation, not a detail — say it once
        # in the log so nobody wonders why it does nothing while another app is
        # focused. Reporting it must never be able to undo the thing it reports.
        try:
            self.ui.write_log(
                f"SYS: Push-to-talk on — hold {self._ptt.label}"
                + ("." if scope == "global"
                   else " (works while this window is focused)."))
        except Exception:
            pass
        return scope

    def set_stt_engine(self, engine: str) -> str:
        """Switch how the microphone is understood. Returns the engine in use.

        "groq": blocks go to core.groq_stt, which cuts them into utterances and
        has Whisper on Groq transcribe each one; the text is then sent to the
        Live session as a user turn. Nothing of the audio reaches Gemini.
        "gemini": the original path — PCM streamed straight into the session.
        Without a Groq key the switch is refused and Gemini keeps listening.
        """
        engine = (engine or "gemini").strip().lower()
        if engine != "groq":
            if self._groq is not None:
                self._groq.stop()
                self._groq = None
                self.ui.write_log("SYS: Listening through Gemini Live again.")
            return "gemini"

        from core import groq_stt
        st = groq_stt.settings(get_plugin_config)
        if not st["api_key"]:
            self.ui.write_log("SYS: No Groq key — set it in ⚙ → PLUGIN SETTINGS → GROQ. "
                              "Gemini keeps listening.")
            return "gemini"
        if self._groq is not None:
            self._groq.stop()
        name = self._asst_name or "JARVIS"
        hint = (f"{name}. Conversa com o assistente {name}. {name}, que horas são? "
                f"{name}, abre o navegador."
                if (st["language"] or "pt").startswith("pt") else
                f"{name}. A conversation with the assistant {name}. {name}, what time is it?")
        self._groq = groq_stt.GroqListener(
            st["api_key"], st["model"], st["language"],
            on_text=self._on_groq_text,
            on_status=lambda m: self.ui.write_log(f"SYS: {m}"),
            sample_rate=SEND_SAMPLE_RATE, prompt=hint)
        self.ui.write_log(f"SYS: Listening with Whisper on Groq ({st['model']}, "
                          f"{st['language'] or 'auto'}) — speech reaches Gemini as text.")
        return "groq"

    @staticmethod
    def _sound_key(word: str) -> str:
        """A consonant skeleton of how the word SOUNDS in Portuguese, so the
        speech-to-text's spelling of a name does not matter: Gervis, Jarves,
        Jarviz, Jarbis and Yarvis all reduce to the same key as Jarvis."""
        import re
        import unicodedata
        w = unicodedata.normalize("NFKD", word.lower())
        w = "".join(ch for ch in w if not ch.isspace() and not unicodedata.combining(ch))
        w = re.sub(r"[^a-z]", "", w)
        w = re.sub(r"g(?=[ei])", "j", w)      # ge/gi sound like je/ji
        w = re.sub(r"c(?=[ei])", "s", w)      # ce/ci sound like se/si
        w = w.replace("ph", "f").replace("ch", "x").replace("sh", "x")
        w = w.replace("y", "j").replace("w", "v").replace("b", "v")
        w = w.replace("z", "s").replace("k", "c").replace("q", "c")
        w = re.sub(r"[aeiouh]", "", w)        # vowels and h carry no identity here
        return re.sub(r"(.)\1+", r"\1", w)    # collapse doubles

    @staticmethod
    def _edit_distance(a: str, b: str) -> int:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    def _addressed_to_me(self, text: str) -> bool:
        """Does the utterance name the assistant? Matched by sound, not by
        spelling: any word whose consonant skeleton is the name's (or one
        edit away from it, same first sound) counts. "Gervis, você sabe…"
        passes; "jardins", "serviço" and "jarro" do not."""
        import re
        import unicodedata
        name = (self._asst_name or "JARVIS")
        folded = unicodedata.normalize("NFKD", text.lower())
        folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
        if unicodedata.normalize("NFKD", name.lower()) and name.lower() in folded:
            return True
        # Known mishearings that no sound rule can reach without also catching
        # ordinary words ("Charles" for Jarvis, but "chaves" must stay out).
        # Extend in config/api_keys.json: "name_aliases": ["charles", "..."].
        aliases = {"charles", "charlie"}
        try:
            extra = json.loads(open(API_CONFIG_PATH, encoding="utf-8").read()).get("name_aliases")
            if isinstance(extra, list):
                aliases |= {str(a).strip().lower() for a in extra if str(a).strip()}
        except Exception:
            pass
        words = set(re.findall(r"[a-z]+", folded))
        if words & aliases:
            return True
        key = self._sound_key(name)
        if len(key) < 2:
            return name.lower() in folded
        for word in re.findall(r"[a-zA-Z\u00C0-\u024F]+", text):
            k = self._sound_key(word)
            if not k:
                continue
            if k == key:
                return True
            if len(key) >= 3 and k[0] == key[0] and self._edit_distance(k, key) <= 1:
                return True
        return False

    def _in_followup_window(self) -> bool:
        """True only when JARVIS's last reply ended in a question and it
        finished saying it less than _FOLLOWUP_AFTER_QUESTION_S ago."""
        last = (getattr(self, "_last_out_logged", "") or "").strip()
        if not last.endswith(("?", "?\"", "?»", "?)")):
            return False
        ended = getattr(self, "_reply_ended_at", 0.0)
        if not ended:
            return False
        return (time.monotonic() - ended) <= self._FOLLOWUP_AFTER_QUESTION_S

    @staticmethod
    def _squash(text: str) -> str:
        """Lowercase, no accents, letters and digits only — so 'JARVI S, abre o
        navegador.' and 'jarvis abre o navegador' compare equal."""
        import re
        import unicodedata
        t = unicodedata.normalize("NFKD", (text or "").lower())
        t = "".join(ch for ch in t if not unicodedata.combining(ch))
        return re.sub(r"[^a-z0-9]+", "", t)

    def _is_blocked_phrase(self, text: str) -> str:
        """The configured phrase this transcript IS (whole utterance, exact
        after squashing case, accents, punctuation and spaces), or ''."""
        hay = self._squash(text)
        if not hay:
            return ""
        for phrase in get_blocked_phrases():
            key = self._squash(phrase)
            if key and key == hay:
                return phrase
        return ""

    def _is_good_morning(self, text: str) -> bool:
        """True when the utterance is the assistant's name plus "bom dia" (or
        "good morning") and nothing else — in either order, any punctuation.
        The name may be misheard the same way _addressed_to_me allows."""
        import re
        import unicodedata
        folded = unicodedata.normalize("NFKD", (text or "").lower())
        folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
        squashed = "".join(re.findall(r"[a-z0-9]+", folded))
        if not squashed:
            return False
        # Whisper sometimes splits the name ("JARVI S"), so compare on the
        # squashed string: the greeting at one end, only the name at the other.
        for greet in ("bomdia", "goodmorning", "buenosdias"):
            if squashed.endswith(greet):
                name_part = squashed[:-len(greet)]
            elif squashed.startswith(greet):
                name_part = squashed[len(greet):]
            else:
                continue
            if name_part and self._addressed_to_me(name_part):
                return True
        return False

    def replay_briefing(self, why: str = "") -> bool:
        """Thread-safe: run the morning routine again — music, greeting,
        channel check and AI news — even if it was already given today.
        When the model itself asked for it (why == "tool") it is already
        answering the greeting, so the routine's own greeting is skipped."""
        loop = getattr(self, "_loop", None)
        if not loop or not self.session:
            return False
        try:
            self.ui.write_log("SYS: Bom dia → replaying the morning briefing"
                              + (f" ({why})" if why else "") + ".")
        except Exception:
            pass
        self._last_user_speech = time.monotonic()
        asyncio.run_coroutine_threadsafe(
            self._send_startup_briefing(force=True, greet=(why != "tool")), loop)
        return True

    def _on_groq_text(self, text: str) -> None:
        """An utterance came back from Groq — runs on its worker thread."""
        loop = getattr(self, "_loop", None)
        if not loop or not self.session:
            return
        blocked = self._is_blocked_phrase(text)
        if blocked:
            try:
                self.ui.write_log(f"SYS: Blocked phrase ignored («{blocked}»): «{text[:60]}»")
            except Exception:
                pass
            return
        # "Jarvis, bom dia" is handled here, before any model: the morning
        # routine itself is the answer, so nothing is sent for a reply.
        if self._is_good_morning(text):
            try:
                self.ui.write_log(f"YOU: {text}")
            except Exception:
                pass
            self.replay_briefing()
            return
        if get_name_gate_enabled() and not self._addressed_to_me(text):
            if self._in_followup_window():
                try:
                    self.ui.write_log("SYS: (taken as a reply to me — no name needed)")
                except Exception:
                    pass
            else:
                try:
                    self.ui.write_log(f"SYS: Ignored (not addressed to {self._asst_name}): «{text[:60]}»")
                except Exception:
                    pass
                return
        self._last_user_speech = time.monotonic()
        _usage.note("gemini", turns=1)
        try:
            self.ui.write_log(f"YOU: {text}")
        except Exception:
            pass

        async def _send():
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": text}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Groq] send failed: {e}")

        try:
            asyncio.run_coroutine_threadsafe(_send(), loop)
        except Exception as e:
            print(f"[Groq] send failed: {e}")

    def _on_ptt(self, held: bool) -> None:
        """Chord pressed or released — may arrive on the hotkey thread."""
        self._ptt_held = held
        if not held and self._groq is not None:
            self._groq.flush()      # the key going up ends the utterance
        if held:
            # Holding the key is also a way to wake it, so push-to-talk works
            # without having to say the wake word first.
            if self._wake_enabled and not self._awake:
                self._awake = True
                self._last_user_speech = time.monotonic()
        try:
            self.ui.set_state("LISTENING" if held else "SLEEPING")
        except Exception:
            pass

    def interrupt(self) -> None:
        """Stop JARVIS mid-speech: drain queued audio and open mic immediately."""
        self._interrupted = True
        q = self.audio_in_queue
        if q:
            drained = 0
            while True:
                try:
                    q.get_nowait()
                    drained += 1
                except Exception:
                    break
            if drained:
                print(f"[JARVIS] ✋ Interrupted — {drained} audio chunks discarded")
        self.set_speaking(False)
        # The words we were about to mouth are never going to be spoken now.
        self._visemes.reset()
        self._play_cursor = 0.0     # next batch starts a fresh timeline
        if self._turn_done_event:
            self._turn_done_event.clear()
        self.ui.write_log("SYS: Interrupted — listening...")

    def speak(self, text: str):
        if not self._loop or not self.session:
            return
        _usage.note("gemini", turns=1)
        asyncio.run_coroutine_threadsafe(
            self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": text}]},
                turn_complete=True
            ),
            self._loop
        )

    def speak_error(self, tool_name: str, error: str):
        short = str(error)[:120]
        self.ui.write_log(f"ERR: {tool_name} — {short}")
        self.speak(f"Sir, {tool_name} encountered an error. {short}")

    def _build_config(self) -> types.LiveConnectConfig:
        from datetime import datetime

        # Load customization from config
        try:
            _cfg = json.loads(open(API_CONFIG_PATH, encoding="utf-8").read())
            self._asst_name = (_cfg.get("assistant_name") or "JARVIS").strip()
            _user_name = (_cfg.get("user_name") or "").strip()
        except Exception:
            self._asst_name = "JARVIS"
            _user_name = ""

        memory     = load_memory()
        mem_str    = format_memory_for_prompt(memory)
        sys_prompt = _load_system_prompt()

        now      = datetime.now()
        time_str = now.strftime("%A, %B %d, %Y — %I:%M %p")
        time_ctx = (
            f"[CURRENT DATE & TIME]\n"
            f"Right now it is: {time_str}\n"
            f"Use this to calculate exact times for reminders.\n\n"
        )

        # Identity injection — overrides any hardcoded name in prompt.txt
        # Address form is a property of the language being spoken, so it is
        # stated as a principle rather than a two-language lookup — the model
        # already knows the respectful register of whatever language it is in.
        _addr = (f"ADDRESS: Always call the user '{_user_name}'."
                 if _user_name
                 else 'ADDRESS: Address the user with the ordinary respectful form '
                      'for a superior in the language you are currently speaking — '
                      '"sir" in English, its everyday equivalent in any other '
                      'language. Never an archaic or aristocratic form, and never '
                      'the form from a different language than the one you are '
                      'speaking in this sentence.')
        identity_ctx = (
            f"[IDENTITY]\n"
            f"Your name is {self._asst_name}. "
            f"Always refer to yourself as {self._asst_name}.\n"
            f"{_addr}\n\n"
        )

        # Everything the model is told about *itself* is derived here, not
        # written into prompt.txt: the name comes from config, the platform from
        # the host, the capability list from the registries that were just
        # discovered. Rename the assistant, add a plugin or move to another OS
        # and this follows without anyone editing a prompt.
        _all_decls = (TOOL_DECLARATIONS
                      + self._action_registry.get_tool_declarations()
                      + self._plugin_registry.get_tool_declarations())
        _names = {(d.get("name") if isinstance(d, dict) else getattr(d, "name", ""))
                  for d in _all_decls}
        sys_prompt = _render_prompt(sys_prompt, {
            "assistant_name": self._asst_name,
            "platform": f"{_platform.system()} {_platform.release()}".strip(),
            "capabilities": _describe_tools(_all_decls),
            "limits": _describe_limits(
                has_vision="screen_process" in _names,
                has_mic=True,
            ),
        })

        parts = [time_ctx, identity_ctx]
        if mem_str:
            parts.append(mem_str)
        parts.append(sys_prompt)
        parts.append(_PACING_CTX)
        if get_acid_humor_enabled():
            parts.append(_ACID_HUMOR_CTX)
        # Kept for the economy brain, which sends the same text as the
        # OpenAI-style system message instead of a Live system_instruction.
        self._system_prompt_text = "\n".join(parts)
        self._all_tool_decls = list(_all_decls)

        cfg = dict(
            response_modalities=["AUDIO"],
            output_audio_transcription={},
            input_audio_transcription={},
            system_instruction="\n".join(parts),
            tools=[{"function_declarations": _all_decls}],
            # Asks the server to issue session-resumption handles. Enabled but
            # NOT yet used: no handle is captured from the resumption update and
            # none is passed back here, so a reconnect starts a fresh session
            # rather than continuing the old one. Left on because the handles
            # cost nothing and are the prerequisite for doing it properly.
            # Hand back the handle captured from the last session_resumption
            # update. `handle=None` is exactly the old behaviour (ask for
            # handles, start fresh), so the first connect of a run is unchanged.
            session_resumption=types.SessionResumptionConfig(
                handle=self._resume_handle
            ),
            # Sliding-window compression: session never dies from a full context
            # window — JARVIS can stay in one conversation for hours
            context_window_compression=types.ContextWindowCompressionConfig(
                sliding_window=types.SlidingWindow(),
            ),
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=get_voice()
                    )
                )
            ),
        )
        if self._enhanced_live:
            # Proactive audio: JARVIS stays silent when speech isn't addressed
            # to it (background chatter, talking to someone else in the room).
            # (Affective dialog was dropped: gemini-3.1-flash-live does not
            #  support it, and it never reliably detected tone in practice.
            #  To restore it on a 2.5 native-audio model, add back:
            #  cfg["enable_affective_dialog"] = True )
            if get_proactive_audio_enabled():
                cfg["proactivity"] = types.ProactivityConfig(proactive_audio=True)

        if self._tuned_live:
            cfg.update(self._tuning_config())

        return types.LiveConnectConfig(**cfg)

    def _tuning_config(self) -> dict:
        """The optional knobs, kept apart so one bad field can be dropped wholesale.

        Every one of these is a preview-API field. If a future model release
        stops accepting any of them the connection fails at setup, so the run
        loop turns `_tuned_live` off and reconnects on the plain config rather
        than leaving the user with an assistant that will not start.
        """
        out: dict = {}

        # How long the server waits through a pause before deciding your turn is
        # over. This — not the size of the prompt — is what most of the delay
        # before a reply actually is, and the default has to suit everybody, so
        # it is necessarily cautious.
        turn = get_turn_tuning()
        if turn.get("enabled", True):
            detect = types.AutomaticActivityDetection(
                silence_duration_ms=turn["silence_ms"],
                prefix_padding_ms=turn["prefix_ms"],
            )
            if turn["end_sensitivity"] == "high":
                detect.end_of_speech_sensitivity = types.EndSensitivity.END_SENSITIVITY_HIGH
            elif turn["end_sensitivity"] == "low":
                detect.end_of_speech_sensitivity = types.EndSensitivity.END_SENSITIVITY_LOW
            if turn["start_sensitivity"] == "high":
                detect.start_of_speech_sensitivity = types.StartSensitivity.START_SENSITIVITY_HIGH
            elif turn["start_sensitivity"] == "low":
                detect.start_of_speech_sensitivity = types.StartSensitivity.START_SENSITIVITY_LOW
            out["realtime_input_config"] = types.RealtimeInputConfig(
                automatic_activity_detection=detect)

        # Screenshots and camera frames are tokenised at this resolution and then
        # stay in the session's context. 'medium' keeps on-screen text legible
        # for a fraction of a full-resolution frame.
        res = get_media_resolution()
        if res != "default":
            out["media_resolution"] = {
                "low":    types.MediaResolution.MEDIA_RESOLUTION_LOW,
                "medium": types.MediaResolution.MEDIA_RESOLUTION_MEDIUM,
                "high":   types.MediaResolution.MEDIA_RESOLUTION_HIGH,
            }[res]

        # Thinking is left at the server default deliberately. Forcing the budget
        # to zero was measured on gemini-3.1-flash-live over interleaved trials
        # and did not make the first word arrive sooner — this model does not
        # appear to deliberate on the Live path, so pinning the field only adds a
        # way for a future release to behave differently. Set "thinking_enabled"
        # in config/api_keys.json to true to let it reason instead.
        if get_thinking_enabled():
            out["thinking_config"] = types.ThinkingConfig(thinking_budget=-1)

        return out

    async def _execute_tool(self, fc) -> types.FunctionResponse:
        name = fc.name
        args = dict(fc.args or {})

        print(f"[JARVIS] 🔧 {name}  {args}")
        self.ui.set_state("THINKING")


        if name == "save_memory":
            category = args.get("category", "notes")
            key      = args.get("key", "")
            value    = args.get("value", "")
            if key and value:
                update_memory({category: {key: {"value": value}}})
                print(f"[Memory] 💾 save_memory: {category}/{key} = {value}")
            if not self.ui.muted:
                self.ui.set_state("LISTENING")
            return types.FunctionResponse(
                id=fc.id, name=name,
                response={"result": "ok", "silent": True}
            )

        loop   = asyncio.get_event_loop()
        result = "Done."

        try:
            if name == "recall_memory":
                # Local file search: no network, no second model. Kept out of
                # the executor deliberately — it is a dictionary scan over a few
                # hundred short strings, and a thread hop would cost more than
                # the work itself.
                result = search_memory(args.get("query", ""), limit=8)

            elif name == "undo":
                if str(args.get("action", "")).lower().strip() == "list":
                    items = undo_stack.history()
                    result = ("Things I can undo, most recent first:\n"
                              + "\n".join(f"{i+1}. {t}" for i, t in enumerate(items))
                              ) if items else "I have not changed anything I can undo yet."
                else:
                    result = await loop.run_in_executor(None, undo_stack.undo_last)

            elif name == "screen_process":
                import time as _t_mod
                _now = _t_mod.monotonic()
                _cooldown = 4.0  # seconds — covers echo window after speaking ends
                if self._vision_busy or (_now - self._vision_last_time) < _cooldown:
                    _wait = max(0, _cooldown - (_now - self._vision_last_time))
                    print(f"[Vision] ⏳ Cooldown active ({_wait:.1f}s remaining) — ignoring duplicate call")
                    result = "Vision is still processing the previous request. I will not call this again."
                else:
                    self._vision_busy      = True
                    self._vision_last_time = _now
                    angle     = args.get("angle", "screen").lower()
                    user_text = args.get("text", "What do you see?")
                    if angle == "camera":
                        img_b, mime_t = await loop.run_in_executor(None, _capture_camera)
                        self.ui.start_camera_stream()
                        self._vision_cam_active = True
                        print(f"[Vision] 📷 Camera: {len(img_b):,} bytes")
                        _stall = "camera"
                    else:
                        img_b, mime_t = await loop.run_in_executor(None, _capture_screen)
                        print(f"[Vision] 🖥️  Screen: {len(img_b):,} bytes")
                        _stall = "screen"
                    self._pending_vision = (img_b, mime_t, user_text, angle)
                    # The image is attached to this same exchange, so there is
                    # nothing to stall for and nothing to announce. Asking for an
                    # acknowledgement here is what produced two spoken answers —
                    # the model filled that turn by answering the question from
                    # imagination, then answered it again once it could see.
                    result = (
                        f"[VISION_ACTIVE] {_stall.capitalize()} captured and attached to this "
                        f"same exchange. Do not acknowledge and do not answer yet — the image "
                        f"is arriving with this result. Reply once, from what you actually see "
                        f"in it."
                    )

            elif name == "close_camera":
                self.ui.stop_camera_stream()
                result = "Camera closed."

            elif name == "system_status":
                r = await loop.run_in_executor(None, get_system_status)
                result = str(r)

            elif name == "manage_monitor":
                action = args.get("action", "").lower().strip()
                topic  = args.get("topic", "").strip()
                if action == "add" and topic:
                    result = await asyncio.to_thread(add_monitor, topic)
                elif action == "remove" and topic:
                    result = await asyncio.to_thread(remove_monitor, topic)
                elif action == "list":
                    topics = await asyncio.to_thread(list_monitors)
                    result = ("Monitoring: " + ", ".join(topics)) if topics else "No topics are being monitored."
                else:
                    result = "Specify action (add/remove/list) and a topic."

            elif name == "shutdown_jarvis":
                self.ui.write_log("SYS: Shutdown requested.")
                async def _do_shutdown():
                    await self._save_session_summary()
                    if self.session:
                        try:
                            await self.session.send_client_content(
                                turns={"role": "user", "parts": [{"text": "Say a brief natural goodbye to the user."}]},
                                turn_complete=True,
                            )
                        except Exception:
                            pass
                    await asyncio.sleep(1.5)
                    import os as _os
                    _os._exit(0)
                asyncio.create_task(_do_shutdown())

            elif self._action_registry.has(name):
                # file_processor: fall back to the currently-uploaded file when none is given
                if name == "file_processor" and not args.get("file_path") and self.ui.current_file:
                    args["file_path"] = self.ui.current_file
                _ctx = {"player": self.ui, "speak": self.speak,
                        "response": None, "session_memory": None}
                r = await loop.run_in_executor(None, lambda: self._action_registry.run(name, args, _ctx))
                result = r or "Done."
                # web_search: mirror results to the on-screen content panel
                if (name == "web_search" and r
                        and not r.startswith("No results")
                        and not r.startswith("Search failed")):
                    _mode  = args.get("mode", "search")
                    _query = args.get("query") or ", ".join(args.get("items", []))
                    _label = f"{_mode.upper()} — {_query[:38]}" if _query else _mode.upper()
                    self.ui.show_content(_label, r)

            else:
                if self._plugin_registry.has(name):
                    r = await loop.run_in_executor(
                        None,
                        lambda: self._plugin_registry.run(name, args, player=self.ui, session_memory=None)
                    )
                    result = r or "Done."
                else:
                    result = f"Unknown tool: {name}"

        except Exception as e:
            result = f"Tool '{name}' failed: {e}"
            traceback.print_exc()
            self.speak_error(name, e)

        if not self.ui.muted:
            self.ui.set_state("LISTENING")

        print(f"[JARVIS] 📤 {name} → {str(result)[:80]}")

        # A tool that declared itself NON_BLOCKING also says when its answer may
        # re-enter the conversation. Without this the model finishes whatever it
        # was saying and then reads the result out on top of it — which, for
        # something like a phone call already ringing, is exactly the noise the
        # non-blocking call was meant to avoid. Tools that declared nothing get
        # the API default and behave as they always have.
        _sched = (self._action_registry.scheduling(name)
                  or self._plugin_registry.scheduling(name))
        _extra = {"scheduling": _sched} if _sched else {}
        return types.FunctionResponse(
            id=fc.id, name=name,
            response={"result": result},
            **_extra
        )

    async def _send_realtime(self):
        while True:
            msg = await self.out_queue.get()
            # Gemini 3.x Live rejects the old realtime_input.media_chunks field
            # (what `media=...` maps to) and closes the socket with a 1007. Send
            # mic / phone PCM through the new `audio` field instead. Queue items
            # are {"data": <bytes>, "mime_type": <str>} from _listen_audio and
            # the phone relay.
            await self.session.send_realtime_input(
                audio=types.Blob(
                    data=msg["data"],
                    mime_type=msg.get("mime_type", "audio/pcm"),
                )
            )
            # 16 kHz int16 mono → 32 000 bytes per second of microphone.
            _usage.note("gemini", audio_in_s=len(msg["data"]) / 32000.0)

    async def _listen_audio(self):
        print("[JARVIS] 🎤 Mic started")
        loop = asyncio.get_event_loop()

        def callback(indata, frames, time_info, status):
            # ── Wake-word gate ───────────────────────────────────────────────
            # While asleep, the mic audio NEVER goes to Gemini (nothing is
            # streamed, so JARVIS can't respond to speech not addressed to it and
            # nothing leaves the machine). Frames are instead handed to the local
            # detector, which runs its model in ITS OWN thread — the cost here is
            # only a queue push, so the audio path is never slowed. When wake word
            # is off (default) or we're awake, this is a single boolean check.
            if self._wake_enabled and not self._awake:
                det = self._wake_detector
                if det is not None:
                    det.feed(indata)
                return
            with self._speaking_lock:
                jarvis_speaking = self._is_speaking

            # ── Barge-in ─────────────────────────────────────────────────────
            # While JARVIS talks the mic is not streamed, but it is still worth
            # listening to locally: if the user starts speaking, cut the answer
            # short the way a person would stop when interrupted.
            #
            # The whole difficulty is echo — on speakers the mic hears JARVIS.
            # So the test is not "is the mic loud" but "is the mic louder than
            # the echo of what we are playing right now", sustained long enough
            # that a cough or a keystroke cannot trigger it.
            if jarvis_speaking:
                # Nothing is streamed while JARVIS talks.
                #
                # Interrupting by voice used to live here: `EchoGuard` can pick a
                # user out from under our own echo, and `core/echo.py` still does
                # that for the tail below. Re-enabling is small — classify each
                # block here and call interrupt() after `required_blocks` of
                # agreement — but it depends on the listener's room, so it stays
                # out until it can be tried on real hardware.
                return

            # ── Echo tail ────────────────────────────────────────────────────
            # The speaking flag has dropped but the speakers have not finished.
            # Sending this to the model is how an assistant hears itself, decides
            # it was addressed, and answers its own last sentence. The microphone
            # stays OPEN — the guard only drops blocks that are our own voice, so
            # replying the instant it stops still works.
            if self._tail_active():
                try:
                    if not self._echo.is_user_speech(
                            indata, SEND_SAMPLE_RATE, _pcm_level(indata)):
                        return
                    self._tail_until = 0.0      # a real voice ends the tail early
                except Exception:
                    return
            elif self._echo._hist:
                self._echo.reset()

            # ── Push-to-talk ─────────────────────────────────────────────────
            # When it is on the microphone is closed by default and the chord
            # opens it, which is the whole point: nothing leaves the machine
            # unless you are holding the key.
            if self._ptt_enabled and not self._ptt_held:
                return
            
            if not self.ui.muted and not self._phone_active:
                # ── Groq listening ───────────────────────────────────────
                # The block goes to the local segmenter instead of the
                # session. Same gates as above apply, so nothing is heard
                # while asleep, muted, or (with push-to-talk) unheld.
                if self._groq is not None:
                    try:
                        self._groq.feed(indata)
                        self.ui.set_audio_level(_pcm_level(indata))
                    except Exception:
                        pass
                    return
                data = indata.tobytes()
                loop.call_soon_threadsafe(
                    self.out_queue.put_nowait,
                    {"data": data, "mime_type": "audio/pcm"}
                )
                # Feed the live mic level to the HUD so the waveform reacts to
                # the user's actual voice while listening. Purely cosmetic — any
                # failure here must never disturb the mic.
                try:
                    self.ui.set_audio_level(_pcm_level(indata))
                except Exception:
                    pass

        try:
            def _open_mic(dev):
                return sd.InputStream(
                    samplerate=SEND_SAMPLE_RATE,
                    channels=CHANNELS,
                    dtype="int16",
                    blocksize=CHUNK_SIZE,
                    device=dev,
                    callback=callback,
                )

            # Which microphone. resolve() returns None for "system default" and
            # for a saved device that is no longer present — so a headset
            # unplugged since the last run falls back to the built-in mic
            # instead of raising on startup and taking the session with it.
            _mic_name = get_input_device()
            _mic_dev  = audio_devices.resolve(_mic_name, "input")
            if _mic_dev is not None:
                print(f"[JARVIS] 🎤 Input device: {_mic_name}")
            try:
                _mic_stream = _open_mic(_mic_dev)
            except Exception as _e:
                # A device the picker listed but the driver will not open right
                # now — exclusive mode, a webcam already in use, a virtual mic
                # whose source went away. Chosen hardware failing must never
                # mean the assistant cannot hear at all.
                if _mic_dev is None:
                    raise
                print(f"[JARVIS] ⚠️  Mic '{_mic_name}' failed: {_e} — using default")
                self.ui.write_log(
                    f"SYS: Microphone '{_mic_name}' unavailable — using system default."
                )
                _mic_stream = _open_mic(None)

            with _mic_stream:
                print("[JARVIS] 🎤 Mic stream open")
                while True:
                    await asyncio.sleep(0.1)
        except Exception as e:
            print(f"[JARVIS] ❌ Mic: {e}")
            raise

    async def _flush_pending_vision(self) -> bool:
        """Send a captured frame immediately after its tool response.

        The frame is already in hand by the time `screen_process` returns — the
        capture happened inside the tool call. The old flow still made the model
        speak a turn first and only injected the image on that turn's
        turn_complete, which cost a whole extra round trip AND produced two
        spoken answers: one improvised without the picture, then the real one.
        Sending it here means the model has the tool result and the image before
        it generates anything, so the user gets one answer, sooner.
        """
        if not (self._pending_vision and self.session):
            return False

        import base64 as _b64
        img_b, mime_t, question, angle = self._pending_vision
        self._pending_vision = None
        b64 = _b64.b64encode(img_b).decode("ascii")
        print(f"[Vision] 📤 {len(img_b):,} bytes (angle={angle}) → main session")

        # Label the source. Without it the image arrives carrying nothing but
        # the user's own sentence, and a screenshot of this app — which has a
        # face in the middle of it — got read as a photo of the user. What the
        # label *means* is explained once, in the generated [SELF] block.
        src = ("[IMAGE SOURCE: WEBCAM]" if angle == "camera"
               else "[IMAGE SOURCE: SCREEN CAPTURE]")
        await self.session.send_client_content(
            turns={"role": "user", "parts": [
                {"inline_data": {"mime_type": mime_t, "data": b64}},
                {"text": f"{src}\n\n{question}"},
            ]},
            turn_complete=True,
        )

        if self._vision_cam_active:
            # Camera: stay busy until JARVIS has finished speaking the answer,
            # then close the preview.
            self._vision_cam_active    = False
            self._vision_close_pending = True
        else:
            self._vision_busy = False
        return True

    async def _receive_audio(self):
        print("[JARVIS] 👂 Recv started")
        out_buf, in_buf = [], []

        try:
            while True:
                async for response in self.session.receive():

                    # ── Session resumption ───────────────────────────────────
                    # The server sends this periodically. `resumable` goes false
                    # while a turn is mid-flight — replaying a handle from that
                    # moment is what the flag exists to prevent — so only
                    # resumable handles are kept. This is three lines and it is
                    # the entire fix for "every reconnect forgets everything".
                    _sru = getattr(response, "session_resumption_update", None)
                    if _sru is not None:
                        if getattr(_sru, "resumable", False) and getattr(_sru, "new_handle", None):
                            if self._resume_handle is None:
                                print("[JARVIS] 🔗 Session resumption armed")
                            self._resume_handle = _sru.new_handle

                    if response.data:
                        if self._interrupted:
                            pass  # discard: interrupted
                        else:
                            if self._turn_done_event and self._turn_done_event.is_set():
                                self._turn_done_event.clear()
                            # Split into ~50 ms chunks so interrupt() stops audio within 50 ms
                            # (24000 Hz × 2 bytes/sample × 0.05 s = 2400 bytes per slice)
                            _audio_data = response.data
                            # 24 kHz int16 mono → 48 000 bytes per second of voice.
                            _usage.note("gemini", audio_out_s=len(_audio_data) / 48000.0)
                            _SLICE = 2400
                            for _i in range(0, len(_audio_data), _SLICE):
                                self.audio_in_queue.put_nowait(_audio_data[_i : _i + _SLICE])

                    if response.server_content:
                        sc = response.server_content

                        if sc.output_transcription and sc.output_transcription.text:
                            txt = _clean_transcript(sc.output_transcription.text)
                            # A turn that involves a tool call passes through
                            # several turn_completes, and the API re-sends the
                            # tail of the transcript across them. Comparing only
                            # against the previous chunk missed that — once
                            # out_buf had been flushed and emptied, the repeat
                            # sailed straight back in, which logged the answer
                            # twice AND made the avatar mouth it twice.
                            if txt and not _is_repeat_chunk(txt, out_buf):
                                out_buf.append(txt)
                                # Hand the words to the mouth as they arrive, so
                                # the avatar can form the consonants the audio
                                # alone cannot show. Pure string work — it adds
                                # nothing measurable to the response path.
                                self._visemes.feed_text(txt)
                                # …and to the hologram cards, which advance to
                                # the subject he has just started talking about.
                                self.ui.holo_transcript(txt)

                        if sc.input_transcription and sc.input_transcription.text:
                            txt = _clean_transcript(sc.input_transcription.text)
                            if txt:
                                in_buf.append(txt)
                                self._last_user_speech = time.monotonic()

                        if sc.turn_complete:
                            if self._turn_done_event:
                                self._turn_done_event.set()

                            # If this turn_complete ends an interrupted response, clear the
                            # flag and skip all further processing for that turn.
                            if self._interrupted:
                                self._interrupted = False
                                in_buf  = []
                                out_buf = []
                                self._visemes.reset()
                                continue

                            full_in = " ".join(in_buf).strip()
                            if full_in:
                                self._last_out_logged = ""   # new exchange
                                self.ui.write_log(f"You: {full_in}")
                                self._session_log.append(f"User: {full_in}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "user",
                                        "text": full_in,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            in_buf = []

                            full_out = " ".join(out_buf).strip()
                            # Second line of defence: even if a repeat slips
                            # into a *fresh* buffer after a flush, never log the
                            # same answer (or a tail of it) twice in a row.
                            if full_out and len(full_out) >= _REPEAT_MIN and self._last_out_logged:
                                if full_out in self._last_out_logged:
                                    full_out = ""
                            if full_out:
                                self._last_out_logged = full_out
                                self.ui.write_log(f"{self._asst_name}: {full_out}")
                                self._session_log.append(f"{self._asst_name}: {full_out}")
                                if self._dashboard:
                                    asyncio.create_task(self._dashboard.broadcast({
                                        "type": "log", "speaker": "jarvis",
                                        "text": full_out,
                                        "ts": datetime.now().isoformat(),
                                    }))
                            out_buf = []

                            if self._vision_close_pending:
                                # This turn_complete IS the vision answer — close camera + release busy flag
                                self._vision_close_pending = False
                                self._vision_busy = False
                                async def _cam_close():
                                    await asyncio.sleep(2.0)
                                    self.ui.stop_camera_stream()
                                asyncio.create_task(_cam_close())

                    if response.tool_call:
                        fn_responses = []
                        for fc in response.tool_call.function_calls:
                            print(f"[JARVIS] 📞 {fc.name}")
                            fr = await self._execute_tool(fc)
                            fn_responses.append(fr)
                        await self.session.send_tool_response(
                            function_responses=fn_responses
                        )
                        await self._flush_pending_vision()
        except Exception as e:
            print(f"[JARVIS] ❌ Recv: {e}")
            traceback.print_exc()
            raise

    async def _play_audio(self):
        print("[JARVIS] 🔊 Play started")

        _spk_name = get_output_device()
        _spk_dev  = audio_devices.resolve(_spk_name, "output")
        if _spk_dev is not None:
            print(f"[JARVIS] 🔊 Output device: {_spk_name}")

        def _open_spk(dev):
            st = sd.RawOutputStream(
                samplerate=RECEIVE_SAMPLE_RATE,
                channels=CHANNELS,
                dtype="int16",
                blocksize=CHUNK_SIZE,
                device=dev,
            )
            st.start()
            return st

        try:
            stream = _open_spk(_spk_dev)
        except Exception as _e:
            # A chosen output that the host API accepts by name but refuses to
            # open (exclusive mode, wrong sample rate, device asleep) must not
            # cost the user their voice. Fall back to the default and say so.
            if _spk_dev is None:
                raise
            print(f"[JARVIS] ⚠️  Output device '{_spk_name}' failed: {_e} — using default")
            self.ui.write_log(f"SYS: Speaker '{_spk_name}' unavailable — using system default.")
            stream = _open_spk(None)

        # Ask the device how far behind the speakers actually are, rather than
        # assuming. This is what the echo tail is sized from, so a machine with a
        # large audio buffer gets a correspondingly longer guard — and one with a
        # tiny buffer is not penalised with a delay it does not need.
        try:
            lat = float(getattr(stream, "latency", 0.0) or 0.0)
            if 0.0 < lat < 1.0:
                self._out_latency = lat
            print(f"[JARVIS] 🔊 Output latency {self._out_latency*1000:.0f} ms "
                  f"→ echo tail {(self._out_latency + _TAIL_MARGIN)*1000:.0f} ms")
        except Exception:
            pass

        try:
            while True:
                try:
                    chunk = await asyncio.wait_for(
                        self.audio_in_queue.get(),
                        timeout=0.1
                    )
                except asyncio.TimeoutError:
                    if (
                        self._turn_done_event
                        and self._turn_done_event.is_set()
                        and self.audio_in_queue.empty()
                    ):
                        self.set_speaking(False)
                        self._turn_done_event.clear()
                    continue

                self.set_speaking(True)

                # Batch all immediately-available chunks into one write to reduce
                # thread-pool round-trips (was one asyncio.to_thread per 50ms slice).
                # Cap at ~200 ms so interrupt() still stops audio within ~200 ms.
                batch = bytearray(chunk)
                while len(batch) < 9600:   # 9600 bytes ≈ 200 ms at 24 kHz / 16-bit mono
                    try:
                        batch.extend(self.audio_in_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                # Drive the HUD waveform and the avatar's mouth from JARVIS's
                # own voice. The batch is up to 200 ms long, so we hand over a
                # *schedule* of 20 ms viseme frames instead of a single averaged
                # level and let the HUD play it out in step with the audio.
                try:
                    pcm = np.frombuffer(bytes(batch), dtype=np.int16)
                    hop = _VIS_HOP / RECEIVE_SAMPLE_RATE
                    frames = _pcm_visemes(pcm, sr=RECEIVE_SAMPLE_RATE)
                    # When does this batch become audible? The stream was
                    # started at launch and its callback has been pulling
                    # silence ever since, so the first bytes of a reply reach
                    # the speaker about one callback period later — NOT one
                    # buffer later. `stream.latency` reports the buffer's
                    # capacity, which is how much can be queued ahead, and on
                    # Windows that is commonly 300-500 ms. Anchoring on it put
                    # the entire schedule a buffer late; that is the half second
                    # of lag, and it grew with whatever the device reported.
                    #
                    # After the anchor nothing needs measuring: the device
                    # consumes at exactly realtime, so each batch sounds one
                    # batch-duration after the one before it. The cursor is
                    # re-anchored only when it leaves the range physically
                    # possible — behind `now` means the device drained and this
                    # batch starts a fresh stretch of speech, while further
                    # ahead than the buffer can hold means it has drifted.
                    now = time.time()
                    horizon = self._out_latency + _CURSOR_SLACK
                    if not (now <= self._play_cursor <= now + horizon):
                        self._play_cursor = now + _FIRST_SOUND
                    at = self._play_cursor
                    # Advance by the batch's own duration whether or not it
                    # yielded frames, so a block too short to analyse cannot
                    # shift everything after it out of step with the audio.
                    self._play_cursor += pcm.size / RECEIVE_SAMPLE_RATE
                    if frames:
                        frames = self._visemes.frames(frames, hop)
                        self.ui.push_visemes(frames, hop, at)
                        # Barge-in needs to know what we are playing, not just
                        # how loud: the guard subtracts this from the microphone.
                        self._out_level = max(f[0] for f in frames)
                        self._echo.note_output(pcm, RECEIVE_SAMPLE_RATE,
                                               self._out_level)
                    else:
                        lvl = _pcm_level(pcm)
                        self.ui.set_audio_level(lvl)
                        self._out_level = lvl
                        self._echo.note_output(pcm, RECEIVE_SAMPLE_RATE, lvl)
                except Exception:
                    pass

                try:
                    await asyncio.to_thread(stream.write, bytes(batch))
                except (RuntimeError, asyncio.CancelledError):
                    break   # executor shutting down — exit cleanly
        except Exception as e:
            print(f"[JARVIS] ❌ Play: {e}")
            raise
        finally:
            self.set_speaking(False)
            stream.stop()
            stream.close()

    # ── Morning briefing ────────────────────────────────────────────────────────

    def _briefing_allowed(self) -> bool:
        """The start-up briefing fires once per launch, only with MORNING BRIEF
        on and the assistant awake. When it is skipped, the log says why — a
        silent skip looks exactly like a bug."""
        if self._briefing_sent:
            return False
        if not get_brief_enabled():
            self.ui.write_log("SYS: Morning briefing skipped — MORNING BRIEF is OFF (🎛 to turn it on).")
            self._briefing_sent = True          # say it once per launch
            return False
        if not self._awake:
            self.ui.write_log("SYS: Morning briefing skipped — waiting for the wake word.")
            return False
        return True

    async def _send_startup_briefing(self, force: bool = False, greet: bool = True) -> None:
        """Wrapper: a briefing that fails must say so in the log, never vanish."""
        try:
            await self._send_startup_briefing_inner(force=force, greet=greet)
        except Exception as e:
            try:
                self.ui.write_log(f"ERR: Briefing failed — {str(e)[:140]}")
            except Exception:
                pass
            print(f"[Briefing] {e!r}")

    async def _send_startup_briefing_inner(self, force: bool = False, greet: bool = True) -> None:
        """
        Two-phase briefing optimized for speed:
          Phase 1 — instant greeting (no tools) → speech starts in <1s
          Phase 2 — news pre-fetched in a background thread while Phase 1 plays,
                    delivered as ready text (no Gemini tool-call round-trip) and
                    shown on the UI content panel. Waits for turn_complete event
                    instead of a fixed sleep so there is no unnecessary gap.
        """
        # A intro de abertura (core/intro.py) toca antes da saudação: espera
        # ela terminar, com teto para nunca segurar o briefing para sempre.
        _deadline = time.monotonic() + 30.0
        while self.ui.intro_active() and time.monotonic() < _deadline:
            await asyncio.sleep(0.2)

        memory   = load_memory()
        identity = memory.get("identity", {})

        def _val(k: str) -> str:
            e = identity.get(k, {})
            return (e.get("value", "") if isinstance(e, dict) else str(e)).strip()

        # Sem idioma na memória (primeira abertura), o padrão deste fork é o
        # português do Brasil; a partir da primeira fala do usuário, vale a
        # língua dele (ver [LANGUAGE] em core/prompt.txt).
        lang = _val("language") or "Brazilian Portuguese (pt-BR)"
        name = _val("name")
        time_str = datetime.now().strftime("%H:%M")
        lang_clause0 = (f" Speak in {lang}, then follow the user's own language "
                        f"from their first reply onward." if lang else "")
        name_clause0 = f" Address the user as {name}." if name else ""

        # ── Once a day ───────────────────────────────────────────────────
        # The channel check and the AI news are a morning thing: the first
        # launch of the day gets the full briefing, every later launch the
        # same day gets a one-line greeting. The date is kept on disk so a
        # restart does not reset it.
        # `force` (the user said "Jarvis, bom dia") replays the whole routine
        # — music, greeting, channel check, AI news — however many times.
        today = datetime.now().strftime("%Y-%m-%d")
        if _briefing_done_today(today) and not force:
            self.ui.write_log("SYS: Briefing already given today — short greeting only.")
            p0 = (f"Greet the user in one short sentence, mention it is {time_str}, "
                  f"and say you are at their disposal. No news, no agenda recap — the "
                  f"briefing was already given today. Do not call any tools."
                  f"{lang_clause0}{name_clause0}")
            if self.session:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": p0}]},
                    turn_complete=True,
                )
            return
        self.ui.write_log("SYS: Morning briefing starting…")

        # ── Morning music ────────────────────────────────────────────────
        # The first launch of the day opens with the user's track, and the
        # greeting is held until the music has reached MORNING_SYNC_MS so the
        # voice lands on the opening instead of over a silent intro. The stream
        # is resolved off-thread; if anything fails the briefing simply goes
        # on without music.
        # Bounded: a stalled download or a YouTube hiccup must never hold the
        # greeting hostage (it did once — silently).
        try:
            await asyncio.wait_for(self._start_morning_music(), timeout=30.0)
        except asyncio.TimeoutError:
            self.ui.write_log("SYS: Morning music is taking too long — going on without it.")
        except Exception as e:
            self.ui.write_log(f"SYS: Morning music skipped — {str(e)[:80]}")

        # Start the lookups immediately — they run while phase 1 plays. Two
        # sources, both chosen by the user (core/briefing_sources.py): the
        # YouTube channel's videos published today, and news on the AI topics
        # they follow. Not the generic world headlines.
        loop = asyncio.get_event_loop()
        news_future = loop.run_in_executor(None, _collect_briefing)

        await asyncio.sleep(0.3)
        if not self.session:
            return

        # ── Phase 1: instant greeting ─────────────────────────────────────────
        # The briefing fires before the user has said anything, so the
        # remembered language is the only signal there is. It is a starting
        # point, not a setting: the moment they reply, their language wins.
        lang_clause = (f" Speak this greeting in {lang}, then follow the "
                       f"user's own language from their first reply onward."
                       if lang else "")
        name_clause = f" Address the user as {name}." if name else ""

        # Inject last session context if available — pop removes it so it's never repeated
        last = await asyncio.to_thread(pop_last_session)
        session_clause = ""
        if last:
            try:
                _delta = (datetime.now() - datetime.strptime(last["date"], "%Y-%m-%d")).days
                _when  = "earlier today" if _delta == 0 else ("yesterday" if _delta == 1 else f"{_delta} days ago")
            except Exception:
                _when = "last time"
            session_clause = (
                f" Also briefly and naturally mention that {_when}: {last['summary']}"
            )

        p1 = (
            f"Greet the user warmly, mention it is {time_str}, and say you are checking "
            f"whether their YouTube channel posted anything today and the latest AI news.{session_clause} "
            f"Keep it to 2 short sentences max. Do not call any tools.{lang_clause}{name_clause}"
        )

        # Clear the turn-done event so we can wait for Phase 1 to finish
        if self._turn_done_event:
            self._turn_done_event.clear()

        try:
            self.ui.holo_cards([{
                "title": "BRIEFING",
                "lines": [datetime.now().strftime("%H:%M  ·  %A, %d %B").lower(),
                          "agenda · rastracking · canal · notícias",
                          "coletando…"],
                "hold": 40}])
        except Exception:
            pass
        if greet:
            await self.session.send_client_content(
                turns={"role": "user", "parts": [{"text": p1}]},
                turn_complete=True,
            )
            print("[JARVIS] Briefing phase 1 (greeting) sent.")
            _mark_briefing_done(today)             # only now does the day count as briefed
        else:
            print("[JARVIS] Briefing phase 1 skipped (the model already greeted).")
            _mark_briefing_done(today)

        # ── Phase 2: fire as soon as Phase 1 audio is done ───────────────────
        async def _deliver_news():
            try:
                lang_str = (f" Speak in {lang} unless the user has since "
                            f"spoken another language, in which case use theirs."
                            if lang else "")

                # Wait for news fetch (already running) and Phase 1 turn-complete
                # in parallel — whichever takes longer determines the wait time
                news_done   = asyncio.wrap_future(news_future)
                turn_waited = False
                if self._turn_done_event:
                    try:
                        await asyncio.wait_for(self._turn_done_event.wait(), timeout=6.0)
                        turn_waited = True
                    except asyncio.TimeoutError:
                        pass

                # Extra buffer: turn_complete fires when Gemini finishes *generating*
                # Phase 1, but audio may still be playing.  Waiting a beat here
                # prevents Phase 2 audio from arriving while Phase 1 is mid-sentence
                # (which sounds like a "repeated first response" to the user).
                if turn_waited:
                    await asyncio.sleep(0.8)
                else:
                    await asyncio.sleep(1.0)

                try:
                    found = await asyncio.wait_for(news_done, timeout=30.0)
                except Exception as e:
                    self.ui.write_log(f"SYS: Briefing lookups timed out/failed: {e!r}")
                    found = {"youtube": {"channel": "", "videos": [], "error": str(e)}, "news": ""}

                if not self.session:
                    return

                yt        = found.get("youtube") or {}
                yt_text   = _fmt_youtube(yt)
                agenda    = (found.get("agenda") or "").strip()
                tracking  = (found.get("tracking") or "").strip()
                news_text = found.get("news") or ""
                news_ok   = bool(news_text) and not news_text.startswith(
                    ("No news found", "Search failed", "Please provide"))
                n_vids    = len(yt.get("videos") or [])

                panel = ((("AGENDA\n" + agenda + "\n\n") if agenda else "")
                         + (("RASTRACKING\n" + tracking + "\n\n") if tracking else "")
                         + "YOUTUBE\n" + yt_text + "\n\n"
                         + (news_text if news_ok else "AI news: nothing could be fetched right now."))
                self.ui.show_content("BRIEFING — channel + AI news", panel)

                if not news_ok:
                    self.ui.write_log(
                        f"SYS: AI news unavailable — backend returned: {news_text[:120]!r}")

                # ── One subject per turn ─────────────────────────────────────
                # Everything in a single prompt came out as one breathless
                # paragraph. Each subject is now its own short turn: the card
                # for it goes up, he says it in one or two calm sentences, the
                # audio is allowed to finish, a breath, then the next one.
                yt_rule = (
                    f"The channel posted {n_vids} new video(s) today: name each title. "
                    if n_vids else
                    ("Say in a few words that the channel has not posted anything today. "
                     if not yt.get("error") else
                     "Say in a few words that the channel could not be checked right now. ")
                )
                news_rule = (
                    "From the AI news below, pick the ONE item that matters most — a new "
                    "model launched by OpenAI (GPT) or Anthropic (Claude), or a change in LLM "
                    "prices — summarise it in one sentence, and say the full list is on screen. "
                    if news_ok else
                    "Say briefly that the AI news could not be fetched. "
                )
                cards = _briefing_cards(agenda, found.get("tracking_data") or {},
                                        yt, news_text if news_ok else "")
                by_title = {c["title"]: c for c in cards}
                steps: list[tuple[str, str, list]] = []   # (text block, rule, cards)
                if agenda:
                    steps.append((f"Agenda:\n{agenda}",
                                  "The agenda: name today's events with their times and the "
                                  "tasks due today; overdue tasks in one clause if any. ",
                                  [by_title.get("AGENDA DE HOJE")]))
                if tracking:
                    t_lines = tracking.splitlines()
                    steps.append((f"Rastracking (today):\n{t_lines[0]}",
                                  "Say how much was tracked today and the number of sales. "
                                  "The amount is already spelled out in words: read it exactly "
                                  "as written, slowly, whole reais, never cents. One sentence. ",
                                  [by_title.get("TRACKEADO HOJE")]))
                    if len(t_lines) > 1:
                        steps.append((f"Support:\n{t_lines[1]}",
                                      "Say how many clients are waiting for a support reply; if "
                                      "nobody is waiting, say so in three words. Do NOT read any "
                                      "subject. One sentence. ",
                                      [by_title.get("SUPORTE")]))
                steps.append((f"YouTube channel check:\n{yt_text}", yt_rule,
                              [c for c in cards if c["title"].startswith("CANAL")]))
                steps.append((f"AI news:\n{news_text if news_ok else '(none)'}", news_rule,
                              [c for c in cards if c["title"].startswith("IA")]))

                n_steps = len(steps)
                for i, (block, rule, step_cards) in enumerate(steps, 1):
                    if not self.session:
                        return
                    step_cards = [c for c in step_cards if c]
                    if step_cards:
                        try:
                            self.ui.holo_cards(step_cards)
                        except Exception:
                            pass
                    p2 = (
                        f"[BRIEFING {i}/{n_steps}]\n{block}\n\n{rule}"
                        "Only this subject, nothing else: do not announce what comes next, "
                        "do not greet, do not say 'sir' as an opener, do not call any tools. "
                        "Deliver it unhurried, one fact per sentence, a short pause after "
                        "each number."
                        + (" This is the last item: you may close with a brief word." if i == n_steps else "")
                        + f"{lang_str}"
                    )
                    if self._turn_done_event:
                        self._turn_done_event.clear()
                    await self.session.send_client_content(
                        turns={"role": "user", "parts": [{"text": p2}]},
                        turn_complete=True,
                    )
                    print(f"[JARVIS] Briefing item {i}/{n_steps} sent.")
                    # Wait for the words to be generated, then for the voice to
                    # actually finish, then one breath before the next subject.
                    if self._turn_done_event:
                        try:
                            await asyncio.wait_for(self._turn_done_event.wait(), timeout=40.0)
                        except asyncio.TimeoutError:
                            pass
                    await asyncio.sleep(0.4)
                    for _ in range(600):                       # ≤ 60 s of audio
                        if not self._is_speaking or self._interrupted:
                            break
                        await asyncio.sleep(0.1)
                    if self._interrupted:
                        return
                    await asyncio.sleep(1.1)
                print("[JARVIS] Briefing phase 2 (news) done.")
            except Exception as e:
                print(f"[Briefing] Phase 2 error: {e}")
                print(f"[JARVIS] Briefing phase 2 failed: {e}")
                self.ui.write_log("SYS: Could not fetch the news for the briefing.")

        asyncio.create_task(_deliver_news())

    # ── Session memory ──────────────────────────────────────────────────────────

    async def _save_session_summary(self) -> None:
        """Summarise the current session in 1-2 sentences and save to long_term.json."""
        log = self._session_log
        if len(log) < 3:          # need at least one exchange to be worth saving
            return
        self._session_log = []    # reset immediately so the next session starts clean

        memory = load_memory()
        lang_entry = memory.get("identity", {}).get("language", {})
        lang = (lang_entry.get("value", "") if isinstance(lang_entry, dict) else str(lang_entry)).strip()
        lang = lang or "English"

        convo = "\n".join(log[-40:])   # cap at last 40 turns to stay within token budget
        prompt = (
            f"Summarize this conversation in 1-2 sentences in {lang}. "
            "Focus on what the user accomplished or discussed. "
            "Output ONLY the summary text, nothing else:\n\n" + convo
        )
        try:
            from core import gemini
            summary = await asyncio.to_thread(
                gemini.text, prompt, gemini.SMART, None, 30_000,
            )
            if summary:
                save_session_summary(summary, lang)
        except Exception as e:
            print(f"[Memory] ⚠️ Session summary failed: {e}")

    # ── System monitor ──────────────────────────────────────────────────────────

    async def _run_system_monitor(self) -> None:
        """Background task: voice alerts when metrics exceed thresholds."""
        while True:
            await asyncio.sleep(10)
            alert = await asyncio.to_thread(self._sys_monitor.check)
            if not alert or not self.session or not self._awake:
                continue
            # Don't interrupt an active conversation
            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking or (time.monotonic() - self._last_user_speech) < 10:
                continue
            try:
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": alert}]},
                    turn_complete=True,
                )
            except Exception as e:
                print(f"[Monitor] ⚠️ Could not send alert: {e}")

    # ── Background monitor ──────────────────────────────────────────────────────

    async def _run_background_monitor(self) -> None:
        """Check user-configured topics once per day; speak alerts when new headlines appear."""
        await asyncio.sleep(300)          # wait 5 min after startup before first check
        while True:
            if self.session and self._awake:
                # Don't interrupt if user spoke recently or JARVIS is mid-sentence
                with self._speaking_lock:
                    speaking = self._is_speaking
                recent_speech = (time.monotonic() - self._last_user_speech) < 30
                if not speaking and not recent_speech:
                    try:
                        alerts = await asyncio.to_thread(monitor_check_all)
                        memory = load_memory()
                        lang_e = memory.get("identity", {}).get("language", {})
                        lang   = (lang_e.get("value", "") if isinstance(lang_e, dict) else str(lang_e)).strip() or "English"
                        for alert in alerts:
                            msg = (
                                f"{alert}\n\n"
                                f"Inform the user about this development naturally in {lang}. "
                                "One brief sentence only."
                            )
                            await self.session.send_client_content(
                                turns={"role": "user", "parts": [{"text": msg}]},
                                turn_complete=True,
                            )
                            print("[JARVIS] Monitor alert sent.")
                            await asyncio.sleep(6)   # gap between consecutive alerts
                    except Exception as e:
                        print(f"[Monitor] ⚠️ Background check error: {e}")
            await asyncio.sleep(1800)     # check every 30 minutes

    # ── Proactive mode ──────────────────────────────────────────────────────────

    async def _run_proactive_mode(self) -> None:
        """
        Background task: periodically checks if the user has been silent long enough,
        then hands time + memory context to Gemini so it can decide what (if anything)
        to say proactively. No hardcoded rules — Gemini makes the call.
        """
        while True:
            await asyncio.sleep(60)   # evaluate once per minute

            if not self.session or not self._awake:
                continue

            with self._speaking_lock:
                speaking = self._is_speaking
            if speaking:
                continue

            if not self._proactive.should_trigger(self._last_user_speech):
                continue

            self._proactive.mark_triggered()

            try:
                memory       = await asyncio.to_thread(load_memory)
                monitors     = await asyncio.to_thread(list_monitors)
                recent_turns = self._session_log[-8:] if self._session_log else []
                prompt = self._proactive.build_prompt(
                    memory       = memory,
                    monitors     = monitors or None,
                    recent_turns = recent_turns or None,
                )
                await self.session.send_client_content(
                    turns={"role": "user", "parts": [{"text": prompt}]},
                    turn_complete=True,
                )
                print("[JARVIS] Proactive check-in.")
            except Exception as e:
                print(f"[Proactive] ⚠️ {e}")

    # ── Phone audio relay ────────────────────────────────────────────────────────

    async def _relay_phone_audio(self) -> None:
        """Forward phone mic PCM chunks from dashboard queue into the Gemini Live session."""
        q = self._dashboard._phone_audio_queue
        while True:
            try:
                chunk = await asyncio.wait_for(q.get(), timeout=1.0)
            except asyncio.TimeoutError:
                # No audio for 1 s → phone mic inactive, give PC mic back
                self._phone_active = False
                continue
            self._phone_active = True   # phone is streaming — silence PC mic
            with self._speaking_lock:
                speaking = self._is_speaking
            if not speaking and not self.ui.muted:
                try:
                    self.out_queue.put_nowait(chunk)
                except asyncio.QueueFull:
                    pass

    def _on_phone_connected(self) -> None:
        self.ui.write_log("SYS: Phone connected via Remote Dashboard.")
        self.ui.notify_phone_connected()

    # ── dashboard command relay ─────────────────────────────────────────────

    async def _process_dashboard_commands(self) -> None:
        while True:
            try:
                text = await asyncio.wait_for(
                    self._dashboard._command_queue.get(), timeout=0.5
                )
                if not text:
                    continue
                # Wait up to 8s for session to become ready after a wake
                for _ in range(80):
                    if self.session:
                        break
                    await asyncio.sleep(0.1)
                if self.session:
                    # A remote command is deliberate control and the phone user
                    # has no desktop WAKE button — so it wakes JARVIS if asleep.
                    if self._wake_enabled and not self._awake:
                        self.wake(reason="remote command")
                    await self.session.send_client_content(
                        turns={"role": "user", "parts": [{"text": text}]},
                        turn_complete=True,
                    )
                    self.ui.write_log(f"[Web]: {text}")
                else:
                    print(f"[Dashboard] Dropped command (no session): {text}")
            except asyncio.TimeoutError:
                pass
            except Exception as e:
                print(f"[Dashboard] Command error: {e}")
                await asyncio.sleep(0.5)

    # ── main loop ───────────────────────────────────────────────────────────

    async def run(self):
        self._loop = asyncio.get_event_loop()
        self._reconnect_event = asyncio.Event()

        # ── Wire the shared core services to the interface ───────────────────
        # The confirmation gate is useless without a way to ask, and a memory
        # trim is invisible without a way to say so. Both are bound once here
        # rather than passed down through every action signature.
        confirm_gate.bind(
            show = self.ui.show_confirm,
            hide = self.ui.hide_confirm,
            log  = self.ui.write_log,
        )
        set_trim_notifier(self.ui.write_log)

        # Tell the device picker the exact rates the streams open at, from the
        # constants that actually open them — so it can never list a device that
        # cannot be opened at them.
        audio_devices.configure(SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE)

        # Enumerate audio devices off-thread. The settings drawer must never pay
        # for host-API enumeration on the Qt thread.
        audio_devices.prefetch()

        # Start dashboard (optional — needs: pip install fastapi "uvicorn[standard]" cryptography)
        try:
            from dashboard.server import DashboardServer
            self._dashboard = DashboardServer()
            self._dashboard.set_connect_callback(self._on_phone_connected)
            asyncio.create_task(self._dashboard.serve())
            # Runs for the whole lifetime, not just inside an active session
            asyncio.create_task(self._process_dashboard_commands())
        except Exception as e:
            print(f"[Dashboard] Disabled: {e}")
            self._dashboard = None

        while True:
            # ── Economy brain: no Gemini session at all ──────────────────
            if get_brain_engine() == "economy":
                try:
                    await self._run_economy()
                except KeyboardInterrupt:
                    raise
                except BaseException as e:
                    if _is_reconnect_signal(e):
                        print("[JARVIS] Voluntary reconnect requested.")
                        continue
                    print(f"[Economy] stopped: {e!r}")
                    traceback.print_exc()
                    self.ui.set_state("SLEEPING")
                    await asyncio.sleep(3)
                continue

            try:
                print("[JARVIS] Connecting...")
                self.ui.set_state("THINKING")
                # Pick the rung to open the conversation on. A model resting
                # off a quota limit is skipped; the name is published back to
                # LIVE_MODEL so plugins follow whatever is actually in use.
                global LIVE_MODEL
                LIVE_MODEL = _gemini.live_model()
                live_model = LIVE_MODEL
                print(f"[JARVIS] Live model: {live_model}")

                _resumed_with = self._resume_handle is not None
                config = self._build_config()

                # Fresh client on every reconnect — avoids stale HTTP session state
                # v1alpha carries proactive audio; if it gets rejected we fall
                # back to v1beta.
                client = genai.Client(
                    api_key=_get_api_key(),
                    http_options={"api_version": "v1alpha" if self._enhanced_live else "v1beta"}
                )

                async with (
                    client.aio.live.connect(model=live_model, config=config) as session,
                    asyncio.TaskGroup() as tg,
                ):
                    self.session          = session
                    self.audio_in_queue   = asyncio.Queue()
                    self.out_queue        = asyncio.Queue(maxsize=200)
                    self._turn_done_event = asyncio.Event()

                    # Reset transient state that must not carry over from a previous session
                    self._pending_vision       = None
                    self._vision_cam_active    = False
                    self._vision_close_pending = False
                    self._vision_busy          = False
                    self._vision_last_time     = 0.0
                    self._interrupted          = False

                    print("[JARVIS] Connected.")
                    try:
                        if _gemini.key_count() > 1:
                            print(f"[Gemini] accounts — {_gemini.key_status()}")
                            if _gemini.key_index() > 1:
                                self.ui.write_log(
                                    f"SYS: Connected on Gemini fallback {_gemini.key_index() - 1} "
                                    "(the primary is resting).")
                    except Exception:
                        pass
                    if _resumed_with:
                        # Say it plainly: the difference between "it reconnected"
                        # and "it reconnected and still knows what we were doing"
                        # is the whole point, and it is invisible otherwise.
                        self.ui.write_log("SYS: Reconnected — conversation restored.")

                    # Wake word: if enabled, come up ASLEEP (mic gated, silent)
                    # until the user says "Hey Jarvis" or taps wake in the UI.
                    if self._wake_enabled:
                        self._ensure_wake_detector()
                        self._awake = False
                        self.ui.set_state("SLEEPING")
                        self.ui.write_log("SYS: JARVIS online — sleeping. Say 'Hey Jarvis' to wake me.")
                    else:
                        self._awake = True
                        self.ui.set_state("LISTENING")
                        self.ui.write_log("SYS: JARVIS online.")

                    if self._dashboard:
                        await self._dashboard.broadcast({"type": "status", "state": "active"})

                    self._reconnect_event.clear()  # ignore requests from before this session
                    tg.create_task(self._watch_reconnect())
                    tg.create_task(self._send_realtime())
                    tg.create_task(self._listen_audio())
                    tg.create_task(self._receive_audio())
                    tg.create_task(self._play_audio())
                    tg.create_task(self._run_system_monitor())
                    tg.create_task(self._run_background_monitor())
                    tg.create_task(self._run_proactive_mode())
                    tg.create_task(self._run_sleep_watch())
                    tg.create_task(self._run_reminder_watch())
                    if self._dashboard:
                        tg.create_task(self._relay_phone_audio())

                    # Morning briefing — fires once per process launch (if enabled).
                    # Skipped in wake-word mode: it comes up asleep, and a briefing
                    # would mean talking while "asleep".
                    if self._briefing_allowed():
                        self._briefing_sent = True
                        tg.create_task(self._send_startup_briefing())

            except KeyboardInterrupt:
                raise
            except SystemExit:
                raise
            except BaseException as e:
                # Catches both Exception and BaseExceptionGroup (Python 3.11+
                # TaskGroup raises BaseExceptionGroup when tasks are cancelled
                # externally, which `except Exception` would miss, letting the
                # exception escape the while-loop and causing asyncio.run() to
                # start shutdown — resulting in "executor after shutdown" errors).
                # Voluntary reconnect (voice change) — not an error. Rebuild the
                # session immediately with no backoff and no scary logs.
                if _is_reconnect_signal(e):
                    print("[JARVIS] Voluntary reconnect requested.")
                    if not _keep_context_of(e):
                        # A deliberate clean slate (voice change) — drop the
                        # handle so the next connect really does start empty.
                        self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                # A resumption handle the server will not accept — expired, or
                # belonging to a session it has since dropped. Without this, the
                # same dead handle would be replayed on every retry and the
                # assistant would never come back at all: the feature meant to
                # survive a reconnect would be the thing preventing one. Drop it
                # once and let the next attempt start clean.
                if _resumed_with and (
                    "resum" in str(e).lower()
                    or "handle" in str(e).lower()
                    or "INVALID_ARGUMENT" in str(e)
                    or "NOT_FOUND" in str(e)
                ):
                    print("[JARVIS] 🔗 Resumption handle rejected — starting a fresh session")
                    self.ui.write_log("SYS: Could not restore the conversation — starting fresh.")
                    self._resume_handle = None
                    self._conn_backoff = 0
                    continue

                err_str = str(e)
                print(f"[JARVIS] Error ({type(e).__name__}): {e}")
                traceback.print_exc()

                # Out of quota, or this model is not available to this key —
                # step down the ladder and reconnect straight away. This is the
                # difference between "JARVIS is quieter today" and "JARVIS does
                # not start today": one model means one daily limit, and the
                # limit always arrives mid-conversation.
                # Several accounts configured? A quota or credit wall on this
                # one is the moment to hand the SAME model to the next account,
                # which keeps the better model instead of stepping down.
                # getattr-guarded: an older core/gemini.py left in place must
                # not take the whole run loop down with an AttributeError.
                _credit = getattr(_gemini, "is_credit_error", lambda e: False)
                _rotate = getattr(_gemini, "rotate_key", lambda reason="": 0)
                _nkeys  = getattr(_gemini, "key_count", lambda: 1)
                if _gemini.is_quota_error(err_str) or _credit(err_str):
                    n = _rotate("credit depleted" if _credit(err_str) else "quota exhausted")
                    if n:
                        self.ui.write_log(
                            f"SYS: Gemini {'primary' if n == 1 else 'fallback ' + str(n - 1)} "
                            f"account in use — the previous one is "
                            f"{'out of credit' if _credit(err_str) else 'out of quota'} "
                            f"and will be retried later.")
                        self._conn_backoff = 0
                        continue
                    if _credit(err_str):
                        self.ui.write_log(
                            "ERR: Gemini prepaid credit is depleted on this account. "
                            "Add credit, or another key in config/api_keys.json "
                            "(\"gemini_api_keys\"). Retrying in 30s.")
                        self._conn_backoff = 30
                        await asyncio.sleep(30)
                        continue

                if _gemini.note_live_failure(live_model, err_str):
                    nxt = _gemini.live_model()
                    self.ui.write_log(
                        f"SYS: Switching to {nxt.split('/')[-1]} — the previous "
                        f"model is out of quota."
                        if nxt != live_model else
                        "SYS: Every live model is rate-limited — retrying.")
                    self._conn_backoff = 0 if nxt != live_model else 15
                    if nxt == live_model:
                        await asyncio.sleep(self._conn_backoff)
                    continue

                # Turn-taking / media / thinking knobs rejected by the server
                # (preview API drift) — drop them first, because they are the
                # newest fields and the cheapest to lose. Proactive audio is
                # tried again on the next pass if the error persists.
                if self._tuned_live and (
                    "INVALID_ARGUMENT" in err_str
                    or "Unknown name" in err_str
                    or "unexpected keyword" in err_str
                    or "realtime_input" in err_str.lower()
                    or "media_resolution" in err_str.lower()
                    or "thinking" in err_str.lower()
                ):
                    self._tuned_live = False
                    print("[JARVIS] Live tuning rejected — reconnecting without it.")
                    continue

                # Proactive audio rejected by the server (preview API drift) —
                # drop it and reconnect with the plain config.
                if self._enhanced_live and (
                    "INVALID_ARGUMENT" in err_str
                    or "proactiv" in err_str.lower()
                    or "Unknown name" in err_str
                    or "unexpected keyword" in err_str
                ):
                    self._enhanced_live = False
                    self.ui.write_log(
                        "SYS: Proactive audio unavailable — reconnecting without it."
                    )
                    continue

                # A voice this Live model does not serve. The catalogue beyond
                # the five shipped names is offered as experimental, so a
                # rejection here is expected occasionally: go back to the
                # default voice, say so, and reconnect rather than loop.
                if "voice" in err_str.lower() and get_voice() != DEFAULT_VOICE:
                    bad = get_voice()
                    save_voice(DEFAULT_VOICE)
                    self.ui.write_log(
                        f"SYS: Voice {bad} is not available on this model — "
                        f"back to {DEFAULT_VOICE}.")
                    self._conn_backoff = 0
                    continue

                # Invalid API key — stop hammering the API, prompt re-configuration
                if "API key not valid" in err_str or "1007" in err_str:
                    n = _rotate("key rejected")
                    if n:
                        self.ui.write_log(
                            f"ERR: A Gemini key was rejected — using "
                            f"{'the primary' if n == 1 else 'fallback ' + str(n - 1)} instead.")
                        self._conn_backoff = 0
                        continue
                    self.ui.write_log("ERR: API key invalid — please re-enter your key.")
                    self.ui.set_state("SLEEPING")
                    self.ui.prompt_reconfig()
                    while not self.ui._win._ready:
                        await asyncio.sleep(1)
                    print("[JARVIS] New API key saved — reconnecting...")
                    _conn_backoff = 3
                    continue

                # Network / timeout errors — log clearly and back off
                is_net_err = any(k in err_str for k in (
                    "TimeoutError", "timed out", "getaddrinfo", "CancelledError",
                    "ConnectionRefusedError", "OSError", "Cannot connect",
                ))
                if is_net_err:
                    _conn_backoff = min(getattr(self, "_conn_backoff", 3) * 2, 60)
                    self._conn_backoff = _conn_backoff
                    self.ui.write_log(
                        f"NET: Connection failed — retrying in {_conn_backoff}s. "
                        "(a VPN may be required)"
                    )
                else:
                    self._conn_backoff = 3
            finally:
                self.session = None
                # Only save if there was a real conversation (≥3 turns)
                if len(self._session_log) >= 3:
                    asyncio.create_task(self._save_session_summary())

            self.set_speaking(False)
            self.ui.set_state("SLEEPING")

            if self._dashboard:
                await self._dashboard.broadcast({"type": "status", "state": "sleeping"})

            delay = getattr(self, "_conn_backoff", 3)
            print(f"[JARVIS] Reconnecting in {delay}s...")
            await asyncio.sleep(delay)

def main():
    ui = JarvisUI("face.png")

    def runner():
        ui.wait_for_api_key()
        jarvis = JarvisLive(ui)
        try:
            asyncio.run(jarvis.run())
        except KeyboardInterrupt:
            print("\n🔴 Shutting down...")

    threading.Thread(target=runner, daemon=True).start()
    ui.root.mainloop()

if __name__ == "__main__":
    main()
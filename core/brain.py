"""
The assistant's brain — an OpenAI-compatible chat model writes the replies and
the assistant's own voice reads them. It is the only mode: Gemini Live was
retired because its audio only comes in Google's voices.

    microphone → Whisper on Groq (core.groq_stt, free)
              → a chat model over /chat/completions (Gemini by default,
                or DeepSeek, Groq, OpenRouter, anything OpenAI-compatible)
              → the cloned voice of voz_local/referencia.wav, synthesised on
                this PC (voz_local/servidor_voz.py) → the speakers

Everything else the assistant does stays: the persona prompt, memory, every
action and plugin (they are offered to the model as OpenAI tools), reminders,
the briefing, the avatar's lip-sync (the voice is played through the same
queue Gemini's audio used). What is lost: barge-in by voice, vision tools
(deepseek-chat cannot see), and Gemini's native hearing of tone.

Settings come from ⚙ → PLUGIN SETTINGS → ECONOMY BRAIN (plugins/brain_economy.py),
stored under plugin_config.brain in config/api_keys.json.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable

import requests

NAMESPACE = "brain"
TIMEOUT = 60
MAX_REPLY_CHARS = 12000        # a spoken reply never needs more; guards runaway streams

PRESETS = {
    "deepseek":   {"base_url": "https://api.deepseek.com/v1",
                   "model": "deepseek-chat",
                   "key_url": "https://platform.deepseek.com/api_keys"},
    "groq":       {"base_url": "https://api.groq.com/openai/v1",
                   "model": "openai/gpt-oss-120b",
                   "key_url": "https://console.groq.com/keys"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1",
                   "model": "deepseek/deepseek-chat",
                   "key_url": "https://openrouter.ai/keys"},
    "gemini":     {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
                   "model": "gemini-3.6-flash",
                   "key_url": "https://aistudio.google.com/app/apikey"},
    "custom":     {"base_url": "http://localhost:11434/v1", "model": "llama3.2", "key_url": ""},
}
PROVIDERS = tuple(PRESETS)
DEFAULT_PROVIDER = "gemini"

# The assistant has exactly one voice: the clone of voz_local/referencia.wav,
# synthesised on this PC (Chatterbox Multilingual, served by
# voz_local/servidor_voz.py — GPU when there is one, CPU otherwise). No
# account, no credits, nothing leaves the machine, and nothing to choose.
_ROOT = Path(__file__).resolve().parent.parent
LOCAL_TTS_URL = os.environ.get("JARVIS_LOCAL_TTS_URL", "http://127.0.0.1:5005")
LOCAL_TTS_PY = (_ROOT / "voz_local" / ".venv" / "Scripts" / "python.exe" if sys.platform == "win32"
                else _ROOT / "voz_local" / ".venv" / "bin" / "python")
LOCAL_TTS_SERVER = _ROOT / "voz_local" / "servidor_voz.py"

# Tools that only make sense when the model can receive an image.
_VISION_TOOLS = {"screen_process", "close_camera"}


def settings(get_plugin_config: Callable[[str], dict]) -> dict:
    cfg = get_plugin_config(NAMESPACE)
    provider = str(cfg.get("provider") or DEFAULT_PROVIDER).strip().lower()
    if provider not in PRESETS:
        provider = DEFAULT_PROVIDER
    preset = PRESETS[provider]
    base = str(cfg.get("base_url") or "").strip().rstrip("/") or preset["base_url"]
    model = str(cfg.get("model") or "").strip() or preset["model"]
    key = str(cfg.get("api_key") or "").strip()
    if not key and provider == "gemini":
        # Out of the box the brain uses the Gemini key typed on the first-run
        # screen, so a fresh clone works without opening PLUGIN SETTINGS.
        try:
            from memory.config_manager import get_gemini_key
            key = str(get_gemini_key() or "").strip()
        except Exception:
            pass
    return {"provider": provider, "base_url": base, "model": model, "api_key": key}


def voice_label(st: dict | None = None) -> str:
    return "JARVIS (voz_local/referencia.wav)"


# ── tools: Gemini declarations → OpenAI tools ────────────────────────────────

def _lower_schema(node):
    if isinstance(node, dict):
        out = {}
        for k, v in node.items():
            if k == "type" and isinstance(v, str):
                out[k] = v.lower()
            elif k in ("properties", "items") or isinstance(v, (dict, list)):
                out[k] = _lower_schema(v)
            else:
                out[k] = v
        return out
    if isinstance(node, list):
        return [_lower_schema(x) for x in node]
    return node


def to_openai_tools(decls: list) -> list[dict]:
    out = []
    for d in decls:
        if not isinstance(d, dict):
            d = {"name": getattr(d, "name", ""), "description": getattr(d, "description", ""),
                 "parameters": getattr(d, "parameters", None)}
        name = d.get("name")
        if not name or name in _VISION_TOOLS:
            continue
        params = _lower_schema(d.get("parameters") or {"type": "object", "properties": {}})
        params.setdefault("type", "object")
        params.setdefault("properties", {})
        out.append({"type": "function", "function": {
            "name": name, "description": (d.get("description") or "")[:1024],
            "parameters": params}})
    return out


# ── chat ─────────────────────────────────────────────────────────────────────

# A sentence ends at punctuation FOLLOWED BY WHITESPACE. Matching the end of
# the buffer too cut "R$ 381.000,00" into "R$ 381." and "000,00" whenever a
# streamed chunk happened to stop on the dot — the voice then read two
# nonsense numbers. The tail is flushed when the stream ends anyway.
_SENT_END = re.compile(r"(?<=[.!?…])\s+|\n+")


class ContextTooLong(RuntimeError):
    """The provider refused the request because the conversation sent is
    bigger than the model's window. The caller drops the history and retries."""


def chat_stream(messages: list, tools: list | None, st: dict,
                on_sentence: Callable[[str], None],
                should_stop: Callable[[], bool] = lambda: False) -> tuple[str, list[dict]]:
    """Stream one reply. Calls on_sentence() for each finished sentence as it
    arrives; returns (full_content, tool_calls) where tool_calls are
    [{"id", "name", "arguments": dict}]. Raises on HTTP errors."""
    payload = {"model": st["model"], "messages": messages, "stream": True,
               "max_tokens": 600, "temperature": 0.6}
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"
    headers = {"Content-Type": "application/json"}
    if st["api_key"]:
        headers["Authorization"] = f"Bearer {st['api_key']}"
    if st["provider"] == "openrouter":
        headers["HTTP-Referer"] = "https://github.com/FatihMakes/Mark-LV"
        headers["X-Title"] = "MARK LV"

    content, buf = "", ""
    frags: dict[int, dict] = {}
    with requests.post(f"{st['base_url']}/chat/completions", json=payload,
                       headers=headers, timeout=TIMEOUT, stream=True) as r:
        if r.status_code >= 400:
            body = r.text[:300]
            low = body.lower()
            if r.status_code == 400 and ("context length" in low or "maximum context" in low
                                         or "too many tokens" in low or "token limit" in low):
                raise ContextTooLong(f"{r.status_code}: {body}")
            raise requests.HTTPError(f"{r.status_code}: {body}", response=r)
        for raw in r.iter_lines():
            if should_stop():
                break
            if not raw:
                continue
            line = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}
            piece = delta.get("content") or ""
            if piece:
                # Some OpenAI-compatible gateways stream the WHOLE text so far
                # in every chunk instead of the new bit. Appending that would
                # square the reply; keep only what is new.
                if content and len(piece) > len(content) and piece.startswith(content):
                    piece = piece[len(content):]
                if len(content) + len(piece) > MAX_REPLY_CHARS:
                    piece = piece[: max(0, MAX_REPLY_CHARS - len(content))]
                    if not piece:
                        break
                content += piece
                buf += piece
                # hand over finished sentences, keep the open tail
                parts = _SENT_END.split(buf)
                if len(parts) > 1:
                    for s in parts[:-1]:
                        s = s.strip()
                        if s:
                            on_sentence(s)
                    buf = parts[-1]
            for tc in delta.get("tool_calls") or []:
                i = int(tc.get("index", 0))
                f = frags.setdefault(i, {"id": "", "name": "", "arguments": ""})
                if tc.get("id"):
                    f["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    f["name"] += fn["name"]
                if fn.get("arguments"):
                    f["arguments"] += fn["arguments"]
    tail = buf.strip()
    if tail and not should_stop():
        on_sentence(tail)
    calls = []
    for i in sorted(frags):
        f = frags[i]
        try:
            args = json.loads(f["arguments"] or "{}")
        except json.JSONDecodeError:
            args = {}
        if f["name"]:
            calls.append({"id": f["id"] or f"call_{i}", "name": f["name"],
                          "arguments": args if isinstance(args, dict) else {}})
    return content, calls


# ── voice: local clone (voz_local/servidor_voz.py) → 24 kHz int16 PCM ────────

def local_tts_up() -> bool:
    try:
        return requests.get(f"{LOCAL_TTS_URL}/health", timeout=2).ok
    except Exception:
        return False


def ensure_local_tts(wait_s: float = 360.0) -> bool:
    """Start the local voice server if it is not running, then wait until the
    model is loaded (the first start downloads it, later ones take seconds)."""
    if local_tts_up():
        return True
    if not LOCAL_TTS_PY.exists() or not LOCAL_TTS_SERVER.exists():
        raise RuntimeError("Local voice not installed (voz_local/.venv missing)")
    flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    subprocess.Popen([str(LOCAL_TTS_PY), str(LOCAL_TTS_SERVER)], cwd=str(_ROOT),
                     creationflags=flags, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t0 = time.monotonic()
    while time.monotonic() - t0 < wait_s:
        if local_tts_up():
            return True
        time.sleep(1)
    return False


def tts_local_pcm24k(text: str) -> bytes:
    import miniaudio

    if not local_tts_up() and not ensure_local_tts():
        raise RuntimeError("Local voice server did not start (see voz_local/servidor_voz.log)")
    r = requests.post(f"{LOCAL_TTS_URL}/tts", json={"text": text}, timeout=TIMEOUT)
    r.raise_for_status()
    if not r.content:
        return b""
    dec = miniaudio.decode(r.content, output_format=miniaudio.SampleFormat.SIGNED16,
                           nchannels=1, sample_rate=24000)
    return bytes(dec.samples)


def synth_pcm24k(text: str, st: dict | None = None) -> bytes:
    """One sentence → 24 kHz PCM in the assistant's voice. There is no other
    voice to fall back to: if the server died, it is started again and the
    sentence retried once."""
    try:
        return tts_local_pcm24k(text)
    except requests.exceptions.ConnectionError:
        print("[Brain] Voice server not answering — restarting it")
        if not ensure_local_tts():
            raise RuntimeError("Voice server did not come back (see voz_local/servidor_voz.log)")
        return tts_local_pcm24k(text)


def missing_packages(st: dict | None = None) -> list[str]:
    out = []
    need = [("miniaudio", "miniaudio")]
    for mod, pkg in need:
        try:
            __import__(mod)
        except Exception:
            out.append(pkg)
    return out


def test_connection(values: dict):
    """⚙ TEST CONNECTION: key accepted, model listed, voice packages present."""
    fake = dict(values)
    st = settings(lambda ns: fake)
    notes = []
    missing = missing_packages(st)
    if missing:
        notes.append("voice packages missing — run: py -m pip install " + " ".join(missing))
    if not missing:
        try:
            t0 = time.monotonic()
            pcm = synth_pcm24k("Teste de voz.")
            notes.append(f"Voice OK ({len(pcm) / 48000.0:.1f} s of audio "
                         f"in {time.monotonic() - t0:.1f} s)")
        except Exception as e:
            return False, f"Voice failed: {e}"
    if not st["api_key"] and st["provider"] != "custom":
        return False, "Enter the API key first. " + " ".join(notes)
    headers = {"Authorization": f"Bearer {st['api_key']}"} if st["api_key"] else {}
    try:
        r = requests.get(f"{st['base_url']}/models", headers=headers, timeout=15)
        if r.status_code == 401:
            return False, "The provider rejected the key (401)."
        r.raise_for_status()
        # Gemini lists its ids as "models/<name>"; compare the bare names.
        ids = {str(m.get("id")).removeprefix("models/") for m in (r.json().get("data") or [])}
        if ids and st["model"] not in ids:
            notes.append(f"model '{st['model']}' not in the provider's list — check the name")
    except Exception as e:
        return False, f"Connection failed: {e}"

    # The list says the key works; only a real reply says the MODEL works for
    # this key (credits, access, tool support). One tiny request, with a tool
    # attached, because that is exactly what the assistant will send.
    probe_tool = [{"type": "function", "function": {
        "name": "noop", "description": "does nothing",
        "parameters": {"type": "object", "properties": {}}}}]
    try:
        got: list[str] = []
        content, calls = chat_stream(
            [{"role": "system", "content": "Reply with the single word OK."},
             {"role": "user", "content": "Ready?"}],
            probe_tool, st, got.append)
        reply = (content or " ".join(got)).strip()[:40] or ("(tool call)" if calls else "(empty)")
        ok_line = (f"Connected to {st['provider']}, model {st['model']} answered: "
                   f"\"{reply}\".")
        return (not missing), ok_line + (" " + "; ".join(notes) if notes else "")
    except requests.HTTPError as e:
        body = ""
        try:
            body = e.response.text[:200] if e.response is not None else str(e)[:200]
        except Exception:
            body = str(e)[:200]
        return False, (f"The key works but model {st['model']} refused: {body}. "
                       "On OpenRouter a 402 means no credits — add some or use a ':free' "
                       "model such as deepseek/deepseek-chat:free; a 404 means the model "
                       "id is wrong.")
    except Exception as e:
        return False, f"Model test failed: {e}"

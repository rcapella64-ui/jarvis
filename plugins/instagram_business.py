"""
Instagram (conta Comercial ou Criador) — métricas, comentários e mensagens.

Usa a API oficial da Meta "Instagram API com login do Instagram"
(graph.instagram.com): não precisa de Página do Facebook. O token fica em
⚙ → PLUGIN SETTINGS → INSTAGRAM. Para gerar: developers.facebook.com → criar
app → produto Instagram → "API setup with Instagram login" → adicionar a conta
→ gerar token com as permissões instagram_business_basic,
instagram_business_manage_comments e instagram_business_manage_messages.

Só leitura. Cada parte (perfil, comentários, mensagens) é buscada à parte: se
uma permissão faltar, as outras continuam e o erro daquela parte é informado.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import requests

NAMESPACE = "instagram"
API = "https://graph.instagram.com"
TIMEOUT = 20

PLUGIN = {
    "name": "instagram_summary",
    "description": (
        "Reads the user's business Instagram account (read-only): followers and the "
        "latest posts with likes and comments, recent comments, and recent direct "
        "messages. Use when the user asks about their Instagram (e.g. the Vitale Living "
        "account): new comments, DMs, followers, how posts are doing. Report only what "
        "this tool returns; parts that failed are marked as such — say so, do not guess."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "hours": {"type": "INTEGER",
                      "description": "Comments and messages from the last N hours (default 24)."},
        },
        "required": [],
    },
}


def _get(path: str, token: str, **params) -> dict:
    r = requests.get(f"{API}/{path}", params={**params, "access_token": token}, timeout=TIMEOUT)
    data = r.json() if r.content else {}
    if r.status_code != 200 or "error" in data:
        msg = (data.get("error") or {}).get("message") or r.text[:120]
        raise RuntimeError(msg)
    return data


def _when(ts: str) -> datetime | None:
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S%z")
    except Exception:
        return None


def _fmt(dt: datetime | None) -> str:
    return dt.astimezone().strftime("%d/%m %H:%M") if dt else "sem data"


def _token(values: dict | None = None) -> str:
    if values is None:
        from memory.config_manager import get_plugin_config
        values = get_plugin_config(NAMESPACE)
    return str(values.get("access_token") or "").strip()


def summary(token: str, hours: int = 24) -> list[str]:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    out: list[str] = []

    # Profile and the latest posts
    media = None
    try:
        me = _get("me", token, fields="username,followers_count,follows_count,media_count")
        out.append(f"Account @{me.get('username')}: {me.get('followers_count', '?')} followers, "
                   f"{me.get('media_count', '?')} posts.")
        media = _get("me/media", token, limit=5,
                     fields="id,caption,timestamp,like_count,comments_count").get("data", [])
        for i, m in enumerate(media, 1):
            cap = (m.get("caption") or "(sem legenda)").replace("\n", " ")[:80]
            out.append(f"  Post {i} ({_fmt(_when(m.get('timestamp', '')))}): "
                       f"{m.get('like_count', '?')} likes, {m.get('comments_count', '?')} comments — {cap}")
    except Exception as e:
        out.append(f"Profile/posts: FAILED ({str(e)[:120]})")

    # Recent comments on those posts. Without the post list there is nothing to
    # look at — "0 comments" would be a made-up answer, so say it was not checked.
    try:
        if media is None:
            raise RuntimeError("NOT CHECKED — the posts could not be read")
        recent = []
        for m in media:
            for c in _get(f"{m['id']}/comments", token, fields="username,text,timestamp",
                          limit=25).get("data", []):
                when = _when(c.get("timestamp", ""))
                if when and when >= cutoff:
                    recent.append((when, c))
        recent.sort(key=lambda x: x[0], reverse=True)
        out.append(f"Comments in the last {hours} h on the latest {len(media)} posts: {len(recent)}")
        for when, c in recent[:10]:
            out.append(f"  @{c.get('username', '?')} ({_fmt(when)}): {(c.get('text') or '')[:150]}")
    except Exception as e:
        out.append(f"Comments: FAILED ({str(e)[:120]})")

    # Direct messages: latest message of each recent conversation
    try:
        convs = _get("me/conversations", token, platform="instagram",
                     fields="updated_time,participants,messages.limit(1){from,message,created_time}",
                     limit=15).get("data", [])
        fresh = [c for c in convs if (_when(c.get("updated_time", "")) or cutoff) >= cutoff]
        out.append(f"Conversations with activity in the last {hours} h: {len(fresh)}")
        for c in fresh[:10]:
            msg = ((c.get("messages") or {}).get("data") or [{}])[0]
            sender = (msg.get("from") or {}).get("username") or "?"
            text = (msg.get("message") or "(mídia ou mensagem sem texto)")[:150]
            out.append(f"  @{sender} ({_fmt(_when(msg.get('created_time', '')))}): {text}")
    except Exception as e:
        out.append(f"Direct messages: FAILED ({str(e)[:120]})")
    return out


def test_connection(values: dict):
    token = _token(values)
    if not token:
        return False, "Paste the access token first."
    try:
        me = _get("me", token, fields="username,account_type")
    except Exception as e:
        return False, f"Token refused: {str(e)[:150]}"
    kind = me.get("account_type", "?")
    if kind not in ("BUSINESS", "MEDIA_CREATOR"):
        return False, f"@{me.get('username')} is a {kind} account — switch it to Business or Creator."
    return True, f"Connected to @{me.get('username')} ({kind})."


PLUGIN_SETTINGS = {
    "namespace": NAMESPACE,
    "title": "INSTAGRAM (conta comercial — métricas, comentários, mensagens)",
    "fields": [
        {"key": "access_token", "label": "Access token (Instagram API, login do Instagram)",
         "type": "password", "placeholder": "IGAA… gerado no app da Meta"},
    ],
    "action": {"label": "TEST CONNECTION", "run": test_connection},
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    token = _token()
    if not token:
        return ("Instagram is not configured (⚙ → PLUGIN SETTINGS → INSTAGRAM needs an access "
                "token). Tell the user; do not guess.")
    hours = max(1, min(int(parameters.get("hours") or 24), 24 * 30))
    return "Instagram (read-only):\n" + "\n".join(summary(token, hours))

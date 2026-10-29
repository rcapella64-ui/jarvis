"""
Caixa de entrada por IMAP — só leitura.

A conta fica em ⚙ → PLUGIN SETTINGS → E-MAIL: endereço e uma SENHA DE APP (não
a senha normal). No Gmail: myaccount.google.com/apppasswords (exige verificação
em duas etapas). A caixa é aberta com EXAMINE e as mensagens lidas com
BODY.PEEK, então nada é marcado como lido, movido ou apagado.

A ferramenta devolve remetente, assunto, data e o começo do texto de cada
e-mail; o modelo resume a partir disso e de mais nada.
"""
from __future__ import annotations

import email
import imaplib
import re
from datetime import datetime, timedelta
from email.header import decode_header, make_header
from email.utils import parsedate_to_datetime

NAMESPACE = "email"
HOSTS = {"gmail": "imap.gmail.com", "outlook": "outlook.office365.com",
         "yahoo": "imap.mail.yahoo.com", "outro": ""}
SNIPPET = 300

PLUGIN = {
    "name": "email_inbox",
    "description": (
        "Reads the user's e-mail inbox (read-only) and returns the recent messages: "
        "sender, subject, date and the start of the text. Use when the user asks about "
        "their e-mail ('tenho e-mail novo?', 'resume meus e-mails de hoje'). Summarise "
        "ONLY what this tool returns; if it reports an error or no messages, say so."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "hours": {"type": "INTEGER",
                      "description": "How far back to look, in hours (default 24)."},
            "unread_only": {"type": "BOOLEAN",
                            "description": "Only unread messages (default true)."},
            "max_messages": {"type": "INTEGER",
                             "description": "At most this many messages (default 10)."},
        },
        "required": [],
    },
}


def _decode(value) -> str:
    try:
        return str(make_header(decode_header(value or ""))).strip()
    except Exception:
        return str(value or "").strip()


def _text_of(msg) -> str:
    """First text/plain part, or text/html stripped of tags."""
    html = ""
    for part in msg.walk() if msg.is_multipart() else [msg]:
        if part.get_content_maintype() == "multipart" or part.get_filename():
            continue
        try:
            payload = part.get_payload(decode=True) or b""
            text = payload.decode(part.get_content_charset() or "utf-8", "replace")
        except Exception:
            continue
        if part.get_content_type() == "text/plain" and text.strip():
            return text
        if part.get_content_type() == "text/html" and not html:
            html = re.sub(r"(?is)<(script|style).*?</\1>|<[^>]+>", " ", text)
    return html


def _settings(values: dict | None = None) -> tuple[str, str, str]:
    if values is None:
        from memory.config_manager import get_plugin_config
        values = get_plugin_config(NAMESPACE)
    provider = str(values.get("provider") or "gmail").strip().lower()
    host = str(values.get("imap_host") or "").strip() or HOSTS.get(provider, "")
    return host, str(values.get("address") or "").strip(), str(values.get("app_password") or "").replace(" ", "")


def fetch(hours: int = 24, unread_only: bool = True, limit: int = 10,
          values: dict | None = None) -> list[dict]:
    host, user, password = _settings(values)
    if not (host and user and password):
        raise RuntimeError("e-mail not configured (⚙ → PLUGIN SETTINGS → E-MAIL)")
    since = (datetime.now() - timedelta(hours=hours)).strftime("%d-%b-%Y")
    with imaplib.IMAP4_SSL(host, timeout=20) as box:
        box.login(user, password)
        box.select("INBOX", readonly=True)            # EXAMINE: never changes flags
        criteria = ["SINCE", since] + (["UNSEEN"] if unread_only else [])
        _, data = box.search(None, *criteria)
        ids = (data[0] or b"").split()[-limit:]
        out = []
        cutoff = datetime.now().astimezone() - timedelta(hours=hours)
        for num in reversed(ids):
            _, parts = box.fetch(num, "(BODY.PEEK[])")
            raw = next((p[1] for p in parts if isinstance(p, tuple)), None)
            if not raw:
                continue
            msg = email.message_from_bytes(raw)
            try:
                when = parsedate_to_datetime(msg.get("Date")).astimezone()
            except Exception:
                when = None
            if when and when < cutoff:
                continue                               # SINCE is per day; trim to the hour
            body = re.sub(r"\s+", " ", _text_of(msg)).strip()
            out.append({"from": _decode(msg.get("From")), "subject": _decode(msg.get("Subject")),
                        "when": when, "snippet": body[:SNIPPET]})
        return out


def test_connection(values: dict):
    try:
        msgs = fetch(hours=24, unread_only=False, limit=5, values=values)
    except imaplib.IMAP4.error as e:
        return False, (f"Login refused ({str(e)[:80]}). Use an APP PASSWORD, not your normal "
                       "password, and check that IMAP is enabled.")
    except Exception as e:
        return False, f"Could not read the inbox: {str(e)[:120]}"
    return True, f"Inbox read: {len(msgs)} message(s) in the last 24 h."


PLUGIN_SETTINGS = {
    "namespace": NAMESPACE,
    "title": "E-MAIL (leitura da caixa de entrada)",
    "fields": [
        {"key": "provider", "label": "Provedor", "type": "choice",
         "options": list(HOSTS), "default": "gmail"},
        {"key": "address", "label": "Endereço de e-mail", "type": "text",
         "placeholder": "voce@gmail.com"},
        {"key": "app_password", "label": "Senha de app (não a senha normal)", "type": "password",
         "placeholder": "Gmail: myaccount.google.com/apppasswords"},
        {"key": "imap_host", "label": "Servidor IMAP (só para 'outro')", "type": "text",
         "placeholder": "imap.seuprovedor.com.br"},
    ],
    "action": {"label": "TEST CONNECTION", "run": test_connection},
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    hours = int(parameters.get("hours") or 24)
    unread = parameters.get("unread_only")
    unread = True if unread is None else bool(unread)
    limit = max(1, min(int(parameters.get("max_messages") or 10), 25))
    try:
        msgs = fetch(hours, unread, limit)
    except Exception as e:
        return f"Could not read the e-mail inbox: {str(e)[:150]}. Tell the user; do not guess."
    scope = f"{'unread ' if unread else ''}e-mails from the last {hours} h"
    if not msgs:
        return f"There are no {scope}."
    lines = []
    for i, m in enumerate(msgs, 1):
        when = m["when"].strftime("%d/%m %H:%M") if m["when"] else "sem data"
        lines.append(f"{i}. From: {m['from']} | {when} | Subject: {m['subject']}\n   {m['snippet']}")
    return f"{len(msgs)} {scope} (newest first):\n" + "\n".join(lines)

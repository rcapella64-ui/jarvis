"""
WhatsApp — resumo das conversas não lidas, só leitura, pelo WhatsApp Web.

Pareamento (uma vez): ⚙ → PLUGIN SETTINGS → WHATSAPP → PAREAR. Abre uma janela
do WhatsApp Web; leia o QR code com o celular (WhatsApp → Aparelhos conectados).
A sessão fica em config/whatsapp_web/ (fora do git — quem tiver essa pasta tem
acesso ao seu WhatsApp).

Leitura: abre o WhatsApp Web sem janela e lê SÓ a lista de conversas — nome,
quantidade de não lidas, hora e a prévia da última mensagem. Nenhuma conversa é
aberta, então ninguém recebe confirmação de leitura e nada fica marcado como
lido. Por isso o resumo vê a última mensagem de cada conversa, não o histórico.

Aviso: os termos do WhatsApp proíbem automação. Só ler a lista é discreto, mas o
risco de bloqueio do número não é zero.
"""
from __future__ import annotations

from pathlib import Path

NAMESPACE = "whatsapp"
PROFILE = Path(__file__).resolve().parent.parent / "config" / "whatsapp_web"
URL = "https://web.whatsapp.com/"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/141.0.0.0 Safari/537.36")

PLUGIN = {
    "name": "whatsapp_unread",
    "description": (
        "Lists the user's WhatsApp chats that have unread messages: contact or group "
        "name, number of unread messages, time and a preview of the LAST message only "
        "(chats are not opened, so earlier messages are not visible). Use when the user "
        "asks about new WhatsApp messages or a summary of them. Report only what this "
        "tool returns; if it says WhatsApp is not paired or failed, say so."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "max_chats": {"type": "INTEGER", "description": "At most this many chats (default 15)."},
        },
        "required": [],
    },
}

# Runs in the page. Chat-list rows live under #pane-side; a row with unread
# messages carries a badge whose aria-label mentions "não lida(s)" / "unread".
_READ_LIST = r"""
(max) => {
  const side = document.querySelector('#pane-side');
  if (!side) return null;
  const rows = side.querySelectorAll('[role="listitem"], [role="row"]');
  const out = [];
  for (const row of rows) {
    const badge = [...row.querySelectorAll('[aria-label]')].find(e =>
      /n[ãa]o lid|unread/i.test(e.getAttribute('aria-label')));
    if (!badge) continue;
    const m = (badge.getAttribute('aria-label').match(/\d+/) || [badge.textContent.trim()])[0];
    const title = row.querySelector('[title]');
    const lines = row.innerText.split('\n').map(s => s.trim()).filter(Boolean);
    const name = ((title && title.getAttribute('title')) || lines[0] || '').trim();
    const time = lines.find(s => /^\d{1,2}:\d{2}$|^ontem$|^yesterday$|\d{2}\/\d{2}/i.test(s)) || '';
    const preview = lines.filter(s => s !== name && s !== time && s !== m &&
                                      !/^\d+\s+(mensage(m|ns)\s+n[ãa]o\s+lidas?|unread)/i.test(s))
                         .join(' ');
    out.push({name, unread: m, time, preview});
    if (out.length >= max) break;
  }
  return out;
}
"""


def _launch(p, headless: bool):
    PROFILE.mkdir(parents=True, exist_ok=True)
    return p.chromium.launch_persistent_context(
        str(PROFILE), headless=headless, user_agent=UA, locale="pt-BR",
        viewport={"width": 1280, "height": 900})


def pair(values: dict | None = None):
    """Settings button: open a visible WhatsApp Web and wait for the QR scan."""
    from playwright.sync_api import sync_playwright
    try:
        with sync_playwright() as p:
            ctx = _launch(p, headless=False)
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(URL)
            try:
                page.wait_for_selector("#pane-side", timeout=180_000)
            except Exception:
                ctx.close()
                return False, "QR code not scanned within 3 minutes. Press PAREAR again."
            page.wait_for_timeout(5000)            # let the session settle before closing
            ctx.close()
        return True, "WhatsApp paired. The assistant can now read your unread chats."
    except Exception as e:
        return False, f"Could not open WhatsApp Web: {str(e)[:120]}"


def unread_chats(max_chats: int = 15) -> list[dict]:
    from playwright.sync_api import sync_playwright
    if not PROFILE.exists():
        raise RuntimeError("WhatsApp is not paired (⚙ → PLUGIN SETTINGS → WHATSAPP → PAREAR)")
    with sync_playwright() as p:
        ctx = _launch(p, headless=True)
        try:
            page = ctx.pages[0] if ctx.pages else ctx.new_page()
            page.goto(URL)
            try:
                page.wait_for_selector("#pane-side, canvas[aria-label]", timeout=60_000)
            except Exception:
                raise RuntimeError("WhatsApp Web did not load in 60 s")
            if not page.query_selector("#pane-side"):
                raise RuntimeError("WhatsApp session expired — pair again "
                                   "(⚙ → PLUGIN SETTINGS → WHATSAPP → PAREAR)")
            page.wait_for_timeout(4000)            # unread badges arrive after the list
            return page.evaluate(_READ_LIST, max_chats) or []
        finally:
            ctx.close()


PLUGIN_SETTINGS = {
    "namespace": NAMESPACE,
    "title": "WHATSAPP (leitura das conversas não lidas)",
    "fields": [],
    "action": {"label": "PAREAR (ler o QR code)", "run": pair},
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    n = max(1, min(int(parameters.get("max_chats") or 15), 40))
    try:
        chats = unread_chats(n)
    except Exception as e:
        return f"Could not read WhatsApp: {str(e)[:150]}. Tell the user; do not guess."
    if not chats:
        return "There are no unread WhatsApp chats."
    lines = [f"{i}. {c['name']} — {c['unread']} unread, {c['time'] or 'sem hora'}: {c['preview']}"
             for i, c in enumerate(chats, 1)]
    return (f"{len(chats)} WhatsApp chat(s) with unread messages (last message preview only):\n"
            + "\n".join(lines))

"""
Notícias da cidade do usuário — Google Notícias (RSS público, sem chave).

A cidade fica em ⚙ → PLUGIN SETTINGS → CITY NEWS. Se não estiver configurada,
o assistente pergunta ao usuário e a primeira cidade dita é gravada. Cada
manchete volta com fonte e data, para o modelo repetir só o que veio daqui.
"""
from urllib.parse import quote_plus

import requests

NAMESPACE = "city_news"
_URL = "https://news.google.com/rss/search?q={q}&hl=pt-BR&gl=BR&ceid=BR:pt-419"
MAX_ITEMS = 6

PLUGIN = {
    "name": "city_news",
    "description": (
        "Latest local news headlines for the user's city (last 2 days, Google News, "
        "Brazilian sources), each with its source and date. Use when the user asks "
        "for news of their city or region ('notícias da minha cidade', 'o que "
        "aconteceu em Campinas'). Pass `city` only when the user names one; otherwise "
        "the configured city is used. Report only the headlines returned."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "city": {"type": "STRING",
                     "description": "City (and state) to search, e.g. 'Campinas SP'. "
                                    "Omit to use the user's configured city."},
        },
        "required": [],
    },
}

PLUGIN_SETTINGS = {
    "namespace": NAMESPACE,
    "title": "CITY NEWS (notícias da sua cidade)",
    "fields": [
        {"key": "city", "label": "Cidade (e estado)", "type": "text",
         "placeholder": "Campinas SP"},
    ],
}


def headlines(city: str, timeout: float = 8.0) -> list[dict]:
    from core.briefing_sources import parse_rss

    r = requests.get(_URL.format(q=quote_plus(f'"{city}" when:2d')), timeout=timeout,
                     headers={"User-Agent": "Mozilla/5.0 (JARVIS city news)"})
    r.raise_for_status()
    seen, out = set(), []
    for item in parse_rss(r.text, max_age_days=2):
        # Google News appends " - <source>" to every title; the source is reported apart.
        src = item.get("source") or ""
        if src and item["title"].endswith(f" - {src}"):
            item["title"] = item["title"][: -len(src) - 3].rstrip()
        key = item["title"].lower()[:80]
        if key not in seen:
            seen.add(key)
            out.append(item)
        if len(out) >= MAX_ITEMS:
            break
    return out


def run(parameters: dict, player=None, session_memory=None) -> str:
    from memory.config_manager import get_plugin_config, save_plugin_config

    stored = str(get_plugin_config(NAMESPACE).get("city") or "").strip()
    city = str(parameters.get("city") or "").strip() or stored
    if not city:
        return ("No city is configured. Ask the user which city they want news for, "
                "then call city_news again with it.")
    if not stored:
        save_plugin_config(NAMESPACE, {"city": city})
    try:
        items = headlines(city)
    except Exception as e:
        return f"Could not reach Google News for {city} ({str(e)[:80]}). Say so; do not guess."
    if not items:
        return f"Google News has no headlines about {city} from the last 2 days."
    lines = []
    for i, it in enumerate(items, 1):
        when = it["when"].strftime("%d/%m %H:%M") if it.get("when") else "sem data"
        src = it.get("source") or "fonte não informada"
        lines.append(f"{i}. {it['title']} — {src}, {when}")
    return f"Latest news about {city} (Google News, last 2 days):\n" + "\n".join(lines)

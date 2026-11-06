"""
Obsidian — buscar, ler e escrever notas no cofre do usuário, direto nos arquivos.

Um cofre do Obsidian é uma pasta de arquivos Markdown: não precisa de plugin,
de internet nem do Obsidian aberto. O cofre é o que estiver em
⚙ → PLUGIN SETTINGS → OBSIDIAN; vazio, usa o cofre aberto no Obsidian (lido de
%APPDATA%/obsidian/obsidian.json).

Escrita só acrescenta: anotar na nota do dia (AAAA-MM-DD.md) ou criar uma nota
nova; se ela já existir, o texto vai para o final. Nada é apagado, renomeado ou
sobrescrito, e nenhum caminho sai da pasta do cofre.
"""
from __future__ import annotations

import difflib
import json
import os
import re
import unicodedata
from datetime import datetime
from pathlib import Path

NAMESPACE = "obsidian"
SKIP_DIRS = {".obsidian", ".trash", ".git"}
MAX_READ = 6000
_FRONTMATTER = re.compile(r"\A---\s*\n.*?\n---\s*\n", re.S)

PLUGIN = {
    "name": "obsidian_notes",
    "description": (
        "The user's Obsidian notes (their personal knowledge base). Actions: "
        "'search' finds notes containing words (use for 'what did I write about X', "
        "'find my notes on X'); 'read' returns one note by its name; 'recent' lists the "
        "notes edited most recently; 'append_daily' adds a line to today's daily note "
        "('anota aí…', 'take a note…'); 'create' makes a new note (or appends if it "
        "exists) in a folder. Answer only from what this tool returns; if a note is not "
        "found, say so. Never claim to have deleted or changed existing text."
    ),
    "parameters": {
        "type": "OBJECT",
        "properties": {
            "action": {"type": "STRING",
                       "description": "search | read | recent | append_daily | create"},
            "query": {"type": "STRING",
                      "description": "search: words to look for. read: the note's name."},
            "text": {"type": "STRING",
                     "description": "append_daily / create: the text to write, in the user's words."},
            "title": {"type": "STRING", "description": "create: the new note's title."},
            "folder": {"type": "STRING",
                       "description": "create: folder inside the vault, e.g. '01_Projetos'. "
                                      "Omit for the vault's root."},
        },
        "required": ["action"],
    },
}


# ── vault ────────────────────────────────────────────────────────────────────

def _open_vault_from_obsidian() -> str:
    """The vault Obsidian has open (or used last), from its own config."""
    cfg = Path(os.environ.get("APPDATA", "")) / "obsidian" / "obsidian.json"
    try:
        vaults = json.loads(cfg.read_text(encoding="utf-8")).get("vaults", {}).values()
    except Exception:
        return ""
    vaults = [v for v in vaults if isinstance(v, dict) and v.get("path")
              and "Obsidian Sandbox" not in v["path"]]
    vaults.sort(key=lambda v: (bool(v.get("open")), v.get("ts", 0)), reverse=True)
    return vaults[0]["path"] if vaults else ""


def vault(values: dict | None = None) -> Path:
    if values is None:
        from memory.config_manager import get_plugin_config
        values = get_plugin_config(NAMESPACE)
    path = str(values.get("vault_path") or "").strip().strip('"') or _open_vault_from_obsidian()
    if not path:
        raise RuntimeError("no Obsidian vault found (⚙ → PLUGIN SETTINGS → OBSIDIAN)")
    p = Path(path).expanduser()
    if not p.is_dir():
        raise RuntimeError(f"the vault folder does not exist: {p}")
    return p.resolve()


def _inside(root: Path, p: Path) -> Path:
    """Resolve `p` and refuse anything outside the vault."""
    p = p.resolve()
    if p != root and root not in p.parents:
        raise RuntimeError("that path is outside the vault")
    return p


def _notes(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for f in filenames:
            if f.lower().endswith(".md"):
                yield Path(dirpath) / f


def _fold(text: str) -> str:
    """Lower case without accents, so 'reuniao' finds 'Reunião'."""
    return "".join(c for c in unicodedata.normalize("NFD", text.lower())
                   if unicodedata.category(c) != "Mn")


def _rel(root: Path, p: Path) -> str:
    return p.relative_to(root).as_posix()[:-3]


def _safe_name(name: str) -> str:
    name = re.sub(r'[\\/:*?"<>|#^\[\]]', " ", name).strip().strip(".")
    return re.sub(r"\s+", " ", name)[:120]


# ── actions ──────────────────────────────────────────────────────────────────

def search(root: Path, query: str, limit: int = 8) -> list[dict]:
    words = [w for w in _fold(query).split() if len(w) > 1]
    if not words:
        return []
    hits = []
    for p in _notes(root):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        title, body = _fold(p.stem), _fold(text)
        if not all(w in title or w in body for w in words):
            continue
        score = sum(3 for w in words if w in title) + sum(body.count(w) for w in words)
        # The snippet comes from the text, not the properties block at the top:
        # "tipo: area cliente: …" says nothing when read out. The properties
        # still count for the match and the score.
        fm = _FRONTMATTER.match(text)
        start = fm.end() if fm else 0
        i = body.find(words[0], start)
        if i < 0:
            i = start
        snippet = re.sub(r"\s+", " ", text[max(start, i - 60): i + 140]).strip()
        hits.append({"note": _rel(root, p), "score": score, "snippet": snippet,
                     "mtime": p.stat().st_mtime})
    hits.sort(key=lambda h: (h["score"], h["mtime"]), reverse=True)
    return hits[:limit]


def find_note(root: Path, name: str) -> Path | None:
    want = _fold(name.strip().removesuffix(".md"))
    notes = list(_notes(root))
    for p in notes:                                     # exact title or path
        if _fold(p.stem) == want or _fold(_rel(root, p)) == want:
            return p
    titles = {_fold(p.stem): p for p in notes}
    close = difflib.get_close_matches(want, list(titles), n=1, cutoff=0.6)
    if close:
        return titles[close[0]]
    contains = [p for p in notes if want and want in _fold(p.stem)]
    return contains[0] if len(contains) == 1 else None


def recent(root: Path, limit: int = 8) -> list[tuple[str, datetime]]:
    notes = sorted(_notes(root), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    return [(_rel(root, p), datetime.fromtimestamp(p.stat().st_mtime)) for p in notes]


def _append(path: Path, text: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    sep = "" if not existing or existing.endswith("\n") else "\n"
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(sep + text.rstrip() + "\n")


def append_daily(root: Path, text: str) -> Path:
    path = _inside(root, root / f"{datetime.now():%Y-%m-%d}.md")
    _append(path, f"- {datetime.now():%H:%M} {text.strip()}")
    return path


def create(root: Path, title: str, text: str, folder: str = "") -> tuple[Path, bool]:
    name = _safe_name(title)
    if not name:
        raise RuntimeError("the note needs a title")
    base = _inside(root, root / folder.strip().strip("/\\")) if folder.strip() else root
    base.mkdir(parents=True, exist_ok=True)
    path = _inside(root, base / f"{name}.md")
    existed = path.exists()
    _append(path, text.strip() if existed else f"# {name}\n\n{text.strip()}")
    return path, existed


# ── settings form ────────────────────────────────────────────────────────────

def test_connection(values: dict):
    try:
        root = vault(values)
        n = sum(1 for _ in _notes(root))
    except Exception as e:
        return False, str(e)
    return True, f"Vault {root.name}: {n} notes."


PLUGIN_SETTINGS = {
    "namespace": NAMESPACE,
    "title": "OBSIDIAN (suas notas)",
    "fields": [
        {"key": "vault_path", "label": "Pasta do cofre (vazio = o cofre aberto no Obsidian)",
         "type": "text", "placeholder": r"C:\Users\voce\Documents\Obsidian Vault"},
    ],
    "action": {"label": "TEST CONNECTION", "run": test_connection},
}


def run(parameters: dict, player=None, session_memory=None) -> str:
    action = str(parameters.get("action") or "").strip().lower()
    query = str(parameters.get("query") or "").strip()
    text = str(parameters.get("text") or "").strip()
    try:
        root = vault()
        if action == "search":
            if not query:
                return "Say what to search for."
            hits = search(root, query)
            if not hits:
                return f"No note in the vault mentions '{query}'."
            return f"Notes that mention '{query}':\n" + "\n".join(
                f"- {h['note']}: …{h['snippet']}…" for h in hits)
        if action == "read":
            p = find_note(root, query)
            if p is None:
                return f"There is no note called '{query}' in the vault."
            body = p.read_text(encoding="utf-8", errors="replace").strip()
            if not body:
                return f"The note {_rel(root, p)} is empty."
            more = f"\n[… {len(body) - MAX_READ} more characters]" if len(body) > MAX_READ else ""
            return f"Note {_rel(root, p)}:\n{body[:MAX_READ]}{more}"
        if action == "recent":
            return "Most recently edited notes:\n" + "\n".join(
                f"- {name} ({when:%d/%m %H:%M})" for name, when in recent(root))
        if action == "append_daily":
            if not text:
                return "Say what to write in today's note."
            p = append_daily(root, text)
            return f"Added to {_rel(root, p)}: {text}"
        if action == "create":
            title = str(parameters.get("title") or "").strip()
            p, existed = create(root, title, text, str(parameters.get("folder") or ""))
            return (f"The note {_rel(root, p)} already existed; the text was added at the end."
                    if existed else f"Created the note {_rel(root, p)}.")
        return "Unknown action. Use search, read, recent, append_daily or create."
    except Exception as e:
        return f"Could not use the Obsidian vault: {str(e)[:150]}. Tell the user; do not guess."

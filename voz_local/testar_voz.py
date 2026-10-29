"""Gera voz_local/teste_voz.wav pelo servidor local e mostra quanto demorou.

Uso (com o Python do Jarvis): .\\.venv\\Scripts\\python.exe voz_local\\testar_voz.py ["frase"]
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import requests  # noqa: E402

from core import brain  # noqa: E402

text = sys.argv[1] if len(sys.argv) > 1 else "Bom dia, senhor. Todos os sistemas estão funcionando."
print("Ligando o servidor de voz (a primeira vez baixa o modelo, uns 3 GB)…", flush=True)
t0 = time.monotonic()
if not brain.ensure_local_tts(wait_s=1800):
    sys.exit("O servidor não ficou pronto — veja voz_local/servidor_voz.log")
print(f"Servidor pronto em {time.monotonic() - t0:.0f} s", flush=True)

t0 = time.monotonic()
r = requests.post(f"{brain.LOCAL_TTS_URL}/tts", json={"text": text}, timeout=120)
r.raise_for_status()
out = Path(__file__).resolve().parent / "teste_voz.wav"
out.write_bytes(r.content)
print(f"Frase gerada em {time.monotonic() - t0:.1f} s -> {out}")

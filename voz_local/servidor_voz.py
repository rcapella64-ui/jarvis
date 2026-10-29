"""Servidor de voz local do Jarvis: clona a voz de voz_local/referencia.wav.

Roda no Python de voz_local/.venv (Chatterbox Multilingual na GPU) e atende só
este PC, em http://127.0.0.1:5005:

    GET  /health          200 quando o modelo já carregou
    POST /tts {"text":…}  devolve um WAV mono 16 bits na voz clonada

O core/brain.py do Jarvis liga este servidor sozinho quando o motor de voz é
"local". Para trocar a voz, substitua referencia.wav (10–30 s de fala limpa)
e reinicie o servidor (ou o Jarvis).

Ajustes opcionais por variável de ambiente:
    JARVIS_VOZ_EXAGERO  0.25–1.0, expressividade (padrão 0.5)
    JARVIS_VOZ_CFG      0.0–1.0, fidelidade ao ritmo da referência (padrão 0.5)
"""

import io
import json
import logging
import os
import sys
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REF = HERE / "referencia.wav"
PORT = 5005
EXAGERO = float(os.environ.get("JARVIS_VOZ_EXAGERO", "0.5"))
CFG = float(os.environ.get("JARVIS_VOZ_CFG", "0.5"))

logging.basicConfig(filename=HERE / "servidor_voz.log", level=logging.INFO,
                    format="%(asctime)s %(message)s")
log = logging.getLogger("voz")

model = None
lock = threading.Lock()


def load() -> None:
    global model
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("carregando Chatterbox Multilingual em %s…", device)
    m = ChatterboxMultilingualTTS.from_pretrained(device=device)
    if not REF.exists():
        raise SystemExit(f"falta o áudio de referência: {REF}")
    m.prepare_conditionals(str(REF), exaggeration=EXAGERO)
    model = m
    log.info("pronto (referência %s)", REF.name)


def synth(text: str) -> bytes:
    with lock:
        wav = model.generate(text, language_id="pt", exaggeration=EXAGERO, cfg_weight=CFG)
    pcm = (wav.squeeze(0).numpy().clip(-1, 1) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(model.sr)
        wf.writeframes(pcm.tobytes())
    return buf.getvalue()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            ok = model is not None
            self._send(200 if ok else 503, b"ok" if ok else b"loading", "text/plain")
        else:
            self._send(404, b"", "text/plain")

    def do_POST(self):
        if self.path != "/tts":
            return self._send(404, b"", "text/plain")
        if model is None:
            return self._send(503, b"loading", "text/plain")
        try:
            n = int(self.headers.get("Content-Length") or 0)
            text = str(json.loads(self.rfile.read(n) or b"{}").get("text") or "").strip()
            if not text:
                return self._send(200, b"", "audio/wav")
            self._send(200, synth(text), "audio/wav")
        except Exception as e:
            log.exception("erro gerando voz")
            self._send(500, str(e).encode("utf-8", "replace"), "text/plain")


def download_only() -> None:
    """`servidor_voz.py --baixar`: fetch the model files without loading them
    (setup.py runs this so the first launch does not wait for the download)."""
    from chatterbox.mtl_tts import REPO_ID
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=REPO_ID, repo_type="model", revision="main",
                      allow_patterns=["ve.pt", "t3_mtl23ls_v2.safetensors", "s3gen.pt",
                                      "grapheme_mtl_merged_expanded_v1.json", "conds.pt",
                                      "Cangjie5_TC.json"],
                      token=os.getenv("HF_TOKEN"))


def main() -> None:
    if "--baixar" in sys.argv:
        download_only()
        return
    try:
        server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    except OSError:
        log.info("porta %d já em uso — outro servidor de voz já está rodando", PORT)
        sys.exit(0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        load()
    except BaseException:
        log.exception("falha ao carregar o modelo")
        server.shutdown()
        raise
    threading.Event().wait()


if __name__ == "__main__":
    main()

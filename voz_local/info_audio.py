"""Mostra duração, taxa e volume do áudio de referência, e gera referencia.wav (mono 24 kHz)."""

import sys
import wave
from pathlib import Path

import miniaudio
import numpy as np

src = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "referencia.mp3")
info = miniaudio.get_file_info(str(src))
print(f"Arquivo: {src.name}")
print(f"Duração: {info.duration:.1f} s | {info.sample_rate} Hz | {info.nchannels} canal(is)")

dec = miniaudio.decode_file(str(src), output_format=miniaudio.SampleFormat.SIGNED16,
                            nchannels=1, sample_rate=24000)
pcm = np.frombuffer(bytes(dec.samples), dtype=np.int16)
print(f"Pico: {np.abs(pcm).max() / 32768:.2f} | RMS: {np.sqrt(np.mean((pcm / 32768) ** 2)):.3f}")

out = src.with_name("referencia.wav")
with wave.open(str(out), "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(24000)
    wf.writeframes(pcm.tobytes())
print(f"Gerado: {out}")

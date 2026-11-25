"""
Captura del dataset propio de acordes (requisito R4), dirigida por el juego.

Rocky muestra un acorde, cuenta 3-2-1 y manda "GRABAR|clase|sesion|guitarrista|guitarra".
Aquí se guardan los siguientes 1.5 s de audio de la guitarra como WAV (16 kHz, 16 bits)
y se anota una fila en metadata.csv.

Estructura en disco (versionada):
    IA/dataset/v1/<clase>/<sesion>_<clase>_<n>.wav
    IA/dataset/v1/metadata.csv   archivo, clase, sesion, guitarrista, guitarra, fuente, rms, pico, fecha
La sesión (una tanda de grabación) es la unidad para separar train/val/test sin fugas.
"""

import csv
import os
import threading
import time
import wave

import numpy as np

VERSION = "v1"
DURACION = 1.5            # s grabados por toma (el entrenamiento recorta ventanas de 1 s)
FS = 16000
NIVEL_MINIMO = 0.004      # RMS por debajo del cual "no se escuchó la guitarra"

AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.normpath(os.path.join(AQUI, "..", "IA", "dataset", VERSION))
_lock = threading.Lock()


def _limpio(t):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in t.strip())[:40] or "anonimo"


def conteos():
    """Tomas guardadas por clase: {'C': 12, 'Am': 9, ...}"""
    import ia_features
    res = {}
    for c in ia_features.CLASES:
        d = os.path.join(CARPETA, c)
        res[c] = len([f for f in os.listdir(d) if f.endswith(".wav")]) if os.path.isdir(d) else 0
    return res


def grabar(anillo, estado, clase, sesion, guitarrista, guitarra, log=print):
    """Espera a que lleguen DURACION segundos nuevos y los guarda. Devuelve (ok, mensaje, rms)."""
    import ia_features
    if clase not in ia_features.CLASES:
        return False, "clase desconocida", 0.0
    n = int(DURACION * FS)
    _, inicio = anillo.ultimas(1)
    limite = time.time() + DURACION + 2.0
    while True:
        _, pos = anillo.ultimas(1)
        if pos - inicio >= n:
            break
        if time.time() > limite:
            return False, "no llega audio de la guitarra", 0.0
        time.sleep(0.02)
    x, pos = anillo.ultimas(pos - inicio)
    x = x[:n] - np.mean(x[:n])
    rms = float(np.sqrt(np.mean(x ** 2)))
    pico = float(np.max(np.abs(x)))
    if clase != "ninguno" and rms < NIVEL_MINIMO:
        return False, "muy bajo: no se escuchó la guitarra", rms
    if pico >= 0.99:
        return False, "saturado: baja el volumen", rms

    sesion, guitarrista, guitarra = _limpio(sesion), _limpio(guitarrista), _limpio(guitarra)
    with _lock:
        d = os.path.join(CARPETA, clase)
        os.makedirs(d, exist_ok=True)
        k = 1
        while os.path.exists(os.path.join(d, f"{sesion}_{clase}_{k:03d}.wav")):
            k += 1
        nombre = f"{sesion}_{clase}_{k:03d}.wav"
        with wave.open(os.path.join(d, nombre), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(FS)
            w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
        meta = os.path.join(CARPETA, "metadata.csv")
        nuevo = not os.path.exists(meta)
        with open(meta, "a", newline="", encoding="utf-8") as fh:
            wr = csv.writer(fh)
            if nuevo:
                wr.writerow(["archivo", "clase", "sesion", "guitarrista", "guitarra", "fuente", "rms", "pico", "fecha"])
            wr.writerow([f"{clase}/{nombre}", clase, sesion, guitarrista, guitarra,
                         estado.get("fuente", ""), f"{rms:.4f}", f"{pico:.4f}", time.strftime("%Y-%m-%d %H:%M:%S")])
    log(f"Dataset {VERSION}: guardado {clase}/{nombre} (nivel {rms:.3f})")
    return True, nombre, rms

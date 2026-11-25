"""
Clasificación de acordes EN TIEMPO REAL en el PC (respaldo y validación de la IA de la ESP32).

Qué hace, cada 250 ms:
  1. Toma el último segundo de audio del buffer circular (el que llena la ESP32 o la interfaz de audio).
  2. Si hay guitarra sonando (RMS sobre la puerta): características 28x48 (ia_features) -> modelo int8
     (el mismo .tflite que se carga en la ESP32) -> probabilidades.
  3. Ventana de votación (estructuras.VotingWindow) -> acorde estable.
  4. Manda a Unity el mismo mensaje que la ESP32:  C1;clase;conf;estable;conf_estable;lat_ms;feat_ms;inf_ms
  5. Guarda cada inferencia en IA/mediciones/pc_<fecha>.csv (latencia, confianza, top-3).

Modos (--ia):
  auto  corre en el PC solo cuando la ESP32 NO está mandando acordes (si la ESP32 los calcula, manda ella).
  pc    siempre en el PC.   esp  nunca (solo reenviar lo de la ESP32).   off  apagado.
"""

import csv
import glob
import os
import socket
import time

import numpy as np

import ia_borde
import ia_features as F
from estructuras import VotingWindow, top_k

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.normpath(os.path.join(AQUI, ".."))
MODELOS = os.path.join(RAIZ, "IA", "modelos")
HOP = 0.25            # s entre inferencias (igual que el firmware)
PUERTA_IA = 0.004     # RMS mínimo para gastar CPU en la red; por debajo se reporta "ninguno"
UNITY = ("127.0.0.1", 5005)


def buscar_modelo(ruta=None):
    """Ruta del .tflite int8 más reciente (o la indicada)."""
    if ruta:
        return ruta
    # Los modelos de prueba (dataset sintético) nunca se cargan solos: darían acordes falsos con una guitarra real.
    c = [r for r in glob.glob(os.path.join(MODELOS, "*", "modelo_int8.tflite"))
         if not os.path.basename(os.path.dirname(r)).startswith("sint")]
    c.sort(key=os.path.getmtime)
    return c[-1] if c else None


def _interprete(ruta):
    for modulo in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            m = __import__(modulo, fromlist=["Interpreter"])
            return m.Interpreter(model_path=ruta)
        except ImportError:
            continue
    import tensorflow as tf
    return tf.lite.Interpreter(model_path=ruta)


class MotorIA:
    def __init__(self, ruta):
        self.ruta = ruta
        self.it = _interprete(ruta)
        self.it.allocate_tensors()
        self.ent = self.it.get_input_details()[0]
        self.sal = self.it.get_output_details()[0]
        self.escala_e, self.cero_e = self.ent["quantization"]
        self.escala_s, self.cero_s = self.sal["quantization"]

    def cuantizar(self, feat):
        return np.clip(np.round(feat / self.escala_e + self.cero_e), -128, 127).astype(np.int8)

    def predecir(self, x):
        """x: 16000 muestras. Devuelve (probabilidades, ms_caracteristicas, ms_inferencia, entrada_int8)."""
        t0 = time.perf_counter()
        feat = F.caracteristicas(x)[None, :, :, None]
        q = self.cuantizar(feat)
        t1 = time.perf_counter()
        self.it.set_tensor(self.ent["index"], q)
        self.it.invoke()
        out = self.it.get_tensor(self.sal["index"])[0].astype(np.float32)
        t2 = time.perf_counter()
        probs = (out - self.cero_s) * self.escala_s
        return probs, (t1 - t0) * 1000, (t2 - t1) * 1000, q


def hilo_ia_pc(anillo, estado, modo="auto", ruta_modelo=None, log=print):
    if modo in ("off", "esp"):
        estado["ia_pc"] = "apagada"
        return
    ruta = buscar_modelo(ruta_modelo)
    if not ruta:
        estado["ia_pc"] = "sin modelo"
        log("IA en el PC: no hay modelo en IA/modelos (entrena con IA/entrenar.py). Solo se usa YIN.")
        return
    try:
        motor = MotorIA(ruta)
    except Exception as e:
        estado["ia_pc"] = "error"
        log("IA en el PC: no pude cargar el modelo:", e.__class__.__name__, e)
        return
    log("IA en el PC lista:", os.path.relpath(ruta, RAIZ))
    voto = VotingWindow(n=5, minimo=3, conf_min=0.5, n_clases=len(F.CLASES))
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    os.makedirs(ia_borde.CARPETA, exist_ok=True)
    archivo = os.path.join(ia_borde.CARPETA, "pc_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
    ninguno = F.CLASES.index("ninguno")
    escrito = False
    proximo = time.time()
    while True:
        proximo += HOP
        espera = proximo - time.time()
        if espera > 0:
            time.sleep(espera)
        else:
            proximo = time.time()
        if modo == "auto" and ia_borde.esp_activa():
            estado["ia_pc"] = "en pausa (calcula la ESP32)"
            continue
        x, pos = anillo.ultimas(F.MUESTRAS)
        if pos < F.MUESTRAS:
            continue                                   # aún no hay 1 s de audio
        rms = float(np.sqrt(np.mean((x[-4000:] - x[-4000:].mean()) ** 2)))
        t0 = time.perf_counter()
        if rms < PUERTA_IA:
            clase, conf, feat_ms, inf_ms, top = ninguno, 1.0, 0.0, 0.0, [(ninguno, 1.0)]
        else:
            probs, feat_ms, inf_ms, _ = motor.predecir(x)
            top = top_k(probs, 3)
            clase, conf = top[0][0], float(max(0.0, min(1.0, top[0][1])))
        voto.agregar(clase, conf)
        estable, conf_voto = voto.votar()
        lat_ms = (time.perf_counter() - t0) * 1000
        estado["ia_pc"] = "activa"
        estado["acorde_pc"] = F.CLASES[estable] if estable >= 0 else ""
        msg = "C1;%s;%.3f;%s;%.2f;%.2f;%.2f;%.2f" % (
            F.CLASES[clase], conf, F.CLASES[estable] if estable >= 0 else "", conf_voto, lat_ms, feat_ms, inf_ms)
        try:
            sock.sendto(msg.encode("ascii"), UNITY)
        except OSError:
            pass
        try:
            with open(archivo, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if not escrito:
                    w.writerow(["fecha", "rms", "clase", "conf", "estable", "conf_voto", "lat_ms", "feat_ms", "inf_ms", "top3"])
                    escrito = True
                w.writerow([time.strftime("%H:%M:%S"), f"{rms:.4f}", F.CLASES[clase], f"{conf:.3f}",
                            F.CLASES[estable] if estable >= 0 else "", f"{conf_voto:.2f}",
                            f"{lat_ms:.2f}", f"{feat_ms:.2f}", f"{inf_ms:.2f}",
                            " ".join(f"{F.CLASES[i]}:{p:.2f}" for i, p in top)])
        except OSError:
            pass

"""
Validación EN LA PLACA: ¿la ESP32-S3 calcula lo mismo que el PC?  (evidencia de R3/R7)

Para cada toma del conjunto de prueba manda 1 s de audio a la ESP32 (comando B2 2B), recibe de vuelta
la entrada int8 que ella calculó (características en C) y la salida int8 de la red (TFLite Micro), y las
compara con lo que calcula el PC (ia_features + el mismo .tflite). Reporta:
  - % de casillas de entrada idénticas (prueba del código C de características)
  - % de tomas con salida idéntica y con el mismo acorde (prueba de TFLite Micro vs PC)
  - exactitud y F1 macro del modelo EN LA PLACA contra las etiquetas reales
Guarda IA/mediciones/validacion_placa_<fecha>.json y .csv

Antes de correrlo: cierra Rocky/rocky.py (el puerto USB solo lo puede abrir un programa).
Uso:  python validar_en_placa.py --puerto COM5 [--dataset v1] [--max 100] [--sintetico]
"""

import argparse
import csv
import json
import os
import struct
import sys
import time
import wave

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.normpath(os.path.join(AQUI, ".."))
sys.path.insert(0, AQUI)
sys.path.insert(0, os.path.join(RAIZ, "IA"))
import ia_features as F      # noqa: E402
import ia_vivo               # noqa: E402

N_ENT = F.TRAMAS * F.BANDAS
N_CLS = len(F.CLASES)


def leer_respuesta(ser, espera=6.0):
    """Busca [A8 8A][n][n bytes] ignorando los paquetes de audio/acorde que sigue mandando la placa."""
    t0, buf = time.time(), bytearray()
    while time.time() - t0 < espera:
        buf += ser.read(max(1, ser.in_waiting))
        i = buf.find(b"\xA8\x8A")
        if i >= 0 and len(buf) >= i + 4:
            n = buf[i + 2] | (buf[i + 3] << 8)
            if len(buf) >= i + 4 + n:
                datos = np.frombuffer(bytes(buf[i + 4:i + 4 + n]), dtype=np.int8)
                return datos[:N_ENT], datos[N_ENT:N_ENT + N_CLS]
        if len(buf) > 200000:
            del buf[:-1000]
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto", required=True)
    ap.add_argument("--dataset", default="v1")
    ap.add_argument("--sintetico", action="store_true")
    ap.add_argument("--max", type=int, default=100)
    ap.add_argument("--modelo")
    a = ap.parse_args()
    import serial
    import entrenar as E
    ruta = ia_vivo.buscar_modelo(a.modelo)
    if not ruta:
        sys.exit("No hay modelo en IA/modelos. Entrena primero con IA/entrenar.py")
    carpeta = os.path.join(RAIZ, "IA", "dataset", "sintetico" if a.sintetico else a.dataset)
    filas = E.cargar_dataset(carpeta)
    if not filas:
        sys.exit(f"No hay tomas en {carpeta}")
    asign, aviso = E.particionar(filas)
    prueba = [f for f in filas if asign[f["archivo"]] == "test"][:a.max]
    print(f"Modelo: {ruta}\nTomas de prueba: {len(prueba)}  ({aviso})")
    motor = ia_vivo.MotorIA(ruta)
    ser = serial.Serial(a.puerto, 921600, timeout=0.05)
    time.sleep(1.5)
    ser.reset_input_buffer()

    reales, p_placa, p_pc, ent_ok, sal_ok, filas_csv, sin_resp = [], [], [], 0, 0, [], 0
    for f in prueba:
        x16 = E.leer_wav(os.path.join(carpeta, f["archivo"]))[:F.MUESTRAS]
        x16 = np.pad(x16, (0, F.MUESTRAS - len(x16)))
        q16 = np.clip(np.round(x16 * 32768), -32768, 32767).astype("<i2")
        ser.write(b"\xB2\x2B" + struct.pack("<H", F.MUESTRAS * 2) + q16.tobytes())
        ent, sal = leer_respuesta(ser)
        if ent is None:
            sin_resp += 1
            continue
        probs, _, _, qpc = motor.predecir(q16.astype(np.float32) / 32768)
        it = motor.it
        salida_pc = np.clip(np.round(probs / motor.escala_s + motor.cero_s), -128, 127).astype(np.int8)
        iguales_ent = int((ent == qpc.reshape(-1)).sum())
        ent_ok += iguales_ent == N_ENT
        sal_ok += bool(np.array_equal(sal, salida_pc))
        y = F.CLASES.index(f["clase"])
        reales.append(y)
        p_placa.append(int(np.argmax(sal)))
        p_pc.append(int(np.argmax(salida_pc)))
        filas_csv.append([f["archivo"], f["clase"], F.CLASES[p_placa[-1]], F.CLASES[p_pc[-1]],
                          N_ENT - iguales_ent, int(np.abs(sal.astype(int) - salida_pc.astype(int)).max())])
    ser.close()
    n = len(reales)
    if n == 0:
        sys.exit("La placa no respondió. ¿Está cargado rocky_ia.ino y cerraste rocky.py?")
    m = E.metricas(np.array(reales), np.array(p_placa), N_CLS)
    rep = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "modelo": os.path.relpath(ruta, RAIZ),
           "dataset": "sintetico" if a.sintetico else a.dataset, "tomas": n, "sin_respuesta": sin_resp,
           "entrada_identica_pct": round(100 * ent_ok / n, 2), "salida_identica_pct": round(100 * sal_ok / n, 2),
           "mismo_acorde_pc_vs_placa_pct": round(100 * float(np.mean(np.array(p_placa) == np.array(p_pc))), 2),
           "exactitud_placa": m["exactitud"], "f1_macro_placa": m["f1_macro"], "matriz_confusion_placa": m["matriz_confusion"]}
    os.makedirs(os.path.join(RAIZ, "IA", "mediciones"), exist_ok=True)
    base = os.path.join(RAIZ, "IA", "mediciones", "validacion_placa_" + time.strftime("%Y%m%d_%H%M%S"))
    json.dump(rep, open(base + ".json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    with open(base + ".csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["archivo", "real", "placa", "pc", "casillas_entrada_distintas", "max_dif_salida"])
        w.writerows(filas_csv)
    print(json.dumps({k: v for k, v in rep.items() if k != "matriz_confusion_placa"}, indent=1, ensure_ascii=False))
    print("Guardado:", base + ".json")


if __name__ == "__main__":
    main()

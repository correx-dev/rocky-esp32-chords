"""
Diagnóstico de la cadena de hardware EN VIVO (guitarra -> TL072 -> PCM1808 -> ESP32-S3 -> USB).

Lee la ESP32 durante unos segundos y mide, con datos reales:
  - paquetes por segundo (esperado 250 con BLOQUE=64 a 16 kHz) y paquetes perdidos (número de secuencia)
  - componente continua (DC), RMS, pico y % de muestras saturadas
  - piso de ruido (RMS en silencio) y nivel con la guitarra tocando -> relación señal/ruido real en dB
  - nota detectada con YIN (para comprobar que afina)
Dos fases guiadas:  1) silencio (guitarra sin tocar)   2) tocar la 6.ª cuerda al aire y las demás.
Guarda IA/mediciones/hardware_<fecha>.json: es la evidencia de que el hardware funciona (R1/R2).

Uso:  python diagnostico_hw.py [--puerto COM5] [--seg 5]      (cerrar rocky.py antes)
"""

import argparse
import json
import os
import struct
import sys
import time

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)


def capturar(ser, seg):
    buf, audio, seqs = bytearray(), [], []
    t0 = time.time()
    while time.time() - t0 < seg:
        buf += ser.read(max(1, ser.in_waiting))
        while True:
            i = buf.find(b"\xA5\x5A")
            if i < 0:
                del buf[:-1]
                break
            if i > 0:
                del buf[:i]
            if len(buf) < 6:
                break
            seq, n = struct.unpack_from("<HH", buf, 2)
            if n == 0 or n > 2048:
                del buf[:2]
                continue
            if len(buf) < 6 + 2 * n:
                break
            audio.append(np.frombuffer(bytes(buf[6:6 + 2 * n]), dtype="<i2"))
            seqs.append(seq)
            del buf[:6 + 2 * n]
    dur = time.time() - t0
    x = np.concatenate(audio).astype(np.float32) / 32768 if audio else np.zeros(0, np.float32)
    perdidos = sum(((b - a - 1) & 0xFFFF) for a, b in zip(seqs, seqs[1:]))
    return x, len(seqs) / dur, perdidos, dur


def resumen(x):
    if len(x) == 0:
        return {}
    dc = float(x.mean())
    y = x - dc
    return {"dc": round(dc, 5), "rms": round(float(np.sqrt(np.mean(y ** 2))), 5), "pico": round(float(np.abs(x).max()), 4),
            "saturadas_pct": round(100 * float(np.mean(np.abs(x) > 0.98)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--puerto")
    ap.add_argument("--seg", type=float, default=5)
    a = ap.parse_args()
    import serial
    import rocky
    puerto = a.puerto or rocky.buscar_puerto()
    if not puerto:
        sys.exit("No encuentro la ESP32. Revisa el cable USB o usa --puerto COMx")
    ser = serial.Serial(puerto, 921600, timeout=0.05)
    time.sleep(1.0)
    ser.reset_input_buffer()
    rep = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "puerto": puerto}

    input("FASE 1: NO toques la guitarra (silencio). Enter para medir el piso de ruido... ")
    x, pps, perd, dur = capturar(ser, a.seg)
    rep["silencio"] = resumen(x)
    rep["paquetes_por_s"] = round(pps, 1)
    rep["paquetes_perdidos_silencio"] = perd
    print("  silencio:", rep["silencio"], f"· {pps:.0f} paquetes/s (esperado 250) · perdidos {perd}")

    input("FASE 2: toca las 6 cuerdas al aire, una por una, con fuerza media. Enter para medir... ")
    x2, pps2, perd2, _ = capturar(ser, a.seg * 2)
    rep["guitarra"] = resumen(x2)
    rep["paquetes_perdidos_guitarra"] = perd2
    if len(x2) > 2048:
        notas = []
        for i in range(0, len(x2) - 2048, 4096):
            seg = x2[i:i + 2048] - x2[i:i + 2048].mean()
            if np.sqrt(np.mean(seg ** 2)) > 0.01:
                f, cl = rocky.yin(seg)
                if f and cl > 0.8:
                    m = int(round(69 + 12 * np.log2(f / 440.0)))
                    notas.append(rocky.NOMBRES[m % 12] + str(m // 12 - 1))
        rep["notas_detectadas"] = sorted(set(notas))
    rs, rg = rep["silencio"].get("rms", 0), rep["guitarra"].get("rms", 0)
    if rs > 0 and rg > 0:
        rep["snr_db"] = round(20 * np.log10(rg / rs), 1)
    print("  guitarra:", rep["guitarra"], "· notas:", rep.get("notas_detectadas"), "· SNR:", rep.get("snr_db"), "dB")
    ser.close()

    avisos = []
    if rep["paquetes_por_s"] < 240:
        avisos.append("Menos de 250 paquetes/s: el USB o la lectura de I2S no van al ritmo.")
    if rep["guitarra"].get("saturadas_pct", 0) > 0.1:
        avisos.append("Hay saturación: baja la ganancia del TL072 o GANANCIA en config.h.")
    if rep["guitarra"].get("rms", 0) < 0.01:
        avisos.append("Señal muy débil: revisa el preamplificador, el cable o sube GANANCIA.")
    if abs(rep["silencio"].get("dc", 0)) > 0.05:
        avisos.append("Componente continua alta: revisa el acoplamiento (condensador de entrada del PCM1808).")
    rep["avisos"] = avisos
    carpeta = os.path.join(AQUI, "..", "IA", "mediciones")
    os.makedirs(carpeta, exist_ok=True)
    ruta = os.path.join(carpeta, "hardware_" + time.strftime("%Y%m%d_%H%M%S") + ".json")
    json.dump(rep, open(ruta, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("Avisos:", avisos or "ninguno")
    print("Guardado:", os.path.normpath(ruta))


if __name__ == "__main__":
    main()

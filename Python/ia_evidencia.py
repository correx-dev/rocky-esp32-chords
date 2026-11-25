"""
Reúne TODA la evidencia real del proyecto en un solo informe (docs/EVIDENCIA.md + docs/evidencia.json).

Recorre: dataset propio, modelos entrenados, mediciones de la ESP32 (recursos), mediciones del PC (latencia en
vivo), validación en la placa, diagnóstico de hardware, firmware, historial de Git y el código. Todo lo que no
existe queda marcado como FALTA (no se inventa nada). Los datos sintéticos se marcan como "no es evidencia".

Uso:  python Python/ia_evidencia.py          (desde la carpeta del proyecto)
"""

import csv
import glob
import json
import os
import subprocess
import sys
import time
import wave

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.normpath(os.path.join(AQUI, ".."))
IA = os.path.join(RAIZ, "IA")
DOCS = os.path.join(RAIZ, "docs")


def leer_json(r):
    try:
        return json.load(open(r, encoding="utf-8"))
    except (OSError, ValueError):
        return None


def dataset():
    res = []
    for d in sorted(glob.glob(os.path.join(IA, "dataset", "*"))):
        if not os.path.isdir(d):
            continue
        nombre = os.path.basename(d)
        meta = os.path.join(d, "metadata.csv")
        filas = list(csv.DictReader(open(meta, encoding="utf-8"))) if os.path.exists(meta) else []
        por_clase, ses, gui, guit, segundos = {}, set(), set(), set(), 0.0
        for f in filas:
            por_clase[f["clase"]] = por_clase.get(f["clase"], 0) + 1
            ses.add(f.get("sesion")); gui.add(f.get("guitarrista")); guit.add(f.get("guitarra"))
            try:
                with wave.open(os.path.join(d, f["archivo"]), "rb") as w:
                    segundos += w.getnframes() / w.getframerate()
            except (OSError, wave.Error):
                pass
        res.append({"version": nombre, "sintetico": nombre.startswith("sint") or "sintetico" in gui, "tomas": len(filas),
                    "por_clase": por_clase, "sesiones": sorted(map(str, ses)), "guitarristas": sorted(map(str, gui)),
                    "guitarras": sorted(map(str, guit)), "minutos_audio": round(segundos / 60, 2)})
    return res


def modelos():
    res = []
    for d in sorted(glob.glob(os.path.join(IA, "modelos", "*"))):
        r = leer_json(os.path.join(d, "reporte.json"))
        if r:
            r["carpeta"] = os.path.basename(d)
            r["sintetico"] = str(r.get("dataset", "")).startswith("sint")
            r.pop("clases", None)
            res.append(r)
    return res


def _stats(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    return {"media": round(float(v.mean()), 3), "desv": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 3), "min": round(float(v.min()), 3),
            "max": round(float(v.max()), 3), "n": int(len(v))} if len(v) else None


def mediciones():
    out = {"recursos_esp32": [], "pc_vivo": [], "validacion_placa": [], "hardware": []}
    for f in sorted(glob.glob(os.path.join(IA, "mediciones", "recursos_*.csv"))):
        filas = list(csv.DictReader(open(f, encoding="utf-8")))
        if filas:
            u = filas[-1]
            out["recursos_esp32"].append({"archivo": os.path.basename(f), "reportes": len(filas), "ultimo": u})
    for f in sorted(glob.glob(os.path.join(IA, "mediciones", "pc_*.csv"))):
        filas = list(csv.DictReader(open(f, encoding="utf-8")))
        act = [r for r in filas if float(r["inf_ms"]) > 0]
        out["pc_vivo"].append({"archivo": os.path.basename(f), "inferencias": len(act),
                               "latencia_total_ms": _stats([float(r["lat_ms"]) for r in act]),
                               "caracteristicas_ms": _stats([float(r["feat_ms"]) for r in act]),
                               "red_ms": _stats([float(r["inf_ms"]) for r in act])})
    for f in sorted(glob.glob(os.path.join(IA, "mediciones", "validacion_placa_*.json"))):
        r = leer_json(f)
        if r:
            r["archivo"] = os.path.basename(f); r.pop("matriz_confusion_placa", None); out["validacion_placa"].append(r)
    for f in sorted(glob.glob(os.path.join(IA, "mediciones", "hardware_*.json"))):
        r = leer_json(f)
        if r:
            r["archivo"] = os.path.basename(f); out["hardware"].append(r)
    return out


def git():
    try:
        sh = lambda *a: subprocess.run(["git", "-C", RAIZ, *a], capture_output=True, text=True, timeout=10).stdout.strip()
        if not sh("rev-parse", "--is-inside-work-tree"):
            return None
        return {"remoto": sh("config", "--get", "remote.origin.url") or None, "commits": int(sh("rev-list", "--count", "HEAD") or 0),
                "ultimo": sh("log", "-1", "--format=%h %ad %s", "--date=short")}
    except Exception:
        return None


def codigo():
    res = {}
    for carpeta, pats in (("Python", ["*.py"]), ("Firmware/rocky_ia", ["*.ino", "*.h"]), ("IA", ["*.py"]), ("tests", ["*.py", "*.cpp"])):
        n = 0
        for p in pats:
            for f in glob.glob(os.path.join(RAIZ, carpeta, p)):
                n += sum(1 for _ in open(f, encoding="utf-8", errors="ignore"))
        res[carpeta] = n
    return res


def pct(v):
    try:
        return f"{float(v):.2f} %"
    except (TypeError, ValueError):
        return "?"


def main():
    d, m, me, g, c = dataset(), modelos(), mediciones(), git(), codigo()
    reales_d = [x for x in d if not x["sintetico"]]
    reales_m = [x for x in m if not x["sintetico"]]
    L = [f"# Evidencia del proyecto Rocky — generada {time.strftime('%Y-%m-%d %H:%M')}", "",
         "Todo lo que aparece aquí se leyó de archivos reales del proyecto. **FALTA** = no existe todavía (no se inventa).", ""]

    L += ["## 1. Dataset propio (R4)"]
    if not reales_d:
        L.append("**FALTA**: no hay un dataset real en `IA/dataset/`. Graba con el modo «Entrena a la IA» del juego.")
    for x in d:
        L.append(f"- `{x['version']}`{' — **SINTÉTICO: solo prueba el flujo, NO es evidencia**' if x['sintetico'] else ''}: "
                 f"{x['tomas']} tomas, {x['minutos_audio']} min de audio, {len(x['sesiones'])} sesiones, "
                 f"guitarristas {x['guitarristas']}, guitarras {x['guitarras']}")
        L.append("  - por clase: " + ", ".join(f"{k}={v}" for k, v in sorted(x["por_clase"].items())))
    L.append("")

    L += ["## 2. Modelo entrenado (R3)"]
    if not reales_m:
        L.append("**FALTA**: no hay un modelo entrenado con el dataset real.")
    for x in m:
        f32, i8 = x["float32"], x["int8"]
        L.append(f"- `{x['carpeta']}`{' — **SINTÉTICO, no es evidencia**' if x['sintetico'] else ''}: {x['parametros']} parámetros, "
                 f"{x['epocas']} épocas, {x['segundos_entrenamiento']} s.")
        L.append(f"  - exactitud test float32 {f32['exactitud']:.3f} / int8 {i8['exactitud']:.3f} · F1 macro {f32['f1_macro']:.3f} / {i8['f1_macro']:.3f} "
                 f"· pérdida por cuantización {x['perdida_por_cuantizacion_pp']} pp · tamaño {f32['tamano_tflite_bytes'] / 1024:.1f} KB -> {i8['tamano_tflite_bytes'] / 1024:.1f} KB (x{x['reduccion_tamano']})")
        L.append(f"  - split: {x.get('split', '')}; ventanas {x['ventanas']}")
    L.append("")

    L += ["## 3. Recursos medidos en la ESP32-S3 (R7)"]
    if not me["recursos_esp32"]:
        L.append("**FALTA**: no hay `IA/mediciones/recursos_*.csv`. Carga `Firmware/rocky_ia`, conecta la ESP32 a Rocky y déjala ~10 min.")
    for r in me["recursos_esp32"]:
        u = r["ultimo"]
        L.append(f"- `{r['archivo']}` ({r['reportes']} reportes, último): latencia {u.get('lat_ms')} ± {u.get('lat_ms_desv')} ms (n={u.get('n')}), "
                 f"características {u.get('feat_ms')} ms, red {u.get('inf_ms')} ms, ciclos/inferencia {u.get('ciclos_inf')}")
        L.append(f"  - Flash app {pct(u.get('flash_app_pct'))} de la partición · modelo {pct(u.get('modelo_pct_flash_chip'))} del chip · RAM interna {pct(u.get('ram_interna_pct'))} · "
                 f"arena {u.get('arena_usada')} B ({pct(u.get('arena_pct'))}) · carga núcleo 0 {u.get('carga_nucleo0_pct')} % · potencia {u.get('potencia_mw')} mW")
    L.append("")

    L += ["## 4. Validación en la placa: ¿calcula lo mismo que el PC?"]
    if not me["validacion_placa"]:
        L.append("**FALTA**: ejecuta `python Python/validar_en_placa.py --puerto COMx`.")
    for r in me["validacion_placa"]:
        L.append(f"- `{r['archivo']}`: {r['tomas']} tomas · entrada idéntica {r['entrada_identica_pct']} % · salida idéntica {r['salida_identica_pct']} % · "
                 f"mismo acorde PC/placa {r['mismo_acorde_pc_vs_placa_pct']} % · **exactitud en la placa {r['exactitud_placa']:.3f}**, F1 {r['f1_macro_placa']:.3f}")
    L.append("")

    L += ["## 5. Tiempo real en el PC (latencia de la IA en vivo)"]
    if not me["pc_vivo"]:
        L.append("**FALTA**: abre Rocky con la guitarra conectada y toca; se crea `IA/mediciones/pc_*.csv`.")
    for r in me["pc_vivo"]:
        L.append(f"- `{r['archivo']}`: {r['inferencias']} inferencias · latencia {r['latencia_total_ms']} · características {r['caracteristicas_ms']} · red {r['red_ms']}")
    L.append("")

    L += ["## 6. Hardware (diagnóstico real de la cadena de audio)"]
    if not me["hardware"]:
        L.append("**FALTA**: ejecuta `python Python/diagnostico_hw.py` con la guitarra conectada.")
    for r in me["hardware"]:
        L.append(f"- `{r['archivo']}`: {r.get('paquetes_por_s')} paquetes/s · perdidos {r.get('paquetes_perdidos_silencio', 0)}+{r.get('paquetes_perdidos_guitarra', 0)} · "
                 f"ruido RMS {r.get('silencio', {}).get('rms')} · guitarra RMS {r.get('guitarra', {}).get('rms')} · SNR {r.get('snr_db')} dB · notas {r.get('notas_detectadas')} · avisos {r.get('avisos')}")
    L.append("")

    L += ["## 7. Repositorio y código"]
    L.append(f"- Git: {g if g else '**FALTA**: la carpeta no es un repositorio Git (el requisito pide un repositorio público)'}")
    L.append("- Líneas de código por carpeta: " + ", ".join(f"{k}={v}" for k, v in c.items()))
    L.append("")
    os.makedirs(DOCS, exist_ok=True)
    open(os.path.join(DOCS, "EVIDENCIA.md"), "w", encoding="utf-8").write("\n".join(L))
    json.dump({"dataset": d, "modelos": m, "mediciones": me, "git": g, "codigo": c}, open(os.path.join(DOCS, "evidencia.json"), "w", encoding="utf-8"),
              indent=1, ensure_ascii=False, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()

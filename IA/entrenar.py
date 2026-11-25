"""
Entrenamiento del clasificador de acordes para la ESP32-S3 (requisitos R3 y R4).

Uso:
    python entrenar.py                     # entrena con IA/dataset/v1
    python entrenar.py --version v2        # otra versión del dataset
    python entrenar.py --sintetico         # prueba del flujo con acordes generados por código (NO es el dataset real)

Qué hace:
  1. Lee el dataset propio (WAV de 1.5 s grabados con el modo «Entrena a la IA» + metadata.csv).
  2. Separa train / val / test POR SESIÓN de grabación, para que tomas casi idénticas no queden
     a la vez en entrenamiento y en prueba (fuga de datos).
  3. Extrae las características (Python/ia_features.py: 28 tramas x 48 semitonos).
  4. Entrena una CNN pequeña en Keras (float32).
  5. La convierte a TensorFlow Lite cuantizada en int8 (pesos y activaciones).
  6. Compara exactitud y F1 del modelo float contra el int8 (la pérdida por cuantización).
  7. Exporta Firmware/rocky_ia/modelo.h para compilarlo en la ESP32 y un reporte (JSON + Markdown + figuras).
"""

import argparse
import csv
import json
import os
import sys
import time
import wave

import numpy as np

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)
sys.path.insert(0, os.path.join(RAIZ, "Python"))
import ia_features as F  # noqa: E402

FS = F.FS
SEMILLA = 42


# ---------------------------------------------------------------- dataset
def leer_wav(ruta):
    with wave.open(ruta, "rb") as w:
        x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return x


def cargar_dataset(carpeta):
    filas = []
    meta = os.path.join(carpeta, "metadata.csv")
    if os.path.exists(meta):
        with open(meta, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if os.path.exists(os.path.join(carpeta, r["archivo"])):
                    filas.append(r)
    else:                                           # sin CSV: la carpeta es la clase, el prefijo la sesión
        for c in F.CLASES:
            d = os.path.join(carpeta, c)
            if os.path.isdir(d):
                for a in sorted(os.listdir(d)):
                    if a.endswith(".wav"):
                        filas.append({"archivo": f"{c}/{a}", "clase": c, "sesion": a.split("_")[0], "guitarrista": "?"})
    return filas


def particionar(filas, por="sesion"):
    """Asigna train/val/test por grupo (sesión o guitarrista). Devuelve dict archivo->partición y un aviso."""
    grupos = sorted({f[por] for f in filas})
    aviso = ""
    if len(grupos) >= 3:
        rng = np.random.default_rng(SEMILLA)
        orden = list(rng.permutation(grupos))
        test, val = {orden[0]}, {orden[1]}
        return {f["archivo"]: ("test" if f[por] in test else "val" if f[por] in val else "train") for f in filas}, \
            f"split por {por}: test={sorted(map(str, test))}, val={sorted(map(str, val))}, train={sorted(map(str, set(grupos) - test - val))}"
    # Pocas sesiones: estratificado por clase (riesgo de fuga, se documenta)
    aviso = (f"AVISO: solo hay {len(grupos)} {por}(es); se usó un split estratificado por archivo. "
             f"Graba al menos 3 sesiones para evitar fugas de datos.")
    rng = np.random.default_rng(SEMILLA)
    asign = {}
    for c in F.CLASES:
        archivos = [f["archivo"] for f in filas if f["clase"] == c]
        rng.shuffle(archivos)
        n = len(archivos)
        nt, nv = max(1, round(n * 0.15)) if n >= 5 else 0, max(1, round(n * 0.15)) if n >= 5 else 0
        for i, a in enumerate(archivos):
            asign[a] = "test" if i < nt else "val" if i < nt + nv else "train"
    return asign, aviso


def ventanas(x, entrenamiento, rng):
    """Recortes de 1 s de una toma de 1.5 s. En train, desplazados y con ruido (aumento de datos)."""
    n = F.MUESTRAS
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    sobra = len(x) - n
    if entrenamiento:
        inicios = np.linspace(0, sobra, 6).astype(int)
    else:
        inicios = np.linspace(0, sobra, 3).astype(int)    # como en el firmware: ventanas cada ~250 ms
    salida = []
    for i in inicios:
        v = x[i:i + n].copy()
        if entrenamiento:
            v *= rng.uniform(0.3, 1.5)
            v += rng.normal(0, rng.uniform(0.0005, 0.004), n).astype(np.float32)
        salida.append(F.caracteristicas(v))
    return salida


def construir_xy(filas, asign, carpeta):
    rng = np.random.default_rng(SEMILLA)
    datos = {p: ([], []) for p in ("train", "val", "test")}
    for f in filas:
        p = asign[f["archivo"]]
        x = leer_wav(os.path.join(carpeta, f["archivo"]))
        y = F.CLASES.index(f["clase"])
        for feat in ventanas(x, p == "train", rng):
            datos[p][0].append(feat)
            datos[p][1].append(y)
    return {p: (np.array(X, dtype=np.float32)[..., None], np.array(Y, dtype=np.int64)) for p, (X, Y) in datos.items()}


# ---------------------------------------------------------------- dataset sintético (solo para probar el flujo)
FORMAS = {"C": [-1, 3, 2, 0, 1, 0], "D": [-1, -1, 0, 2, 3, 2], "E": [0, 2, 2, 1, 0, 0], "G": [3, 2, 0, 0, 0, 3],
          "A": [-1, 0, 2, 2, 2, 0], "Am": [-1, 0, 2, 2, 1, 0], "Em": [0, 2, 2, 0, 0, 0], "Dm": [-1, -1, 0, 2, 3, 1],
          "F": [1, 3, 3, 2, 1, 1], "B7": [-1, 2, 1, 2, 0, 2]}
AL_AIRE = [40, 45, 50, 55, 59, 64]


def cuerda(midi, dur, rng):
    """Karplus-Strong: imita una cuerda pulsada."""
    f = 440 * 2 ** ((midi - 69) / 12) * (1 + rng.normal(0, 0.003))
    p = int(FS / f)
    buf = rng.uniform(-1, 1, p)
    out = np.zeros(int(dur * FS))
    for i in range(len(out)):
        out[i] = buf[i % p]
        buf[i % p] = 0.996 * 0.5 * (buf[i % p] + buf[(i + 1) % p])
    return out


def generar_sintetico(carpeta, por_clase=40, sesiones=4):
    rng = np.random.default_rng(1)
    os.makedirs(carpeta, exist_ok=True)
    with open(os.path.join(carpeta, "metadata.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["archivo", "clase", "sesion", "guitarrista", "guitarra", "fuente", "rms", "pico", "fecha"])
        for c in F.CLASES:
            os.makedirs(os.path.join(carpeta, c), exist_ok=True)
            for k in range(por_clase):
                ses = f"s{k % sesiones + 1}"
                dur = 1.5
                x = np.zeros(int(dur * FS))
                if c == "ninguno":
                    x += rng.normal(0, rng.uniform(0.001, 0.02), len(x))
                    if rng.random() < 0.5:                       # golpe apagado
                        x[: 800] += rng.normal(0, 0.2, 800) * np.exp(-np.arange(800) / 150)
                else:
                    ataque = int(rng.uniform(0.0, 0.25) * FS)
                    for s, tr in enumerate(FORMAS[c]):
                        if tr < 0:
                            continue
                        retardo = ataque + int(s * rng.uniform(0.005, 0.02) * FS)
                        nota = cuerda(AL_AIRE[s] + tr, dur, rng) * rng.uniform(0.5, 1.0)
                        x[retardo:] += nota[: len(x) - retardo]
                    x = x / (np.max(np.abs(x)) + 1e-9) * rng.uniform(0.2, 0.8)
                    x += rng.normal(0, 0.003, len(x))
                nombre = f"{ses}_{c}_{k:03d}.wav"
                with wave.open(os.path.join(carpeta, c, nombre), "wb") as wv:
                    wv.setnchannels(1)
                    wv.setsampwidth(2)
                    wv.setframerate(FS)
                    wv.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
                w.writerow([f"{c}/{nombre}", c, ses, "sintetico", "karplus", "sintetico", "", "", ""])


# ---------------------------------------------------------------- modelo
def crear_modelo(nclases):
    import tensorflow as tf
    L = tf.keras.layers
    m = tf.keras.Sequential([
        L.Input(shape=(F.TRAMAS, F.BANDAS, 1)),
        L.Conv2D(16, 3, padding="same", activation="relu"),
        L.MaxPooling2D(2),
        L.Conv2D(32, 3, padding="same", activation="relu"),
        L.MaxPooling2D(2),
        L.Conv2D(32, 3, padding="same", activation="relu"),
        L.GlobalAveragePooling2D(),
        L.Dropout(0.3),
        L.Dense(nclases, activation="softmax"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return m


def metricas(y, p, n):
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    return {"exactitud": float(accuracy_score(y, p)),
            "f1_macro": float(f1_score(y, p, average="macro", labels=list(range(n)), zero_division=0)),
            "matriz_confusion": confusion_matrix(y, p, labels=list(range(n))).tolist()}


def evaluar_tflite(contenido, X):
    import tensorflow as tf
    it = tf.lite.Interpreter(model_content=contenido)
    it.allocate_tensors()
    ent, sal = it.get_input_details()[0], it.get_output_details()[0]
    escala, cero = ent["quantization"]
    preds = []
    t0 = time.perf_counter()
    for x in X:
        if ent["dtype"] == np.int8:
            q = np.clip(np.round(x / escala + cero), -128, 127).astype(np.int8)
        else:
            q = x.astype(np.float32)
        it.set_tensor(ent["index"], q[None])
        it.invoke()
        preds.append(int(np.argmax(it.get_tensor(sal["index"])[0])))
    ms = (time.perf_counter() - t0) / max(len(X), 1) * 1000
    return np.array(preds), ms, (float(escala), int(cero))


def exportar_header(contenido, ruta, info):
    bytes_c = ",".join(f"0x{b:02x}" for b in contenido)
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("// Generado por IA/entrenar.py: NO editar a mano.\n")
        fh.write(f"// Dataset {info['dataset']} · {info['fecha']} · exactitud int8 en test {info['exactitud_int8']:.3f}\n")
        fh.write("#pragma once\n#include <stdint.h>\n\n")
        fh.write(f"#define IA_MODELO_BYTES {len(contenido)}\n")
        fh.write("alignas(16) static const unsigned char g_modelo[IA_MODELO_BYTES] = {\n")
        for i in range(0, len(bytes_c), 96 * 5):
            fh.write("  " + bytes_c[i:i + 96 * 5] + "\n")
        fh.write("};\n")


def figuras(hist, mats, salida):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(hist["loss"], label="train"); ax[0].plot(hist["val_loss"], label="val"); ax[0].set_title("Loss"); ax[0].legend()
    ax[1].plot(hist["accuracy"], label="train"); ax[1].plot(hist["val_accuracy"], label="val"); ax[1].set_title("Accuracy"); ax[1].legend()
    for a in ax:
        a.set_xlabel("Epoch")
    fig.tight_layout(); fig.savefig(os.path.join(salida, "curvas_entrenamiento.png"), dpi=150); plt.close(fig)
    for nombre, m in mats.items():
        m = np.array(m)
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(F.CLASES))); ax.set_xticklabels(F.CLASES, rotation=45)
        ax.set_yticks(range(len(F.CLASES))); ax.set_yticklabels(F.CLASES)
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                if m[i, j]:
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=7, color="white" if m[i, j] > m.max() / 2 else "black")
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"Confusion matrix ({nombre}, test)")
        fig.tight_layout(); fig.savefig(os.path.join(salida, f"matriz_confusion_{nombre}.png"), dpi=150); plt.close(fig)


# ---------------------------------------------------------------- principal
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v1")
    ap.add_argument("--por", default="sesion", choices=["sesion", "guitarrista"])
    ap.add_argument("--epocas", type=int, default=60)
    ap.add_argument("--sintetico", action="store_true", help="probar el flujo con acordes sintéticos")
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEMILLA)

    if args.sintetico:
        carpeta = os.path.join(AQUI, "dataset", "sintetico")
        if not os.path.exists(os.path.join(carpeta, "metadata.csv")):
            print("Generando dataset sintético de prueba...")
            generar_sintetico(carpeta)
        etiqueta = "sintetico"
    else:
        carpeta = os.path.join(AQUI, "dataset", args.version)
        etiqueta = args.version
    filas = cargar_dataset(carpeta)
    if not filas:
        sys.exit(f"No hay tomas en {carpeta}. Graba el dataset con el modo «Entrena a la IA» de Rocky.")

    asign, aviso = particionar(filas, args.por)
    print(aviso)
    conteo = {p: {c: 0 for c in F.CLASES} for p in ("train", "val", "test")}
    for f in filas:
        conteo[asign[f["archivo"]]][f["clase"]] += 1
    for p in conteo:
        print(f"  {p:5s}", conteo[p])

    print("Extrayendo características...")
    D = construir_xy(filas, asign, carpeta)
    Xtr, ytr = D["train"]
    Xva, yva = D["val"] if len(D["val"][0]) else D["train"]
    Xte, yte = D["test"] if len(D["test"][0]) else D["val"]
    print(f"  ventanas: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}")

    n = len(F.CLASES)
    pesos = {i: len(ytr) / (n * max(1, int((ytr == i).sum()))) for i in range(n)}
    modelo = crear_modelo(n)
    modelo.summary()
    cb = [tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_loss")]
    t0 = time.time()
    h = modelo.fit(Xtr, ytr, validation_data=(Xva, yva), epochs=args.epocas, batch_size=32, class_weight=pesos,
                   callbacks=cb, verbose=2)
    t_entreno = time.time() - t0

    p_float = np.argmax(modelo.predict(Xte, verbose=0), axis=1)
    m_float = metricas(yte, p_float, n)

    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    tfl_float = conv.convert()

    def representativo():
        idx = np.random.default_rng(0).choice(len(Xtr), min(300, len(Xtr)), replace=False)
        for i in idx:
            yield [Xtr[i:i + 1]]

    conv = tf.lite.TFLiteConverter.from_keras_model(modelo)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    conv.representative_dataset = representativo
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8
    conv.inference_output_type = tf.int8
    # Hallazgo del proyecto de heladas en la misma placa: con cuantización por canal en las capas densas, TFLite Micro
    # (Chirale) daba salidas distintas a las del PC. Se desactiva solo para las densas; las conv siguen por canal.
    conv._experimental_disable_per_channel_quantization_for_dense_layers = True
    tfl_int8 = conv.convert()

    p_tfl_float, ms_float, _ = evaluar_tflite(tfl_float, Xte)
    p_int8, ms_int8, cuant = evaluar_tflite(tfl_int8, Xte)
    m_int8 = metricas(yte, p_int8, n)

    fecha = time.strftime("%Y%m%d_%H%M")
    salida = os.path.join(AQUI, "modelos", f"{etiqueta}_{fecha}")
    os.makedirs(salida, exist_ok=True)
    modelo.save(os.path.join(salida, "modelo_float.keras"))
    open(os.path.join(salida, "modelo_float.tflite"), "wb").write(tfl_float)
    open(os.path.join(salida, "modelo_int8.tflite"), "wb").write(tfl_int8)

    reporte = {
        "dataset": etiqueta, "fecha": fecha, "split": aviso, "tomas_por_particion": conteo,
        "ventanas": {"train": int(len(Xtr)), "val": int(len(Xva)), "test": int(len(Xte))},
        "clases": F.CLASES, "parametros": int(modelo.count_params()), "epocas": len(h.history["loss"]),
        "segundos_entrenamiento": round(t_entreno, 1),
        "float32": {**m_float, "tamano_tflite_bytes": len(tfl_float), "ms_por_inferencia_pc": round(ms_float, 3)},
        "int8": {**m_int8, "tamano_tflite_bytes": len(tfl_int8), "ms_por_inferencia_pc": round(ms_int8, 3),
                 "escala_entrada": cuant[0], "cero_entrada": cuant[1]},
        "acuerdo_float_vs_int8": float(np.mean(p_int8 == p_tfl_float)),
        "perdida_por_cuantizacion_pp": round(100 * (m_float["exactitud"] - m_int8["exactitud"]), 2),
        "reduccion_tamano": round(len(tfl_float) / len(tfl_int8), 2),
    }
    json.dump(reporte, open(os.path.join(salida, "reporte.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    figuras(h.history, {"float32": m_float["matriz_confusion"], "int8": m_int8["matriz_confusion"]}, salida)

    md = [f"# Model report — {etiqueta} ({fecha})", "", aviso, "",
          "| | float32 | int8 |", "|---|---|---|",
          f"| Accuracy (test) | {m_float['exactitud']:.3f} | {m_int8['exactitud']:.3f} |",
          f"| Macro F1 (test) | {m_float['f1_macro']:.3f} | {m_int8['f1_macro']:.3f} |",
          f"| TFLite size | {len(tfl_float) / 1024:.1f} KB | {len(tfl_int8) / 1024:.1f} KB |",
          f"| PC inference | {ms_float:.2f} ms | {ms_int8:.2f} ms |", "",
          f"Parameters: {modelo.count_params()} · quantization loss: {reporte['perdida_por_cuantizacion_pp']} pp · "
          f"float/int8 agreement: {reporte['acuerdo_float_vs_int8']:.3f} · size reduction ×{reporte['reduccion_tamano']}"]
    open(os.path.join(salida, "reporte.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")

    firmware = os.path.join(RAIZ, "Firmware", "rocky_ia")
    if os.path.isdir(firmware) and not args.sintetico:
        exportar_header(tfl_int8, os.path.join(firmware, "modelo.h"),
                        {"dataset": etiqueta, "fecha": fecha, "exactitud_int8": m_int8["exactitud"]})
        F.exportar_filtros_c(os.path.join(firmware, "filtros.h"))
        print("Firmware actualizado: Firmware/rocky_ia/modelo.h")
    elif args.sintetico:
        exportar_header(tfl_int8, os.path.join(salida, "modelo.h"),
                        {"dataset": etiqueta, "fecha": fecha, "exactitud_int8": m_int8["exactitud"]})
    print("\n".join(md))
    print("Resultados en", salida)


if __name__ == "__main__":
    main()

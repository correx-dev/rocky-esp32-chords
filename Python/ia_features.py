"""
Características de audio para el clasificador de acordes (compartidas por PC y ESP32-S3).

Lo que hace, igual en Python (entrenamiento) y en C (firmware):
  1. Toma 1 s de audio a 16 kHz (16000 muestras).
  2. Lo corta en 28 ventanas de 2048 muestras, una cada 512 (ventana de Hann).
  3. FFT de cada ventana -> energía por frecuencia.
  4. Agrupa la energía en 48 bandas de UN SEMITONO cada una (de Mi2 a Re#6): la guitarra
     afina en semitonos, así que cada banda equivale a una "tecla" del piano.
  5. log10, se resta el máximo del clip (así no importa si tocas fuerte o suave),
     se recorta a 6 décadas y se escala a 0..1.
Resultado: una imagen de 28 x 48 (tiempo x semitono) que la red neuronal clasifica.

El firmware no recalcula los filtros: los copia de filtros.h, generado aquí, para que
los números sean idénticos.
"""

import numpy as np

FS = 16000
MUESTRAS = 16000          # 1 s por inferencia
NFFT = 2048
SALTO = 512
TRAMAS = 1 + (MUESTRAS - NFFT) // SALTO       # 28
MIDI_MIN = 40             # Mi2 (6.ª cuerda al aire)
BANDAS = 48               # hasta MIDI 87 (Re#6)
PISO_DB = 6.0             # décadas que se conservan bajo el máximo
EPS = 1e-10

CLASES = ["C", "D", "E", "G", "A", "Am", "Em", "Dm", "F", "B7", "ninguno"]


def ventana_hann():
    n = np.arange(NFFT)
    return (0.5 - 0.5 * np.cos(2 * np.pi * n / NFFT)).astype(np.float32)


def banco_semitonos():
    """Matriz (BANDAS x NFFT/2+1) de filtros triangulares de un semitono, normalizados (suma 1)."""
    freqs = np.arange(NFFT // 2 + 1) * FS / NFFT
    centros = 440.0 * 2.0 ** ((np.arange(MIDI_MIN - 1, MIDI_MIN + BANDAS + 1) - 69) / 12.0)
    W = np.zeros((BANDAS, len(freqs)), dtype=np.float32)
    for k in range(BANDAS):
        izq, cen, der = centros[k], centros[k + 1], centros[k + 2]
        sube = (freqs - izq) / (cen - izq)
        baja = (der - freqs) / (der - cen)
        w = np.clip(np.minimum(sube, baja), 0, None)
        if w.sum() == 0:                      # en graves la banda es más estrecha que un bin de la FFT
            w[int(round(cen * NFFT / FS))] = 1.0
        W[k] = w / w.sum()
    return W


_HANN = ventana_hann()
_W = banco_semitonos()


def caracteristicas(x):
    """x: 16000 muestras float (-1..1). Devuelve float32 (TRAMAS, BANDAS) en 0..1."""
    x = np.asarray(x, dtype=np.float32)
    if len(x) < MUESTRAS:
        x = np.pad(x, (0, MUESTRAS - len(x)))
    x = x[:MUESTRAS] - np.mean(x[:MUESTRAS])
    tramas = np.stack([x[i * SALTO: i * SALTO + NFFT] * _HANN for i in range(TRAMAS)])
    pot = np.abs(np.fft.rfft(tramas, axis=1)).astype(np.float64) ** 2
    e = pot @ _W.T.astype(np.float64)
    f = np.log10(e + EPS)
    f = f - f.max()
    f = np.clip(f, -PISO_DB, 0.0)
    return ((f + PISO_DB) / PISO_DB).astype(np.float32)


def exportar_filtros_c(ruta):
    """Escribe filtros.h con la ventana y el banco de filtros en formato disperso."""
    lineas = ["// Generado por Python/ia_features.py: NO editar a mano.",
              "#pragma once", "#include <stdint.h>", "",
              f"#define IA_FS {FS}", f"#define IA_MUESTRAS {MUESTRAS}", f"#define IA_NFFT {NFFT}",
              f"#define IA_SALTO {SALTO}", f"#define IA_TRAMAS {TRAMAS}", f"#define IA_BANDAS {BANDAS}",
              f"#define IA_PISO {PISO_DB:.1f}f", f"#define IA_EPS {EPS:.1e}f", ""]
    inicio, largo, pesos = [], [], []
    for k in range(BANDAS):
        nz = np.nonzero(_W[k])[0]
        a, b = int(nz[0]), int(nz[-1])
        inicio.append(a)
        largo.append(b - a + 1)
        pesos.extend(_W[k, a:b + 1].tolist())
    lineas.append("static const uint16_t IA_FILTRO_INICIO[IA_BANDAS] = {" + ", ".join(map(str, inicio)) + "};")
    lineas.append("static const uint16_t IA_FILTRO_LARGO[IA_BANDAS] = {" + ", ".join(map(str, largo)) + "};")
    lineas.append(f"#define IA_NPESOS {len(pesos)}")
    lineas.append("static const float IA_FILTRO_PESOS[IA_NPESOS] = {" +
                  ", ".join(f"{p:.9e}f" for p in pesos) + "};")
    lineas.append("static const char* const IA_CLASES[] = {" + ", ".join(f'"{c}"' for c in CLASES) + "};")
    lineas.append(f"#define IA_NCLASES {len(CLASES)}")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lineas) + "\n")

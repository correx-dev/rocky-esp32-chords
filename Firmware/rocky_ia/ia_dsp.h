// Características de audio en C: las MISMAS cuentas que Python/ia_features.py (verificado en el PC con
// tests/test_firmware_host.cpp contra Python). Entrada: 16000 muestras int16 a 16 kHz. Salida: 28 x 48 en 0..1.
//
// Optimización (Arquitectura del Computador): la FFT de una señal real desperdicia la mitad del cálculo.
// Se empaquetan DOS tramas reales en una FFT compleja (una en la parte real, otra en la imaginaria) y luego
// se separan sus espectros con la simetría X[k] = conj(X[N-k]). Así 28 tramas cuestan 14 FFT de 2048 puntos.
#pragma once
#include <math.h>
#include <stdint.h>
#include <string.h>
#include "filtros.h"

struct IaDsp {
  float hann[IA_NFFT];
  float cosT[IA_NFFT / 2], sinT[IA_NFFT / 2];
  uint16_t rev[IA_NFFT];
  float re[IA_NFFT], im[IA_NFFT];

  void init() {
    const float PI2 = 6.283185307179586f;
    for (int n = 0; n < IA_NFFT; n++) hann[n] = 0.5f - 0.5f * cosf(PI2 * n / IA_NFFT);   // Hann periódica (igual que Python)
    for (int k = 0; k < IA_NFFT / 2; k++) {
      cosT[k] = cosf(PI2 * k / IA_NFFT);
      sinT[k] = -sinf(PI2 * k / IA_NFFT);
    }
    int bits = 0;
    while ((1 << bits) < IA_NFFT) bits++;
    for (int i = 0; i < IA_NFFT; i++) {
      int r = 0;
      for (int b = 0; b < bits; b++) if (i & (1 << b)) r |= 1 << (bits - 1 - b);
      rev[i] = (uint16_t)r;
    }
  }

  // FFT compleja in-place radix-2 (decimación en el tiempo). O(N log N): 11 etapas de 1024 mariposas.
  void fft() {
    for (int i = 0; i < IA_NFFT; i++) {
      int j = rev[i];
      if (j > i) {
        float t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (int len = 2; len <= IA_NFFT; len <<= 1) {
      int mitad = len >> 1, paso = IA_NFFT / len;
      for (int i = 0; i < IA_NFFT; i += len) {
        for (int k = 0; k < mitad; k++) {
          float wr = cosT[k * paso], wi = sinT[k * paso];
          int a = i + k, b = a + mitad;
          float tr = re[b] * wr - im[b] * wi;
          float ti = re[b] * wi + im[b] * wr;
          re[b] = re[a] - tr; im[b] = im[a] - ti;
          re[a] += tr;        im[a] += ti;
        }
      }
    }
  }

  // x: IA_MUESTRAS int16.  out: IA_TRAMAS * IA_BANDAS floats (fila = trama, columna = semitono).
  void caracteristicas(const int16_t *x, float *out) {
    int64_t suma = 0;
    for (int i = 0; i < IA_MUESTRAS; i++) suma += x[i];
    const float media = (float)((double)suma / IA_MUESTRAS);        // en unidades int16
    const float esc = 1.0f / 32768.0f;
    float maximo = -1e30f;

    for (int par = 0; par < IA_TRAMAS / 2; par++) {
      const int ta = 2 * par, tb = 2 * par + 1;
      for (int n = 0; n < IA_NFFT; n++) {
        re[n] = ((float)x[ta * IA_SALTO + n] - media) * esc * hann[n];
        im[n] = ((float)x[tb * IA_SALTO + n] - media) * esc * hann[n];
      }
      fft();
      // Para cada banda, acumular la energía de las dos tramas a la vez.
      const uint16_t *pini = IA_FILTRO_INICIO, *plar = IA_FILTRO_LARGO;
      int w = 0;
      for (int b = 0; b < IA_BANDAS; b++) {
        float ea = 0, eb = 0;
        for (int j = 0; j < plar[b]; j++) {
          int k = pini[b] + j, nk = (IA_NFFT - k) & (IA_NFFT - 1);
          float zr = re[k], zi = im[k], cr = re[nk], ci = im[nk];
          float ar = 0.5f * (zr + cr), ai = 0.5f * (zi - ci);       // espectro de la trama A
          float br = 0.5f * (zi + ci), bi = 0.5f * (cr - zr);       // espectro de la trama B
          float wgt = IA_FILTRO_PESOS[w++];
          ea += wgt * (ar * ar + ai * ai);
          eb += wgt * (br * br + bi * bi);
        }
        float la = log10f(ea + IA_EPS), lb = log10f(eb + IA_EPS);
        out[ta * IA_BANDAS + b] = la;
        out[tb * IA_BANDAS + b] = lb;
        if (la > maximo) maximo = la;
        if (lb > maximo) maximo = lb;
      }
    }
    // Restar el máximo del clip, recortar a IA_PISO décadas y escalar a 0..1
    for (int i = 0; i < IA_TRAMAS * IA_BANDAS; i++) {
      float f = out[i] - maximo;
      if (f < -IA_PISO) f = -IA_PISO;
      out[i] = (f + IA_PISO) / IA_PISO;
    }
  }
};

// Cuantiza 0..1 al int8 que espera el modelo: q = round(x / escala + cero), saturado a [-128, 127].
static inline int8_t ia_cuantizar(float x, float escala, int cero) {
  long q = lrintf(x / escala) + cero;
  return (int8_t)(q > 127 ? 127 : (q < -128 ? -128 : q));
}

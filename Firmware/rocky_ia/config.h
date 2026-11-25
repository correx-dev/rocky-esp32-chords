// Configuración del firmware Rocky IA (ESP32-S3-N16R8 + PCM1808 + preamplificador TL072).
#pragma once

// ---- Pines I2S (PCM1808 en modo maestro 48 kHz; la ESP32 es esclava) ----
#define PIN_BCLK 4
#define PIN_WS   5
#define PIN_DIN  6

// ---- Audio ----
#define FS_ENTRADA   48000        // PCM1808 en M-48K
#define DECIMACION   3            // 48 kHz -> 16 kHz
#define BLOQUE       64           // muestras a 16 kHz por paquete de audio (4 ms)
#define GANANCIA     1.0f         // súbela (p.ej. 4.0) si la guitarra va directa sin preamplificador
#define BAUDIOS      921600

// ---- IA ----
#define HOP_MUESTRAS 4000         // una inferencia cada 250 ms (ventana de 1 s con 75 % de solape)
#define TENSOR_ARENA_BYTES (64 * 1024)   // el firmware informa cuánto usa de verdad (arena_usada)
#define VOTO_N       5            // ventana de votación
#define VOTO_MIN     3            // votos mínimos para cambiar de acorde
#define VOTO_CONF    0.5f         // confianza media mínima
#define PUERTA_RMS   0.004f       // por debajo de este nivel hay "silencio"
#define SALTAR_SILENCIO 0         // 1: no ejecutar la red con silencio y reportar "ninguno" (ahorra energía)
#define PERIODO_RECURSOS_MS 10000 // cada cuánto se manda el reporte de Flash/RAM/latencia (R7)

// ---- Medición de energía opcional (INA219 en serie con los 5 V; mismo bus I2C que usaste en el nodo de heladas) ----
#define USAR_INA219  0
#define PIN_SDA      8
#define PIN_SCL      9

// ---- Núcleos y prioridades ----
#define NUCLEO_ADQ   1            // adquisición de audio (tiempo real estricto)
#define NUCLEO_IA    0            // inferencia
#define PRIO_ADQ     10
#define PRIO_IA      2

/*
  Rocky IA — ESP32-S3: adquisición de audio + clasificador de acordes en la placa (TFLite Micro, int8)
  =====================================================================================================
  Placa:  ESP32S3 Dev Module (Arduino-ESP32 core 3.x)  ·  N16R8: Flash 16 MB, PSRAM OPI
  Librería: "Chirale_TensorFlowLite" (Administrador de librerías) — la misma que ya funcionó en el nodo de heladas.

  Cableado (igual que esp32_streaming): PCM1808 en M-48K (OP1 SHORT, OP2 OPEN) e I2S-24 (OP4 OPEN)
    BCLK -> GPIO 4 · LRCK -> GPIO 5 · DOUT del PCM1808 -> GPIO 6 · GND común

  Qué hace (dos núcleos):
    NÚCLEO 1 · tarea de adquisición (prioridad alta, tiempo real)
        I2S 48 kHz/24 bit -> biquads Butterworth 4.º orden (6.5 kHz) -> diezmado x3 -> 16 kHz/16 bit
        -> buffer circular (RingBuffer, estructuras.h)  y  paquete de audio por USB  [A5 5A][seq][n][muestras]
    NÚCLEO 0 · tarea de IA (cada 250 ms)
        últimas 16000 muestras -> características 28x48 (ia_dsp.h) -> int8 -> modelo (TFLite Micro)
        -> ventana de votación (VotingWindow) -> paquete de acorde [A6 6A] (18 bytes)
        cada 10 s -> reporte de recursos JSON [A7 7A][len][json]  (evidencia del requisito R7)

  Modo de validación (lo usa Python/validar_en_placa.py): el PC manda  [B2 2B][len=32000 u16][16000 int16]
  y la placa responde  [A8 8A][n u16][1344 int8 de entrada del modelo][11 int8 de salida],
  para comprobar que la placa calcula lo mismo que el PC.

  IMPORTANTE: este firmware no imprime texto: el puerto USB es solo para paquetes binarios.
  Los paquetes se escriben bajo un mutex para que los dos núcleos no mezclen sus bytes.

  Archivos generados desde Python (no se editan a mano):
    filtros.h  <- Python/ia_features.py (exportar_filtros_c)      modelo.h <- IA/entrenar.py
*/

#include <Arduino.h>
#include <math.h>
#include "driver/i2s_std.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include <Chirale_TensorFlowLite.h>
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "config.h"
#include "filtros.h"
#include "modelo.h"
#include "estructuras.h"
#include "ia_dsp.h"

#if USAR_INA219
#include <Wire.h>
#include <Adafruit_INA219.h>
Adafruit_INA219 ina219(0x40);
bool inaListo = false;
#endif

// ---------------------------------------------------------------- datos compartidos ----
static RingBuffer<int16_t, 32768> anillo;       // 2 s de audio a 16 kHz (potencia de 2)
static SemaphoreHandle_t mutexSerial;

static IaDsp dsp;
static int16_t ventana[IA_MUESTRAS];
static float caract[IA_TRAMAS * IA_BANDAS];

alignas(16) static uint8_t arena[TENSOR_ARENA_BYTES];
static tflite::MicroInterpreter *interprete = nullptr;
static TfLiteTensor *entrada = nullptr;
static TfLiteTensor *salida = nullptr;
static bool modeloListo = false;
static float escalaE = 1, escalaS = 1;
static int ceroE = 0, ceroS = 0;

static VotingWindow<VOTO_N, IA_NCLASES> votacion(VOTO_MIN, VOTO_CONF);
static RunningStats statLat, statFeat, statInf, statCiclos;

// ---------------------------------------------------------------- envío de paquetes ----
static void enviar(const uint8_t *p, size_t n) {
  xSemaphoreTake(mutexSerial, portMAX_DELAY);
  Serial.write(p, n);
  xSemaphoreGive(mutexSerial);
}

// ---------------------------------------------------------------- filtro de audio ----
struct Biquad {
  float b0, b1, b2, a1, a2, z1 = 0, z2 = 0;
  void pasaBajos(float fc, float q, float fs) {
    float w = 2.0f * PI * fc / fs, cw = cosf(w), alpha = sinf(w) / (2.0f * q);
    float a0 = 1.0f + alpha;
    b0 = (1.0f - cw) / 2.0f / a0;
    b1 = (1.0f - cw) / a0;
    b2 = b0;
    a1 = -2.0f * cw / a0;
    a2 = (1.0f - alpha) / a0;
  }
  inline float procesar(float x) {          // forma directa II transpuesta
    float y = b0 * x + z1;
    z1 = b1 * x - a1 * y + z2;
    z2 = b2 * x - a2 * y;
    return y;
  }
};
static Biquad lp1, lp2;

static i2s_chan_handle_t rx;

static void iniciarI2S() {
  i2s_chan_config_t cc = I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_SLAVE);
  cc.dma_desc_num = 8;
  cc.dma_frame_num = 256;                   // 8 x 256 tramas = 43 ms de colchón
  ESP_ERROR_CHECK(i2s_new_channel(&cc, NULL, &rx));
  i2s_std_config_t cfg = {
    .clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(FS_ENTRADA),
    .slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_STEREO),
    .gpio_cfg = { .mclk = I2S_GPIO_UNUSED, .bclk = (gpio_num_t)PIN_BCLK, .ws = (gpio_num_t)PIN_WS,
                  .dout = I2S_GPIO_UNUSED, .din = (gpio_num_t)PIN_DIN, .invert_flags = { false, false, false } },
  };
  ESP_ERROR_CHECK(i2s_channel_init_std_mode(rx, &cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(rx));
}

// ---------------------------------------------------------------- tarea 1: adquisición (núcleo 1) ----
static void tareaAdquisicion(void *) {
  static int32_t crudo[BLOQUE * DECIMACION * 2];
  static int16_t muestras[BLOQUE];
  static uint8_t paquete[6 + BLOQUE * 2];
  uint16_t seq = 0;
  paquete[0] = 0xA5; paquete[1] = 0x5A;
  for (;;) {
    size_t leidos = 0, porLeer = sizeof(crudo);
    uint8_t *dst = (uint8_t *)crudo;
    while (porLeer > 0) {
      if (i2s_channel_read(rx, dst, porLeer, &leidos, portMAX_DELAY) != ESP_OK) continue;
      dst += leidos;
      porLeer -= leidos;
    }
    for (int k = 0; k < BLOQUE; k++) {
      float y = 0;
      for (int j = 0; j < DECIMACION; j++) {
        int32_t m = crudo[(k * DECIMACION + j) * 2] >> 8;        // canal izquierdo, 24 bits
        y = lp2.procesar(lp1.procesar((float)m / 8388608.0f));
      }
      float v = y * GANANCIA * 32767.0f;                          // nos quedamos con 1 de cada 3 muestras filtradas
      if (v > 32767.0f) v = 32767.0f;
      if (v < -32768.0f) v = -32768.0f;
      muestras[k] = (int16_t)lrintf(v);
    }
    anillo.push(muestras, BLOQUE);                                // productor del buffer circular
    memcpy(paquete + 6, muestras, sizeof(muestras));
    paquete[2] = seq & 0xFF; paquete[3] = seq >> 8;
    paquete[4] = BLOQUE & 0xFF; paquete[5] = BLOQUE >> 8;
    seq++;
    enviar(paquete, sizeof(paquete));
  }
}

// ---------------------------------------------------------------- modelo ----
static bool iniciarModelo() {
  const tflite::Model *modelo = tflite::GetModel(g_modelo);
  if (modelo->version() != TFLITE_SCHEMA_VERSION) return false;
  // Solo se enlazan las 5 operaciones que usa la red (ahorra Flash): conv, max-pool, promedio global, densa y softmax.
  static tflite::MicroMutableOpResolver<5> resolutor;
  resolutor.AddConv2D();
  resolutor.AddMaxPool2D();
  resolutor.AddMean();
  resolutor.AddFullyConnected();
  resolutor.AddSoftmax();
  static tflite::MicroInterpreter interp(modelo, resolutor, arena, TENSOR_ARENA_BYTES);
  interprete = &interp;
  if (interprete->AllocateTensors() != kTfLiteOk) return false;
  entrada = interprete->input(0);
  salida = interprete->output(0);
  if (entrada->type != kTfLiteInt8 || salida->type != kTfLiteInt8) return false;
  if (entrada->bytes != IA_TRAMAS * IA_BANDAS || salida->bytes != IA_NCLASES) return false;
  escalaE = entrada->params.scale; ceroE = entrada->params.zero_point;
  escalaS = salida->params.scale;  ceroS = salida->params.zero_point;
  return true;
}

// Cuantiza las características al tensor de entrada.
static void cargarEntrada() {
  for (int i = 0; i < IA_TRAMAS * IA_BANDAS; i++) entrada->data.int8[i] = ia_cuantizar(caract[i], escalaE, ceroE);
}

// Ejecuta la red sobre la entrada ya cargada. Devuelve ciclos de CPU de Invoke().
static int32_t ejecutar(uint32_t *us) {
  uint32_t c0 = ESP.getCycleCount();
  uint32_t t0 = micros();
  interprete->Invoke();
  *us = micros() - t0;
  return (int32_t)(ESP.getCycleCount() - c0);
}

// ---------------------------------------------------------------- reporte de recursos (R7) ----
static void enviarRecursos(double cargaPct, double potenciaMw, double corrienteMa) {
  static char json[900];
  int n = snprintf(json, sizeof(json),
    "{\"flash_app\":%u,\"flash_particion\":%u,\"flash_chip\":%u,\"modelo_bytes\":%u,"
    "\"ram_interna_total\":%u,\"ram_interna_libre\":%u,\"ram_interna_min_libre\":%u,\"psram_libre\":%u,"
    "\"arena_usada\":%u,\"arena_reservada\":%u,"
    "\"lat_ms\":%.3f,\"lat_ms_desv\":%.3f,\"feat_ms\":%.3f,\"inf_ms\":%.3f,\"inf_ms_desv\":%.3f,"
    "\"ciclos_inf\":%.0f,\"n\":%u,\"carga_nucleo0_pct\":%.2f,\"cpu_mhz\":%u,\"hop_ms\":%d,"
    "\"potencia_mw\":%.1f,\"corriente_ma\":%.1f,\"estado\":\"%s\"}",
    (unsigned)ESP.getSketchSize(), (unsigned)(ESP.getSketchSize() + ESP.getFreeSketchSpace()),
    (unsigned)ESP.getFlashChipSize(), (unsigned)IA_MODELO_BYTES,
    (unsigned)heap_caps_get_total_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_INTERNAL),
    (unsigned)heap_caps_get_minimum_free_size(MALLOC_CAP_INTERNAL), (unsigned)heap_caps_get_free_size(MALLOC_CAP_SPIRAM),
    (unsigned)(modeloListo ? interprete->arena_used_bytes() : 0), (unsigned)TENSOR_ARENA_BYTES,
    statLat.mean / 1000.0, statLat.stddev() / 1000.0, statFeat.mean / 1000.0, statInf.mean / 1000.0, statInf.stddev() / 1000.0,
    statCiclos.mean, (unsigned)statInf.n, cargaPct, (unsigned)getCpuFrequencyMhz(), (int)(1000L * HOP_MUESTRAS / IA_FS),
    potenciaMw, corrienteMa, modeloListo ? "ok" : "error_modelo");
  if (n <= 0 || n >= (int)sizeof(json)) return;
  uint8_t cab[4] = {0xA7, 0x7A, (uint8_t)(n & 0xFF), (uint8_t)(n >> 8)};
  xSemaphoreTake(mutexSerial, portMAX_DELAY);
  Serial.write(cab, 4);
  Serial.write((const uint8_t *)json, n);
  xSemaphoreGive(mutexSerial);
}

// ---------------------------------------------------------------- comando de validación ----
// [B2 2B][len u16 = 32000][16000 int16]  ->  [A8 8A][n u16][entrada int8 x1344][salida int8 x11]
static void atenderComandos() {
  static int estado = 0;               // 0: esperando B2, 1: esperando 2B, 2: longitud, 3: datos
  static uint32_t cuenta = 0, inicio = 0;
  static uint8_t lenb[2];
  while (Serial.available()) {
    uint8_t b = Serial.read();
    if (estado == 0) { if (b == 0xB2) estado = 1; continue; }
    if (estado == 1) { estado = (b == 0x2B) ? 2 : 0; cuenta = 0; continue; }
    if (estado == 2) {
      lenb[cuenta++] = b;
      if (cuenta == 2) {
        estado = ((lenb[0] | (lenb[1] << 8)) == IA_MUESTRAS * 2) ? 3 : 0;
        cuenta = 0; inicio = millis();
      }
      continue;
    }
    ((uint8_t *)ventana)[cuenta++] = b;
    if (cuenta == IA_MUESTRAS * 2) {
      estado = 0;
      if (!modeloListo) return;
      dsp.caracteristicas(ventana, caract);
      cargarEntrada();
      uint32_t us;
      ejecutar(&us);
      static uint8_t resp[4 + IA_TRAMAS * IA_BANDAS + IA_NCLASES];
      const uint16_t n = IA_TRAMAS * IA_BANDAS + IA_NCLASES;
      resp[0] = 0xA8; resp[1] = 0x8A; resp[2] = n & 0xFF; resp[3] = n >> 8;
      memcpy(resp + 4, entrada->data.int8, IA_TRAMAS * IA_BANDAS);
      memcpy(resp + 4 + IA_TRAMAS * IA_BANDAS, salida->data.int8, IA_NCLASES);
      enviar(resp, sizeof(resp));
      return;
    }
    if (millis() - inicio > 3000) { estado = 0; cuenta = 0; }
  }
  if (estado == 3 && millis() - inicio > 3000) estado = 0;
}

// ---------------------------------------------------------------- tarea 2: IA (núcleo 0) ----
static void tareaIA(void *) {
  dsp.init();
  modeloListo = iniciarModelo();
#if USAR_INA219
  Wire.begin(PIN_SDA, PIN_SCL);
  inaListo = ina219.begin();
#endif
  uint32_t proximo = IA_MUESTRAS;                 // total de muestras en el que toca la siguiente inferencia
  uint32_t ultimoReporte = millis(), ocupadoUs = 0, desdeUs = micros();
  for (;;) {
    atenderComandos();
    uint32_t total = anillo.total();
    if (total < proximo) { vTaskDelay(pdMS_TO_TICKS(4)); continue; }
    proximo += HOP_MUESTRAS;
    if (proximo < total) proximo = total + HOP_MUESTRAS;     // si nos atrasamos, saltar (no acumular retraso)

    uint32_t t0 = micros();
    anillo.last(ventana, IA_MUESTRAS);
    // Nivel de la última cuarta parte de la ventana, para la puerta de silencio
    int64_t s2 = 0, s1 = 0;
    for (int i = IA_MUESTRAS * 3 / 4; i < IA_MUESTRAS; i++) { s1 += ventana[i]; s2 += (int64_t)ventana[i] * ventana[i]; }
    int nn = IA_MUESTRAS / 4;
    float rms = sqrtf(fmaxf(0.0f, (float)s2 / nn - (float)s1 / nn * (float)s1 / nn)) / 32768.0f;

    int clase = IA_NCLASES - 1;                     // "ninguno" es la última clase
    float conf = 1.0f;
    uint32_t featUs = 0, infUs = 0;
    int32_t ciclos = 0;
    if (modeloListo && !(SALTAR_SILENCIO && rms < PUERTA_RMS)) {
      uint32_t tf = micros();
      dsp.caracteristicas(ventana, caract);
      cargarEntrada();
      featUs = micros() - tf;
      ciclos = ejecutar(&infUs);
      int mejor = 0;
      for (int k = 1; k < IA_NCLASES; k++) if (salida->data.int8[k] > salida->data.int8[mejor]) mejor = k;
      clase = mejor;
      conf = (salida->data.int8[mejor] - ceroS) * escalaS;
      if (conf < 0) conf = 0;
      if (conf > 1) conf = 1;
      statFeat.add(featUs);
      statInf.add(infUs);
      statCiclos.add(ciclos);
    }
    votacion.add(clase, conf);
    float fr;
    int estable = votacion.vote(&fr);
    uint32_t latUs = micros() - t0;
    if (modeloListo) statLat.add(latUs);
    ocupadoUs += latUs;

    uint8_t p[18] = {0xA6, 0x6A, (uint8_t)clase, (uint8_t)lrintf(conf * 255.0f),
                     (uint8_t)(estable >= 0 ? estable : 255), (uint8_t)lrintf(fr * 255.0f)};
    memcpy(p + 6, &latUs, 4); memcpy(p + 10, &featUs, 4); memcpy(p + 14, &infUs, 4);
    enviar(p, sizeof(p));

    if (millis() - ultimoReporte >= PERIODO_RECURSOS_MS) {
      uint32_t ahora = micros();
      double carga = 100.0 * ocupadoUs / (double)(ahora - desdeUs);
      double mw = 0, ma = 0;
#if USAR_INA219
      if (inaListo) { for (int i = 0; i < 20; i++) { ma += ina219.getCurrent_mA(); mw += ina219.getPower_mW(); } ma /= 20; mw /= 20; }
#endif
      enviarRecursos(carga, mw, ma);
      ultimoReporte = millis(); ocupadoUs = 0; desdeUs = ahora;
    }
  }
}

// ---------------------------------------------------------------- arranque ----
void setup() {
  Serial.setRxBufferSize(4096);
  Serial.begin(BAUDIOS);
  mutexSerial = xSemaphoreCreateMutex();
  lp1.pasaBajos(6500.0f, 0.5412f, FS_ENTRADA);    // dos biquads en cascada = Butterworth de 4.º orden
  lp2.pasaBajos(6500.0f, 1.3065f, FS_ENTRADA);
  iniciarI2S();
  xTaskCreatePinnedToCore(tareaAdquisicion, "adq", 4096, NULL, PRIO_ADQ, NULL, NUCLEO_ADQ);
  xTaskCreatePinnedToCore(tareaIA, "ia", 12288, NULL, PRIO_IA, NULL, NUCLEO_IA);
}

void loop() { vTaskDelay(pdMS_TO_TICKS(1000)); }

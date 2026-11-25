# Firmware `rocky_ia` (ESP32-S3-N16R8)

| Archivo | Qué es |
|---|---|
| `rocky_ia.ino` | Programa principal: 2 tareas (adquisición en núcleo 1, IA en núcleo 0) |
| `config.h` | Pines, ganancia, hop, arena, ventana de votación, INA219 opcional |
| `estructuras.h` | `RingBuffer`, `VotingWindow`, `RunningStats` (propias, probadas en el PC) |
| `ia_dsp.h` | FFT radix-2 con 2 tramas reales por FFT compleja + banco de semitonos + log (verificado contra Python) |
| `filtros.h` | Generado por `Python/ia_features.py`: ventana y banco de filtros. No editar |
| `modelo.h` | Generado por `IA/entrenar.py`. **El incluido es de PRUEBA (datos sintéticos): reemplázalo entrenando con tu dataset** |

## Compilar y cargar (Windows)
```
arduino-cli compile --fqbn esp32:esp32:esp32s3:CDCOnBoot=cdc,FlashSize=16M,PSRAM=opi,PartitionScheme=app3M_fat9M_16MB -u -p COM9 Firmware/rocky_ia
```
Las opciones del `--fqbn` (nombres exactos) conviene verificarlas con `arduino-cli board details -b esp32:esp32:esp32s3`.
Con el conector **USB** (nativo) usa `CDCOnBoot=cdc`; con el conector **COM/UART** déjalo en `default`.
Aviso: el firmware **no imprime texto** por USB (solo paquetes binarios); no abras el Monitor Serie mientras Rocky esté abierto.

## Protocolo USB (little-endian)
| Sentido | Paquete | Contenido |
|---|---|---|
| ESP -> PC | `A5 5A` seq u16, n u16, n x int16 | audio 16 kHz (64 muestras = 4 ms) |
| ESP -> PC | `A6 6A` clase, conf, voto, conf_voto (4 x u8), lat_us, feat_us, inf_us (3 x u32) | acorde (18 bytes), cada 250 ms |
| ESP -> PC | `A7 7A` len u16, JSON | recursos: Flash, RAM, arena, latencia, ciclos, carga, potencia (cada 10 s) |
| PC -> ESP | `B2 2B` len u16 (=32000), 16000 x int16 | validación: calcula la entrada y la salida de la red |
| ESP -> PC | `A8 8A` n u16, 1344 x int8, 11 x int8 | respuesta a la validación |

## Qué verificar al cargarlo por primera vez
1. `python Python/diagnostico_hw.py` -> ~250 paquetes/s, 0 perdidos, SNR razonable.
2. `python Python/validar_en_placa.py --puerto COMx` -> entrada idéntica ~100 %, salida idéntica ~100 %.
   Si la salida difiere, es el problema de cuantización por canal que ya viste en el nodo de heladas
   (`entrenar.py` ya desactiva la cuantización por canal en las capas densas).
3. Abrir Rocky: en `IA/mediciones/recursos_*.csv` aparecen los datos de R7.
4. Si `estado` dice `error_modelo`: sube `TENSOR_ARENA_BYTES` en `config.h` o revisa que `modelo.h` sea del modelo actual.

## Límites conocidos (honestidad)
- Este código NO se compiló para la ESP32 en el entorno donde se escribió (sin acceso a las herramientas de Arduino).
  Sí se verificó: la sintaxis contra cabeceras simuladas, las estructuras y la FFT/características en C contra Python.
  Los nombres de funciones de `Chirale_TensorFlowLite` y de ESP-IDF se tomaron de tu sketch del nodo de heladas.
- Las latencias y porcentajes de R7 hay que medirlos en la placa; este código solo los reporta.

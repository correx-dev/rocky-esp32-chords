# Rocky — descripción técnica del sistema (base para el informe)

## 1. En qué consiste
Rocky es un sistema para aprender guitarra al estilo de un videojuego. La guitarra eléctrica pasa por un
preamplificador (TL072), un convertidor analógico-digital (PCM1808) y una **ESP32-S3-N16R8** que filtra y reduce el
audio y **clasifica en la propia placa el acorde que se toca** con una red neuronal convolucional cuantizada (TinyML).
Un servidor en Python y un juego en Unity (estilo Rocksmith) comparan lo que suena con la partitura y dan
retroalimentación inmediata. Además se extrae de la web (Wikipedia) la información de cada canción para ambientar el juego.

## 2. Qué hace el prototipo hoy (según el código)
- Captura audio de guitarra a 48 kHz / 24 bit y lo lleva a 16 kHz / 16 bit con filtro Butterworth de 4.º orden.
- Detecta la **nota** (algoritmo YIN, 60 veces/s) y el **acorde** (red neuronal, 4 veces/s, 11 clases: C, D, E, G, A, Am, Em, Dm, F, B7, ninguno).
- Estabiliza el acorde con una ventana de votación (5 predicciones, mínimo 3 votos, confianza media >= 0.5).
- Convierte canciones (MP3, Guitar Pro, MIDI, MusicXML, Rocksmith) en niveles (`.chart.json`).
- Graba un dataset propio con el modo «Entrena a la IA» y entrena/cuantiza el modelo (`IA/entrenar.py`).
- Mide por sí mismo sus recursos en la placa (Flash, RAM, arena, latencia, ciclos, carga) y los guarda en CSV.

## 3. Flujo de datos y tipos de datos
```
Guitarra -> TL072 (x10) -> PCM1808 -> ESP32-S3 [núcleo 1: I2S -> biquads -> /3 -> buffer circular -> USB]
                                      [núcleo 0: 1 s -> características 28x48 -> int8 -> CNN -> votación -> USB]
         USB (921600 baudios) -> rocky.py -> UDP 5005 -> Unity        Unity -> UDP 5006 -> rocky.py (órdenes)
```
| Dato | Tipo y formato | Dónde |
|---|---|---|
| Audio | int16 mono 16 kHz; paquete `A5 5A` seq, n, muestras (4 ms) | USB, buffer circular de 2 s |
| Toma del dataset | WAV 16 kHz 16 bit, 1.5 s | `IA/dataset/v1/<clase>/<sesion>_<clase>_<n>.wav` |
| Metadatos | CSV: archivo, clase, sesion, guitarrista, guitarra, fuente, rms, pico, fecha | `IA/dataset/v1/metadata.csv` |
| Características | matriz 28 x 48 float32 en 0..1 (tiempo x semitono, MIDI 40–87); int8 en la placa | memoria |
| Modelo | TFLite int8 (~21 KB), arreglo C `modelo.h` | `IA/modelos/*/`, Flash de la ESP32 |
| Predicción de acorde | 18 bytes: clase, confianza, estable, confianza del voto, 3 tiempos en µs | USB -> UDP `C1;...` |
| Recursos de la placa | JSON cada 10 s -> CSV (Flash, RAM, arena, latencias, ciclos, carga, potencia) | `IA/mediciones/recursos_*.csv` |
| Latencia en vivo del PC | CSV por inferencia | `IA/mediciones/pc_*.csv` |
| Canciones | JSON con notas, técnicas, secciones | `Canciones/*.chart.json` |
| Información web | JSON: artista, álbum, año, géneros, ambiente | `Canciones/*.info.json` |

## 4. Cómo se obtienen los datos (R4)
Rocky muestra un acorde, cuenta 3-2-1 y graba 1.5 s. Rechaza tomas muy bajas (RMS < 0.004, salvo «ninguno») o saturadas (pico >= 0.99).
Cada toma lleva sesión, guitarrista y guitarra. **Las particiones train/val/test se hacen por sesión** (o por guitarrista con `--por guitarrista`):
tomas casi idénticas de una misma tanda nunca quedan a la vez en entrenamiento y prueba. En entrenamiento se aumenta con desplazamientos,
cambios de volumen (x0.3–1.5) y ruido. La clase «ninguno» incluye silencio, ruido y golpes apagados.

## 5. Modelo y entrenamiento (R3)
Características: Hann 2048, FFT, 48 filtros triangulares de un semitono, log10, resta del máximo, recorte a 6 décadas -> 0..1.
Red: Conv2D(16) -> MaxPool -> Conv2D(32) -> MaxPool -> Conv2D(32) -> GlobalAveragePooling -> Dropout 0.3 -> Dense(11) con softmax (~14 400 parámetros).
Entrenamiento: Adam 2e-3, pesos por clase, parada temprana. Cuantización: int8 posterior al entrenamiento con 300 muestras representativas;
se reporta la pérdida de exactitud y el tamaño (float32 vs int8) y, con `validar_en_placa.py`, la exactitud real en la placa.

## 6. Estructuras de datos propias (R6)
| Estructura | Dónde se usa | Operaciones y costo | Alternativa y por qué no |
|---|---|---|---|
| Buffer circular | audio (Python y ESP32) | insertar bloque O(k), leer últimas n O(n), memoria fija, sin copiar ni mover | arreglo con desplazamiento: O(N) por bloque; lista enlazada: asigna memoria y fragmenta el heap |
| Ventana de votación | estabiliza el acorde (Python y ESP32) | agregar O(1), votar O(N + C) | usar solo la última predicción: el acorde parpadea con cada duda de la red |
| Tabla hash (encadenamiento, hash polinómico, crece al 75 %) | caché de información web por canción | poner/buscar/borrar O(1) promedio, O(n) peor caso | lista de pares: O(n) por búsqueda en cada canción |
| Montículo mínimo + top-k | los 3 acordes más probables de cada inferencia | insertar/extraer O(log k), top-k en O(C log k) | ordenar todo: O(C log C) |
| Estadística de Welford | latencia media y desviación en la ESP32 | O(1) memoria por muestra | guardar todas las latencias: memoria creciente |
Pruebas: `tests/test_estructuras.py` (compara con listas/diccionarios de referencia) y `tests/test_firmware_host.cpp` (las versiones en C++).

## 7. Cómo se relaciona con las tres materias
- **Sistemas de Adquisición y Procesamiento de Datos:** cadena analógica-digital, muestreo 48 kHz, filtro anti-aliasing, diezmado, protocolo de paquetes con número de secuencia (detección de pérdidas), dataset versionado con metadatos, scraping, ETL.
- **Arquitectura del Computador:** dos núcleos (adquisición en tiempo real / IA), I2S con DMA, FFT con dos tramas reales por FFT compleja (mitad de FFT), cuantización int8, arena de tensores, medición de Flash/RAM/ciclos, mutex para el puerto compartido, buffer productor-consumidor sin bloqueo.
- **Estructuras de Datos:** sección 6.

## 8. Mejoras y sugerencias
1. **Datos:** al menos 3 sesiones y, si se puede, un segundo guitarrista, para medir generalización. Es lo que más cambiará el resultado.
2. **Alcance:** el sistema clasifica 10 acordes abiertos. Documentar esta limitación (no reconoce acordes con cejilla fuera de F, ni inversiones). Si falta tiempo, mejor un modelo sólido con pocos acordes que uno amplio y flojo.
3. **Confusiones previsibles:** A/Am, E/Em y D/Dm difieren en una sola nota; mirar la matriz de confusión y, si hace falta, grabar más tomas de esos pares.
4. **Latencia:** la ventana de 1 s implica que el acorde se confirma ~0.5–1 s después de tocarlo. Para el juego conviene mostrar la nota (YIN, inmediata) y confirmar el acorde con la red. Se puede bajar `HOP_MUESTRAS` o la ventana si se reentrena.
5. **Protección del PCM1808:** como ya hubo un daño en la placa, comprobar el nivel máximo a la entrada del convertidor (acoplamiento por condensador, polarización a mitad de rango, y diodos de sujeción con una resistencia en serie) antes de tocar fuerte.
6. **Energía:** añadir un INA219 (`USAR_INA219 1`) para reportar mJ por inferencia, como pide R7.
7. **Reproducibilidad:** publicar el repositorio (código + dataset o script de captura + `docs/EVIDENCIA.md`) y versionar cada dataset (`v1`, `v2`).
8. **Consentimiento:** si graba otra persona, guardar su autorización por escrito junto con `metadata.csv`.

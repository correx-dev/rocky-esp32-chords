# Rocky — on-device chord recognition for guitar learning (ESP32-S3 + TinyML)

Integrative project, 2nd semester, Data and AI Engineering.
A guitar goes through a TL072 preamplifier and a PCM1808 ADC into an **ESP32-S3-N16R8**, which filters and
decimates the audio (48 kHz -> 16 kHz) and **classifies the chord on the device** with an int8 TensorFlow Lite Micro
CNN (11 classes: C, D, E, G, A, Am, Em, Dm, F, B7, none). A Python server and a Unity game (Rocksmith-style)
turn this into real-time feedback for learning.

| Folder | Content |
|---|---|
| `Firmware/rocky_ia/` | ESP32-S3 firmware (two cores: acquisition / inference) — see its README |
| `Python/` | Game server, real-time inference, dataset recorder, board validation, hardware diagnostic |
| `IA/` | Training and int8 quantization (`entrenar.py`), datasets, models, measurements |
| `tests/` | Data-structure tests (Python and C++), C-vs-Python feature check, simulated real-time test |
| `docs/` | `GUIA_PASO_A_PASO.md`, `DESCRIPCION_TECNICA.md`, `EVIDENCIA.md` (generated) |

Quick start: `powershell -ExecutionPolicy Bypass -File instalar_herramientas.ps1`, then follow `docs/GUIA_PASO_A_PASO.md`.

Own data structures (no libraries): circular buffer, voting window, hash table, min-heap (Python and C++).
Status of every result (what is measured and what is missing): `docs/EVIDENCIA.md`.

Data and consent: only recordings of the authors or with written consent of the player are published.
Songs and Rocksmith files are not included.

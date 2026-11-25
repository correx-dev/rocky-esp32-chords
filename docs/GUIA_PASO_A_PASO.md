# Guía paso a paso: de la guitarra a la evidencia del informe

Orden recomendado. Cada paso deja un archivo que `Python/ia_evidencia.py` recoge solo.

| # | Paso | Comando / acción | Evidencia que deja |
|---|---|---|---|
| 0 | Instalar herramientas | `powershell -ExecutionPolicy Bypass -File instalar_herramientas.ps1` | Python y Arduino CLI listos |
| 1 | Probar el software sin hardware | `python -m unittest tests/test_estructuras.py` y `python tests/test_tiempo_real_simulado.py Am` | Estructuras y protocolo OK |
| 2 | Cargar el firmware | ver `Firmware/rocky_ia/README.md` | Tamaño del programa (lo imprime el compilador) |
| 3 | Probar el hardware real | `python Python/diagnostico_hw.py` (guitarra conectada, Rocky cerrado) | `IA/mediciones/hardware_*.json` (paquetes/s, ruido, SNR, notas) |
| 4 | Grabar el dataset propio | Abrir Rocky -> «Entrena a la IA». Mínimo recomendado: **3 sesiones**, **>= 40 tomas por acorde por sesión**, 1 guitarrista distinto si puedes | `IA/dataset/v1/**.wav` + `metadata.csv` |
| 5 | Entrenar y cuantizar | `python IA/entrenar.py` | `IA/modelos/v1_*/reporte.md`, matrices de confusión, curvas, `Firmware/rocky_ia/modelo.h` |
| 6 | Volver a cargar el firmware | igual que el paso 2 (ahora con el modelo real) | modelo en la placa |
| 7 | Validar la placa contra el PC | `python Python/validar_en_placa.py --puerto COMx` (Rocky cerrado) | `validacion_placa_*.json`: exactitud REAL en la placa |
| 8 | Medir recursos y latencia en vivo | Abrir Rocky con la ESP32 y tocar ~10 min | `recursos_*.csv` (Flash, RAM, arena, latencia media ± desv, ciclos, carga) y `pc_*.csv` |
| 9 | Energía (opcional) | `USAR_INA219 1` en `config.h` y un INA219 en serie con los 5 V | `potencia_mw` / `corriente_ma` en `recursos_*.csv` |
| 10 | Reunir la evidencia | `python Python/ia_evidencia.py` | `docs/EVIDENCIA.md` y `docs/evidencia.json` |

Con el paso 10 completo se puede escribir el informe final sin inventar ningún número.

## Cómo usar Rocky con la guitarra en tiempo real
- `python Python/rocky.py` (o abrir el juego, que lo arranca). Opciones nuevas:
  - `--ia auto` (por defecto): si la ESP32 manda acordes, usa los suyos; si no, el PC clasifica con el mismo modelo.
  - `--ia pc` / `--ia esp` / `--ia off`; `--modelo ruta.tflite` para elegir otro modelo.
- Cada 5 s el servidor imprime el nivel, los paquetes recibidos/perdidos y el acorde estable.
- Unity recibe `R1;...` (nota, YIN, 60 veces/s) y `C1;clase;conf;estable;conf_estable;lat_ms;feat_ms;inf_ms` (acorde, 4 veces/s).

## Cosas que no se pueden saber desde aquí (hay que medirlas)
Exactitud real, latencias en la placa, porcentajes de Flash/RAM, SNR de la cadena y si el hardware quedó sano
tras el problema del PCM1808. Ningún archivo de este paquete contiene esos números a propósito.

"""Lo ejecuta el juego de Unity al abrirse: instala lo que falte y arranca el servidor."""
print("ROCKY-OK", flush=True)

import importlib.util
import subprocess
import sys

REQUISITOS = {
    "numpy": "numpy", "serial": "pyserial", "scipy": "scipy", "soundfile": "soundfile",
    "librosa": "librosa", "guitarpro": "PyGuitarPro", "mido": "mido", "sounddevice": "sounddevice",
    "requests": "requests", "bs4": "beautifulsoup4",
}
# La IA en vivo en el PC es opcional (intérprete TFLite liviano): si no se instala, el juego sigue con YIN.
OPCIONALES = {"ai_edge_litert": "ai-edge-litert"}

faltan = [paquete for modulo, paquete in REQUISITOS.items() if importlib.util.find_spec(modulo) is None]
if faltan:
    print("Instalando componentes (solo la primera vez, tarda unos minutos)...", flush=True)
    r = subprocess.call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *faltan])
    if r != 0:
        print("ERROR: no pude instalar " + ", ".join(faltan) + ". Revisa tu conexión a internet.", flush=True)
        sys.exit(4)

faltan_op = [p for m, p in OPCIONALES.items() if importlib.util.find_spec(m) is None
             and importlib.util.find_spec("tensorflow") is None and importlib.util.find_spec("tflite_runtime") is None]
if faltan_op:
    subprocess.call([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", *faltan_op])

import rocky

rocky.main()

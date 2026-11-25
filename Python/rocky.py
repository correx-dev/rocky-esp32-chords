"""
ROCKY — servidor de audio para el juego de Unity
=================================================
Hace dos cosas a la vez:

1) EN VIVO: recibe el audio de la guitarra (ESP32-S3 + PCM1808 por USB, o el
   micrófono/línea del PC con --mic), detecta las notas y se las manda a Unity
   por UDP (127.0.0.1:5005) unas 60 veces por segundo.

2) CANCIONES: vigila la carpeta "Canciones". Cada MP3/WAV/OGG/FLAC que pongas
   ahí se analiza solo y se le crea un archivo .chart.json con las notas.
   Unity lo muestra en el menú apenas termina.
   Si además pones una partitura con el MISMO nombre (Guitar Pro .gp3/.gp4/.gp5,
   MusicXML o MIDI), las notas salen de la partitura, que es exacta.
   Una partitura sola (sin MP3) también sirve: se le crea un audio de guitarra.

Uso:
    python rocky.py                  (lo normal)
    python rocky.py --puerto COM5    (si no encuentra la ESP32 sola)
    python rocky.py --mic            (usar micrófono/entrada de línea del PC)
    python rocky.py --solo-analizar  (solo convertir canciones y salir)

Requisitos:  pip install numpy pyserial librosa soundfile
             (para --mic además: pip install sounddevice)
"""

import argparse
import json
import os
import socket
import struct
import sys
import threading
import time

import numpy as np

from estructuras import RingBuffer

FS = 16000
BAUD = 921600
HEADER = b"\xA5\x5A"
UDP_DESTINO = ("127.0.0.1", 5005)
EXT_AUDIO = (".mp3", ".wav", ".ogg", ".flac", ".m4a")
CUERDAS_AL_AIRE = [40, 45, 50, 55, 59, 64]      # E2 A2 D3 G3 B3 E4
NOMBRES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


# ===========================================================================
#  DETECCIÓN EN VIVO
# ===========================================================================
def yin(x, fs=FS, fmin=70.0, fmax=1400.0, umbral=0.15):
    """Frecuencia fundamental (Hz) y claridad, o (None, 0)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    tau_min = int(fs / fmax)
    tau_max = int(fs / fmin)
    w = n - tau_max
    if w <= 0:
        return None, 0.0
    nfft = 1 << int(np.ceil(np.log2(n + w)))
    corr = np.fft.irfft(np.fft.rfft(x, nfft) * np.conj(np.fft.rfft(x[:w], nfft)), nfft)[: tau_max + 1]
    c = np.concatenate(([0.0], np.cumsum(x * x)))
    taus = np.arange(tau_max + 1)
    d = c[w] + (c[taus + w] - c[taus]) - 2.0 * corr
    d[0] = 0.0
    dn = np.ones_like(d)
    dn[1:] = d[1:] * taus[1:] / np.maximum(np.cumsum(d[1:]), 1e-12)
    cand = np.flatnonzero(dn[tau_min:tau_max] < umbral)
    if len(cand) == 0:
        return None, 0.0
    tau = int(cand[0]) + tau_min
    while tau + 1 < tau_max and dn[tau + 1] < dn[tau]:
        tau += 1
    claridad = float(1.0 - dn[tau])
    a, b, cc = d[tau - 1], d[tau], d[tau + 1]
    den = a - 2 * b + cc
    t = tau + 0.5 * (a - cc) / den if abs(den) > 1e-12 else float(tau)
    return fs / t, claridad


# Tabla para la detección de varias notas a la vez (acordes)
_NFFT = 8192
_VENT = np.hanning(2048)
_MIDIS = np.arange(38, 90)
_ARM = np.arange(1, 6)
_PESOS = np.array([1.0, 0.8, 0.6, 0.45, 0.35])
_BINS = np.rint(440.0 * 2 ** ((_MIDIS[:, None] - 69) / 12.0) * _ARM[None, :] * _NFFT / FS).astype(int)
_BINS_OK = _BINS < (_NFFT // 2 - 2)
_BINS = np.where(_BINS_OK, _BINS, 0)


def notas_presentes(x2048, maximo=8):
    """Lista de notas MIDI que probablemente están sonando (sirve para acordes)."""
    mag = np.abs(np.fft.rfft(x2048 * _VENT, _NFFT))
    mag = np.sqrt(mag)
    # tomar el máximo en ±1 bin para tolerar pequeñas desafinaciones
    m3 = np.maximum(mag, np.maximum(np.roll(mag, 1), np.roll(mag, -1)))
    arm = m3[_BINS] * _BINS_OK
    # una nota solo cuenta si su fundamental está presente de verdad
    sal = (arm * _PESOS).sum(axis=1) * (arm[:, 0] > 0.35 * arm.max())
    if sal.max() <= 0:
        return []
    umbral = 0.6 * sal.max()
    notas = []
    for i in np.argsort(-sal)[: maximo * 2]:
        if sal[i] < umbral:
            break
        izq = sal[i - 1] if i > 0 else 0
        der = sal[i + 1] if i + 1 < len(sal) else 0
        if sal[i] >= izq and sal[i] >= der:
            notas.append(int(_MIDIS[i]))
        if len(notas) >= maximo:
            break
    return notas


MONITOR = None                 # retorno de la guitarra con efectos (se crea al primer uso)


def monitor():
    global MONITOR
    if MONITOR is None:
        import efectos
        MONITOR = efectos.Monitor()
    return MONITOR


def renderizar_seccion(velocidad, inicio, fin, origen, destino):
    """Recorta un tramo de la canción y lo estira en el tiempo SIN cambiar la afinación."""
    try:
        import glob
        import librosa
        import soundfile as sf
        y, sr = librosa.load(origen, sr=32000, mono=True, offset=max(0.0, inicio), duration=max(0.5, fin - inicio))
        if abs(velocidad - 1.0) > 0.01:
            y = librosa.effects.time_stretch(y, rate=velocidad)
        os.makedirs(os.path.dirname(destino), exist_ok=True)
        for viejo in glob.glob(os.path.join(os.path.dirname(destino), "sec_*")):
            if os.path.abspath(viejo) != os.path.abspath(destino):
                try:
                    os.remove(viejo)
                except OSError:
                    pass
        sf.write(destino + ".tmp", y, sr, format="WAV", subtype="PCM_16")
        os.replace(destino + ".tmp", destino)
        log("Sección lista: %.0f %% de velocidad, %.1f s" % (velocidad * 100, len(y) / sr))
    except Exception as e:
        log("No pude preparar la sección:", e)
        try:
            with open(destino + ".error", "w", encoding="utf-8") as fh:
                fh.write(str(e))
        except OSError:
            pass


ANILLO = None                  # audio de la guitarra (lo crea main)
ESTADO = {}


def responder(msg):
    """Respuesta corta a Unity por el mismo canal de las notas (UDP 5005)."""
    try:
        socket.socket(socket.AF_INET, socket.SOCK_DGRAM).sendto(msg.encode("utf-8"), UDP_DESTINO)
    except OSError:
        pass


def _grabar_toma(clase, sesion, guitarrista, guitarra):
    import ia_dataset
    try:
        ok, detalle, rms = ia_dataset.grabar(ANILLO, ESTADO, clase, sesion, guitarrista, guitarra, log)
        cuenta = ia_dataset.conteos().get(clase, 0)
        responder("D1;%s;%s;%d;%.4f;%s" % ("ok" if ok else "error", clase, cuenta, rms, detalle.replace(";", ",")))
    except Exception as e:
        log("No pude grabar la toma:", e)
        responder("D1;error;%s;0;0;%s" % (clase, str(e).replace(";", ",")))


def manejar_control(texto):
    """Órdenes que manda el juego: MON;1;0.8  (retorno on/off y volumen)  ·  TONO;g=0.8;wah=1;..."""
    if texto.startswith("GRABAR|"):              # modo «Entrena a la IA»: grabar una toma etiquetada
        p = (texto.strip().split("|") + ["", "", "", "", ""])[:5]
        threading.Thread(target=_grabar_toma, args=(p[1], p[2], p[3], p[4]), daemon=True).start()
        return
    if texto.strip() == "DATASET?":
        import ia_dataset
        c = ia_dataset.conteos()
        responder("D2;" + ",".join(f"{k}:{v}" for k, v in c.items()) + ";" + ia_dataset.VERSION)
        return
    if texto.startswith("SECCION|"):
        p = texto.strip().split("|")
        threading.Thread(target=renderizar_seccion, args=(float(p[1]), float(p[2]), float(p[3]), p[4], p[5]), daemon=True).start()
        return
    partes = texto.strip().split(";")
    try:
        if partes[0] == "MON":
            m = monitor()
            quiere = partes[1] == "1"
            m.cadena.configurar(vol=float(partes[2]))
            if quiere and not m.activo and time.time() >= getattr(m, "reintentar_en", 0):
                m.reintentar_en = time.time() + 10      # si falla, no insistir cada segundo
                if m.encender():
                    log("Retorno de guitarra activado (%d Hz)" % m.fs_salida)
                elif not getattr(m, "avisado", False):
                    m.avisado = True
                    log("No pude abrir la salida de audio para el retorno:", m.error)
            elif not quiere and m.activo:
                m.apagar()
                log("Retorno de guitarra desactivado")
        elif partes[0] == "TONO":
            valores = dict(p.split("=", 1) for p in partes[1:] if "=" in p)
            monitor().cadena.configurar(**{k: float(v) for k, v in valores.items()})
    except Exception as e:
        log("Orden de control inválida:", texto, e)


class Anillo:
    """Últimos 2 s de audio. Envuelve el RingBuffer propio (estructuras.py) con un candado para los hilos."""

    def __init__(self):
        self.rb = RingBuffer(FS * 2)
        self.lock = threading.Lock()

    def agregar(self, x, retorno=True):
        if retorno and MONITOR is not None and MONITOR.activo:
            MONITOR.alimentar(x)
        with self.lock:
            self.rb.agregar(x)

    def ultimas(self, n):
        with self.lock:
            return self.rb.ultimas(n), self.rb.total


def buscar_puerto():
    import serial.tools.list_ports
    puertos = list(serial.tools.list_ports.comports())
    for vid in (0x303A, 0x1A86, 0x10C4):     # Espressif, CH340, CP210x
        for p in puertos:
            if p.vid == vid:
                return p.device
    return None


CAB_ACORDE = b"\xA6\x6A"       # predicción del clasificador que corre en la ESP32
CAB_RECURSOS = b"\xA7\x7A"     # reporte de Flash/RAM/latencia de la ESP32


def hilo_serial(anillo, puerto_fijo, estado):
    """Lee los paquetes de la ESP32 y se reconecta solo si se desconecta."""
    import serial
    import ia_borde
    while True:
        puerto = puerto_fijo or buscar_puerto()
        if puerto is None:
            if not estado.get("avisado"):
                estado["avisado"] = True
                log("Servidor listo. Buscando la ESP32 o una interfaz de audio...")
            if not estado.get("interfaz"):
                estado["fuente"] = "buscando ESP32 o interfaz de audio..."
            time.sleep(2)
            continue
        try:
            ser = serial.Serial(puerto, BAUD, timeout=0.05)
        except Exception as e:
            estado["fuente"] = f"no pude abrir {puerto} ({e.__class__.__name__})"
            time.sleep(2)
            continue
        log(f"ESP32 conectada en {puerto}")
        estado["fuente"] = f"ESP32 en {puerto}"
        estado["esp32"] = True                      # la ESP32 manda: la interfaz de audio se ignora
        buf = bytearray()
        try:
            while True:
                data = ser.read(max(1, ser.in_waiting))
                if not data:
                    continue
                buf += data
                while True:
                    # Tres tipos de paquete: audio (A5 5A), acorde de la IA (A6 6A), recursos (A7 7A)
                    pos = [(buf.find(h), h) for h in (HEADER, CAB_ACORDE, CAB_RECURSOS)]
                    pos = [(i, h) for i, h in pos if i >= 0]
                    if not pos:
                        del buf[:-1]
                        break
                    i, cab = min(pos)
                    if i > 0:
                        del buf[:i]
                    if cab == CAB_ACORDE:
                        if len(buf) < 18:
                            break
                        if not ia_borde.paquete_valido(buf):
                            del buf[:2]
                            continue
                        ia_borde.recibir_acorde(bytes(buf[:18]))
                        del buf[:18]
                        continue
                    if cab == CAB_RECURSOS:
                        if len(buf) < 4:
                            break
                        largo = buf[2] | (buf[3] << 8)
                        if largo == 0 or largo > 2048:
                            del buf[:2]
                            continue
                        if len(buf) < 4 + largo:
                            break
                        texto = bytes(buf[4:4 + largo])
                        if not texto.startswith(b"{"):
                            del buf[:2]
                            continue
                        ia_borde.recibir_recursos(texto, estado)
                        del buf[:4 + largo]
                        continue
                    if len(buf) < 6:
                        break
                    seq, n = struct.unpack_from("<HH", buf, 2)
                    if n == 0 or n > 2048:
                        del buf[:2]
                        continue
                    ult = estado.get("seq")
                    if ult is not None:                       # paquetes que no llegaron (USB saturado, cable)
                        salto = (seq - ult - 1) & 0xFFFF
                        if salto < 2000:
                            estado["perdidos"] = estado.get("perdidos", 0) + salto
                        else:                                 # número absurdo: la lectura se desalineó; no es pérdida real
                            estado["resincronizaciones"] = estado.get("resincronizaciones", 0) + 1
                    estado["seq"] = seq
                    estado["paquetes"] = estado.get("paquetes", 0) + 1
                    total = 6 + 2 * n
                    if len(buf) < total:
                        break
                    m = np.frombuffer(bytes(buf[6:total]), dtype="<i2")
                    del buf[:total]
                    anillo.agregar(m.astype(np.float32) / 32768.0)
        except Exception:
            estado["esp32"] = False
            estado["avisado"] = False
            log("Se desconectó la ESP32; reintentando...")
            try:
                ser.close()
            except Exception:
                pass
            time.sleep(1)


class _A16k:
    """Convierte el audio de la tarjeta (44.1/48/96 kHz) a los 16 kHz del detector, bloque a bloque."""

    def __init__(self, fs):
        from scipy.signal import butter
        self.sos = butter(8, min(0.9, 7200.0 / (fs / 2.0)), output="sos")
        self.zi = np.zeros((self.sos.shape[0], 2))
        self.razon = fs / float(FS)
        self.fase = -1.0
        self.ult = 0.0

    def __call__(self, x):
        from scipy.signal import sosfilt
        if len(x) == 0:
            return np.zeros(0, dtype=np.float32)
        y, self.zi = sosfilt(self.sos, x.astype(np.float64), zi=self.zi)
        ext = np.concatenate(([self.ult], y))
        pos = np.arange(self.fase, len(y) - 1, self.razon)
        self.fase = (pos[-1] + self.razon if len(pos) else self.fase) - len(y)
        self.ult = float(y[-1])
        return np.interp(pos + 1.0, np.arange(len(ext)), ext).astype(np.float32)


_ENT_NUNCA = ("stereo mix", "mezcla est", "mapper", "asignador", "primary", "primario", "loopback",
              "what u hear", "virtual", "steam", "nvidia", "voicemeeter", "cable output",
              "ngenuity", "sonar", "nahimic", "broadcast", "obs-")
_ENT_INTERNA = ("realtek", "array", "varios micr", "integrad", "built-in", "intel", "amd ", "webcam", "camera",
                "cámara", "camara", "bluetooth", "hands-free", "manos libres", "headset", "auricular", "airpods")
_ENT_BUENA = ("usb", "interface", "interfaz", "line", "línea", "linea", "guitar", "instrument", "codec",
              "scarlett", "focusrite", "behringer", "umc", "m-audio", "m-track", "presonus", "audiobox",
              "steinberg", "ur22", "rocksmith", "real tone", "irig", "teyun", "q-2", "analog", "input", "entrada")
_API_PUNTOS = (("wasapi", 0.5), ("mme", 0.3), ("directsound", 0.2), ("wdm", 0.1))


def _candidatas_entrada(sd, permitir_micro, preferida):
    """Entradas de audio ordenadas de más a menos probable que sea la interfaz de la guitarra."""
    apis = [a["name"] for a in sd.query_hostapis()]
    lista = []
    for i, d in enumerate(sd.query_devices()):
        if d["max_input_channels"] < 1:
            continue
        nombre = d["name"]
        bajo = nombre.lower()
        api = apis[d["hostapi"]] if d["hostapi"] < len(apis) else "?"
        if "asio" in api.lower():
            continue                                   # ASIO toma la interfaz en exclusiva; no hace falta
        if any(t in bajo for t in _ENT_NUNCA):
            continue
        puntos = 0.0
        if any(t in bajo for t in _ENT_BUENA):
            puntos += 3
        if any(t in bajo for t in _ENT_INTERNA):
            puntos -= 4
        if "micr" in bajo:
            puntos -= 1
        if preferida and preferida in bajo:
            puntos += 100
        elif puntos < 0 and not permitir_micro:
            continue                                   # micrófono del portátil: oiría la canción, no la guitarra
        puntos += next((v for k, v in _API_PUNTOS if k in api.lower()), 0.0)
        lista.append((puntos, i, nombre, api, int(d["max_input_channels"]), int(d["default_samplerate"] or 48000)))
    lista.sort(key=lambda c: -c[0])
    return lista


def hilo_interfaz(anillo, estado, permitir_micro=False):
    """Usa una interfaz de audio USB (o la entrada de línea) como guitarra. Elige sola el dispositivo
    y el canal donde suena la guitarra, y se reconecta si se desenchufa.
    Para forzar un dispositivo: escribe parte de su nombre en Python/entrada.txt."""
    import queue
    try:
        import sounddevice as sd
    except Exception as e:
        log("No pude usar las entradas de audio del PC:", e)
        return
    aqui = os.path.dirname(os.path.abspath(__file__))
    ult_lista = None
    while True:
        preferida = ""
        try:
            with open(os.path.join(aqui, "entrada.txt"), encoding="utf-8-sig") as fh:
                preferida = fh.read().strip().strip('"').lower()
        except OSError:
            pass
        try:
            # refrescar la lista de dispositivos (solo si el retorno no está sonando: comparten motor)
            if MONITOR is None or MONITOR.stream is None:
                sd._terminate()
                sd._initialize()
            cands = _candidatas_entrada(sd, permitir_micro, preferida)
        except Exception as e:
            log("No pude listar las entradas de audio:", e)
            time.sleep(5)
            continue
        resumen = [f"{c[2]} [{c[3]}, {c[4]} canales, {c[5]} Hz]" for c in cands]
        if resumen != ult_lista:
            ult_lista = resumen
            log("Entradas de audio que puedo usar: " + (" | ".join(resumen) if resumen else "ninguna"))

        cola = queue.Queue(maxsize=400)
        abierto = None
        # Todo lo urgente ocurre dentro del callback de la tarjeta (sin pasar por otro hilo):
        # convertir a 16 kHz, elegir la entrada con guitarra y mandar el retorno a los audífonos.
        # El hilo de abajo solo entrega el audio al detector de notas, que no tiene prisa.
        ctx = {}

        def cb(indata, frames, t, status, cola=cola, ctx=ctx):
            try:
                if not ctx.get("listo"):
                    cola.put_nowait(None)                  # señal de "sí llega audio"
                    return
                if estado.get("esp32"):
                    return
                n = indata.shape[1]
                ctx["energia"] = 0.98 * ctx["energia"] + 0.02 * np.sqrt(np.mean(indata ** 2, axis=0))
                e = ctx["energia"]
                mejor = int(np.argmax(e))
                if mejor != ctx["canal"] and e[mejor] > 2.0 * e[ctx["canal"]] + 1e-4:
                    ctx["canal"] = mejor
                    ctx["aviso"] = mejor
                salidas = [ctx["conv"][c](indata[:, c]) for c in range(n)]
                x = salidas[ctx["canal"]]
                if len(x):
                    if MONITOR is not None and MONITOR.activo:
                        MONITOR.alimentar(x)
                    cola.put_nowait(x)
            except queue.Full:
                pass
            except Exception as ex:                       # nunca dejar que el callback se caiga
                ctx["fallo"] = repr(ex)

        for puntos, idx, nombre, api, maxc, fs0 in cands:
            for fs in dict.fromkeys([fs0, 48000, 44100]):
                for canales in dict.fromkeys([min(maxc, 2), maxc, 1]):
                    if canales < 1 or canales > maxc:
                        continue
                    while not cola.empty():
                        cola.get_nowait()
                    ctx.clear()
                    ctx.update(listo=False, canal=0, energia=np.zeros(canales),
                               conv=[_A16k(fs) for _ in range(canales)])
                    st = None
                    try:
                        st = sd.InputStream(samplerate=fs, channels=canales, dtype="float32", device=idx,
                                            latency="low", blocksize=0, callback=cb)
                        st.start()
                        cola.get(timeout=1.5)          # ¿de verdad llega audio?
                        abierto = (st, nombre, api, fs, canales)
                    except Exception as e:
                        ult_error = f"{nombre} [{api}] {fs} Hz x{canales}: {e.__class__.__name__} {e}"
                        estado["error_entrada"] = ult_error
                        try:
                            if st is not None:
                                st.close()
                        except Exception:
                            pass
                    if abierto:
                        break
                if abierto:
                    break
            if abierto:
                break

        if not abierto:
            if cands and estado.get("error_entrada") != estado.get("error_avisado"):
                estado["error_avisado"] = estado.get("error_entrada")
                log("No pude abrir ninguna entrada de audio. Último intento:", estado.get("error_entrada"))
            time.sleep(4)
            continue

        st, nombre, api, fs, canales = abierto
        ctx["listo"] = True
        estado["interfaz"] = True
        estado["lat_entrada"] = float(st.latency)
        log(f"Interfaz de audio lista: {nombre} [{api}] a {fs} Hz, {canales} canal(es), "
            f"búfer de entrada {st.latency * 1000:.0f} ms")
        ult_rep = 0.0
        try:
            while True:
                try:
                    x = cola.get(timeout=2.0)
                except queue.Empty:
                    if estado.get("esp32"):
                        continue
                    raise RuntimeError("la interfaz dejó de enviar audio")
                if not st.active:
                    raise RuntimeError("la interfaz se detuvo")
                if "fallo" in ctx:
                    log("Aviso en la entrada de audio:", ctx.pop("fallo"))
                if "aviso" in ctx:
                    log(f"Guitarra detectada en la entrada {ctx.pop('aviso') + 1} de la interfaz")
                if x is None:
                    continue
                estado["fuente"] = f"{nombre} (entrada {ctx['canal'] + 1})"
                anillo.agregar(x, retorno=False)       # el retorno ya sonó desde el callback
                if MONITOR is not None and MONITOR.activo and time.time() - ult_rep > 30:
                    ult_rep = time.time()
                    total = (st.latency + MONITOR.latencia_salida) * 1000 + 3
                    log(f"Retraso estimado del retorno: {total:.0f} ms "
                        f"(entrada {st.latency * 1000:.0f} + salida {MONITOR.latencia_salida * 1000:.0f} + 3)")
        except Exception as e:
            log(f"Se perdió la interfaz de audio ({e}); buscando de nuevo...")
        estado["interfaz"] = False
        try:
            st.stop()
            st.close()
        except Exception:
            pass
        time.sleep(2)


def bucle_envio(anillo, estado, puerta=0.004):
    """Analiza el audio nuevo y manda un mensaje UDP por cada bloque de 16 ms."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    ult_pos = 0
    rms_hist = [0.0] * 4
    ult_onset = 0.0
    ult_info = time.time()
    ult_nota = "--"
    while True:
        x, pos = anillo.ultimas(2048)
        if pos - ult_pos < 256:
            time.sleep(0.003)
            continue
        ult_pos = pos
        x = x - x.mean()
        ult = x[-1024:]
        rms = float(np.sqrt(np.mean(ult[-512:] ** 2)))

        midi_f, claridad, notas, onset = -1.0, 0.0, [], 0
        if rms > puerta:
            f, claridad = yin(ult)
            if f is not None:
                midi_f = 69 + 12 * np.log2(f / 440.0)
                ult_nota = NOMBRES[int(round(midi_f)) % 12] + str(int(round(midi_f)) // 12 - 1)
            notas = notas_presentes(x)
            ahora = time.time()
            if rms > 1.6 * max(rms_hist[0], puerta) and ahora - ult_onset > 0.07:
                onset = 1
                ult_onset = ahora
        rms_hist = rms_hist[1:] + [rms]

        msg = "R1;%.2f;%.4f;%.2f;%d;%s" % (midi_f, rms, claridad, onset, ",".join(map(str, notas)))
        try:
            sock.sendto(msg.encode("ascii"), UDP_DESTINO)
        except OSError:
            pass

        if time.time() - ult_info > 5:
            ult_info = time.time()
            extra = ""
            if estado.get("paquetes"):
                extra = f" · paquetes {estado['paquetes']} (perdidos {estado.get('perdidos', 0)})"
            if estado.get("acorde_pc"):
                extra += f" · acorde {estado['acorde_pc']}"
            log(f"{estado['fuente']} · nivel {rms:.3f} · última nota {ult_nota}{extra}")


# ===========================================================================
#  ANÁLISIS DE CANCIONES  (MP3 -> .chart.json)
# ===========================================================================
def asignar_cuerdas(midis):
    """Elige cuerda y traste para cada nota, moviendo la mano lo menos posible."""
    resultado = []
    mano = 3.0
    for m in midis:
        mejor = None
        for s, base in enumerate(CUERDAS_AL_AIRE):
            f = m - base
            if 0 <= f <= 17:
                costo = 0.3 if f == 0 else abs(f - mano) + (0.5 if f > 12 else 0)
                if mejor is None or costo < mejor[0]:
                    mejor = (costo, s, f)
        _, s, f = mejor
        if f > 0:
            mano = 0.7 * mano + 0.3 * f
        resultado.append((s, f))
    return resultado


def analizar_cancion(ruta):
    """Devuelve el diccionario del chart para un archivo de audio."""
    import librosa
    from scipy.ndimage import median_filter

    sr = 22050
    hop = 512
    y, _ = librosa.load(ruta, sr=sr, mono=True)
    duracion = len(y) / sr

    # Separar la parte melódica de la percusión para que la batería no confunda
    y_arm = librosa.effects.harmonic(y, margin=2.0)

    f0, sonoro, _ = librosa.pyin(y_arm, fmin=65.0, fmax=1320.0, sr=sr,
                                 frame_length=2048, hop_length=hop)
    midi = np.where(sonoro & ~np.isnan(f0), np.rint(librosa.hz_to_midi(np.nan_to_num(f0, nan=440.0))), -1)
    midi = median_filter(midi, size=5, mode="nearest").astype(int)

    onsets = librosa.onset.onset_detect(y=y, sr=sr, hop_length=hop, backtrack=False)
    es_onset = np.zeros(len(midi), dtype=bool)
    es_onset[onsets[onsets < len(midi)]] = True

    # Tramos de la misma nota; un ataque nuevo dentro del tramo lo divide
    notas = []
    i = 0
    n = len(midi)
    while i < n:
        if midi[i] < 0:
            i += 1
            continue
        j = i + 1
        while j < n and midi[j] == midi[i] and not (es_onset[j] and j - i >= 4):
            j += 1
        if j - i >= 4:                                   # al menos ~90 ms
            notas.append((i * hop / sr, (j - i) * hop / sr, int(midi[i])))
        i = j

    # Limitar densidad para que sea tocable
    filtradas = []
    for t, d, m in notas:
        if filtradas and t - filtradas[-1][0] < 0.16:
            continue
        while m < 40:
            m += 12
        while m > 76:
            m -= 12
        filtradas.append((t, d, m))

    posiciones = asignar_cuerdas([m for _, _, m in filtradas])
    _, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop)
    beats_t = librosa.frames_to_time(beats, sr=sr, hop_length=hop)

    # Partes de la canción: se agrupan los tramos que suenan parecido
    sec_t = []
    try:
        if len(beats) > 32:
            croma = librosa.feature.chroma_stft(y=y_arm, sr=sr, hop_length=hop)
            croma_b = librosa.util.sync(croma, beats, aggregate=np.median)
            k = int(np.clip(round(duracion / 25.0), 3, 12))
            limites = librosa.segment.agglomerative(croma_b, k)
            marcos = librosa.util.fix_frames(beats, x_min=0)
            sec_t = sorted(set(round(float(librosa.frames_to_time(marcos[min(i, len(marcos) - 1)], sr=sr, hop_length=hop)), 3) for i in limites))
    except Exception:
        sec_t = []

    return {
        "seccionesT": sec_t,
        "seccionesNombre": ["Parte %d" % (i + 1) for i in range(len(sec_t))],
        "title": os.path.splitext(os.path.basename(ruta))[0],
        "audio": os.path.basename(ruta),
        "duration": round(duracion, 2),
        "generator": "rocky-auto-1",
        "beats": [round(float(b), 3) for b in beats_t],
        "notes": [
            {"t": round(t, 3), "d": round(d if d > 0.7 else 0.0, 3), "s": s, "f": f, "m": m}
            for (t, d, m), (s, f) in zip(filtradas, posiciones)
        ],
    }


def ruta_chart(ruta_audio):
    return os.path.splitext(ruta_audio)[0] + ".chart.json"


def _guardar_chart(destino, chart):
    tmp = destino + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(chart, fh, ensure_ascii=False)
    os.replace(tmp, destino)


def procesar_partituras(carpeta, archivos):
    """Convierte las partituras/tablaturas de la carpeta en charts exactos."""
    import partituras
    hechas = 0
    for nombre in archivos:
        base, ext = os.path.splitext(nombre)
        ext = ext.lower()
        ruta = os.path.join(carpeta, nombre)
        destino = os.path.join(carpeta, base + ".chart.json")
        aviso = ruta + ".error"
        if ext == ".psarc":
            continue                                   # los maneja desempaquetar_psarc con el Toolkit
        if ext in partituras.EXT_NO_SOPORTADA:
            if not os.path.exists(aviso):
                log(f"«{nombre}»: este formato no lo puedo leer. Expórtalo como Guitar Pro 5 (.gp5), MusicXML o MIDI.")
                open(aviso, "w").close()
            continue
        if ext not in partituras.EXT_PARTITURA or os.path.exists(aviso):
            continue
        # ¿ya está hecho a partir de esta misma partitura?
        if os.path.exists(destino):
            try:
                with open(destino, encoding="utf-8") as fh:
                    previo = json.load(fh)
                if previo.get("source") == nombre and previo.get("source_mtime") == int(os.path.getmtime(ruta)):
                    continue
                if previo.get("generator") == "manual":
                    continue
            except Exception:
                pass
        try:
            notas, beats, titulo = partituras.leer_partitura(ruta)
            audio = next((base + e for e in EXT_AUDIO if os.path.exists(os.path.join(carpeta, base + e))), None)
            propio = audio is None
            if propio:
                audio = base + ".wav"
                partituras.sintetizar(notas, os.path.join(carpeta, audio))
            _guardar_chart(destino, {
                "title": base,
                "audio": audio,
                "duration": round(max(n["t"] + n["d"] for n in notas) + 3.0, 2) if propio else 0,
                "generator": "partitura",
                "source": nombre,
                "source_mtime": int(os.path.getmtime(ruta)),
                "afinacion": list(partituras.ULTIMA_AFINACION),
                "seccionesT": [t for t, _ in partituras.ULTIMAS_SECCIONES],
                "seccionesNombre": [n for _, n in partituras.ULTIMAS_SECCIONES],
                "beats": beats,
                "notes": notas,
            })
            log(f"Partitura «{nombre}»: {len(notas)} notas exactas" + (" (audio de guitarra creado)" if propio else f" (sobre «{audio}»)"))
            hechas += 1
        except ImportError as e:
            log(f"Para leer «{nombre}» falta instalar una librería: pip install PyGuitarPro mido  ({e})")
            return hechas
        except Exception as e:
            log(f"No pude leer la partitura «{nombre}»: {e}")
            with open(aviso, "w", encoding="utf-8") as fh:
                fh.write(str(e))
    return hechas


_packer_cache = {"buscado": False, "ruta": None, "avisado": False}


def buscar_packer():
    """Ubica packer.exe del Rocksmith Custom Song Toolkit que el usuario tenga instalado.
    Se puede fijar la ruta escribiéndola en Python/toolkit.txt."""
    import glob
    if _packer_cache["buscado"]:
        return _packer_cache["ruta"]
    _packer_cache["buscado"] = True
    aqui = os.path.dirname(os.path.abspath(__file__))
    candidatos = []
    cfg = os.path.join(aqui, "toolkit.txt")
    if os.path.exists(cfg):
        with open(cfg, encoding="utf-8") as fh:
            linea = fh.read().strip().strip('"')
        candidatos += [linea, os.path.join(linea, "packer.exe")]
    junto = os.path.join(os.path.dirname(aqui), "Toolkit")          # carpeta «Toolkit» junto al juego
    candidatos += [os.path.join(junto, "packer.exe")]
    candidatos += sorted(glob.glob(os.path.join(junto, "*", "packer.exe")), reverse=True)
    casa = os.path.expanduser("~")
    for base in ("Downloads", "Descargas", "Desktop", "Escritorio", "Documents", "Documentos", ""):
        for patron in ("rstoolkit*", "RocksmithToolkit*", "Rocksmith*Toolkit*"):
            candidatos += sorted(glob.glob(os.path.join(casa, base, patron, "packer.exe")), reverse=True)
            candidatos += sorted(glob.glob(os.path.join(casa, base, patron, "*", "packer.exe")), reverse=True)
    for c in candidatos:
        if c and os.path.isfile(c):
            _packer_cache["ruta"] = c
            log("Toolkit de Rocksmith encontrado:", c)
            break
    return _packer_cache["ruta"]


def _marca_vigente(ruta_marca):
    """True si el marcador (.listo / .error) es de esta versión del importador."""
    try:
        with open(ruta_marca, encoding="utf-8") as fh:
            return fh.readline().strip() == "v" + VERSION_IMPORTADOR
    except OSError:
        return False


def desempaquetar_psarc(carpeta, archivos):
    """Le pide al Toolkit instalado que desempaquete cada .psarc nuevo (también en subcarpetas)
    y lo importa. Se desempaqueta en una carpeta temporal de ruta corta: Windows falla con rutas largas."""
    import shutil
    import subprocess
    import tempfile
    hechas = 0
    pendientes = []
    for raiz_dir, dirs, ficheros in os.walk(carpeta):
        dirs[:] = [d for d in dirs if not d.startswith("_")]
        if os.path.relpath(raiz_dir, carpeta).count(os.sep) >= 3:
            dirs[:] = []
        enminus = set(f.lower() for f in ficheros)
        for f in sorted(ficheros):
            if f.lower().endswith(".psarc"):
                pendientes.append((raiz_dir, f, enminus))
    for raiz_dir, nombre, nombres in pendientes:
        ruta = os.path.join(raiz_dir, nombre)
        if _marca_vigente(ruta + ".listo") or _marca_vigente(ruta + ".error"):
            continue
        base = nombre[:-6]
        es_mac = base.lower().endswith("_m")
        if es_mac and (base[:-2] + "_p.psarc").lower() in nombres:
            continue                                   # misma canción en versión PC: basta con esa
        packer = buscar_packer()
        if packer is None:
            if not _packer_cache["avisado"]:
                _packer_cache["avisado"] = True
                log("Hay archivos .psarc pero no encontré el Rocksmith Custom Song Toolkit. "
                    "Descomprímelo en Descargas o escribe su carpeta en Python/toolkit.txt.")
            return hechas
        temporal = tempfile.mkdtemp(prefix="rk")
        log(f"Desempaquetando «{nombre}» con el Toolkit...")
        try:
            # copia con nombre corto para que ninguna ruta interna pase el límite de Windows
            copia = os.path.join(temporal, "c_m.psarc" if es_mac else "c_p.psarc")
            shutil.copyfile(ruta, copia)
            salida = os.path.join(temporal, "o")
            os.makedirs(salida)
            r = subprocess.run(
                [packer, "-u", "-i", copia, "-o", salida, "-v=RS2014", "-f=" + ("Mac" if es_mac else "Pc"), "-d", "-x"],
                cwd=os.path.dirname(packer), capture_output=True, text=True, errors="replace", timeout=600,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            texto = ((r.stdout or "") + "\n" + (r.stderr or "")).strip()
            n = importar_carpeta_rocksmith(carpeta, salida, base)
            if not n:
                raise RuntimeError("el Toolkit no produjo los arreglos (código %d): %s" % (r.returncode, texto[-400:]))
            with open(ruta + ".listo", "w", encoding="utf-8") as fh:
                fh.write("v" + VERSION_IMPORTADOR + "\n")
            try:
                os.remove(ruta + ".error")
            except OSError:
                pass
            hechas += n
        except Exception as e:
            log(f"No pude desempaquetar «{nombre}»: {e}")
            with open(ruta + ".error", "w", encoding="utf-8") as fh:
                fh.write("v" + VERSION_IMPORTADOR + "\n" + str(e))
        finally:
            shutil.rmtree(temporal, ignore_errors=True)
    return hechas


def descomprimir_zips(carpeta, archivos):
    """Un .zip en Canciones: se sacan a la carpeta las canciones, partituras y .psarc que traiga."""
    import zipfile
    import partituras
    utiles = EXT_AUDIO + partituras.EXT_PARTITURA + (".psarc",)
    for nombre in archivos:
        if not nombre.lower().endswith(".zip"):
            continue
        ruta = os.path.join(carpeta, nombre)
        if os.path.exists(ruta + ".listo") or os.path.exists(ruta + ".error"):
            continue
        try:
            t1 = os.path.getsize(ruta)
            time.sleep(0.5)
            if os.path.getsize(ruta) != t1:
                continue                               # todavía se está copiando
            sacados = 0
            with zipfile.ZipFile(ruta) as z:
                for miembro in z.infolist():
                    limpio = os.path.basename(miembro.filename.replace("\\", "/"))
                    if miembro.is_dir() or not limpio or not limpio.lower().endswith(utiles):
                        continue
                    destino = os.path.join(carpeta, limpio)
                    if not os.path.exists(destino):
                        with z.open(miembro) as origen, open(destino + ".tmp", "wb") as fh:
                            while True:
                                trozo = origen.read(1 << 20)
                                if not trozo:
                                    break
                                fh.write(trozo)
                        os.replace(destino + ".tmp", destino)
                    sacados += 1
            if not sacados:
                raise ValueError("el zip no trae canciones, partituras ni archivos .psarc")
            open(ruta + ".listo", "w").close()
            log(f"Zip «{nombre}»: {sacados} archivo(s) extraídos")
        except Exception as e:
            log(f"No pude abrir el zip «{nombre}»: {e}")
            with open(ruta + ".error", "w", encoding="utf-8") as fh:
                fh.write(str(e))


VERSION_IMPORTADOR = "6"       # al subirla, las custom songs ya importadas se vuelven a importar


def _nombre_seguro(texto):
    return "".join(c for c in texto if c not in '<>:"/\\|?*').strip() or "Cancion"


def importar_carpeta_rocksmith(carpeta, sub, nombre):
    """Importa una custom song ya desempaquetada que está en «sub». Devuelve cuántos arreglos creó
    (0 si la carpeta no es una custom song). Lanza un error si lo es pero está incompleta."""
    import shutil
    import partituras
    arreglos, audios, wem = [], [], False
    manifiestos = {}
    for raiz_dir, _, archivos in os.walk(sub):
        for a in archivos:
            ruta = os.path.join(raiz_dir, a)
            bajo = a.lower()
            if bajo.endswith(".xml") and partituras.es_rocksmith_xml(ruta):
                arreglos.append(ruta)
            elif bajo.endswith((".ogg", ".wav", ".mp3")) and "preview" not in bajo:
                audios.append(ruta)
            elif bajo.endswith(".wem"):
                wem = True
            elif bajo.endswith(".json"):
                manifiestos[os.path.splitext(a)[0].lower()] = ruta
    if not arreglos:
        return 0
    if not audios:
        raise ValueError("encontré los arreglos pero no el audio en .ogg/.wav" +
                         (" (solo hay .wem: conviértelo a .ogg al desempaquetar)" if wem else ""))
    audio = max(audios, key=os.path.getsize)
    leidos = []
    for ruta in sorted(arreglos):
        try:
            notas, beats, info = partituras.leer_rocksmith_xml(ruta)
            if notas:
                mani = manifiestos.get(os.path.splitext(os.path.basename(ruta))[0].lower())
                info["equipos"] = partituras.tonos_de_manifiesto(mani) if mani else {}
                leidos.append((notas, beats, info))
        except Exception:
            pass                      # voces, luces y otros XML que no son de guitarra
    if not leidos:
        raise ValueError("ningún arreglo de la carpeta tiene notas de guitarra")
    varios = len(leidos) > 1
    usados = set()
    audio_comun = None
    for notas, beats, info in leidos:
        arreglo = info["arreglo"] or "Guitarra"
        etiqueta = {"lead": "Lead", "rhythm": "Rítmica", "combo": "Combo", "bass": "Bajo"}.get(arreglo.lower(), arreglo)
        titulo = " - ".join(x for x in (info["artista"], info["titulo"] or nombre) if x)
        base = _nombre_seguro(titulo + (" (" + etiqueta + ")" if varios else ""))
        k = 2
        while base.lower() in usados:           # p. ej. dos guitarras rítmicas
            base = _nombre_seguro(titulo + " (" + etiqueta + " " + str(k) + ")")
            k += 1
        usados.add(base.lower())
        if audio_comun is None:                  # un solo audio compartido por todos los arreglos
            os.makedirs(os.path.join(carpeta, "_audio"), exist_ok=True)
            audio_comun = "_audio/" + _nombre_seguro(titulo) + os.path.splitext(audio)[1].lower()
            destino_audio = os.path.join(carpeta, audio_comun)
            if not os.path.exists(destino_audio):
                shutil.copyfile(audio, destino_audio)
        tonos = info.get("tonos") or []
        equipos = info.get("equipos") or {}
        _guardar_chart(os.path.join(carpeta, base + ".chart.json"), {
            "title": base, "audio": audio_comun, "duration": 0,
            "generator": "manual", "source": nombre,
            "beats": beats, "notes": notas,
            "afinacion": [int(a) for a in (info.get("afinacion") or [0] * 6)],
            "bajo": arreglo.lower() == "bass",
            "seccionesT": [t for t, _ in (info.get("secciones") or [])],
            "seccionesNombre": [n for _, n in (info.get("secciones") or [])],
            "tonosT": [t for t, _ in tonos],
            "tonosNombre": [n for _, n in tonos],
            "tonosFx": [partituras.tono_a_fx(equipos.get(n), n) for _, n in tonos],
        })
        log(f"Custom song «{base}»: {len(notas)} notas exactas")
    return len(leidos)


def procesar_carpetas_rocksmith(carpeta):
    """Subcarpetas de Canciones con una custom song ya desempaquetada (XML de arreglos + audio)."""
    hechas = 0
    for nombre in sorted(os.listdir(carpeta)):
        sub = os.path.join(carpeta, nombre)
        if not os.path.isdir(sub) or nombre.startswith("_") or os.path.exists(os.path.join(sub, "rocky.error")):
            continue
        marca = os.path.join(sub, "rocky.listo")
        if os.path.exists(marca):
            try:
                with open(marca) as fh:
                    if fh.read().strip() == VERSION_IMPORTADOR:
                        continue
            except OSError:
                continue
        try:
            n = importar_carpeta_rocksmith(carpeta, sub, nombre)
            if n:
                hechas += n
                with open(marca, "w") as fh:
                    fh.write(VERSION_IMPORTADOR)
        except Exception as e:
            log(f"No pude importar la carpeta «{nombre}»: {e}")
            with open(os.path.join(sub, "rocky.error"), "w", encoding="utf-8") as fh:
                fh.write(str(e))
    return hechas


def analizar_pendientes(carpeta):
    hechas = 0
    try:
        archivos = sorted(os.listdir(carpeta))
    except FileNotFoundError:
        os.makedirs(carpeta, exist_ok=True)
        return 0
    try:
        descomprimir_zips(carpeta, archivos)
        archivos = sorted(os.listdir(carpeta))
    except Exception as e:
        log("Error revisando zips:", e)
    hechas += procesar_partituras(carpeta, archivos)
    try:
        hechas += desempaquetar_psarc(carpeta, archivos)
        hechas += procesar_carpetas_rocksmith(carpeta)
        archivos = sorted(os.listdir(carpeta))
    except Exception as e:
        log("Error revisando carpetas de custom songs:", e)
    for nombre in archivos:
        if not nombre.lower().endswith(EXT_AUDIO):
            continue
        ruta = os.path.join(carpeta, nombre)
        destino = ruta_chart(ruta)
        if os.path.exists(destino) or os.path.exists(destino + ".error"):
            continue
        # esperar a que el archivo termine de copiarse
        try:
            t1 = os.path.getsize(ruta)
            time.sleep(0.5)
            if os.path.getsize(ruta) != t1:
                continue
        except OSError:
            continue
        log(f"Analizando «{nombre}» (puede tardar 1-3 minutos)...")
        marca = destino + ".analizando"
        try:
            open(marca, "w").close()
            t0 = time.time()
            chart = analizar_cancion(ruta)
            _guardar_chart(destino, chart)
            log(f"Listo «{nombre}»: {len(chart['notes'])} notas en {time.time() - t0:.0f} s")
            hechas += 1
        except Exception as e:
            log(f"No pude analizar «{nombre}»: {e}")
            with open(destino + ".error", "w", encoding="utf-8") as fh:
                fh.write(str(e))
        finally:
            try:
                os.remove(marca)
            except OSError:
                pass
    return hechas


def hilo_canciones(carpeta):
    threading.Thread(target=hilo_info_web, args=(carpeta,), daemon=True).start()
    while True:
        try:
            analizar_pendientes(carpeta)
        except Exception as e:
            log("Error revisando canciones:", e)
        time.sleep(2)


def hilo_info_web(carpeta):
    """Web scraping: completa artista, álbum, año y género de cada canción desde Wikipedia."""
    import info_web
    espera = 5
    while True:
        time.sleep(espera)
        try:
            info_web.completar_carpeta(carpeta, log)
            espera = 10
        except ImportError as e:
            log("Falta una librería para buscar en Wikipedia:", e)
            espera = 3600
        except Exception as e:                 # sin internet o Wikipedia no responde: reintentar más tarde
            log("No pude consultar Wikipedia (se reintenta en 10 min):", e.__class__.__name__, e)
            espera = 600


# ===========================================================================
PUERTO_CONTROL = 5006


def instancia_unica():
    """Solo un servidor a la vez: el nuevo le pide al anterior que se cierre."""
    for intento in range(3):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind(("127.0.0.1", PUERTO_CONTROL))
        except OSError:
            s.close()
            aviso = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            aviso.sendto(b"SALIR", ("127.0.0.1", PUERTO_CONTROL))
            aviso.close()
            time.sleep(1.5)
            continue

        def escuchar():
            while True:
                try:
                    datos, _ = s.recvfrom(8192)
                except OSError:
                    continue
                if datos == b"SALIR":
                    os._exit(0)
                try:
                    manejar_control(datos.decode("utf-8", "ignore"))
                except Exception:
                    pass

        threading.Thread(target=escuchar, daemon=True).start()
        return True
    return False


def vigilar_padre(pid):
    """Si el juego (Unity) se cierra, el servidor también."""
    def vivo():
        if os.name == "nt":
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x00100000, False, pid)        # SYNCHRONIZE
            if not h:
                return False
            sigue = k.WaitForSingleObject(h, 0) == 0x102     # WAIT_TIMEOUT
            k.CloseHandle(h)
            return sigue
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    def bucle():
        while vivo():
            time.sleep(2)
        os._exit(0)

    threading.Thread(target=bucle, daemon=True).start()


def _iniciar_ia_pc(anillo, estado, modo, modelo):
    try:
        import ia_vivo
        ia_vivo.hilo_ia_pc(anillo, estado, modo, modelo, log)
    except ImportError as e:
        log("IA en el PC desactivada (falta una librería):", e)


def main():
    aqui = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description="Servidor de audio de Rocky")
    ap.add_argument("--puerto", help="puerto serie de la ESP32 (ej. COM5)")
    ap.add_argument("--mic", action="store_true", help="permitir también el micrófono del PC como entrada")
    ap.add_argument("--canciones", default=os.path.join(aqui, "..", "Canciones"))
    ap.add_argument("--solo-analizar", action="store_true")
    ap.add_argument("--puerta", type=float, default=0.004, help="nivel mínimo para detectar")
    ap.add_argument("--padre", type=int, default=0, help="PID del juego; al cerrarse, el servidor termina")
    ap.add_argument("--ia", choices=["auto", "pc", "esp", "off"], default="auto",
                    help="quién clasifica acordes: auto (la ESP32 si lo hace; si no, el PC), pc, esp u off")
    ap.add_argument("--modelo", help="ruta de un .tflite int8 (por defecto, el más reciente de IA/modelos)")
    args = ap.parse_args()

    carpeta = os.path.abspath(args.canciones)
    os.makedirs(carpeta, exist_ok=True)
    log("Carpeta de canciones:", carpeta)

    if args.solo_analizar:
        n = analizar_pendientes(carpeta)
        log(f"Canciones analizadas: {n}")
        return

    if not instancia_unica():
        log("Ya hay otro servidor de Rocky abierto y no se cerró. Ciérralo e inténtalo de nuevo.")
        sys.exit(3)
    if args.padre:
        vigilar_padre(args.padre)

    threading.Thread(target=hilo_canciones, args=(carpeta,), daemon=True).start()

    global ANILLO, ESTADO
    anillo = Anillo()
    estado = {"fuente": "iniciando..."}
    ANILLO, ESTADO = anillo, estado
    if not args.mic:
        threading.Thread(target=hilo_serial, args=(anillo, args.puerto, estado), daemon=True).start()
    threading.Thread(target=hilo_interfaz, args=(anillo, estado, args.mic), daemon=True).start()

    threading.Thread(target=_iniciar_ia_pc, args=(anillo, estado, args.ia, args.modelo), daemon=True).start()

    log("Enviando notas a Unity por UDP %s:%d. Deja esta ventana abierta. Ctrl+C para salir." % UDP_DESTINO)
    try:
        bucle_envio(anillo, estado, args.puerta)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

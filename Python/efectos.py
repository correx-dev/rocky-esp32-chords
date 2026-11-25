"""
Retorno de la guitarra con efectos (pedalera) para Rocky.

    Cadena   -> procesa el audio de la guitarra en bloques (16 kHz):
                puerta de ruido -> wah automático -> saturación (ganancia) -> gabinete
                -> chorus -> delay -> reverb -> volumen
    Monitor  -> lo reproduce por los parlantes/audífonos del PC con poca latencia.

Los efectos son genéricos y propios (no imitan equipos concretos). Un tono se describe con:
    g       ganancia 0..1   (0 limpio, 0.5 crunch, 0.8+ distorsión)
    wah, chorus, delay, reverb   0 o 1
"""

import threading

import numpy as np
from scipy.signal import lfilter

FS = 16000


def _biquad(tipo, fc, q, fs=FS, ganancia_db=0.0):
    w = 2 * np.pi * min(fc, fs * 0.45) / fs
    cw, sw = np.cos(w), np.sin(w)
    alpha = sw / (2 * q)
    if tipo == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif tipo == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif tipo == "bp":
        b = [alpha, 0, -alpha]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:  # "pk": realce o recorte en una banda
        A = 10 ** (ganancia_db / 40)
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    b = np.array(b) / a[0]
    a = np.array(a) / a[0]
    return b, a


class _Filtro:
    def __init__(self, b, a):
        self.b, self.a = b, a
        self.z = np.zeros(2)

    def __call__(self, x):
        y, self.z = lfilter(self.b, self.a, x, zi=self.z)
        return y


class _Retardo:
    """Línea de retardo circular para delay/reverb (el bloque debe ser más corto que el retardo)."""

    def __init__(self, muestras):
        self.buf = np.zeros(muestras)
        self.pos = 0

    def leer(self, n):
        idx = (self.pos + np.arange(n)) % len(self.buf)
        return self.buf[idx], idx

    def avanzar(self, n):
        self.pos = (self.pos + n) % len(self.buf)


class Cadena:
    def __init__(self):
        self.lock = threading.Lock()
        self.p = {"g": 0.15, "wah": 0, "chorus": 0, "delay": 0, "reverb": 1, "vol": 0.8}
        self.hp_entrada = _Filtro(*_biquad("hp", 70, 0.7))
        self.pre = _Filtro(*_biquad("hp", 250, 0.7))            # aprieta los graves antes de saturar
        self.gab_lp = _Filtro(*_biquad("lp", 4200, 0.8))        # gabinete: corta agudos ásperos
        self.gab_lp2 = _Filtro(*_biquad("lp", 5500, 0.7))
        self.gab_pk = _Filtro(*_biquad("pk", 2200, 1.2, ganancia_db=3.0))
        self.gab_hp = _Filtro(*_biquad("hp", 85, 0.7))
        self.wah_z = np.zeros(2)
        self.env = 0.0
        self.puerta = 0.0
        self.ult = 0.0                                           # última muestra (sobremuestreo)
        self.cho_hist = np.zeros(1024)
        self.cho_fase = 0.0
        self.delay = _Retardo(int(0.36 * FS))
        self.combs = [_Retardo(int(t * FS)) for t in (0.0297, 0.0371, 0.0411, 0.0437)]
        self.comb_lp = [0.0] * 4
        self.g_suave = self.p["g"]

    def configurar(self, **kw):
        with self.lock:
            for k, v in kw.items():
                if k in self.p:
                    self.p[k] = float(v)

    def procesar(self, x):
        with self.lock:
            p = dict(self.p)
        x = self.hp_entrada(np.asarray(x, dtype=np.float64))
        n = len(x)
        rms = float(np.sqrt(np.mean(x * x))) if n else 0.0

        # --- Puerta de ruido (más estricta cuanta más ganancia) ---
        umbral = 0.0015 + 0.004 * p["g"]
        objetivo = 1.0 if rms > umbral else 0.0
        paso = 0.5 if objetivo > self.puerta else 0.08
        nueva = self.puerta + (objetivo - self.puerta) * paso
        x = x * np.linspace(self.puerta, nueva, n)
        self.puerta = nueva

        # --- Wah automático: un filtro que se abre según la fuerza con que tocas ---
        if p["wah"] >= 0.5:
            self.env = max(rms, self.env * 0.9)
            fc = 450 + 1900 * min(1.0, self.env * 9.0)
            b, a = _biquad("bp", fc, 3.5)
            y, self.wah_z = lfilter(b, a, x, zi=self.wah_z)
            x = 0.25 * x + 2.2 * y

        # --- Saturación: de limpio a distorsión según la ganancia ---
        self.g_suave += (p["g"] - self.g_suave) * 0.3
        g = self.g_suave
        impulso = 2.0 + (g ** 2) * 400.0
        apretado = self.pre(x)
        mezcla = min(1.0, g * 1.6)
        s = (1 - mezcla) * x + mezcla * apretado
        # sobremuestreo x2 para que la distorsión suene menos áspera
        ant = np.concatenate(([self.ult], s[:-1]))
        self.ult = s[-1] if n else self.ult
        medio = 0.5 * (ant + s)
        y = 0.5 * (np.tanh(medio * impulso) + np.tanh(s * impulso))
        if g > 0.6:                                              # segunda etapa para ganancias altas
            y = np.tanh(y * (1.0 + (g - 0.6) * 6.0))
        y *= 0.55 / (1.0 + 2.5 * g) + 0.12                        # compensar volumen

        # --- Gabinete ---
        if g > 0.25:
            y = self.gab_pk(self.gab_lp2(self.gab_lp(y)))
        else:
            y = self.gab_lp2(y)
        y = self.gab_hp(y)

        # --- Chorus: copia con retardo que oscila ---
        if p["chorus"] >= 0.5:
            todo = np.concatenate((self.cho_hist, y))
            t = self.cho_fase + np.arange(n) / FS
            ret = (0.014 + 0.004 * np.sin(2 * np.pi * 0.8 * t)) * FS
            pos = len(self.cho_hist) + np.arange(n) - ret
            y_mod = np.interp(pos, np.arange(len(todo)), todo)
            self.cho_fase = (self.cho_fase + n / FS) % (1 / 0.8)
            self.cho_hist = todo[-len(self.cho_hist):]
            y = 0.7 * y + 0.6 * y_mod
        else:
            self.cho_hist = np.concatenate((self.cho_hist, y))[-len(self.cho_hist):]

        # --- Delay (eco) ---
        eco, idx = self.delay.leer(n)
        eco = eco.copy()
        if p["delay"] >= 0.5:
            self.delay.buf[idx] = y + 0.38 * eco
            y = y + 0.33 * eco
        else:
            self.delay.buf[idx] = 0.0
        self.delay.avanzar(n)

        # --- Reverb sencilla (cuatro ecos cortos realimentados) ---
        if p["reverb"] >= 0.5:
            mojado = np.zeros(n)
            for i, c in enumerate(self.combs):
                viejo, idx = c.leer(n)
                viejo = viejo.copy()
                c.buf[idx] = y + 0.72 * viejo
                c.avanzar(n)
                mojado += viejo
            y = y + 0.11 * mojado
        else:
            for c in self.combs:
                _, idx = c.leer(n)
                c.buf[idx] = 0.0
                c.avanzar(n)

        return np.tanh(y * p["vol"] * 1.6).astype(np.float32)


class Monitor:
    """Reproduce la guitarra procesada por la salida de audio del PC."""

    def __init__(self):
        self.cadena = Cadena()
        self.activo = False
        self.stream = None
        self.fs_salida = 48000
        self.cola = np.zeros(0, dtype=np.float32)
        self.lock = threading.Lock()
        self.fase = -1.0
        self.ult = 0.0
        self.error = None
        self.lleno = False
        self.frames = 0               # tamaño del bloque que pide la tarjeta
        self.minimos = []             # lo mínimo que quedó en cola en cada bloque (último medio segundo)
        self.latencia_salida = 0.0
        self.colchon = 0.003          # segundos; sube solo si se entrecorta, baja si va estable
        self.estables = 0

    def _callback(self, salida, frames, tiempo, estado):
        with self.lock:
            self.frames = frames
            # colchón mínimo (3 ms) para que no se entrecorte; más colchón = más retraso
            if not self.lleno and len(self.cola) >= frames + int(self.colchon * self.fs_salida):
                self.lleno = True
            if self.lleno and len(self.cola) >= frames:
                bloque = self.cola[:frames]
                self.cola = self.cola[frames:]
                # Si la entrada y la salida van a relojes distintos, la cola crece poco a poco y con ella
                # el retraso. Cuando sobra audio de forma sostenida, se descarta ese sobrante.
                self.minimos.append(len(self.cola))
                if len(self.minimos) >= max(8, int(0.5 * self.fs_salida / max(frames, 1))):
                    sobra = min(self.minimos) - int(self.colchon * self.fs_salida)
                    self.estables += 1
                    if self.estables >= 20 and self.colchon > 0.003:      # 10 s sin cortes: afinar a la baja
                        self.colchon = max(0.003, self.colchon - 0.001)
                        self.estables = 0
                    self.minimos = []
                    if sobra > 0:
                        cola = self.cola[sobra:]
                        if len(cola) > 32:                       # empalme suave para que no haga clic
                            r = np.linspace(0.0, 1.0, 32, dtype=np.float32)
                            cola[:32] = self.cola[:32] * (1 - r) + cola[:32] * r
                        self.cola = cola
            else:
                if self.lleno:                                            # se entrecortó: un poco más de colchón
                    self.colchon = min(0.025, self.colchon + 0.002)
                    self.estables = 0
                bloque = np.zeros(frames, dtype=np.float32)
                self.lleno = False
                self.minimos = []
        salida[:] = 0.0
        salida[:, 0] = bloque
        if salida.shape[1] > 1:
            salida[:, 1] = bloque

    def encender(self):
        if self.stream is not None:
            self.activo = True
            return True
        try:
            import sounddevice as sd
        except Exception as e:
            self.error = str(e)
            return False

        # Probar primero WASAPI (menos retraso) y, si no abre, la salida normal de Windows
        candidatos = []
        try:
            for api in sd.query_hostapis():
                if "WASAPI" in api["name"] and api.get("default_output_device", -1) >= 0:
                    candidatos.append(api["default_output_device"])
        except Exception:
            pass
        candidatos.append(None)

        ultimo = "no hay salida de audio"
        for dispositivo in candidatos:
            try:
                info = sd.query_devices(dispositivo, "output") if dispositivo is not None else sd.query_devices(kind="output")
                fs = int(info["default_samplerate"])
                maximo = int(info["max_output_channels"])
            except Exception as e:
                ultimo = str(e)
                continue
            # WASAPI exige el mismo número de canales que tenga configurado el dispositivo
            for canales in dict.fromkeys([maximo, 2, 1]):
                if canales < 1 or canales > maximo:
                    continue
                try:
                    st = sd.OutputStream(samplerate=fs, channels=canales, dtype="float32", device=dispositivo,
                                         latency="low", blocksize=0, callback=self._callback)
                    st.start()
                    self.latencia_salida = float(st.latency)
                    self.colchon = 0.003
                    self.stream = st
                    self.fs_salida = fs
                    self.activo = True
                    self.error = None
                    return True
                except Exception as e:
                    ultimo = str(e)
        self.error = ultimo
        self.stream = None
        self.activo = False
        return False

    def apagar(self):
        self.activo = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None
        with self.lock:
            self.cola = np.zeros(0, dtype=np.float32)

    def alimentar(self, x16k):
        """Recibe un bloque de la guitarra (16 kHz), le aplica los efectos y lo encola para sonar."""
        if not self.activo or self.stream is None:
            return
        y = self.cadena.procesar(x16k)
        # pasar de 16 kHz a la frecuencia de la tarjeta de sonido (interpolación lineal continua)
        razon = FS / float(self.fs_salida)
        ext = np.concatenate(([self.ult], y))
        pos = np.arange(self.fase, len(y) - 1, razon)          # -1 = última muestra del bloque anterior
        self.fase = (pos[-1] + razon if len(pos) else self.fase) - len(y)
        salida = np.interp(pos + 1.0, np.arange(len(ext)), ext).astype(np.float32)
        self.ult = float(y[-1])
        with self.lock:
            self.cola = np.concatenate((self.cola, salida))
            maximo = max(2 * self.frames, int(0.03 * self.fs_salida))    # tope duro por si algo se atasca
            if len(self.cola) > maximo:
                self.cola = self.cola[-(self.frames + int(self.colchon * self.fs_salida)):]

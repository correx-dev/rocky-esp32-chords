"""
Estructuras de datos propias de Rocky (sin librerías de estructuras: solo listas y números).

Se usan en el sistema así:

  RingBuffer     Buffer circular de muestras de audio. Inserta bloques en O(k) (k = tamaño del bloque)
                 y lee las últimas n muestras en O(n), sin desplazar datos. Es el mismo diseño que
                 usa el firmware de la ESP32 (Firmware/rocky_ia/estructuras.h).
  VotingWindow   Ventana deslizante de las últimas N predicciones del clasificador. Decide el acorde
                 "estable" por mayoría con histéresis, para que el juego no parpadee entre acordes
                 cuando la red duda una vez. Inserción O(1), voto O(C) con C = nº de clases.
  HashTable      Tabla hash con encadenamiento y redimensionado. Guarda la información web de cada
                 canción (clave = "artista|canción") y evita volver a pedir a Wikipedia.
  MinHeap        Montículo binario mínimo. Con TopK se obtienen los k acordes más probables de cada
                 inferencia en O(C log k) sin ordenar todo el vector.

Cada operación indica su complejidad. Las pruebas están en tests/test_estructuras.py.
"""

import numpy as np


# ---------------------------------------------------------------------------------------------
class RingBuffer:
    """Buffer circular de capacidad fija (numpy float32). Sobrescribe lo más viejo cuando se llena."""

    def __init__(self, capacidad, dtype=np.float32):
        if capacidad <= 0:
            raise ValueError("capacidad debe ser > 0")
        self.buf = np.zeros(capacidad, dtype=dtype)
        self.cap = int(capacidad)
        self.total = 0                     # muestras escritas desde el inicio (contador monótono)

    def agregar(self, x):
        """Inserta un bloque. O(len(x)). Si el bloque es mayor que la capacidad, se queda con el final."""
        x = np.asarray(x, dtype=self.buf.dtype)
        if len(x) >= self.cap:
            self.total += len(x)
            self.buf[:] = np.roll(x[-self.cap:], self.total % self.cap)   # conserva la alineación índice = posición mod cap
            return
        pos = self.total % self.cap
        fin = pos + len(x)
        if fin <= self.cap:
            self.buf[pos:fin] = x
        else:
            corte = self.cap - pos
            self.buf[pos:] = x[:corte]
            self.buf[:fin - self.cap] = x[corte:]
        self.total += len(x)

    def ultimas(self, n):
        """Las n muestras más recientes en orden cronológico. O(n). Si hay menos, rellena con ceros al inicio."""
        n = min(int(n), self.cap)
        disponibles = min(self.total, self.cap)
        out = np.zeros(n, dtype=self.buf.dtype)
        m = min(n, disponibles)
        if m == 0:
            return out
        ini = self.total - m                      # índice lógico de la muestra más antigua pedida
        fin = ini % self.cap
        if fin + m <= self.cap:                    # tramo contiguo
            out[n - m:] = self.buf[fin:fin + m]
        else:                                       # el tramo da la vuelta al final del arreglo
            corte = self.cap - fin
            out[n - m:n - m + corte] = self.buf[fin:]
            out[n - m + corte:] = self.buf[:m - corte]
        return out

    def __len__(self):
        return min(self.total, self.cap)


# ---------------------------------------------------------------------------------------------
class VotingWindow:
    """Ventana de votación de las últimas N predicciones (clase, confianza).

    Acorde estable = la clase con más votos si alcanza `minimo` votos de N y su confianza media
    supera `conf_min`. Histéresis: el acorde estable actual solo se cambia si otra clase tiene
    al menos `minimo` votos; si no, se mantiene (evita parpadeo).
    """

    def __init__(self, n=5, minimo=3, conf_min=0.5, n_clases=11):
        self.n, self.minimo, self.conf_min, self.C = n, minimo, conf_min, n_clases
        self.clases = [-1] * n
        self.confs = [0.0] * n
        self.cabeza = 0
        self.cuenta = 0
        self.estable = -1

    def agregar(self, clase, conf):
        """O(1): escribe en la posición `cabeza` (sobrescribe la más vieja)."""
        self.clases[self.cabeza] = int(clase)
        self.confs[self.cabeza] = float(conf)
        self.cabeza = (self.cabeza + 1) % self.n
        self.cuenta = min(self.cuenta + 1, self.n)

    def votar(self):
        """O(N + C). Devuelve (clase_estable, confianza_del_voto) o (-1, 0.0) si aún no hay consenso."""
        if self.cuenta == 0:
            return self.estable, 0.0
        votos = [0] * self.C
        suma = [0.0] * self.C
        for i in range(self.cuenta):
            c = self.clases[i]
            if 0 <= c < self.C:
                votos[c] += 1
                suma[c] += self.confs[i]
        mejor = max(range(self.C), key=lambda k: (votos[k], suma[k]))
        if votos[mejor] >= self.minimo and suma[mejor] / votos[mejor] >= self.conf_min:
            self.estable = mejor
        return self.estable, (votos[self.estable] / self.cuenta if self.estable >= 0 else 0.0)

    def reiniciar(self):
        self.__init__(self.n, self.minimo, self.conf_min, self.C)


# ---------------------------------------------------------------------------------------------
class HashTable:
    """Tabla hash con encadenamiento. Promedio O(1) en poner/buscar/borrar; peor caso O(n).

    Función hash propia: polinómica con base 31 sobre los caracteres de la clave (mod 2^32).
    Se duplica el número de cubetas cuando la carga supera 0.75 (costo amortizado O(1) por inserción).
    """

    def __init__(self, cubetas=16):
        self.cubetas = [[] for _ in range(cubetas)]
        self.n = 0

    @staticmethod
    def hash(clave):
        h = 0
        for ch in str(clave):
            h = (h * 31 + ord(ch)) & 0xFFFFFFFF
        return h

    def _cubeta(self, clave):
        return self.cubetas[self.hash(clave) % len(self.cubetas)]

    def poner(self, clave, valor):
        c = self._cubeta(clave)
        for par in c:
            if par[0] == clave:
                par[1] = valor
                return
        c.append([clave, valor])
        self.n += 1
        if self.n > 0.75 * len(self.cubetas):
            self._crecer()

    def obtener(self, clave, defecto=None):
        for k, v in self._cubeta(clave):
            if k == clave:
                return v
        return defecto

    def contiene(self, clave):
        return any(k == clave for k, _ in self._cubeta(clave))

    def borrar(self, clave):
        c = self._cubeta(clave)
        for i, par in enumerate(c):
            if par[0] == clave:
                c.pop(i)
                self.n -= 1
                return True
        return False

    def _crecer(self):
        viejas = self.cubetas
        self.cubetas = [[] for _ in range(2 * len(viejas))]
        for c in viejas:
            for k, v in c:
                self.cubetas[self.hash(k) % len(self.cubetas)].append([k, v])

    def claves(self):
        return [k for c in self.cubetas for k, _ in c]

    def __len__(self):
        return self.n

    def __contains__(self, clave):
        return self.contiene(clave)

    def __getitem__(self, clave):
        if not self.contiene(clave):
            raise KeyError(clave)
        return self.obtener(clave)

    def __setitem__(self, clave, valor):
        self.poner(clave, valor)


# ---------------------------------------------------------------------------------------------
class MinHeap:
    """Montículo binario mínimo en un arreglo. push/pop O(log n), peek O(1), construcción O(n log n)."""

    def __init__(self):
        self.a = []

    def __len__(self):
        return len(self.a)

    def peek(self):
        return self.a[0]

    def push(self, prioridad, dato=None):
        self.a.append((prioridad, dato))
        i = len(self.a) - 1
        while i > 0:
            p = (i - 1) // 2
            if self.a[p][0] <= self.a[i][0]:
                break
            self.a[p], self.a[i] = self.a[i], self.a[p]
            i = p

    def pop(self):
        a = self.a
        if not a:
            raise IndexError("heap vacío")
        raiz = a[0]
        ultimo = a.pop()
        if a:
            a[0] = ultimo
            i, n = 0, len(a)
            while True:
                l, r, m = 2 * i + 1, 2 * i + 2, i
                if l < n and a[l][0] < a[m][0]:
                    m = l
                if r < n and a[r][0] < a[m][0]:
                    m = r
                if m == i:
                    break
                a[i], a[m] = a[m], a[i]
                i = m
        return raiz


def top_k(valores, k):
    """Los k mayores de `valores` como [(indice, valor), ...] de mayor a menor. O(C log k).

    Usa un MinHeap de tamaño k: si llega un valor mayor que la raíz (el menor de los k), la reemplaza.
    """
    h = MinHeap()
    for i, v in enumerate(valores):
        if len(h) < k:
            h.push(float(v), i)
        elif v > h.peek()[0]:
            h.pop()
            h.push(float(v), i)
    salida = []
    while len(h):
        p, i = h.pop()
        salida.append((i, p))
    return salida[::-1]

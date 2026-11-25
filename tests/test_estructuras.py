"""Pruebas de las estructuras propias (python -m unittest tests/test_estructuras.py)."""
import os, random, sys, unittest
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Python"))
from estructuras import RingBuffer, VotingWindow, HashTable, MinHeap, top_k


class TestRing(unittest.TestCase):
    def test_contra_referencia(self):
        rng = np.random.default_rng(1)
        rb = RingBuffer(100)
        ref = []
        for _ in range(300):
            x = rng.normal(size=int(rng.integers(1, 130))).astype(np.float32)
            rb.agregar(x)
            ref.extend(x.tolist())
            for n in (1, 7, 50, 100):
                esperado = np.array(ref[-n:], dtype=np.float32)
                esperado = np.pad(esperado, (n - len(esperado), 0))
                np.testing.assert_array_equal(rb.ultimas(n), esperado)

    def test_vacio(self):
        self.assertEqual(rb := RingBuffer(8).ultimas(4).sum(), 0)


class TestVoto(unittest.TestCase):
    def test_mayoria_e_histeresis(self):
        v = VotingWindow(n=5, minimo=3, conf_min=0.5, n_clases=11)
        for c in (0, 0, 0, 1, 2):
            v.agregar(c, 0.9)
        self.assertEqual(v.votar()[0], 0)
        v.agregar(1, 0.9); v.agregar(2, 0.9)      # ventana: 0,1,2,1,2 -> nadie llega a 3 votos
        self.assertEqual(v.votar()[0], 0)         # se mantiene el estable anterior
        v.agregar(1, 0.9); v.agregar(1, 0.9)      # ahora 1 tiene 3 votos
        self.assertEqual(v.votar()[0], 1)

    def test_confianza_baja_no_vota(self):
        v = VotingWindow(5, 3, 0.6, 11)
        for _ in range(5):
            v.agregar(4, 0.3)
        self.assertEqual(v.votar()[0], -1)


class TestHash(unittest.TestCase):
    def test_contra_dict(self):
        h, d = HashTable(4), {}
        r = random.Random(7)
        for _ in range(5000):
            k = f"k{r.randint(0, 400)}"
            op = r.random()
            if op < 0.5:
                h.poner(k, op); d[k] = op
            elif op < 0.75:
                self.assertEqual(h.obtener(k), d.get(k))
            else:
                self.assertEqual(h.borrar(k), d.pop(k, None) is not None)
            self.assertEqual(len(h), len(d))
        self.assertEqual(sorted(h.claves()), sorted(d))

    def test_crece(self):
        h = HashTable(2)
        for i in range(100):
            h[f"x{i}"] = i
        self.assertGreaterEqual(len(h.cubetas), 128)
        self.assertEqual(h["x57"], 57)


class TestHeap(unittest.TestCase):
    def test_orden(self):
        r = random.Random(3)
        datos = [r.random() for _ in range(500)]
        h = MinHeap()
        for x in datos:
            h.push(x)
        self.assertEqual([h.pop()[0] for _ in range(500)], sorted(datos))

    def test_top_k(self):
        v = np.random.default_rng(5).random(11)
        esp = sorted(range(11), key=lambda i: -v[i])[:3]
        self.assertEqual([i for i, _ in top_k(v, 3)], esp)


if __name__ == "__main__":
    unittest.main()

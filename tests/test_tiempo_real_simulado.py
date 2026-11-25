"""
Prueba de extremo a extremo SIN hardware: una "ESP32 falsa" (pseudo-terminal) manda audio en tiempo real con el
protocolo A5 5A; rocky.py lo lee por el puerto serie, la IA del PC clasifica y se escuchan los mensajes UDP que
recibiría Unity. Comprueba el protocolo, el buffer circular, la latencia y la ventana de votación.
Uso:  python tests/test_tiempo_real_simulado.py [acorde] [segundos]       (Linux/Mac; en Windows usa com0com)
"""
import os, pty, socket, struct, sys, threading, time
import numpy as np

RAIZ = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(RAIZ, "Python")); sys.path.insert(0, os.path.join(RAIZ, "IA"))
import rocky, ia_vivo, ia_features as F, entrenar as E

acorde = sys.argv[1] if len(sys.argv) > 1 else "Am"
segundos = float(sys.argv[2]) if len(sys.argv) > 2 else 10
rng = np.random.default_rng(3)
x = np.zeros(int(segundos * 16000), np.float32)
pos = int(1.5 * 16000)                                       # 1.5 s de silencio y luego rasgueos cada ~1.5 s
while pos < len(x) - 16000:
    clip = np.zeros(int(1.5 * 16000), np.float32)
    for s, tr in enumerate(E.FORMAS[acorde]):
        if tr >= 0:
            ret = int(s * rng.uniform(0.005, 0.02) * 16000)
            nota = E.cuerda(E.AL_AIRE[s] + tr, 1.5, rng) * rng.uniform(0.5, 1.0)
            clip[ret:] += nota[: len(clip) - ret]
    clip = clip / (np.abs(clip).max() + 1e-9) * rng.uniform(0.2, 0.8)
    n = min(len(clip), len(x) - pos)
    x[pos:pos + n] += clip[:n]
    pos += int(rng.uniform(1.3, 1.6) * 16000)
x += rng.normal(0, 0.002, len(x)).astype(np.float32)
pcm = np.clip(x * 32767, -32768, 32767).astype("<i2")

maestro, esclavo = pty.openpty()
puerto = os.ttyname(esclavo)

def esp32_falsa():
    seq, t0 = 0, time.time()
    for i in range(0, len(pcm) - 64, 64):
        espera = t0 + i / 16000 - time.time()
        if espera > 0:
            time.sleep(espera)
        os.write(maestro, b"\xA5\x5A" + struct.pack("<HH", seq & 0xFFFF, 64) + pcm[i:i + 64].tobytes())
        seq += 1

udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); udp.bind(("127.0.0.1", 5005)); udp.settimeout(0.2)
mensajes = []
def escuchar():
    fin = time.time() + segundos + 2
    while time.time() < fin:
        try:
            mensajes.append((time.time(), udp.recvfrom(512)[0].decode("ascii", "ignore")))
        except socket.timeout:
            pass

estado, anillo = {"fuente": "prueba"}, rocky.Anillo()
rocky.ANILLO, rocky.ESTADO = anillo, estado
for t in (threading.Thread(target=rocky.hilo_serial, args=(anillo, puerto, estado), daemon=True),
          threading.Thread(target=ia_vivo.hilo_ia_pc, args=(anillo, estado, "pc", None, lambda *a: None), daemon=True),
          threading.Thread(target=rocky.bucle_envio, args=(anillo, estado), daemon=True),
          threading.Thread(target=escuchar)):
    t.start()
time.sleep(0.3); inicio = time.time()
esp32_falsa()
time.sleep(1.5)
c1 = [(t - inicio, m.split(";")) for t, m in mensajes if m.startswith("C1")]
r1 = [m for _, m in mensajes if m.startswith("R1")]
print(f"acorde tocado: {acorde} · mensajes C1: {len(c1)} · R1 (notas YIN): {len(r1)} · paquetes: {estado.get('paquetes')} perdidos: {estado.get('perdidos', 0)}")
for t, m in c1[::4]:
    print(f"  t={t:5.2f}s clase={m[1]:8s} conf={m[2]} estable={m[3]:6s} lat={m[5]} ms (feat {m[6]}, red {m[7]})")
activos = [m for t, m in c1 if t > 4.0]                      # ya hay >=2.5 s de rasgueos en la ventana
aciertos = sum(1 for m in activos if m[3] == acorde)
frac = aciertos / max(1, len(activos))
ok = frac >= 0.8
print(f"RESULTADO: {'OK' if ok else 'FALLO'} · acorde estable correcto en {100 * frac:.0f} % de los mensajes desde t=4 s (n={len(activos)})")
sys.exit(0 if ok else 1)

"""
Grabador guiado del dataset de acordes por consola (requisito R4).

Sirve cuando el juego no trae la pantalla «Entrena a la IA»: este programa hace el papel del juego.
Con el servidor (rocky.py / Rocky.exe) ya abierto y la guitarra sonando, muestra un acorde, cuenta
3-2-1 y manda la orden  GRABAR|clase|sesion|guitarrista|guitarra  al servidor (UDP 5006).
El servidor guarda 1.5 s de audio en IA/dataset/v1/<clase>/ y anota metadata.csv (ver ia_dataset.py).
La confirmación vuelve por UDP 5005 como  D1;ok|error;clase;cuenta;rms;detalle.

Uso:
    python Python/grabar_cli.py --guitarrista Ana --guitarra "Yamaha F310" --tomas 30
    python Python/grabar_cli.py --clases C,G,Am --tomas 10 --sesion sesion2

Importante: cierra el juego antes (comparte el puerto 5005 de respuestas) y deja rocky.py abierto.
Cada tanda (sesion) debe grabarse de una vez: el entrenamiento separa train/val/test por sesión.
Si graba otra persona, guarda su autorización por escrito junto con metadata.csv.
"""

import argparse
import datetime
import socket
import sys
import time

CLASES = ["C", "D", "E", "G", "A", "Am", "Em", "Dm", "F", "B7", "ninguno"]
ORDEN = ("127.0.0.1", 5006)
RESPUESTAS = ("127.0.0.1", 5005)
ESPERA_RESPUESTA = 6.0


def abrir_respuestas():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind(RESPUESTAS)
    s.settimeout(0.3)
    return s


def esperar(sock, prefijo, limite=ESPERA_RESPUESTA):
    fin = time.time() + limite
    while time.time() < fin:
        try:
            datos, _ = sock.recvfrom(8192)
        except socket.timeout:
            continue
        except OSError:
            return None
        texto = datos.decode("utf-8", "replace").strip()
        if texto.startswith(prefijo):
            return texto
    return None


def mandar(sock, texto):
    sock.sendto(texto.encode("utf-8"), ORDEN)


def cuenta_regresiva(n=3):
    for i in range(n, 0, -1):
        print("   %d..." % i, end="\r", flush=True)
        time.sleep(1)
    print("   ¡TOCA!      ", flush=True)


def conteos(sock, envio):
    mandar(envio, "DATASET?")
    r = esperar(sock, "D2;", 3.0)
    if not r:
        return None
    partes = r.split(";")
    out = {}
    for par in partes[1].split(","):
        if ":" in par:
            k, v = par.split(":")
            out[k] = int(v)
    return out


def main():
    ap = argparse.ArgumentParser(description="Grabador guiado del dataset de acordes")
    ap.add_argument("--clases", default=",".join(CLASES), help="lista separada por comas (por defecto, las 11)")
    ap.add_argument("--tomas", type=int, default=30, help="tomas por clase en esta tanda")
    ap.add_argument("--sesion", default=datetime.datetime.now().strftime("s%Y%m%d_%H%M"))
    ap.add_argument("--guitarrista", required=True)
    ap.add_argument("--guitarra", required=True)
    ap.add_argument("--pausa", type=float, default=1.0, help="segundos entre tomas")
    a = ap.parse_args()

    clases = [c.strip() for c in a.clases.split(",") if c.strip()]
    malas = [c for c in clases if c not in CLASES]
    if malas:
        sys.exit("Clases desconocidas: %s. Válidas: %s" % (malas, CLASES))

    try:
        rx = abrir_respuestas()
    except OSError as e:
        sys.exit("No pude abrir el puerto 5005 (¿está abierto el juego?). Ciérralo e inténtalo otra vez. %s" % e)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    previo = conteos(rx, tx)
    if previo is None:
        sys.exit("El servidor no responde. Abre rocky.py (o Rocky.exe), conecta la guitarra y vuelve a probar.")

    print("Sesión %s · guitarrista %s · guitarra %s" % (a.sesion, a.guitarrista, a.guitarra))
    print("Tomas ya guardadas:", previo)
    buenas = {c: 0 for c in clases}
    malas_n = {c: 0 for c in clases}
    for c in clases:
        input("\nAcorde %s — prepara la mano y pulsa Enter para empezar (%d tomas)... " % (c, a.tomas))
        hechas, intentos = 0, 0
        while hechas < a.tomas and intentos < a.tomas * 3:
            intentos += 1
            print(" Toma %d/%d de %s" % (hechas + 1, a.tomas, c))
            cuenta_regresiva()
            mandar(tx, "GRABAR|%s|%s|%s|%s" % (c, a.sesion, a.guitarrista, a.guitarra))
            r = esperar(rx, "D1;")
            if r is None:
                print("   Sin respuesta del servidor; la toma no se contó.")
                malas_n[c] += 1
                continue
            p = r.split(";")
            if p[1] == "ok":
                hechas += 1
                buenas[c] += 1
                print("   guardada (total de %s: %s, RMS %s)" % (c, p[3], p[4]))
            else:
                malas_n[c] += 1
                print("   rechazada: %s" % (p[5] if len(p) > 5 else "sin detalle"))
            time.sleep(a.pausa)

    print("\nResumen de esta tanda:")
    for c in clases:
        print("  %-8s buenas=%d rechazadas=%d" % (c, buenas[c], malas_n[c]))
    final = conteos(rx, tx)
    if final:
        print("Total en el dataset:", final)


if __name__ == "__main__":
    main()

"""
Recibe lo que calcula la IA que corre DENTRO de la ESP32-S3 (el PC no clasifica nada).

  - Acordes: los reenvía a Unity como «C1;clase;confianza;clase_estable;latencia_ms».
  - Recursos: el JSON que la ESP manda cada 10 s (Flash, RAM, arena, latencia, ciclos, carga)
    se guarda en IA/mediciones/recursos_<fecha>.csv con los porcentajes sobre el total: es la
    evidencia del requisito R7 y se puede repetir en la sustentación.
"""

import csv
import json
import os
import socket
import struct
import time

import ia_features

AQUI = os.path.dirname(os.path.abspath(__file__))
CARPETA = os.path.normpath(os.path.join(AQUI, "..", "IA", "mediciones"))
UNITY = ("127.0.0.1", 5005)
_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
_archivo = None
_ultimo_log = [0.0]
ULTIMO_ACORDE = [0.0]      # instante del último acorde recibido de la ESP32


def esp_activa(ventana=2.0):
    """True si la ESP32 mandó un acorde en los últimos `ventana` segundos (entonces el PC no clasifica)."""
    return time.time() - ULTIMO_ACORDE[0] < ventana


def _nombre(i):
    return ia_features.CLASES[i] if 0 <= i < len(ia_features.CLASES) else "?"


def paquete_valido(buf):
    n = len(ia_features.CLASES)
    return buf[2] < n and (buf[4] < n or buf[4] == 255)


def recibir_acorde(p):
    ULTIMO_ACORDE[0] = time.time()
    clase, conf, voto, conf_voto = p[2], p[3] / 255.0, p[4], p[5] / 255.0
    lat_us, feat_us, inf_us = struct.unpack_from("<III", p, 6)
    msg = "C1;%s;%.3f;%s;%.2f;%.2f;%.2f" % (_nombre(clase), conf, _nombre(voto) if voto != 255 else "",
                                           lat_us / 1000, feat_us / 1000, inf_us / 1000)
    try:
        _sock.sendto(msg.encode("ascii"), UNITY)
    except OSError:
        pass


def recibir_recursos(texto, estado, log=None):
    global _archivo
    try:
        d = json.loads(texto.decode("ascii", "ignore"))
    except ValueError:
        return
    def pct(a, b):
        return round(100.0 * a / b, 2) if b else 0.0
    d["flash_app_pct"] = pct(d.get("flash_app", 0), d.get("flash_particion", 0))
    d["modelo_pct_flash_chip"] = pct(d.get("modelo_bytes", 0), d.get("flash_chip", 0))
    usada = d.get("ram_interna_total", 0) - d.get("ram_interna_libre", 0)
    d["ram_interna_usada"] = usada
    d["ram_interna_pct"] = pct(usada, d.get("ram_interna_total", 0))
    d["arena_pct"] = pct(d.get("arena_usada", 0), d.get("arena_reservada", 0))
    d["fecha"] = time.strftime("%Y-%m-%d %H:%M:%S")
    estado["recursos_ia"] = d
    os.makedirs(CARPETA, exist_ok=True)
    if _archivo is None:
        _archivo = os.path.join(CARPETA, "recursos_" + time.strftime("%Y%m%d_%H%M%S") + ".csv")
    nuevo = not os.path.exists(_archivo)
    with open(_archivo, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=sorted(d.keys()), extrasaction="ignore")
        if nuevo:
            w.writeheader()
        w.writerow(d)
    if time.time() - _ultimo_log[0] > 60:
        _ultimo_log[0] = time.time()
        print("[IA ESP32] latencia %.1f±%.1f ms (n=%d) · Flash app %.1f %% · RAM interna %.1f %% · arena %d B · carga núcleo 0 %.1f %%"
              % (d.get("lat_ms", 0), d.get("lat_ms_desv", 0), d.get("n", 0), d["flash_app_pct"], d["ram_interna_pct"],
                 d.get("arena_usada", 0), d.get("carga_nucleo0_pct", 0)), flush=True)

"""
Web scraping de información de canciones para Rocky.

Cuando hay una canción nueva, este módulo entra a la página de esa canción en Wikipedia (en inglés),
lee el HTML con BeautifulSoup y extrae de la ficha (la "infobox" de la derecha):
    artista, álbum, año, géneros, duración, sello, compositores.
El primer género reconocido decide el "ambiente" del juego (color del escenario):
    Suave · Pop / Rock · Rock · Metal

Se guarda junto a la chart como  <nombre>.info.json  para no volver a pedirlo.

Buenas prácticas de scraping que se respetan:
  - Solo se piden páginas de artículos (/wiki/...), que el robots.txt de Wikipedia permite.
  - Se envía un User-Agent que identifica al programa.
  - Una petición a la vez, con pausa de 1 s entre peticiones, y resultados en caché.
  - Si no hay internet, el juego sigue igual; se reintenta más tarde.
"""

import json
import os
import re
import time
import urllib.parse

from estructuras import HashTable

BASE = "https://en.wikipedia.org/wiki/"
CABECERAS = {"User-Agent": "RockyGame/1.0 (proyecto universitario; juego de guitarra) python-requests"}
PAUSA = 1.0
REINTENTO_NO_ENCONTRADA = 7 * 24 * 3600      # si no se encontró, volver a buscar en una semana

_ultima_peticion = [0.0]

# Palabras que se escriben en minúscula dentro de títulos en inglés (estilo de Wikipedia)
_MINUSCULAS = {"a", "an", "the", "and", "but", "or", "nor", "for", "so", "yet", "as", "at", "by",
               "in", "of", "off", "on", "per", "to", "up", "via", "with", "from", "into", "onto", "over"}

# Géneros -> ambiente del juego. Se revisan en este orden de prioridad dentro de cada género.
_AMBIENTES = [
    ("Metal", ("metal", "thrash", "djent", "grindcore", "deathcore", "doom", "sludge", "speed")),
    ("Rock", ("rock", "punk", "grunge", "emo", "post-hardcore", "hardcore", "shoegaze", "garage")),
    ("Pop / Rock", ("pop", "dance", "disco", "funk", "hip hop", "hip-hop", "rap", "r&b", "reggaeton", "latin",
                    "electronic", "edm", "house", "synth", "trap", "k-pop", "cumbia", "salsa", "merengue")),
    ("Suave", ("ballad", "soul", "folk", "acoustic", "jazz", "blues", "country", "bossa", "classical",
               "gospel", "lo-fi", "ambient", "bolero", "vallenato", "soft")),
]


def limpiar_titulo(titulo):
    """'Metallica - Enter Sandman (Lead)'  ->  ('Metallica', 'Enter Sandman')"""
    t = titulo
    for _ in range(4):                                   # quitar (Lead), [Official Video], etc. del final
        t2 = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]\s*$", "", t)
        if t2 == t:
            break
        t = t2
    t = re.sub(r"\s+(official|lyrics?|audio|video|hd|4k)\b.*$", "", t, flags=re.I).strip()
    if " - " in t:
        artista, cancion = t.split(" - ", 1)
    else:
        artista, cancion = "", t
    return artista.strip(), cancion.strip(" -\"'")


def _estilo_wiki(texto):
    palabras = texto.split()
    out = []
    for i, p in enumerate(palabras):
        bajo = p.lower()
        if 0 < i < len(palabras) - 1 and bajo in _MINUSCULAS:
            out.append(bajo)
        else:
            out.append(p[:1].upper() + p[1:])
    return " ".join(out)


def candidatos(artista, cancion):
    """Posibles nombres de la página de Wikipedia para esta canción."""
    nombres = []
    primer_artista = re.split(r",| & | and | feat\.? | ft\.? | x ", artista, flags=re.I)[0].strip()
    for forma in dict.fromkeys([cancion, _estilo_wiki(cancion)]):
        if not forma:
            continue
        nombres.append(forma)
        nombres.append(f"{forma} (song)")
        if primer_artista:
            nombres.append(f"{forma} ({primer_artista} song)")
    return list(dict.fromkeys(nombres))


def _pedir(sesion, titulo):
    espera = PAUSA - (time.time() - _ultima_peticion[0])
    if espera > 0:
        time.sleep(espera)
    _ultima_peticion[0] = time.time()
    url = BASE + urllib.parse.quote(titulo.replace(" ", "_"))
    r = sesion.get(url, headers=CABECERAS, timeout=12)
    if r.status_code == 404:
        return None, url
    r.raise_for_status()
    return r.text, r.url


def _texto(nodo):
    """Texto limpio de una celda: sin notas [1], sin espacios repetidos."""
    for sup in nodo.find_all(["sup", "style"]):
        sup.decompose()
    for br in nodo.find_all("br"):
        br.replace_with(", ")
    return re.sub(r"\s+", " ", nodo.get_text(" ", strip=True)).replace(" ,", ",").strip(" ,")


def _lista(td):
    """Celdas con varios valores (géneros, compositores): devuelve una lista."""
    items = [_texto(li) for li in td.find_all("li")]
    if not items:
        items = [x.strip() for x in re.split(r",|·|\n", _texto(td))]
    return [i[:1].upper() + i[1:] for i in items if i]


def leer_ficha(html):
    """Extrae los datos de la infobox de una página de canción. Devuelve None si no es una canción."""
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "html.parser")
    ficha = soup.find("table", class_="infobox")
    if ficha is None:
        return None
    datos = {}
    arriba = ficha.find(class_="infobox-above")
    if arriba:
        datos["titulo"] = _texto(arriba).strip("\"“”")
    for th in ficha.find_all("th", class_="infobox-header"):
        t = _texto(th)
        m = re.match(r"^(Single|Song|Promotional single|Instrumental)( by| from)?\s+(.*)$", t, re.I)
        if m and m.group(2) and m.group(2).strip().lower() == "by":
            datos["artista"] = m.group(3).strip()
            datos["tipo"] = m.group(1)
        m = re.match(r"^from the (?:album|EP|soundtrack|mixtape)\s+(.*)$", t, re.I)
        if m:
            datos["album"] = m.group(1).strip()
    if "artista" not in datos:
        return None                                      # la ficha no es de una canción (es de un álbum, persona...)
    for fila in ficha.find_all("tr"):
        etiqueta = fila.find("th", class_="infobox-label")
        valor = fila.find("td", class_="infobox-data")
        if not etiqueta or not valor:
            continue
        clave = _texto(etiqueta).lower()
        if clave.startswith("released"):
            datos["lanzamiento"] = _texto(valor)
            a = re.search(r"\b(19|20)\d{2}\b", datos["lanzamiento"])
            if a:
                datos["anio"] = int(a.group(0))
        elif clave.startswith("genre"):
            datos["generos"] = _lista(valor)
        elif clave.startswith("length"):
            datos["duracion"] = _texto(valor).split(",")[0]
        elif clave.startswith("label"):
            datos["sello"] = ", ".join(_lista(valor))
        elif clave.startswith("songwriter") or clave.startswith("composer"):
            datos["compositores"] = ", ".join(_lista(valor))
        elif clave.startswith("producer"):
            datos["productores"] = ", ".join(_lista(valor))
    return datos


def ambiente_de(generos):
    for g in generos or []:
        gl = g.lower()
        for nombre, claves in _AMBIENTES:
            if any(c in gl for c in claves):
                return nombre
    return ""


def _mismo_artista(buscado, encontrado):
    if not buscado:
        return True
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()
    b, e = norm(buscado), norm(encontrado)
    primero = norm(re.split(r",| & | and | feat", buscado, flags=re.I)[0])
    return bool(primero) and (primero in e or e in b)


def buscar(titulo, sesion=None):
    """Devuelve un diccionario con la información (encontrado=True/False). Lanza excepción si no hay red."""
    import requests
    sesion = sesion or requests.Session()
    artista, cancion = limpiar_titulo(titulo)
    for nombre in candidatos(artista, cancion):
        html, url = _pedir(sesion, nombre)
        if not html:
            continue
        datos = leer_ficha(html)
        if not datos or not _mismo_artista(artista, datos.get("artista", "")):
            continue
        datos["ambiente"] = ambiente_de(datos.get("generos"))
        datos.update(encontrado=True, fuente=url, consulta=f"{artista} - {cancion}", fecha=int(time.time()))
        return datos
    return {"encontrado": False, "consulta": f"{artista} - {cancion}", "fecha": int(time.time())}


def ruta_info(ruta_chart):
    return ruta_chart[: -len(".chart.json")] + ".info.json" if ruta_chart.endswith(".chart.json") else ruta_chart + ".info.json"


def completar_carpeta(carpeta, log=print):
    """Busca la información de las canciones que aún no la tienen. Varios arreglos de la misma canción
    (Lead, Rítmica, Bajo) comparten una sola búsqueda."""
    import requests
    sesion = requests.Session()
    cache = HashTable()      # tabla hash propia: clave (artista, canción) -> ficha de Wikipedia
    hechas = 0
    for nombre in sorted(os.listdir(carpeta)):
        if not nombre.endswith(".chart.json"):
            continue
        chart = os.path.join(carpeta, nombre)
        destino = ruta_info(chart)
        if os.path.exists(destino):
            try:
                with open(destino, encoding="utf-8") as fh:
                    viejo = json.load(fh)
                if viejo.get("encontrado") or time.time() - viejo.get("fecha", 0) < REINTENTO_NO_ENCONTRADA:
                    continue
            except (OSError, ValueError):
                pass
        try:
            with open(chart, encoding="utf-8") as fh:
                d = json.load(fh)
            if d.get("generator") == "leccion" or str(d.get("title", "")).lower().startswith(("demo", "lección", "leccion")):
                continue
            titulo = d.get("title") or nombre[: -len(".chart.json")]
        except (OSError, ValueError):
            continue
        clave = limpiar_titulo(titulo)
        if clave not in cache:
            cache[clave] = buscar(titulo, sesion)
            info = cache[clave]
            if info["encontrado"]:
                log(f"Wikipedia: «{info['consulta']}» -> {info.get('artista')}, "
                    f"{info.get('anio', '?')}, {', '.join(info.get('generos', [])[:3]) or 'sin género'}")
            else:
                log(f"Wikipedia: no encontré la página de «{info['consulta']}»")
        with open(destino + ".tmp", "w", encoding="utf-8") as fh:
            json.dump(cache[clave], fh, ensure_ascii=False, indent=1)
        os.replace(destino + ".tmp", destino)
        hechas += 1
    return hechas


if __name__ == "__main__":
    import sys
    for t in sys.argv[1:] or ["Metallica - Enter Sandman (Lead)"]:
        print(json.dumps(buscar(t), ensure_ascii=False, indent=1))

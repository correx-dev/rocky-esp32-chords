"""
Lectura de partituras / tablaturas para Rocky.

Formatos:
    Guitar Pro   .gp3 .gp4 .gp5      (trae cuerda y traste exactos)
    MusicXML     .musicxml .xml .mxl (cuerda y traste si el archivo trae tablatura)
    MIDI         .mid .midi          (solo notas; la cuerda y el traste se calculan)

Cada lector devuelve:  (notas, beats, titulo)
    notas = lista de dicts {t, d, s, f, m}   (s: 0 = 6.ª cuerda ... 5 = 1.ª)
    beats = lista de segundos donde cae cada tiempo del compás
"""

import os
import zipfile
import xml.etree.ElementTree as ET

import numpy as np

EXT_PARTITURA = (".gp3", ".gp4", ".gp5", ".musicxml", ".mxl", ".xml", ".mid", ".midi")
EXT_NO_SOPORTADA = (".gp", ".gpx", ".pdf", ".png", ".jpg", ".jpeg")
CUERDAS_AL_AIRE = [40, 45, 50, 55, 59, 64]
QUARTER = 960


# ---------------------------------------------------------------------------
#  Cuerda y traste cuando el archivo no los trae
# ---------------------------------------------------------------------------
def _a_rango(m):
    while m < 40:
        m += 12
    while m > 84:
        m -= 12
    return m


def asignar_posiciones(eventos):
    """eventos: lista de (t, d, midi). Devuelve notas con cuerda/traste tocables.
    Las notas simultáneas (acordes) se reparten en cuerdas distintas."""
    eventos = sorted(eventos, key=lambda e: (round(e[0], 3), e[2]))
    notas = []
    mano = 3.0
    i = 0
    while i < len(eventos):
        j = i
        while j < len(eventos) and abs(eventos[j][0] - eventos[i][0]) < 0.03:
            j += 1
        grupo = eventos[i:j][:6]
        libre_desde = 0
        for t, d, m in grupo:
            m = _a_rango(int(m))
            mejor = None
            for s in range(libre_desde, 6):
                f = m - CUERDAS_AL_AIRE[s]
                if 0 <= f <= 19:
                    costo = (0.3 if f == 0 else abs(f - mano)) + (0.5 if f > 12 else 0)
                    if mejor is None or costo < mejor[0]:
                        mejor = (costo, s, f)
            if mejor is None:
                continue                      # no cabe en el acorde: se omite
            _, s, f = mejor
            libre_desde = s + 1
            if f > 0:
                mano = 0.7 * mano + 0.3 * f
            notas.append({"t": round(eventos[i][0], 3), "d": round(d if d > 0.7 else 0.0, 3), "s": s, "f": f, "m": m})
        i = j
    return notas


# ---------------------------------------------------------------------------
#  Guitar Pro 3 / 4 / 5
# ---------------------------------------------------------------------------
def _orden_compases(headers):
    """Desenrolla las repeticiones (con casillas de 1.ª/2.ª vez)."""
    orden = []
    i, inicio, pasada = 0, 0, 1
    hechas = {}
    guarda = 0
    while i < len(headers) and guarda < 20000:
        guarda += 1
        h = headers[i]
        if getattr(h, "isRepeatOpen", False) and hechas.get(("abre", i)) is None:
            hechas[("abre", i)] = True
            inicio, pasada = i, 1
        alt = getattr(h, "repeatAlternative", 0) or 0
        if alt and not (alt & (1 << (pasada - 1))):
            i += 1
            continue
        orden.append(i)
        veces = getattr(h, "repeatClose", -1)
        if veces is not None and veces > 0:
            if hechas.get(("cierra", i), 0) < veces:
                hechas[("cierra", i)] = hechas.get(("cierra", i), 0) + 1
                pasada += 1
                i = inicio
                continue
            hechas[("cierra", i)] = 0
            hechas.pop(("abre", inicio), None)
            pasada = 1
        i += 1
    return orden


def _tecnicas_gp(n, pulso):
    """Técnicas de una nota de Guitar Pro -> (bits, bend en semitonos)."""
    x, bend = 0, 0.0
    e = n.effect
    try:
        if getattr(e, "palmMute", False): x |= TEC_PALM
        if getattr(e, "harmonic", None) is not None: x |= TEC_ARMONICO
        if getattr(e, "vibrato", False) or getattr(pulso.effect, "vibrato", False): x |= TEC_VIBRATO
        if getattr(e, "tremoloPicking", None) is not None: x |= TEC_TREMOLO
        if getattr(e, "accentuatedNote", False) or getattr(e, "heavyAccentuatedNote", False): x |= TEC_ACENTO
        if getattr(e, "slides", None): x |= TEC_SLIDE
        curva = getattr(e, "bend", None)
        if curva is not None and getattr(curva, "points", None):
            # en Guitar Pro, valor 2 = medio tono (1 semitono)
            bend = round(max(p.value for p in curva.points) / 2.0, 2)
    except Exception:
        pass
    return x, bend


def leer_guitarpro(ruta):
    import guitarpro
    from guitarpro import NoteType

    cancion = guitarpro.parse(ruta)
    candidatas = [p for p in cancion.tracks if not p.isPercussionTrack]
    if not candidatas:
        raise ValueError("el archivo no tiene pistas de guitarra")

    def cuenta(p):
        return sum(len(b.notes) for c in p.measures for v in c.voices for b in v.beats)

    seis = [p for p in candidatas if len(p.strings) == 6]
    pista = max(seis or candidatas, key=cuenta)
    es_seis = len(pista.strings) == 6
    afinacion = {c.number: c.value for c in pista.strings}
    capo = getattr(pista, "offset", 0) or 0

    tempo = float(cancion.tempo)
    t = 0.0
    notas, crudas, beats, secciones = [], [], [], []
    anterior_hammer = {}
    for idx in _orden_compases([c.header for c in pista.measures]):
        compas = pista.measures[idx]
        marcador = getattr(compas.header, "marker", None)
        if marcador is not None and getattr(marcador, "title", ""):
            secciones.append((round(t, 3), str(marcador.title).strip()))
        ts = compas.header.timeSignature
        largo = ts.numerator * QUARTER * 4 // ts.denominator.value

        # Juntar los pulsos de todas las voces con su posición dentro del compás
        pulsos = []
        for voz in compas.voices:
            pos = 0
            for b in voz.beats:
                pulsos.append((pos, b))
                pos += b.duration.time
        pulsos.sort(key=lambda x: x[0])

        seg, ult = t, 0
        tiempos_tick = list(range(0, largo, QUARTER * 4 // ts.denominator.value))
        k = 0
        for pos, b in pulsos:
            while k < len(tiempos_tick) and tiempos_tick[k] <= pos:
                beats.append(seg + (tiempos_tick[k] - ult) * 60.0 / (tempo * QUARTER))
                k += 1
            seg += (pos - ult) * 60.0 / (tempo * QUARTER)
            ult = pos
            mezcla = getattr(b.effect, "mixTableChange", None)
            if mezcla is not None and getattr(mezcla, "tempo", None) is not None and mezcla.tempo.value > 0:
                tempo = float(mezcla.tempo.value)
            dur = b.duration.time * 60.0 / (tempo * QUARTER)
            for n in b.notes:
                if n.type != NoteType.normal:
                    continue                                   # ligaduras, notas muertas, silencios
                m = afinacion.get(n.string, 40) + n.value + capo
                if es_seis:
                    x, bend = _tecnicas_gp(n, b)
                    larga = dur > 0.7 or ((bend > 0 or x & (TEC_SLIDE | TEC_VIBRATO | TEC_TREMOLO)) and dur > 0.12)
                    nueva = {"t": round(seg, 3), "d": round(dur if larga else 0.0, 3),
                             "s": 6 - n.string, "f": int(n.value), "m": int(m)}
                    if x:
                        nueva["x"] = x
                    if bend > 0:
                        nueva["b"] = bend
                    if anterior_hammer.get(n.string):          # la nota que sigue a un ligado se toca sin púa
                        prev_f = anterior_hammer[n.string]
                        nueva["x"] = nueva.get("x", 0) | (TEC_HAMMER if n.value >= prev_f else TEC_PULL)
                    anterior_hammer[n.string] = (n.value or 0.5) if getattr(n.effect, "hammer", False) else None
                    notas.append(nueva)
                else:
                    crudas.append((seg, dur, m))
        while k < len(tiempos_tick):
            beats.append(seg + (tiempos_tick[k] - ult) * 60.0 / (tempo * QUARTER))
            k += 1
        t = seg + (largo - ult) * 60.0 / (tempo * QUARTER)

    if not es_seis:
        notas = asignar_posiciones(crudas)
    titulo = (cancion.title or "").strip()
    ULTIMAS_SECCIONES[:] = secciones
    if es_seis:
        ULTIMA_AFINACION[:] = [int(afinacion.get(6 - i, CUERDAS_AL_AIRE[i]) + capo - CUERDAS_AL_AIRE[i]) for i in range(6)]
    return notas, [round(b, 3) for b in beats], titulo


# ---------------------------------------------------------------------------
#  MIDI
# ---------------------------------------------------------------------------
def leer_midi(ruta):
    import mido

    mid = mido.MidiFile(ruta)
    t = 0.0
    programa = {}
    abiertas = {}
    por_canal = {}
    tempo, cambios = 500000, []
    for msg in mid:                       # recorre todas las pistas ya mezcladas, tiempo en segundos
        t += msg.time
        if msg.type == "set_tempo":
            tempo = msg.tempo
            cambios.append((t, tempo))
        elif msg.type == "program_change":
            programa[msg.channel] = msg.program
        elif msg.type == "note_on" and msg.velocity > 0:
            if msg.channel != 9:
                abiertas[(msg.channel, msg.note)] = t
        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            ini = abiertas.pop((msg.channel, msg.note), None)
            if ini is not None:
                por_canal.setdefault(msg.channel, []).append((ini, t - ini, msg.note))
    if not por_canal:
        raise ValueError("el MIDI no tiene notas (aparte de la batería)")

    # Preferir un canal con sonido de guitarra (programas 24-31); si no, el que más notas tenga
    guitarras = [c for c in por_canal if 24 <= programa.get(c, 0) <= 31]
    canal = max(guitarras or por_canal, key=lambda c: len(por_canal[c]))
    notas = asignar_posiciones(por_canal[canal])

    # Tiempos del compás a partir del tempo
    beats, tb, k = [], 0.0, 0
    actual = 500000
    while tb < t:
        while k < len(cambios) and cambios[k][0] <= tb + 1e-6:
            actual = cambios[k][1]
            k += 1
        beats.append(round(tb, 3))
        tb += actual / 1e6
    return notas, beats, ""


# ---------------------------------------------------------------------------
#  MusicXML
# ---------------------------------------------------------------------------
_PASO = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def _sin_ns(raiz):
    for el in raiz.iter():
        if isinstance(el.tag, str) and "}" in el.tag:
            el.tag = el.tag.split("}", 1)[1]
    return raiz


def leer_musicxml(ruta):
    if ruta.lower().endswith(".mxl"):
        with zipfile.ZipFile(ruta) as z:
            nombres = [n for n in z.namelist() if n.lower().endswith((".xml", ".musicxml")) and not n.startswith("META-INF")]
            if not nombres:
                raise ValueError("el .mxl no contiene una partitura")
            raiz = ET.fromstring(z.read(nombres[0]))
    else:
        raiz = ET.parse(ruta).getroot()
    raiz = _sin_ns(raiz)
    if raiz.tag != "score-partwise":
        raise ValueError("solo se admite MusicXML de tipo score-partwise")

    partes = raiz.findall("part")
    if not partes:
        raise ValueError("la partitura no tiene partes")
    con_tab = [p for p in partes if p.find(".//technical/fret") is not None]
    parte = (con_tab or partes)[0]

    divisiones, tempo = 1, 120.0
    t = 0.0
    con_posicion, crudas, beats = [], [], []
    for compas in parte.findall("measure"):
        pos = 0.0                # en negras desde el inicio del compás
        ult_inicio = 0.0
        largo = 0.0
        for el in compas:
            if el.tag == "attributes":
                d = el.find("divisions")
                if d is not None and d.text:
                    divisiones = max(1, int(float(d.text)))
            elif el.tag in ("direction", "sound"):
                s = el if el.tag == "sound" else el.find("sound")
                if s is not None and s.get("tempo"):
                    tempo = float(s.get("tempo"))
            elif el.tag == "backup":
                pos -= float(el.findtext("duration", "0")) / divisiones
            elif el.tag == "forward":
                pos += float(el.findtext("duration", "0")) / divisiones
            elif el.tag == "note":
                if el.find("grace") is not None:
                    continue
                dur = float(el.findtext("duration", "0")) / divisiones
                acorde = el.find("chord") is not None
                inicio = ult_inicio if acorde else pos
                if not acorde:
                    ult_inicio = pos
                    pos += dur
                largo = max(largo, pos)
                tono = el.find("pitch")
                ligada = any(x.get("type") == "stop" for x in el.findall("tie"))
                if tono is None or el.find("rest") is not None or ligada:
                    continue
                m = 12 * (int(tono.findtext("octave", "4")) + 1) + _PASO[tono.findtext("step", "C")] + int(float(tono.findtext("alter", "0")))
                seg = t + inicio * 60.0 / tempo
                d_seg = dur * 60.0 / tempo
                cuerda, traste = el.findtext(".//technical/string"), el.findtext(".//technical/fret")
                if cuerda and traste and 1 <= int(cuerda) <= 6:
                    con_posicion.append({"t": round(seg, 3), "d": round(d_seg if d_seg > 0.7 else 0.0, 3),
                                         "s": 6 - int(cuerda), "f": int(traste), "m": m})
                else:
                    crudas.append((seg, d_seg, m))
        for b in range(int(round(largo))):
            beats.append(round(t + b * 60.0 / tempo, 3))
        t += largo * 60.0 / tempo

    notas = con_posicion if len(con_posicion) >= len(crudas) else asignar_posiciones(crudas)
    notas.sort(key=lambda n: n["t"])
    titulo = (raiz.findtext(".//work-title") or raiz.findtext(".//movement-title") or "").strip()
    return notas, beats, titulo


# ---------------------------------------------------------------------------
#  Arreglo de Rocksmith 2014 en XML (lo que sale al desempaquetar una custom song)
# ---------------------------------------------------------------------------
# Técnicas de una nota (campo "x" del chart, se pueden combinar)
TEC_HAMMER, TEC_PULL, TEC_SLIDE, TEC_PALM, TEC_MUTE = 1, 2, 4, 8, 16
TEC_ARMONICO, TEC_VIBRATO, TEC_TREMOLO, TEC_TAP, TEC_ACENTO = 32, 64, 128, 256, 512

NOMBRES_SECCION = {
    "intro": "Intro", "verse": "Verso", "chorus": "Coro", "prechorus": "Pre-coro", "bridge": "Puente",
    "solo": "Solo", "riff": "Riff", "outro": "Final", "noguitar": "Sin guitarra", "interlude": "Interludio",
    "breakdown": "Quiebre", "melody": "Melodía", "transition": "Transición", "hook": "Gancho", "variation": "Variación",
}

# Secciones de la última partitura leída con leer_partitura (marcadores de Guitar Pro)
ULTIMAS_SECCIONES = []
# Afinación de la última partitura: semitonos de diferencia por cuerda (6.ª a 1.ª) respecto a la estándar
ULTIMA_AFINACION = [0] * 6


def es_rocksmith_xml(ruta):
    """True si el XML es un arreglo de guitarra/bajo de Rocksmith (raíz <song> con <levels>)."""
    try:
        for _, el in ET.iterparse(ruta, events=("start",)):
            return el.tag == "song"
    except Exception:
        return False
    return False


def leer_rocksmith_xml(ruta):
    """Devuelve (notas, beats, info). Los tiempos ya vienen en segundos del audio."""
    raiz = ET.parse(ruta).getroot()
    if raiz.tag != "song" or raiz.find("levels") is None:
        raise ValueError("no es un arreglo de guitarra de Rocksmith")

    afin = raiz.find("tuning")
    ajuste = [int(afin.get("string%d" % i, "0")) if afin is not None else 0 for i in range(6)]
    plantillas = [[int(p.get("fret%d" % i, "-1")) for i in range(6)] for p in raiz.findall("chordTemplates/chordTemplate")]

    def nota(el):
        s = int(el.get("string", "0"))
        f = int(el.get("fret", "0"))
        if not (0 <= s <= 5) or f < 0:
            return None
        d = float(el.get("sustain", "0") or 0)

        def marca(*nombres):
            return any((el.get(n) or "0") not in ("0", "-1", "") for n in nombres)

        x = 0
        if marca("hammerOn"): x |= TEC_HAMMER
        if marca("pullOff"): x |= TEC_PULL
        if marca("tap"): x |= TEC_TAP
        if marca("palmMute"): x |= TEC_PALM
        if marca("mute"): x |= TEC_MUTE
        if marca("harmonic", "harmonicPinch"): x |= TEC_ARMONICO
        if marca("vibrato"): x |= TEC_VIBRATO
        if marca("tremolo"): x |= TEC_TREMOLO
        if marca("accent"): x |= TEC_ACENTO
        destino = int(el.get("slideTo", "-1") or -1)
        if destino < 0:
            destino = int(el.get("slideUnpitchTo", "-1") or -1)
        bend = float(el.get("bend", "0") or 0)
        tiene_tecnica = destino >= 0 or bend > 0 or (x & (TEC_VIBRATO | TEC_TREMOLO))
        n = {"t": round(float(el.get("time")), 3),
             "d": round(d if (d > 0.7 or (tiene_tecnica and d > 0.12)) else 0.0, 3),
             "s": s, "f": f, "m": CUERDAS_AL_AIRE[s] + ajuste[s] + f}
        if destino >= 0:
            x |= TEC_SLIDE
            n["sl"] = destino
        if bend > 0:
            n["b"] = round(bend, 2)
        if x:
            n["x"] = x
        return n

    def notas_de_nivel(nivel, desde, hasta):
        salida = []
        for el in nivel.findall("notes/note"):
            if desde <= float(el.get("time")) < hasta:
                n = nota(el)
                if n and el.get("ignore", "0") != "1":
                    salida.append(n)
        for ac in nivel.findall("chords/chord"):
            t = float(ac.get("time"))
            if not (desde <= t < hasta):
                continue
            hijas = ac.findall("chordNote")
            if hijas:
                salida += [n for n in (nota(h) for h in hijas) if n]
            else:
                idx = int(ac.get("chordId", "-1"))
                if 0 <= idx < len(plantillas):
                    for s, f in enumerate(plantillas[idx]):
                        if f >= 0:
                            salida.append({"t": round(t, 3), "d": 0.0, "s": s, "f": f, "m": CUERDAS_AL_AIRE[s] + ajuste[s] + f})
        return salida

    niveles = {int(n.get("difficulty", "0")): n for n in raiz.findall("levels/level")}
    if not niveles:
        raise ValueError("el arreglo no tiene notas")
    fin = float(raiz.findtext("songLength", "0") or 0) or 1e9
    frases = raiz.findall("phrases/phrase")
    iteraciones = raiz.findall("phraseIterations/phraseIteration")

    notas = []
    if len(niveles) == 1 or not frases or not iteraciones:
        notas = notas_de_nivel(niveles[max(niveles)], -1e9, 1e9)
    else:
        # Dificultad dinámica: cada frase usa su nivel máximo (la versión completa de la canción)
        for i, it in enumerate(iteraciones):
            desde = float(it.get("time"))
            hasta = float(iteraciones[i + 1].get("time")) if i + 1 < len(iteraciones) else fin + 1
            idf = int(it.get("phraseId", "0"))
            maximo = int(frases[idf].get("maxDifficulty", "0")) if idf < len(frases) else max(niveles)
            nivel = niveles.get(maximo) or niveles[max(k for k in niveles if k <= maximo) if any(k <= maximo for k in niveles) else max(niveles)]
            notas += notas_de_nivel(nivel, desde, hasta)

    beats = [round(float(b.get("time")), 3) for b in raiz.findall("ebeats/ebeat")]
    # Cambios de tono (efectos) a lo largo de la canción
    base = (raiz.findtext("tonebase") or "").strip()
    tonos = [(0.0, base)] if base else []
    for el in raiz.findall("tones/tone"):
        if el.get("name"):
            tonos.append((round(float(el.get("time", "0")), 3), el.get("name").strip()))
    secciones = []
    for el in raiz.findall("sections/section"):
        crudo = (el.get("name") or "").strip().lower()
        nombre = NOMBRES_SECCION.get(crudo, crudo.capitalize() or "Parte")
        secciones.append((round(float(el.get("startTime", "0")), 3), (nombre + " " + (el.get("number") or "")).strip()))
    info = {
        "secciones": secciones,
        "tonos": tonos,
        "titulo": (raiz.findtext("title") or "").strip(),
        "artista": (raiz.findtext("artistName") or "").strip(),
        "arreglo": (raiz.findtext("arrangement") or "").strip(),
        "afinacion": ajuste,
    }
    notas.sort(key=lambda n: (n["t"], n["s"]))
    return notas, beats, info


def tono_a_fx(tono=None, nombre=""):
    """Traduce la lista de equipos de un tono de la canción a los efectos genéricos de Rocky."""
    ganancia, empuje, wah, chorus, delay, reverb = None, False, 0, 0, 0, 0
    for pieza in ((tono or {}).get("GearList") or {}).values():
        if not pieza:
            continue
        clave = str(pieza.get("Key") or pieza.get("PedalKey") or "").lower()
        categoria = str(pieza.get("Category") or "").lower()
        tipo = str(pieza.get("Type") or "").lower()
        perillas = pieza.get("KnobValues") or {}
        if tipo.startswith("amp") or clave.startswith("amp_"):
            for k, v in perillas.items():
                if k.lower().endswith("_gain"):
                    ganancia = float(v) / 100.0
        elif any(x in clave for x in ("drive", "dist", "fuzz", "metal", "boost", "_od")) or categoria == "distortion":
            empuje = True
        if "wah" in clave or categoria == "filter":
            wah = 1
        if any(x in clave for x in ("chorus", "flang", "phas")) or categoria == "modulation":
            chorus = 1
        if any(x in clave for x in ("delay", "echo")) or categoria == "delay":
            delay = 1
        if "reverb" in clave or categoria == "reverb":
            reverb = 1
    n = nombre.lower()
    if ganancia is None:                      # sin datos del equipo: adivinar por el nombre del tono
        if any(x in n for x in ("clean", "limpi", "acoust", "acust")):
            ganancia = 0.15
        elif any(x in n for x in ("lead", "solo", "dist", "metal", "heavy", "fuzz")):
            ganancia = 0.85
        elif any(x in n for x in ("crunch", "drive", "od", "rhythm")):
            ganancia = 0.55
        else:
            ganancia = 0.5
        wah = wah or int("wah" in n)
        delay = delay or int("delay" in n or "echo" in n)
        chorus = chorus or int("chorus" in n)
    if empuje:
        ganancia += 0.1
    ganancia = float(min(1.0, max(0.05, ganancia)))
    if ganancia < 0.35 and not reverb:
        reverb = 1                            # un poco de ambiente en los tonos limpios
    return "g=%.2f;wah=%d;chorus=%d;delay=%d;reverb=%d" % (ganancia, wah, chorus, delay, reverb)


def tonos_de_manifiesto(ruta_json):
    """Devuelve {nombre o clave del tono: dict} a partir del manifiesto JSON del arreglo."""
    import json
    encontrados = {}
    try:
        with open(ruta_json, encoding="utf-8-sig") as fh:
            datos = json.load(fh)
    except Exception:
        return encontrados

    def recorrer(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "Tones" and isinstance(v, list):
                    for t in v:
                        if isinstance(t, dict):
                            for campo in ("Name", "Key"):
                                if t.get(campo):
                                    encontrados[str(t[campo])] = t
                else:
                    recorrer(v)
        elif isinstance(o, list):
            for x in o:
                recorrer(x)

    recorrer(datos)
    return encontrados


def leer_partitura(ruta):
    ULTIMAS_SECCIONES[:] = []
    ULTIMA_AFINACION[:] = [0] * 6
    ext = os.path.splitext(ruta)[1].lower()
    if ext in (".gp3", ".gp4", ".gp5"):
        notas, beats, titulo = leer_guitarpro(ruta)
    elif ext in (".mid", ".midi"):
        notas, beats, titulo = leer_midi(ruta)
    elif ext == ".xml" and es_rocksmith_xml(ruta):
        notas, beats, info = leer_rocksmith_xml(ruta)
        titulo = info["titulo"]
        ULTIMAS_SECCIONES[:] = info["secciones"]
        ULTIMA_AFINACION[:] = info["afinacion"]
    else:
        notas, beats, titulo = leer_musicxml(ruta)
    if not notas:
        raise ValueError("no encontré notas de guitarra en el archivo")
    notas.sort(key=lambda n: (n["t"], n["s"]))
    return notas, beats, titulo


# ---------------------------------------------------------------------------
#  Audio de acompañamiento cuando solo hay partitura (sin MP3)
# ---------------------------------------------------------------------------
def _cuerda_pulsada(m, dur, sr):
    f = 440.0 * 2 ** ((m - 69) / 12.0)
    n = int(sr * dur)
    N = max(2, int(round(sr / f)))
    rng = np.random.default_rng(m)
    buf = np.convolve(rng.uniform(-1, 1, N), np.ones(4) / 4, mode="same")
    out = np.empty(n)
    k = 0
    while k < n:
        q = min(N, n - k)
        out[k:k + q] = buf[:q]
        k += N
        buf = 0.996 * 0.5 * (buf + np.roll(buf, -1))
    return out * np.minimum(1, np.arange(n) / 40.0)


def sintetizar(notas, ruta_wav, sr=22050):
    """Crea un WAV con sonido de guitarra sintética a partir de las notas."""
    import soundfile as sf

    total = max(n["t"] + max(n["d"], 0.5) for n in notas) + 2.5
    y = np.zeros(int(sr * total))
    cache = {}
    for i, n in enumerate(notas):
        sig = notas[i + 1]["t"] - n["t"] if i + 1 < len(notas) else 2.0
        dur = float(np.clip(max(n["d"], sig * 1.5, 0.35), 0.35, 2.5))
        clave = (n["m"], round(dur, 1))
        if clave not in cache:
            cache[clave] = _cuerda_pulsada(n["m"], round(dur, 1) + 0.05, sr)
        w = cache[clave]
        a = int(n["t"] * sr)
        w = w[: len(y) - a]
        y[a:a + len(w)] += 0.4 * w
    pico = np.max(np.abs(y))
    if pico > 0:
        y = y / pico * 0.85
    tmp = ruta_wav + ".tmp"
    sf.write(tmp, y.astype(np.float32), sr, format="WAV", subtype="PCM_16")
    os.replace(tmp, ruta_wav)
    return total

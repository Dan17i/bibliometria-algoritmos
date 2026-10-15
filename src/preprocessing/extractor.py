"""
Módulo de Extracción de Metadatos y Abstracts (Preprocesamiento del Corpus)
---------------------------------------------------------------------------
Extrae de cada artículo científico en PDF los tres campos que consumen los
demás requerimientos del proyecto: título, autores y abstract.

Contexto del problema:
Los 20 artículos del corpus provienen de cuatro editoriales distintas y cada
una maqueta la primera página de forma diferente. El texto plano que produce
un PDF no respeta las columnas visuales, por lo que una extracción ingenua
mezcla afiliaciones, palabras clave y cuerpo del abstract.

Se identificaron tres familias de maquetación:

  1. Elsevier   -> el rótulo aparece espaciado ("A B S T R A C T") y el bloque
                   de Keywords se intercala ANTES del cuerpo del abstract.
  2. Estándar   -> rótulo "Abstract" o "Abstract:" seguido del cuerpo.
  3. Dos columnas (Wiley, Springer) -> el abstract vive en una banda vertical
                   y el texto plano lo intercala con la columna de afiliaciones.

Estrategia:
Se intenta cada vía en orden y se acepta la primera cuyo resultado supere una
validación de calidad. La validación por longitud NO es suficiente: un bloque
de afiliaciones puede tener el mismo número de palabras que un abstract real.

Dependencias: pdfplumber (Python puro, no requiere binarios externos).
"""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata

import pdfplumber

# --------------------------------------------------------------------------
# Patrones de reconocimiento
# --------------------------------------------------------------------------

RE_ELSEVIER = re.compile(r"A\s?B\s?S\s?T\s?R\s?A\s?C\s?T", re.M)
RE_KEYWORDS = re.compile(r"\bKey\s?words?\s*[::]?", re.I)

# Marcas que señalan el final del abstract
RE_FIN = re.compile(
    r"(\bKey\s?words?\b\s*[::·]?"
    r"|^\s*1\.?\s+Introduction"
    r"|^\s*Introduction\s*$"
    r"|©\s*\d{4}"
    r"|^\s*1\s*\|\s*INTRO)",
    re.I | re.M,
)

# Encabezados de afiliación: si el texto empieza así, NO es un abstract
RE_AFILIACION = re.compile(
    r"^\W*(Faculty|Department|University|College|School|Institute|Centre|Center"
    r"|Academy|Correspondence|Program|Graduate|Received|Accepted|\d)",
    re.I,
)

# Ruido típico de la cabecera editorial, a descartar al buscar autores
RE_RUIDO = re.compile(
    r"(doi|http|www\.|journal|homepage|contents lists|sciencedirect|elsevier"
    r"|springer|wiley|received|revised|accepted|published|correspondence"
    r"|review article|research article|^\s*abstract)",
    re.I,
)

# Una línea de autores: dos o más nombres propios seguidos de un separador.
# Entre el apellido y el separador suelen colarse marcas de afiliación
# (superíndices, asteriscos, cruces), por lo que se toleran explícitamente:
# sin esto, 'Laupichler*, Alexandra Aster' no coincide.
MARCAS = r"[\d*†‡§¶∗⁎,]*"
RE_AUTORES = re.compile(
    r"^[A-ZÁÉÍÓÚÑ][\w'’.\-]*"
    r"(?:\s+[A-ZÁÉÍÓÚÑ][\w'’.\-]*){1,4}"
    + MARCAS +
    r"(?:\s*[,|&·]|\s+and\s+)"
)


# --------------------------------------------------------------------------
# Limpieza de texto
# --------------------------------------------------------------------------

def limpiar(texto: str) -> str:
    """
    Normaliza el texto crudo del PDF.

    El guion suave (U+00AD) es el problema silencioso: los PDF lo insertan al
    justificar y, si solo se elimina el carácter, 'devel­opment' se convierte
    en 'devel opment' — dos tokens falsos que contaminan Jaccard y TF-IDF sin
    producir ningún error visible. Por eso se elimina junto con su salto.
    """
    texto = re.sub(r"­\s*\n?\s*", "", texto)              # guion suave + corte
    texto = re.sub(r"[-‐-―]\s*\n\s*(?=[a-z])", "", texto)  # guion de línea
    texto = re.sub(r"\s*\n\s*", " ", texto)
    texto = re.sub(r"\s{2,}", " ", texto)
    return unicodedata.normalize("NFKC", texto).strip()


def es_abstract_valido(texto: str, min_pal: int = 80, max_pal: int = 400) -> bool:
    """
    Valida que el texto extraído sea realmente un abstract.

    Tres comprobaciones, porque la longitud sola deja pasar errores:
      - rango de palabras razonable
      - no empieza con un encabezado de afiliación
      - no es una lista de palabras clave (muchas comas, sin artículos)
    """
    n = len(texto.split())
    if not (min_pal <= n <= max_pal):
        return False
    if RE_AFILIACION.match(texto):
        return False
    cabeza = texto[:200]
    if cabeza.count(",") >= 5 and " the " not in cabeza.lower():
        return False
    return texto[:1].isupper()


# --------------------------------------------------------------------------
# Vías de extracción del abstract
# --------------------------------------------------------------------------

def extraer_abstract_por_coordenadas(pagina) -> str | None:
    """
    Extrae el abstract usando la posición geométrica real de cada palabra.

    Es el método principal. Parsear el texto plano no basta: el PDF no guarda
    columnas, solo coordenadas, y cualquier reconstrucción textual pierde esa
    información. pdfplumber sí expone x0/top por palabra, así que el abstract
    se delimita como una región del plano:

      - Horizontalmente: la banda que arranca en la x del rótulo 'Abstract' y
        termina donde empieza la siguiente columna (o el borde de la página).
        Esto descarta de un golpe las keywords de Elsevier y la columna de
        afiliaciones de Wiley, que viven a la izquierda del rótulo.

      - Verticalmente: desde el rótulo hasta la primera marca de fin
        ('Keywords', 'Introduction') o hasta un salto vertical anómalo, que es
        donde termina el bloque del abstract y empieza el cuerpo del artículo.

    El corte vertical es imprescindible: el cuerpo del artículo ocupa la misma
    banda horizontal que el abstract, así que filtrar solo por columna
    arrastraría el artículo entero.
    """
    palabras = pagina.extract_words(use_text_flow=False, keep_blank_chars=False)
    if not palabras:
        return None

    # --- localizar el rótulo -------------------------------------------------
    # Se guardan por separado la x de inicio (define la columna) y la y de la
    # última letra (define desde dónde leer hacia abajo). En Elsevier el
    # rótulo ocupa ocho "palabras" y tomar la x de la última desplazaría la
    # banda hacia la derecha, cortando el principio de cada renglón.
    x_col = y_rotulo = None
    for i, w in enumerate(palabras):
        limpio = w["text"].strip().rstrip(":：")
        if limpio.lower() == "abstract":
            x_col, y_rotulo = w["x0"], w["bottom"]
            break
        # Elsevier imprime el rótulo letra a letra: A B S T R A C T
        if limpio == "A" and i + 7 < len(palabras):
            seq = "".join(x["text"].strip() for x in palabras[i:i + 8])
            if seq == "ABSTRACT":
                x_col = w["x0"]                      # primera letra
                y_rotulo = palabras[i + 7]["bottom"]  # última letra
                break
    if x_col is None:
        return None

    # --- límite derecho: inicio de la siguiente columna ----------------------
    # Se buscan palabras muy a la derecha que formen otra columna; si no las
    # hay, el límite es el ancho de la página.
    x_limite = pagina.width

    # --- recolectar las palabras de la banda, por debajo del rótulo ---------
    candidatas = [
        w for w in palabras
        if w["top"] >= y_rotulo - 2
        and x_col - 8 <= w["x0"] < x_limite
    ]
    if not candidatas:
        return None
    candidatas.sort(key=lambda w: (round(w["top"], 1), w["x0"]))

    # --- agrupar en renglones y cortar --------------------------------------
    renglones: list[tuple[float, list[str]]] = []
    for w in candidatas:
        if renglones and abs(w["top"] - renglones[-1][0]) <= 3:
            renglones[-1][1].append(w["text"])
        else:
            renglones.append((w["top"], [w["text"]]))

    # Interlineado típico: mediana de los saltos entre renglones consecutivos
    saltos = [renglones[i + 1][0] - renglones[i][0] for i in range(len(renglones) - 1)]
    interlineado = sorted(saltos)[len(saltos) // 2] if saltos else 12.0

    seleccion, acumulado = [], 0
    for i, (top, piezas) in enumerate(renglones):
        linea = " ".join(piezas).strip()
        if not linea:
            continue
        if RE_FIN.match(linea):
            break
        # Salto vertical anómalo = fin del bloque del abstract
        if i > 0 and acumulado > 250:
            if (top - renglones[i - 1][0]) > interlineado * 2.2:
                break
        seleccion.append(linea)
        acumulado += len(linea)

    return limpiar(" ".join(seleccion))


def _texto_poppler(ruta: str) -> str:
    """Texto plano de las dos primeras páginas vía poppler; '' si no está."""
    import shutil
    import subprocess

    if not shutil.which("pdftotext"):
        return ""
    # encoding explícito: sin él, Python decodifica con la codificación local
    # (cp1252 en Windows) y la salida UTF-8 de pdftotext revienta el hilo
    # lector, dejando stdout en None.
    resultado = subprocess.run(
        ["pdftotext", "-f", "1", "-l", "2", ruta, "-"],
        capture_output=True, text=True,
        encoding="utf-8", errors="replace",
    )
    return resultado.stdout or ""


def extraer_abstract_poppler(ruta: str) -> str | None:
    """
    Motor de respaldo basado en pdftotext (poppler).

    Se usa cuando el motor por coordenadas no supera la validación. Hace falta
    porque en algunos PDF del corpus pdfplumber no reconstruye los espacios
    entre palabras ('Withtherapiddevelopment...'), un defecto de la extracción
    que no puede corregirse aguas arriba. pdftotext sí los resuelve.

    Requiere el binario 'pdftotext' en el PATH; si no está, devuelve None y el
    artículo queda marcado para revisión en vez de romper la ejecución.
    """
    import shutil
    import subprocess

    if not shutil.which("pdftotext"):
        return None

    def correr(*extra: str) -> str:
        # encoding explícito: ver la nota en _texto_poppler. En Windows la
        # codificación local es cp1252 y la salida UTF-8 de pdftotext hace
        # fallar la decodificación, dejando stdout en None.
        resultado = subprocess.run(
            ["pdftotext", *extra, "-f", "1", "-l", "2", ruta, "-"],
            capture_output=True, text=True,
            encoding="utf-8", errors="replace",
        )
        return resultado.stdout or ""

    texto, layout = correr(), correr("-layout")

    # Vía A — Elsevier: el bloque de Keywords precede al cuerpo del abstract
    m = RE_ELSEVIER.search(texto)
    if m:
        resto = texto[m.end():]
        kw = RE_KEYWORDS.search(resto)
        if kw:
            lineas = resto[kw.end():].split("\n")
            i = next((k for k, l in enumerate(lineas) if len(l.strip()) > 60), 0)
            cuerpo = "\n".join(lineas[i:])
        else:
            cuerpo = resto
        fin = RE_FIN.search(cuerpo)
        cuerpo = cuerpo[: fin.start()] if fin else cuerpo
        # Sin marca de fin, el abstract es el primer párrafo
        bloque, acum = [], 0
        for linea in cuerpo.split("\n"):
            if not linea.strip():
                if acum > 300:
                    break
                continue
            bloque.append(linea)
            acum += len(linea)
        candidato = limpiar("\n".join(bloque))
        if es_abstract_valido(candidato):
            return candidato

    # Vía B — rótulo en el flujo normal de lectura
    m = re.search(r"\bAbstract\s*[::]?\s*\n?", texto)
    if m:
        cuerpo = texto[m.end():]
        fin = RE_FIN.search(cuerpo)
        candidato = limpiar((cuerpo[: fin.start()] if fin else cuerpo)[:3000])
        if es_abstract_valido(candidato):
            return candidato

    # Vía C — dos columnas: recorte de la banda del rótulo
    m = re.search(r"(?:^|\s{6,})Abstract\s*$", layout, re.M)
    if m:
        col = layout[: m.end()].split("\n")[-1].index("Abstract")
        seleccion, acum, huecos = [], 0, 0
        for linea in layout[m.end():].split("\n"):
            banda = linea[col:].strip() if len(linea) > col else ""
            if not banda:
                huecos += 1
                if huecos >= 4 and acum > 300:
                    break
                continue
            if RE_FIN.match(banda):
                break
            huecos = 0
            seleccion.append(banda)
            acum += len(banda)
        candidato = limpiar(" ".join(seleccion))
        if es_abstract_valido(candidato):
            return candidato

    return None


def _localizar_rotulo(layout: str) -> tuple[int, int] | None:
    """
    Encuentra el rótulo del abstract y devuelve (fin_del_rótulo, columna).

    Se prueban las tres grafías observadas en el corpus, en orden de
    especificidad: la espaciada de Elsevier primero, porque un 'Abstract'
    suelto podría aparecer antes en la cabecera editorial.
    """
    patrones = (
        RE_ELSEVIER,                        # "A B S T R A C T"  (Elsevier)
        re.compile(r"\bAbstract\s*[::]", re.M),   # "Abstract:"   (MDPI)
        re.compile(r"\bAbstract\b", re.M),        # "Abstract"    (resto)
    )
    for patron in patrones:
        m = patron.search(layout)
        if m:
            inicio_linea = layout.rfind("\n", 0, m.start()) + 1
            return m.end(), m.start() - inicio_linea
    return None


def extraer_abstract(layout: str) -> str | None:
    """
    Extrae el abstract recortando la banda vertical donde vive el rótulo.

    Esta única estrategia cubre las tres familias de maquetación porque
    'extract_text(layout=True)' conserva la posición horizontal real de cada
    columna. El recorte por columna resuelve de paso dos problemas que una
    lectura en flujo no puede:

      - En Elsevier, el bloque de Keywords ocupa la columna izquierda y el
        abstract la derecha. Al recortar desde la columna del rótulo, las
        keywords quedan fuera sin necesidad de detectarlas.
      - En Wiley y Springer, la columna de afiliaciones comparte línea física
        con el abstract. Filtrar por sangría fallaría —la sangría es la de la
        columna izquierda—; recortar por posición de carácter, no.

    Cuando el abstract ocupa el ancho completo, la columna es 0 y el recorte
    devuelve la línea entera, que es el comportamiento deseado.

    Las líneas en blanco dentro de la banda son interlineado, no fin de
    párrafo: se toleran hasta tres huecos consecutivos antes de cortar.
    """
    posicion = _localizar_rotulo(layout)
    if posicion is None:
        return None
    fin_rotulo, col = posicion

    seleccion, acumulado, huecos = [], 0, 0
    for linea in layout[fin_rotulo:].split("\n"):
        banda = linea[col:].rstrip() if len(linea) > col else ""
        banda = banda.strip()

        if not banda:
            huecos += 1
            if huecos >= 4 and acumulado > 300:
                break
            continue

        if RE_FIN.match(banda):
            break

        huecos = 0
        seleccion.append(banda)
        acumulado += len(banda)

    return limpiar(" ".join(seleccion))


# --------------------------------------------------------------------------
# Extracción de título y autores
# --------------------------------------------------------------------------

def extraer_titulo(pdf: pdfplumber.PDF, nombre_archivo: str) -> tuple[str, bool]:
    """
    Obtiene el título preferentemente de los metadatos del PDF.

    Devuelve (titulo, revisar). Algunas editoriales dejan el identificador
    interno de producción en ese campo (p. ej. 'ET-04-2023-0117_proof'); en
    ese caso se recurre al nombre del archivo y se marca para revisión.
    """
    meta = (pdf.metadata or {}).get("Title", "") or ""
    meta = meta.strip()

    sospechoso = (
        len(meta) < 20
        or re.search(r"(_proof|\d{3,}\.\.|^untitled|\.(doc|pdf|tex)$)", meta, re.I)
    )
    if meta and not sospechoso:
        return unicodedata.normalize("NFKC", meta), False

    # Respaldo: el nombre del archivo sin el número de orden ni la extensión
    base = re.sub(r"^\d+\.\s*", "", nombre_archivo)
    base = re.sub(r"\.pdf$", "", base, flags=re.I)
    return re.sub(r"\s{2,}", " ", base).strip(), True


def _buscar_autores(texto: str, titulo: str) -> str:
    """Recorre la cabecera buscando la primera línea con forma de autores."""
    corte = texto.find("Abstract")
    cabecera = texto[:corte] if corte > 0 else texto[:2500]
    # Se compara contra el título COMPLETO: cuando el título ocupa dos líneas,
    # la continuación ('Assessment Redesign Principles and Considerations')
    # tiene forma de lista de autores y se colaría como falso positivo.
    palabras_titulo = set(re.findall(r"\w+", titulo.lower()))

    for linea in cabecera.split("\n"):
        linea = linea.strip()
        if not (12 <= len(linea) <= 300):
            continue
        if RE_RUIDO.search(linea):
            continue
        # Descarta la propia línea del título
        if len(palabras_titulo & set(linea.lower().split())) >= 3:
            continue
        if not RE_AUTORES.match(linea):
            continue

        autores = re.sub(r"[\d*†‡§¶∗⁎]+", "", linea)      # quita superíndices
        autores = re.sub(r"\s*[|·]\s*", ", ", autores)     # unifica separadores
        autores = re.sub(r"\s+and\s+", ", ", autores)
        autores = re.sub(r"\s*,\s*", ", ", autores)
        autores = re.sub(r"(,\s*)+", ", ", autores).strip(" ,;")
        if len(autores) >= 10:
            return unicodedata.normalize("NFKC", autores)

    return ""


def extraer_autores(texto: str, titulo: str, texto_alt: str = "") -> tuple[str, bool]:
    """
    Busca la línea de autores entre el título y el inicio del abstract.

    Devuelve (autores, revisar). Es la extracción menos fiable de los tres
    campos, porque la posición de los autores varía mucho entre editoriales;
    por eso siempre se informa si conviene verificarla a mano.

    Se intenta sobre dos reconstrucciones del texto: en varios PDF del corpus
    pdfplumber pega las palabras ('BayodeOgunleye,KudiratZakariyyah') y la
    línea solo es legible en la salida de poppler.
    """
    for fuente in (texto, texto_alt):
        if not fuente:
            continue
        encontrados = _buscar_autores(fuente, titulo)
        if encontrados:
            return encontrados, False
    return "", True


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------

def procesar_pdf(ruta: str) -> dict:
    """Extrae título, autores y abstract de un único artículo."""
    nombre = os.path.basename(ruta)

    with pdfplumber.open(ruta) as pdf:
        paginas = pdf.pages[:2]
        texto = "\n".join((p.extract_text() or "") for p in paginas)
        layout = "\n".join((p.extract_text(layout=True) or "") for p in paginas)
        titulo, rev_titulo = extraer_titulo(pdf, nombre)
        abstract = extraer_abstract_por_coordenadas(pdf.pages[0]) or ""

    autores, rev_autores = extraer_autores(texto, titulo, _texto_poppler(ruta))

    # Cascada de motores: primero el que no necesita binarios externos; si su
    # resultado no pasa la validación, se recurre a poppler, que resuelve los
    # PDF donde pdfplumber no reconstruye los espacios entre palabras.
    motor = "coordenadas"
    if not es_abstract_valido(abstract):
        respaldo = extraer_abstract_poppler(ruta)
        if respaldo:
            abstract, motor = respaldo, "poppler"
        else:
            alterno = extraer_abstract(layout)
            if alterno and es_abstract_valido(alterno):
                abstract, motor = alterno, "layout"

    return {
        "id": "",                       # se asigna en procesar_corpus
        "archivo": nombre,
        "titulo": titulo,
        "autores": autores,
        "texto": abstract,
        "_motor": motor,
        "_palabras": len(abstract.split()),
        "_revisar": sorted(
            n for n, r in (("titulo", rev_titulo),
                           ("autores", rev_autores),
                           ("abstract", not es_abstract_valido(abstract))) if r
        ),
    }


def procesar_corpus(dir_pdfs: str, dir_salida: str) -> list[dict]:
    """
    Procesa todos los PDF de un directorio y escribe abstracts.json y .csv.

    El esquema de salida (id, archivo, titulo, texto) es el que ya consumen
    los módulos de similitud y clustering del equipo; se añade 'autores' para
    el cruce de información que pide el Requerimiento 1.
    """
    def orden(nombre: str) -> int:
        m = re.match(r"(\d+)", nombre)
        return int(m.group(1)) if m else 9999

    archivos = sorted(
        (f for f in os.listdir(dir_pdfs) if f.lower().endswith(".pdf")),
        key=orden,
    )

    registros = []
    for i, nombre in enumerate(archivos, start=1):
        reg = procesar_pdf(os.path.join(dir_pdfs, nombre))
        reg["id"] = f"P{i}"
        registros.append(reg)

    os.makedirs(dir_salida, exist_ok=True)
    ruta_json = os.path.join(dir_salida, "abstracts.json")

    # Protección contra degradación silenciosa: si ya existe un resultado con
    # más abstractos válidos que el recién calculado (por ejemplo, porque esta
    # vez faltó poppler), se conserva como .bak y se avisa. Perder un corpus
    # bueno por correr el script sin una dependencia es un error caro.
    validos_nuevos = sum(1 for r in registros if "abstract" not in r["_revisar"])
    if os.path.exists(ruta_json):
        try:
            with open(ruta_json, encoding="utf-8") as fh:
                previos = json.load(fh)
            validos_previos = sum(
                1 for r in previos if "abstract" not in r.get("_revisar", ["abstract"])
            )
            if validos_previos > validos_nuevos:
                respaldo = ruta_json + ".bak"
                os.replace(ruta_json, respaldo)
                print(f"\n  AVISO: el resultado anterior tenia {validos_previos} "
                      f"abstracts validos y este tiene {validos_nuevos}.")
                print(f"  Se conservo una copia en: {respaldo}\n")
        except (json.JSONDecodeError, OSError):
            pass   # un archivo previo ilegible no debe frenar la ejecución

    with open(ruta_json, "w", encoding="utf-8") as fh:
        json.dump(registros, fh, ensure_ascii=False, indent=2)

    with open(os.path.join(dir_salida, "abstracts.csv"), "w",
              newline="", encoding="utf-8") as fh:
        escritor = csv.writer(fh)
        escritor.writerow(["id", "archivo", "titulo", "autores", "palabras", "texto"])
        for r in registros:
            escritor.writerow([r["id"], r["archivo"], r["titulo"],
                               r["autores"], r["_palabras"], r["texto"]])

    return registros


def main() -> None:
    import shutil
    import sys

    # La consola de Windows usa cp1252 y no puede imprimir los caracteres no
    # latinos que aparecen en varios títulos y nombres de autor del corpus.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass

    raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dir_pdfs = os.path.join(raiz, "docs", "articulos")
    dir_salida = os.path.join(raiz, "data", "processed")

    hay_poppler = bool(shutil.which("pdftotext"))
    if not hay_poppler:
        print("\n  AVISO: 'pdftotext' (poppler) no esta en el PATH.")
        print("  El motor de respaldo queda desactivado y algunos articulos")
        print("  del corpus no podran extraerse. Instalalo con:")
        print("      winget install oschwartz10612.Poppler\n")

    registros = procesar_corpus(dir_pdfs, dir_salida)

    print(f"{'ID':<5}{'MOTOR':<13}{'PAL':<6}{'REVISAR':<18}TITULO")
    print("-" * 104)
    for r in registros:
        marca = ",".join(r["_revisar"]) or "-"
        print(f"{r['id']:<5}{r['_motor']:<13}{r['_palabras']:<6}"
              f"{marca:<18}{r['titulo'][:44]}")
    print("-" * 104)

    validos = sum(1 for r in registros if "abstract" not in r["_revisar"])
    con_autores = sum(1 for r in registros if r["autores"])
    sin_nada = sum(1 for r in registros if not r["_revisar"])
    total = len(registros)

    print(f"Abstracts validos: {validos}/{total}")
    print(f"Autores extraidos: {con_autores}/{total}")
    print(f"Sin incidencias  : {sin_nada}/{total}")
    print(f"Salida           : {dir_salida}")

    if validos < total and not hay_poppler:
        print("\n  Faltan abstracts porque no hay poppler. Instalalo y vuelve a correr.")


if __name__ == "__main__":
    main()

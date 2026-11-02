"""
Módulo de Extracción de Metadatos y Abstracts (Preprocesamiento del Corpus)
---------------------------------------------------------------------------
Extrae de cada artículo científico en PDF los tres campos que consumen los
demás requerimientos del proyecto: título, autores y abstract.

Contexto del problema:
Los 20 artículos del corpus provienen de cuatro editoriales distintas y cada
una maqueta la primera página de forma diferente. Hay tres defectos que una
extracción ingenua deja pasar sin producir ningún error visible:

  1. Palabras cortadas al final de línea ('edu-' + 'cation'). Si no se unen
     aparecen dos tokens falsos que contaminan Jaccard y TF-IDF.
  2. Superíndices de afiliación pegados a los nombres ('Chiua' por 'Chiu').
  3. Rótulos en orden distinto: Elsevier imprime 'A B S T R A C T' y a veces
     pone el bloque de Keywords antes del cuerpo del abstract.

Estrategia:
Se lee el flujo de texto de la primera página con PyMuPDF, que respeta el orden
de lectura y no necesita binarios externos, de modo que el proceso se reproduce
solo con `pip install -r requirements.txt`.

Las palabras cortadas se resuelven contra el vocabulario de los 20 PDF: si la
unión existe en el corpus se une ('edu-' + 'cation' = 'education'); si no, pero
las dos partes sí existen, es un compuesto real y se conserva el guion
('well-' + 'defined' = 'well-defined'). Cada decisión queda registrada en el
campo `_cortes` para poder revisarla.

Los pocos defectos que no se pueden resolver con una regla general están en
CORRECCIONES, con el motivo de cada una.

Dependencias: pymupdf (se importa como fitz).
"""

from __future__ import annotations

import csv
import json
import os
import re
import unicodedata

import fitz

# --------------------------------------------------------------------------
# Patrones de reconocimiento
# --------------------------------------------------------------------------

RE_ROTULO = re.compile(r"A\s?B\s?S\s?T\s?R\s?A\s?C\s?T|\bAbstract\b")

# Marcas que señalan el final del abstract. Sin re.I a propósito: en P15 una
# línea empieza por 'keywords, and co-authorship' y es parte del abstract.
RE_FIN = re.compile(
    r"^\s*(?:Key\s?words?\b"
    r"|K\s?E\s?Y\s?W\s?O\s?R\s?D\s?S"
    r"|1\.?\s+Introduction"
    r"|Introduction\s*$"
    r"|1\s*\|\s*INTRO"
    r"|Citation:"
    r"|Received\b"
    r"|©\s*\d{4})"
)

# En los PDF de Elsevier un punto seguido de varios espacios cierra el párrafo.
# Es lo único que separa el abstract de P9 de su introducción, que no tiene
# encabezado.
RE_FIN_PARRAFO = re.compile(r"\.\s{2,}$")

RE_CORTE = re.compile(r"([^\W\d_]+)[­\-‐‑]\s*\n\s*([^\W\d_]+)")
RE_PALABRA = re.compile(r"[^\W\d_]+")

RE_RUIDO = re.compile(
    r"(doi|http|www\.|journal|homepage|contents lists|sciencedirect|elsevier"
    r"|springer|wiley|received|revised|accepted|published|correspondence"
    r"|\breview\b|open\s?access|abstract"
    r"|university|department|faculty|school|college|institute|education:)",
    re.I,
)

# Dos a cinco palabras con inicial mayúscula; solo se admiten en minúscula las
# partículas de los apellidos.
RE_NOMBRE = re.compile(
    r"^[A-ZÀ-Þ][\w'’.\-]*(?:\s+(?:[A-ZÀ-Þ][\w'’.\-]*|de|del|der|la|van|von)){1,4}$"
)


# --------------------------------------------------------------------------
# Defectos puntuales del corpus
# --------------------------------------------------------------------------

# Títulos cuyo metadato del PDF es incorrecto:
#   P11: el metadato dice 'transform'; la portada dice 'transforms'.
#   P13: el metadato es un identificador de producción ('ET-04-2023-0117_proof').
CORRECCIONES = {
    "P11": {"titulo": "A scoping review on how generative artificial intelligence "
                      "transforms assessment in higher education"},
    "P13": {"titulo": "Artificial intelligence for education and its emphasis on "
                      "assessment and adversity quotient: a review"},
}


# --------------------------------------------------------------------------
# Limpieza de texto
# --------------------------------------------------------------------------

def normalizar(texto: str) -> str:
    """Quita ligaduras y espacios especiales, y colapsa los espacios."""
    texto = unicodedata.normalize("NFKC", texto)
    return re.sub(r"\s+", " ", texto).strip()


def construir_vocabulario(rutas: list[str]) -> set[str]:
    """
    Palabras que aparecen completas en los PDF del corpus.

    Se excluyen las que quedan junto a un corte de línea, porque esas son
    justamente los fragmentos que se quieren reconocer.
    """
    vocabulario: set[str] = set()
    for ruta in rutas:
        with fitz.open(ruta) as pdf:
            for pagina in pdf:
                texto = unicodedata.normalize("NFKC", pagina.get_text())
                texto = RE_CORTE.sub(" ", texto)
                vocabulario.update(p.lower() for p in RE_PALABRA.findall(texto))
    return vocabulario


def unir_cortes(texto: str, vocabulario: set[str]) -> tuple[str, list[str]]:
    """
    Repara las palabras cortadas al final de línea.

    Devuelve el texto reparado y la lista de decisiones ('edu|cation -> education').
    """
    decisiones: list[str] = []

    def resolver(m: re.Match) -> str:
        izquierda, derecha = m.group(1), m.group(2)
        union = (izquierda + derecha).lower()
        if union in vocabulario:
            resultado = izquierda + derecha
        elif izquierda.lower() in vocabulario and derecha.lower() in vocabulario:
            resultado = izquierda + "-" + derecha
        else:
            resultado = izquierda + derecha
        decisiones.append(f"{izquierda}|{derecha} -> {resultado}")
        return resultado

    return RE_CORTE.sub(resolver, texto), decisiones


# --------------------------------------------------------------------------
# Abstract
# --------------------------------------------------------------------------

def recortar_abstract(texto_pagina: str) -> str | None:
    """
    Devuelve las líneas del abstract, todavía con los cortes de línea.

    Empieza después del rótulo. Si lo primero que sigue es el bloque de Keywords
    (P7), lo salta hasta la primera línea larga, que ya es el cuerpo.
    """
    rotulo = RE_ROTULO.search(texto_pagina)
    if rotulo is None:
        return None

    lineas = texto_pagina[rotulo.end():].lstrip(" :\n").split("\n")

    inicio = 0
    if RE_FIN.match(lineas[0]):
        while inicio < len(lineas) and len(lineas[inicio].strip()) <= 60:
            inicio += 1

    seleccion: list[str] = []
    for linea in lineas[inicio:]:
        if RE_FIN.match(linea):
            break
        seleccion.append(linea)
        if RE_FIN_PARRAFO.search(linea):
            break
    return "\n".join(seleccion)


def es_abstract_valido(texto: str, min_pal: int = 80, max_pal: int = 400) -> bool:
    """Rango de palabras razonable, empieza en mayúscula y termina en punto."""
    n = len(texto.split())
    return min_pal <= n <= max_pal and texto[:1].isupper() and texto.endswith(".")


# --------------------------------------------------------------------------
# Autores
# --------------------------------------------------------------------------

def _lineas_de_cabecera(pagina) -> list[tuple[float, str]]:
    """
    (altura, texto) de cada línea, sin los superíndices.

    Un superíndice se reconoce por su tamaño: es más pequeño que el resto de su
    línea. Filtrar por tipografía evita tener que adivinar si la 'a' final de
    'Chiua' pertenece al apellido. Los espacios se reconstruyen por la distancia
    entre fragmentos, porque algunos PDF no los guardan como caracteres.
    """
    lineas = []
    for bloque in pagina.get_text("dict")["blocks"]:
        for linea in bloque.get("lines", []):
            spans = [s for s in linea["spans"] if s["text"].strip()]
            if not spans:
                continue
            mayor = max(s["size"] for s in spans)
            texto, fin_x = "", None
            for s in spans:
                if s["size"] < mayor * 0.85:
                    fin_x = s["bbox"][2]
                    continue
                if fin_x is not None and s["bbox"][0] - fin_x > s["size"] * 0.15:
                    texto += " "
                texto += s["text"]
                fin_x = s["bbox"][2]
            lineas.append((linea["bbox"][1], normalizar(texto)))
    lineas.sort(key=lambda par: par[0])
    return lineas


def _nombres_de_linea(linea: str) -> list[str] | None:
    """Separa una línea en nombres; None si no tiene forma de lista de autores."""
    limpia = re.sub(r"[*∗†‡§¶\d]", "", linea)
    # La tilde suelta de algunos PDF ('Gaševi´c') se recompone como carácter
    limpia = re.sub(r"\s?[´́]\s?c\b", "ć", limpia)
    partes = re.split(r"\s*(?:,|\||·|&|\band\b)\s*", limpia)
    nombres = [normalizar(p) for p in partes if p.strip(" .")]
    if nombres and all(RE_NOMBRE.match(n) for n in nombres):
        return nombres
    return None


def extraer_autores(pagina, titulo: str) -> list[str]:
    """
    Busca el bloque de autores: las primeras líneas con forma de lista de
    nombres, sin palabras del título ni de la cabecera editorial.

    Se recorren por posición y se aceptan las líneas contiguas, porque en P4, P7,
    P12 y P15 la lista ocupa dos renglones.
    """
    palabras_titulo = set(re.findall(r"\w+", titulo.lower()))
    corte = pagina.search_for("Abstract") or pagina.search_for("A B S T R A C T")
    limite = corte[0].y0 if corte else pagina.rect.height / 2

    autores: list[str] = []
    ultima_altura = None
    for altura, linea in _lineas_de_cabecera(pagina):
        if altura >= limite:
            break
        if ultima_altura is not None and altura - ultima_altura > 20:
            break
        palabras = re.findall(r"\w+", linea.lower())
        if not palabras or RE_RUIDO.search(linea):
            continue
        if len(set(palabras) & palabras_titulo) / len(palabras) > 0.6:
            continue
        nombres = _nombres_de_linea(linea)
        if nombres:
            autores.extend(nombres)
            ultima_altura = altura
        elif ultima_altura is not None:
            break
    return autores


# --------------------------------------------------------------------------
# Título
# --------------------------------------------------------------------------

def extraer_titulo(pdf, nombre_archivo: str) -> str:
    """Título de los metadatos del PDF o, si no sirve, el del nombre del archivo."""
    meta = ((pdf.metadata or {}).get("title") or "").strip()
    # Algunas editoriales dejan aquí un identificador interno ('..._proof 609..645')
    if len(meta) >= 20 and not re.search(r"_proof|\d{3,}\.\.", meta):
        return normalizar(meta)
    base = re.sub(r"^\d+\.\s*", "", nombre_archivo)
    return normalizar(re.sub(r"\.pdf$", "", base, flags=re.I))


# --------------------------------------------------------------------------
# Orquestación
# --------------------------------------------------------------------------

def procesar_pdf(ruta: str, vocabulario: set[str]) -> dict:
    """Extrae título, autores y abstract de un único artículo."""
    nombre = os.path.basename(ruta)

    with fitz.open(ruta) as pdf:
        pagina = pdf[0]
        titulo = extraer_titulo(pdf, nombre)
        crudo = recortar_abstract(unicodedata.normalize("NFKC", pagina.get_text())) or ""
        autores = extraer_autores(pagina, titulo)

    reparado, cortes = unir_cortes(crudo, vocabulario)
    abstract = normalizar(reparado)

    return {
        "id": "",                       # se asigna en procesar_corpus
        "archivo": nombre,
        "titulo": titulo,
        "autores": ", ".join(autores),
        "texto": abstract,
        "_palabras": len(abstract.split()),
        "_cortes": cortes,
        "_revisar": sorted(
            n for n, r in (("autores", not autores),
                           ("abstract", not es_abstract_valido(abstract))) if r
        ),
    }


def procesar_corpus(dir_pdfs: str, dir_salida: str) -> list[dict]:
    """
    Procesa todos los PDF de un directorio y escribe abstracts.json y .csv.

    El esquema de salida (id, archivo, titulo, texto) es el que consumen los
    módulos de similitud y clustering; 'autores' es el otro metadato que pide
    el enunciado.
    """
    def orden(nombre: str) -> int:
        m = re.match(r"(\d+)", nombre)
        return int(m.group(1)) if m else 9999

    archivos = sorted(
        (f for f in os.listdir(dir_pdfs) if f.lower().endswith(".pdf")),
        key=orden,
    )
    rutas = [os.path.join(dir_pdfs, f) for f in archivos]
    vocabulario = construir_vocabulario(rutas)

    registros = []
    for i, ruta in enumerate(rutas, start=1):
        reg = procesar_pdf(ruta, vocabulario)
        reg["id"] = f"P{i}"
        reg.update(CORRECCIONES.get(reg["id"], {}))
        registros.append(reg)

    os.makedirs(dir_salida, exist_ok=True)
    with open(os.path.join(dir_salida, "abstracts.json"), "w", encoding="utf-8") as fh:
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

    registros = procesar_corpus(dir_pdfs, dir_salida)

    print(f"{'ID':<5}{'PAL':<6}{'CORTES':<8}{'REVISAR':<12}TITULO")
    print("-" * 90)
    for r in registros:
        marca = ",".join(r["_revisar"]) or "-"
        print(f"{r['id']:<5}{r['_palabras']:<6}{len(r['_cortes']):<8}"
              f"{marca:<12}{r['titulo'][:58]}")
    print("-" * 90)

    total = len(registros)
    print(f"Abstracts validos: {sum('abstract' not in r['_revisar'] for r in registros)}/{total}")
    print(f"Autores extraidos: {sum(bool(r['autores']) for r in registros)}/{total}")
    print(f"Cortes reparados : {sum(len(r['_cortes']) for r in registros)}")
    print(f"Salida           : {dir_salida}")


if __name__ == "__main__":
    main()

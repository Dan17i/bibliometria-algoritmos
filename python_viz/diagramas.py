"""
Diagramas de diseño del proyecto
--------------------------------
Dibuja con matplotlib los diagramas de la sección 2 del documento técnico:

  - paquetes.png: los paquetes y sus dependencias.
  - modulos_*.png: cada módulo con sus funciones. El proyecto no define clases: está
    organizado en módulos de funciones, y estos diagramas ocupan ese lugar.
  - secuencia_*.png: qué llama a qué para obtener cada resultado.

Los diagramas de paquetes y de módulos se leen del código con `ast`, así que no se
pueden desactualizar. Los de secuencia se escriben a mano en SECUENCIAS, y
tests/test_diagramas.py verifica que cada llamada dibujada exista en el código.

    python -m python_viz.diagramas
"""

from __future__ import annotations

import ast
import glob
import math
import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from python_viz.estilo import TINTA_PRIMARIA, TINTA_SECUNDARIA, TINTA_TENUE, aplicar_estilo, color_serie

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AZUL = "#1f4e79"
FONDO_AZUL = "#e4edf7"
BORDE = "#c8cabb"

# Quien aparece en los diagramas de secuencia. El segundo valor es el archivo que se
# verifica contra el código; None si es algo externo al proyecto.
PARTICIPANTES = {
    "consola": ("python -m", None),
    "nav": ("Navegador", None),
    "srv": ("servidor.py", "app/servidor.py"),
    "exp": ("explicaciones.py", "app/explicaciones.py"),
    "exp_c": ("explicaciones_\nclustering.py", "app/explicaciones_clustering.py"),
    "exp_p": ("explicaciones_\npatrones.py", "app/explicaciones_patrones.py"),
    "traza": ("traza_clasicos.py", "src/similarity/traza_clasicos.py"),
    "lev": ("levenshtein.py", "src/similarity/classic/levenshtein.py"),
    "nw": ("needleman_\nwunsch.py", "src/similarity/classic/needleman_wunsch.py"),
    "jac": ("jaccard.py", "src/similarity/classic/jaccard.py"),
    "cos": ("cosine.py", "src/similarity/classic/cosine.py"),
    "tok": ("tokenizador.py", "src/similarity/classic/tokenizador.py"),
    "met": ("metrics.py", "src/similarity/metrics.py"),
    "emb": ("embeddings.py", "src/similarity/ai/embeddings.py"),
    "dem": ("demostracion.py", "src/similarity/ai/demostracion.py"),
    "tra": ("transformacion.py", "src/preprocessing/transformacion.py"),
    "mod": ("demo_modelos.py", "src/similarity/ai/demo_modelos.py"),
    "pre": ("precalculo.py", "src/similarity/ai/precalculo.py"),
    "st": ("Modelo\n(caja negra)", None),
    "ext": ("extractor.py", "src/preprocessing/extractor.py"),
    "pdf": ("PyMuPDF", None),
    "c20": ("demo_corpus20.py", "src/similarity/ai/demo_corpus20.py"),
    "exp2": ("export.py", "src/clustering/export.py"),
    "lnk": ("linkage.py", "src/clustering/linkage.py"),
    "eva": ("evaluation.py", "src/clustering/evaluation.py"),
    "den": ("dendrograma.py", "python_viz/dendrograma.py"),
    "kmp": ("kmp.py", "src/pattern/kmp.py"),
    "bm": ("boyer_moore.py", "src/pattern/boyer_moore.py"),
    "rk": ("rabin_karp.py", "src/pattern/rabin_karp.py"),
    "run": ("runner.py", "src/benchmark/runner.py"),
}


def llamada(origen: str, destino: str, firma: str) -> tuple:
    """Una llamada a una función de otro módulo, o del mismo. Se verifica contra el código."""
    return ("llamada", origen, destino, firma)


def respuesta(origen: str, destino: str, texto: str) -> tuple:
    return ("respuesta", origen, destino, texto)


def externo(origen: str, destino: str, texto: str) -> tuple:
    """Una petición HTTP, un comando o una llamada a una librería. No se verifica."""
    return ("externo", origen, destino, texto)


def nota(quien: str, texto: str) -> tuple:
    """Lo que un participante hace por dentro, sin llamar a otra función."""
    return ("nota", quien, quien, texto)


_ABRIR_DEMO = [
    externo("nav", "srv", "POST /api/demostracion {algoritmo, ids}"),
    llamada("srv", "exp", "demostracion(clave, texto_a, texto_b, ...)"),
]
_CERRAR_DEMO = [
    nota("exp", "arma la idea, las fórmulas y los pasos con los valores"),
    respuesta("exp", "srv", "resultado"),
    llamada("srv", "srv", "_lectura_en_el_corpus(clave, valor)"),
    respuesta("srv", "nav", "JSON"),
    nota("nav", "KaTeX escribe las fórmulas"),
]

# clave: (título, participantes, mensajes)
SECUENCIAS = {
    "preprocesamiento": (
        "Preprocesamiento de los 20 PDF",
        ["consola", "ext", "pdf"],
        [
            externo("consola", "ext", "python -m src.preprocessing.extractor"),
            llamada("ext", "ext", "procesar_corpus(dir_pdfs, dir_salida)"),
            llamada("ext", "ext", "construir_vocabulario(rutas)"),
            externo("ext", "pdf", "fitz.open(ruta) y get_text(), de los 20 PDF"),
            respuesta("pdf", "ext", "texto de cada página"),
            nota("ext", "por cada PDF:"),
            llamada("ext", "ext", "procesar_pdf(ruta, vocabulario)"),
            llamada("ext", "ext", "extraer_titulo(pdf, nombre)"),
            llamada("ext", "ext", "recortar_abstract(texto_pagina)"),
            llamada("ext", "ext", "extraer_autores(pagina, titulo)"),
            llamada("ext", "ext", "unir_cortes(crudo, vocabulario)"),
            llamada("ext", "ext", "normalizar(reparado)"),
            nota("ext", "aplica CORRECCIONES y escribe abstracts.json y abstracts.csv"),
            respuesta("ext", "consola", "resumen por artículo"),
        ],
    ),
    "comparar": (
        "Comparar dos o más artículos",
        ["nav", "srv", "jac", "cos", "met"],
        [
            externo("nav", "srv", "POST /api/similitud {ids}"),
            nota("srv", "Levenshtein y NW: lee similitudes_pd.json"),
            llamada("srv", "srv", "_similitud_del_par(clave, id_a, id_b)"),
            llamada("srv", "jac", "jaccard_similarity(a, b)"),
            respuesta("jac", "srv", "similitud"),
            llamada("srv", "cos", "tfidf_cosine_similarity(a, b, TEXTOS, IDF)"),
            respuesta("cos", "srv", "pasos con el coseno"),
            llamada("srv", "met", "cosine_steps(vector_a, vector_b)"),
            respuesta("met", "srv", "coseno de GloVe y de MiniLM"),
            respuesta("srv", "nav", "seis valores, o una matriz por algoritmo"),
        ],
    ),
    "levenshtein": (
        "Levenshtein: demostración de un par",
        ["nav", "srv", "exp", "traza", "lev"],
        _ABRIR_DEMO + [
            llamada("exp", "traza", "traza_levenshtein(a, b, lado)"),
            llamada("traza", "lev", "levenshtein_distance(a, b)"),
            nota("lev", "llena la matriz (m + 1) × (n + 1)"),
            respuesta("lev", "traza", "distancia, matriz"),
            llamada("traza", "traza", "_esquinas(a, b, matriz, lado)"),
            llamada("traza", "traza", "operaciones_levenshtein(a, b, matriz)"),
            respuesta("traza", "exp", "distancia, esquinas, operaciones"),
        ] + _CERRAR_DEMO,
    ),
    "needleman_wunsch": (
        "Needleman-Wunsch: demostración de un par",
        ["nav", "srv", "exp", "traza", "nw"],
        _ABRIR_DEMO + [
            llamada("exp", "traza", "traza_needleman(a, b, lado)"),
            llamada("traza", "nw", "needleman_wunsch(a, b)"),
            nota("nw", "llena la matriz con el máximo de tres caminos"),
            respuesta("nw", "traza", "puntaje, matriz"),
            llamada("traza", "nw", "normalized_needleman_wunsch(a, b)"),
            respuesta("nw", "traza", "similitud en [0, 1]"),
            llamada("traza", "traza", "operaciones_needleman(a, b, matriz)"),
            respuesta("traza", "exp", "puntaje, esquinas, operaciones"),
        ] + _CERRAR_DEMO,
    ),
    "jaccard": (
        "Jaccard: demostración de un par",
        ["nav", "srv", "exp", "traza", "jac", "tok"],
        _ABRIR_DEMO + [
            llamada("exp", "traza", "traza_jaccard(a, b)"),
            llamada("traza", "jac", "jaccard_similarity(a, b)"),
            llamada("jac", "tok", "tokenizar(texto)"),
            llamada("jac", "tok", "terminos_unicos(tokens)"),
            respuesta("tok", "jac", "un conjunto por texto"),
            nota("jac", "intersección y unión con un diccionario"),
            respuesta("jac", "traza", "similitud, intersección, unión"),
            respuesta("traza", "exp", "conjuntos y tamaños"),
        ] + _CERRAR_DEMO,
    ),
    "coseno_tfidf": (
        "Coseno TF-IDF: demostración de un par",
        ["nav", "srv", "exp", "traza", "cos", "tok", "met"],
        _ABRIR_DEMO + [
            llamada("exp", "traza", "traza_tfidf(a, b, corpus, idf)"),
            llamada("traza", "cos", "tfidf_cosine_similarity(a, b, corpus, idf)"),
            llamada("cos", "tok", "tokenizar(texto)"),
            respuesta("tok", "cos", "tokens de cada texto"),
            nota("cos", "pesos tf · idf, producto punto y normas"),
            llamada("cos", "met", "angle_degrees(coseno)"),
            respuesta("met", "cos", "ángulo"),
            respuesta("cos", "traza", "todos los pasos"),
            respuesta("traza", "exp", "términos que más aportan"),
        ] + _CERRAR_DEMO,
    ),
    "glove": (
        "GloVe: demostración de un par",
        ["nav", "srv", "exp", "tra", "emb", "dem", "met"],
        _ABRIR_DEMO + [
            llamada("exp", "exp", "_vectores_de_palabras()"),
            llamada("exp", "tra", "transformar_abstract(texto)"),
            respuesta("tra", "exp", "abstract transformado"),
            llamada("exp", "emb", "average_word_vectors(texto, palabras, STOPWORDS)"),
            llamada("emb", "met", "mean_vector(vectores)"),
            respuesta("met", "emb", "vector promedio"),
            respuesta("emb", "exp", "tokens, vectores, promedio"),
            llamada("exp", "dem", "compare_vectors_steps(va, vb)"),
            llamada("dem", "met", "cosine_steps(va, vb)"),
            llamada("dem", "met", "euclidean_steps(va, vb)"),
            respuesta("dem", "exp", "producto punto, normas, coseno, ángulo"),
        ] + _CERRAR_DEMO,
    ),
    "minilm": (
        "MiniLM: los vectores se calculan una vez, y se comparan en cada consulta",
        ["pre", "mod", "st", "nav", "srv", "exp", "dem"],
        [
            nota("pre", "python -m src.similarity.ai.precalculo"),
            llamada("pre", "pre", "guardar_vectores()"),
            llamada("pre", "mod", "vectores_del_corpus(textos)"),
            llamada("mod", "mod", "sentence_embeddings(transformados)"),
            externo("mod", "st", "model.encode(textos)"),
            respuesta("st", "mod", "un vector de 384 dimensiones por abstract"),
            respuesta("mod", "pre", "vectores"),
            nota("pre", "escribe data/processed/vectores.json"),
            externo("nav", "srv", "POST /api/demostracion {algoritmo, ids}"),
            llamada("srv", "exp", "demostracion(clave, texto_a, texto_b, ...)"),
            llamada("exp", "dem", "compare_vectors_steps(va, vb)"),
            respuesta("dem", "exp", "producto punto, normas, coseno, ángulo"),
            respuesta("exp", "srv", "resultado"),
            respuesta("srv", "nav", "JSON"),
        ],
    ),
    "clustering": (
        "Agrupamiento: de los abstracts a los dendrogramas",
        ["c20", "mod", "met", "exp2", "lnk", "eva", "den"],
        [
            nota("c20", "python -m src.similarity.ai.demo_corpus20"),
            llamada("c20", "c20", "representaciones(textos)"),
            llamada("c20", "mod", "vectores_del_corpus(textos)"),
            respuesta("mod", "c20", "vectores de los abstracts transformados"),
            llamada("c20", "met", "pairwise_matrix(vectores, cosine_distance)"),
            respuesta("met", "c20", "matriz de distancias n × n"),
            llamada("c20", "exp2", "export_merges_json(ruta, dist, ids)"),
            llamada("exp2", "exp2", "build_export(dist, labels, linkages)"),
            llamada("exp2", "lnk", "hierarchical_clustering(dist, enlace)"),
            respuesta("lnk", "exp2", "n − 1 fusiones"),
            llamada("exp2", "eva", "cophenetic_correlation(dist, fusiones)"),
            respuesta("eva", "exp2", "CPCC"),
            nota("exp2", "escribe fusiones_<representación>.json"),
            llamada("c20", "c20", "evaluar(dist)"),
            llamada("c20", "eva", "best_k_by_silhouette(dist, fusiones)"),
            respuesta("eva", "c20", "mejor corte y silueta"),
            nota("den", "python -m python_viz.dendrograma"),
            llamada("den", "den", "renderizar(ruta_json, dir_salida, sufijo, fuente)"),
            llamada("den", "den", "dibujar_dendrograma(ax, fusiones, etiquetas, color)"),
        ],
    ),
    "agrupamiento_web": (
        "Agrupamiento: demostración en la página",
        ["nav", "srv", "exp_c", "lnk", "eva", "met"],
        [
            externo("nav", "srv", "POST /api/demostracion-agrupamiento"),
            llamada("srv", "exp_c", "demostracion(clave, enlace, vectores, textos, ids)"),
            llamada("exp_c", "exp_c", "matriz_de_distancias(clave, vectores, textos)"),
            llamada("exp_c", "met", "pairwise_matrix(vectores, cosine_distance)"),
            respuesta("met", "exp_c", "matriz de distancias"),
            llamada("exp_c", "exp_c", "_corrida(dist, enlace)"),
            nota("exp_c", "se repite para los tres enlaces:"),
            llamada("exp_c", "lnk", "hierarchical_clustering(dist, enlace, return_steps=True)"),
            respuesta("lnk", "exp_c", "fusiones y pasos"),
            llamada("exp_c", "eva", "cophenetic_correlation(dist, fusiones)"),
            llamada("exp_c", "eva", "best_k_by_silhouette(dist, fusiones)"),
            respuesta("eva", "exp_c", "CPCC y silueta"),
            llamada("exp_c", "exp_c", "_conclusiones(clave, enlace, corridas)"),
            respuesta("exp_c", "srv", "pasos y conclusiones"),
            respuesta("srv", "nav", "JSON"),
        ],
    ),
    "buscar": (
        "Búsqueda de una cadena en los 20 abstracts",
        ["nav", "srv", "kmp", "bm", "rk"],
        [
            externo("nav", "srv", "POST /api/buscar {patron}"),
            nota("srv", "une los 20 abstracts, sin transformarlos"),
            llamada("srv", "kmp", "buscar(corpus_unido, patron)"),
            respuesta("kmp", "srv", "posiciones"),
            llamada("srv", "bm", "buscar(corpus_unido, patron)"),
            respuesta("bm", "srv", "posiciones"),
            llamada("srv", "rk", "buscar(corpus_unido, patron)"),
            respuesta("rk", "srv", "posiciones"),
            nota("srv", "comprueba que los tres coinciden"),
            respuesta("srv", "nav", "ocurrencias, comparaciones y tiempos"),
        ],
    ),
    "kmp": (
        "KMP: demostración",
        ["nav", "srv", "exp_p", "kmp"],
        [
            externo("nav", "srv", "POST /api/demostracion-patron {algoritmo, patron}"),
            llamada("srv", "exp_p", "fragmento(CORPUS, patron)"),
            respuesta("exp_p", "srv", "artículo y texto de la traza"),
            llamada("srv", "exp_p", "demostracion(clave, texto, patron)"),
            llamada("exp_p", "exp_p", "_kmp(texto, patron)"),
            llamada("exp_p", "kmp", "construir_lps_con_pasos(patron)"),
            respuesta("kmp", "exp_p", "tabla LPS"),
            llamada("exp_p", "kmp", "buscar_con_pasos(texto, patron)"),
            respuesta("kmp", "exp_p", "cada comparación"),
            respuesta("exp_p", "srv", "pasos y tablas"),
            llamada("srv", "exp_p", "comparacion_en_corpus(corpus_unido, patron)"),
            respuesta("exp_p", "srv", "tiempos y conclusiones"),
            respuesta("srv", "nav", "JSON"),
        ],
    ),
    "boyer_moore": (
        "Boyer-Moore: demostración",
        ["nav", "srv", "exp_p", "bm"],
        [
            externo("nav", "srv", "POST /api/demostracion-patron {algoritmo, patron}"),
            llamada("srv", "exp_p", "fragmento(CORPUS, patron)"),
            respuesta("exp_p", "srv", "artículo y texto de la traza"),
            llamada("srv", "exp_p", "demostracion(clave, texto, patron)"),
            llamada("exp_p", "exp_p", "_boyer_moore(texto, patron)"),
            llamada("exp_p", "bm", "tabla_caracter_malo(patron)"),
            llamada("exp_p", "bm", "tabla_sufijo_bueno(patron)"),
            respuesta("bm", "exp_p", "las dos tablas"),
            llamada("exp_p", "bm", "buscar_con_pasos(texto, patron, max_pasos)"),
            respuesta("bm", "exp_p", "cada alineación y su salto"),
            respuesta("exp_p", "srv", "pasos y tablas"),
            llamada("srv", "exp_p", "comparacion_en_corpus(corpus_unido, patron)"),
            respuesta("exp_p", "srv", "tiempos y conclusiones"),
            respuesta("srv", "nav", "JSON"),
        ],
    ),
    "rabin_karp": (
        "Rabin-Karp: demostración",
        ["nav", "srv", "exp_p", "rk"],
        [
            externo("nav", "srv", "POST /api/demostracion-patron {algoritmo, patron}"),
            llamada("srv", "exp_p", "fragmento(CORPUS, patron)"),
            respuesta("exp_p", "srv", "artículo y texto de la traza"),
            llamada("srv", "exp_p", "demostracion(clave, texto, patron)"),
            llamada("exp_p", "exp_p", "_rabin_karp(texto, patron)"),
            llamada("exp_p", "rk", "preprocesar(texto, patron)"),
            respuesta("rk", "exp_p", "peso y hash del patrón"),
            llamada("exp_p", "rk", "buscar_con_pasos(texto, patron, max_pasos)"),
            respuesta("rk", "exp_p", "hash de cada ventana"),
            llamada("exp_p", "rk", "buscar_con_pasos(texto, patron, max_pasos, primo)"),
            respuesta("rk", "exp_p", "colisiones con un primo pequeño"),
            respuesta("exp_p", "srv", "pasos y tablas"),
            llamada("srv", "exp_p", "comparacion_en_corpus(corpus_unido, patron)"),
            respuesta("exp_p", "srv", "tiempos y conclusiones"),
            respuesta("srv", "nav", "JSON"),
        ],
    ),
}


# --------------------------------------------------------------------------
# Diagramas de secuencia
# --------------------------------------------------------------------------

def dibujar_secuencia(clave: str, dir_salida: str) -> str:
    """
    Dibuja un diagrama de secuencia. Solo hay un mensaje por renglón, así que un texto
    largo puede pasar sobre otras columnas sin chocar con nada.
    """
    titulo, participantes, mensajes = SECUENCIAS[clave]
    columnas = {p: i for i, p in enumerate(participantes)}
    n = len(participantes)
    paso_x = 1.42 if n >= 6 else 1.75 if n == 5 else 2.1
    alto = 0.4
    ultimo = len(participantes) - 1
    fig, ax = plt.subplots(figsize=(paso_x * ultimo + 2.2, alto * (len(mensajes) + 2) + 1.0))
    x = lambda p: columnas[p] * paso_x
    y_fin = -alto * (len(mensajes) + 0.6)

    for p in participantes:
        nombre, archivo = PARTICIPANTES[p]
        ax.plot([x(p), x(p)], [0, y_fin], color=BORDE, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        ax.add_patch(FancyBboxPatch(
            (x(p) - paso_x * 0.48, 0.02), paso_x * 0.96, 0.6, boxstyle="round,pad=0,rounding_size=0.06",
            facecolor="white" if archivo is None else FONDO_AZUL, edgecolor=BORDE if archivo is None else AZUL,
            linewidth=1.1, zorder=3,
        ))
        ax.text(x(p), 0.32, nombre, ha="center", va="center", fontsize=8.6, fontweight="bold", zorder=4)

    for i, (tipo, origen, destino, texto) in enumerate(mensajes):
        y = -alto * (i + 0.8)
        hacia_la_izquierda = columnas[origen] > ultimo / 2
        if tipo == "nota":
            ax.text(x(origen) + (-0.1 if hacia_la_izquierda else 0.1), y, texto, fontsize=9, style="italic",
                    color=TINTA_SECUNDARIA, va="center", ha="right" if hacia_la_izquierda else "left", zorder=4,
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "#f6f6f2", "edgecolor": "none"})
            continue

        if origen == destino:
            ax.add_patch(FancyArrowPatch(
                (x(origen), y + 0.1), (x(origen), y - 0.1), connectionstyle="arc3,rad=-1.6",
                arrowstyle="-|>", mutation_scale=8, color=AZUL, linewidth=1, zorder=3,
            ))
            ax.text(x(origen) + (-0.14 if hacia_la_izquierda else 0.26), y, texto, fontsize=9, va="center",
                    ha="right" if hacia_la_izquierda else "left", family="monospace", color=TINTA_PRIMARIA, zorder=4,
                    bbox={"boxstyle": "round,pad=0.1", "facecolor": "white", "edgecolor": "none"})
            continue

        es_respuesta = tipo == "respuesta"
        ax.add_patch(FancyArrowPatch(
            (x(origen), y), (x(destino), y), arrowstyle="->" if es_respuesta else "-|>",
            mutation_scale=9, color=TINTA_TENUE if es_respuesta else AZUL, linewidth=1,
            linestyle=(0, (3, 2)) if es_respuesta else "solid", shrinkA=0, shrinkB=0, zorder=3,
        ))
        ax.text((x(origen) + x(destino)) / 2, y + 0.035, texto, fontsize=9, ha="center", va="bottom",
                family="monospace" if tipo == "llamada" else None,
                color=TINTA_SECUNDARIA if es_respuesta else TINTA_PRIMARIA, zorder=2,
                bbox={"boxstyle": "round,pad=0.06", "facecolor": "white", "edgecolor": "none"})

    margen = max(1.0, paso_x * 0.55)
    ax.set_xlim(-margen, paso_x * ultimo + margen)
    ax.set_ylim(y_fin - 0.1, 0.7)
    ax.axis("off")

    ruta = os.path.join(dir_salida, f"secuencia_{clave}.png")
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------
# Lectura del código
# --------------------------------------------------------------------------

def funciones_de(ruta: str) -> list[str]:
    """Funciones definidas en el nivel superior de un módulo, en orden de aparición."""
    with open(os.path.join(RAIZ, ruta), encoding="utf-8") as fh:
        arbol = ast.parse(fh.read())
    return [n.name for n in arbol.body if isinstance(n, ast.FunctionDef)]


def llamadas_de(ruta: str) -> set[str]:
    """
    Nombres que el módulo llama en alguna parte: directos, con punto o a través de un
    diccionario de funciones como {"kmp": _kmp, ...}[clave](texto, patron).
    """
    with open(os.path.join(RAIZ, ruta), encoding="utf-8") as fh:
        arbol = ast.parse(fh.read())
    nombres = set()
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call):
            f = nodo.func
            nombres.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None))
        elif isinstance(nodo, ast.Dict):
            nombres.update(v.id for v in nodo.values if isinstance(v, ast.Name))
    return nombres


def _paquete_de(modulo: str) -> str | None:
    """El paquete del diagrama al que pertenece un módulo importado."""
    if modulo == "src.similarity.metrics":
        return "metrics"
    if modulo.startswith("src.similarity.traza"):
        return "classic"
    for prefijo, paquete in (
        ("src.preprocessing", "preprocessing"), ("src.similarity.classic", "classic"),
        ("src.similarity.ai", "ai"), ("src.clustering", "clustering"), ("src.pattern", "pattern"),
        ("src.benchmark", "benchmark"), ("python_viz", "python_viz"), ("app", "app"),
    ):
        if modulo == prefijo or modulo.startswith(prefijo + "."):
            return paquete
    return None


def imports_entre_modulos() -> dict[str, set[str]]:
    """Por cada módulo del proyecto, los módulos del proyecto que importa."""
    rutas = {}
    for carpeta in ("src", "app", "python_viz"):
        for ruta in glob.glob(os.path.join(RAIZ, carpeta, "**", "*.py"), recursive=True):
            relativa = os.path.relpath(ruta, RAIZ).replace(os.sep, "/")
            rutas[relativa[:-3].replace("/", ".").removesuffix(".__init__")] = ruta

    grafo = {}
    for modulo, ruta in rutas.items():
        with open(ruta, encoding="utf-8") as fh:
            arbol = ast.parse(fh.read())
        grafo[modulo] = set()
        for nodo in ast.walk(arbol):
            if isinstance(nodo, ast.ImportFrom) and nodo.module:
                candidatos = [nodo.module] + [f"{nodo.module}.{a.name}" for a in nodo.names]
            elif isinstance(nodo, ast.Import):
                candidatos = [a.name for a in nodo.names]
            else:
                candidatos = []
            grafo[modulo].update(c for c in candidatos if c in rutas and c != modulo)
    return grafo


def dependencias() -> set[tuple[str, str]]:
    """Pares (paquete que importa, paquete importado), leídos de los imports reales."""
    pares = set()
    for carpeta in ("src", "app", "python_viz"):
        for ruta in glob.glob(os.path.join(RAIZ, carpeta, "**", "*.py"), recursive=True):
            relativa = os.path.relpath(ruta, RAIZ).replace(os.sep, "/")
            origen = _paquete_de(relativa[:-3].replace("/", "."))
            with open(ruta, encoding="utf-8") as fh:
                arbol = ast.parse(fh.read())
            for nodo in ast.walk(arbol):
                modulos = []
                if isinstance(nodo, ast.ImportFrom) and nodo.module:
                    modulos = [nodo.module] + [f"{nodo.module}.{a.name}" for a in nodo.names]
                for modulo in modulos:
                    destino = _paquete_de(modulo)
                    if origen and destino and origen != destino:
                        pares.add((origen, destino))
    return pares


# --------------------------------------------------------------------------
# Diagrama de paquetes
# --------------------------------------------------------------------------

# En este orden, alrededor de una elipse. Una recta entre dos puntos de una elipse no
# pasa por ningún otro, así que ninguna flecha queda oculta detrás de otra caja.
PAQUETES = {
    "app": ("app/", "página web, servidor\ny demostraciones"),
    "benchmark": ("src/benchmark/", "medición de tiempos"),
    "pattern": ("src/pattern/", "KMP, Boyer-Moore,\nRabin-Karp"),
    "clustering": ("src/clustering/", "enlaces, evaluación\ny exportación"),
    "metrics": ("src/similarity/metrics.py", "coseno, normas,\nlogaritmo, ángulo"),
    "classic": ("src/similarity/classic/", "los cuatro clásicos\ny tokenizador"),
    "ai": ("src/similarity/ai/", "embeddings, comparación\ny caso de estudio"),
    "preprocessing": ("src/preprocessing/", "extractor y\ntransformación"),
    "python_viz": ("python_viz/", "dendrogramas, gráficas\ny diagramas"),
}


def _posiciones_de_paquetes() -> dict[str, tuple[float, float]]:
    """Reparte los paquetes por la elipse, con app/ arriba."""
    posiciones = {}
    claves = list(PAQUETES)
    for i, clave in enumerate(claves):
        angulo = math.pi / 2 - 2 * math.pi * i / len(claves)
        posiciones[clave] = (5.0 * math.cos(angulo), 3.5 * math.sin(angulo))
    return posiciones


def dibujar_paquetes(dir_salida: str) -> str:
    fig, ax = plt.subplots(figsize=(10.6, 8.2))
    ancho, alto = 3.05, 0.98
    posiciones = _posiciones_de_paquetes()

    def choca(a: str, b: str, curvatura: float) -> bool:
        """Si la curva de a hacia b pasa por encima de la caja de un tercero."""
        (xa, ya), (xb, yb) = punto_borde(a, b), punto_borde(b, a)
        xc = (xa + xb) / 2 + curvatura * (yb - ya)
        yc = (ya + yb) / 2 - curvatura * (xb - xa)
        for k in range(41):
            t = k / 40
            x = (1 - t) ** 2 * xa + 2 * (1 - t) * t * xc + t ** 2 * xb
            y = (1 - t) ** 2 * ya + 2 * (1 - t) * t * yc + t ** 2 * yb
            for otro, (xo, yo) in posiciones.items():
                if otro not in (a, b) and abs(x - xo) < ancho / 2 + 0.08 and abs(y - yo) < alto / 2 + 0.08:
                    return True
        return False

    def punto_borde(a: str, b: str) -> tuple[float, float]:
        xa, ya = posiciones[a]
        xb, yb = posiciones[b]
        dx, dy = xb - xa, yb - ya
        escala = min(abs((ancho / 2) / dx) if dx else 9e9, abs((alto / 2) / dy) if dy else 9e9)
        return xa + dx * escala, ya + dy * escala

    # Cada paquete que importa tiene su color, para seguir sus flechas entre los cruces
    origenes = sorted({origen for origen, _ in dependencias()}, key=list(PAQUETES).index)
    for origen, destino in sorted(dependencias()):
        curvatura = 0.05
        for candidata in (0.05, 0.12, -0.12, 0.2, -0.2, 0.3, -0.3, 0.42, -0.42):
            if not choca(origen, destino, candidata):
                curvatura = candidata
                break
        ax.add_patch(FancyArrowPatch(
            punto_borde(origen, destino), punto_borde(destino, origen), arrowstyle="-|>",
            mutation_scale=12, color=color_serie(origenes.index(origen)), linewidth=1.2, shrinkA=2, shrinkB=2,
            connectionstyle=f"arc3,rad={curvatura}", zorder=1,
        ))

    for clave, (nombre, descripcion) in PAQUETES.items():
        x, y = posiciones[clave]
        color = color_serie(origenes.index(clave)) if clave in origenes else BORDE
        ax.add_patch(FancyBboxPatch(
            (x - ancho / 2, y - alto / 2), ancho, alto, boxstyle="round,pad=0,rounding_size=0.08",
            facecolor="white", edgecolor=color, linewidth=1.8, zorder=2,
        ))
        ax.text(x, y + 0.25, nombre, ha="center", va="center", fontsize=9.6, fontweight="bold", zorder=3)
        ax.text(x, y - 0.15, descripcion, ha="center", va="center", fontsize=8.6, color=TINTA_SECUNDARIA, zorder=3)

    ax.set_xlim(-6.9, 6.9)
    ax.set_ylim(-4.3, 4.3)
    ax.axis("off")
    ruta = os.path.join(dir_salida, "paquetes.png")
    fig.savefig(ruta)
    plt.close(fig)
    return ruta


# --------------------------------------------------------------------------
# Diagramas de módulos
# --------------------------------------------------------------------------

GRUPOS_DE_MODULOS = {
    "similitud": ("Similitud textual y preprocesamiento", [
        "src/preprocessing/extractor.py", "src/preprocessing/transformacion.py",
        "src/similarity/classic/tokenizador.py", "src/similarity/classic/levenshtein.py",
        "src/similarity/classic/needleman_wunsch.py", "src/similarity/classic/jaccard.py",
        "src/similarity/classic/cosine.py", "src/similarity/metrics.py",
        "src/similarity/traza_clasicos.py", "src/similarity/ai/embeddings.py",
        "src/similarity/ai/stopwords.py", "src/similarity/ai/compare.py",
    ]),
    "ia_y_clustering": ("Corridas de IA y agrupamiento", [
        "src/similarity/ai/demostracion.py", "src/similarity/ai/demo_modelos.py",
        "src/similarity/ai/precalculo.py", "src/similarity/ai/caso_estudio.py",
        "src/similarity/ai/demo_corpus20.py", "src/similarity/ai/demo_clasicos20.py",
        "src/similarity/ai/demo_flujo.py", "src/clustering/linkage.py",
        "src/clustering/evaluation.py", "src/clustering/export.py",
    ]),
    "patrones": ("Búsqueda de patrones y medición", [
        "src/pattern/kmp.py", "src/pattern/boyer_moore.py", "src/pattern/rabin_karp.py",
        "src/pattern/buscador.py", "src/pattern/traza.py", "src/benchmark/runner.py",
    ]),
    "app_y_viz": ("Aplicación y visualización", [
        "app/servidor.py", "app/explicaciones.py", "app/explicaciones_clustering.py",
        "app/explicaciones_patrones.py", "app/lanzar.py", "python_viz/dendrograma.py",
        "python_viz/graficas.py", "python_viz/trazas.py", "python_viz/arquitectura.py",
        "python_viz/diagramas.py", "python_viz/estilo.py",
    ]),
}


def dibujar_modulos(clave: str, dir_salida: str) -> str:
    """
    Una caja por módulo, al estilo de una clase de UML: el nombre arriba y sus
    funciones debajo. Las privadas llevan «-» y las públicas «+».
    """
    _, modulos = GRUPOS_DE_MODULOS[clave]
    columnas = 3
    ancho, renglon = 3.5, 0.23
    alturas = [0.0] * columnas
    cajas = []
    for ruta in modulos:
        funciones = funciones_de(ruta)
        alto = 0.62 + renglon * len(funciones) + 0.1
        c = alturas.index(min(alturas))
        cajas.append((c, alturas[c], alto, ruta, funciones))
        alturas[c] += alto + 0.22

    total = max(alturas)
    fig, ax = plt.subplots(figsize=(columnas * ancho + 0.5, total * 0.92 + 0.9))
    for c, y0, alto, ruta, funciones in cajas:
        x = c * (ancho + 0.1)
        y = total - y0
        ax.add_patch(FancyBboxPatch(
            (x, y - alto), ancho, alto, boxstyle="round,pad=0,rounding_size=0.05",
            facecolor="white", edgecolor=AZUL, linewidth=1,
        ))
        ax.add_patch(plt.Rectangle((x + 0.01, y - 0.56), ancho - 0.02, 0.54, facecolor=FONDO_AZUL, edgecolor="none"))
        ax.text(x + ancho / 2, y - 0.17, "«módulo»", ha="center", va="center", fontsize=7, color=TINTA_SECUNDARIA)
        ax.text(x + ancho / 2, y - 0.39, os.path.basename(ruta), ha="center", va="center", fontsize=9.6, fontweight="bold")
        ax.plot([x, x + ancho], [y - 0.57, y - 0.57], color=AZUL, linewidth=0.8)
        for i, f in enumerate(funciones):
            visibilidad = "-" if f.startswith("_") else "+"
            nombre = f"{visibilidad} {f}()"
            ax.text(x + 0.1, y - 0.72 - i * renglon, nombre, fontsize=8.8, family="monospace", va="center",
                    color=TINTA_SECUNDARIA if f.startswith("_") else TINTA_PRIMARIA)
        if not funciones:
            ax.text(x + 0.1, y - 0.72, "(solo datos)", fontsize=8.8, va="center", color=TINTA_TENUE)

    ax.set_xlim(-0.1, columnas * (ancho + 0.1))
    ax.set_ylim(-0.1, total + 0.1)
    ax.axis("off")
    ruta_salida = os.path.join(dir_salida, f"modulos_{clave}.png")
    fig.savefig(ruta_salida)
    plt.close(fig)
    return ruta_salida


def main() -> None:
    aplicar_estilo()
    dir_salida = os.path.join(RAIZ, "data", "figuras")
    print(dibujar_paquetes(dir_salida))
    for clave in GRUPOS_DE_MODULOS:
        print(dibujar_modulos(clave, dir_salida))
    for clave in SECUENCIAS:
        print(dibujar_secuencia(clave, dir_salida))


if __name__ == "__main__":
    main()

"""
Aplicación web del proyecto (Requerimiento 4: despliegue)
==========================================================
Servidor Flask que expone los algoritmos del proyecto para consultarlos en
vivo desde el navegador.

QUÉ SE CALCULA EN VIVO Y QUÉ NO
-------------------------------
Los cuatro algoritmos clásicos y los tres de búsqueda de patrones son código
puro sin dependencias: se ejecutan en cada petición sobre los textos elegidos.

Los dos enfoques de IA necesitan cargar modelos preentrenados que ocupan
varios GB. La aplicación los usa si están instalados (`requirements-ai.txt`)
y, si no, lo dice en la respuesta en vez de fallar o de mostrar un valor
inventado.

Los dendrogramas se sirven como imágenes ya renderizadas: el clustering de 20
abstracts con tres enlaces no cambia entre peticiones, así que recalcularlo en
cada carga sería gasto sin beneficio.

EJECUCIÓN
---------
    python -m app.servidor

y abrir http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import os
import time

from flask import Flask, jsonify, request, send_from_directory

from src.pattern import boyer_moore, kmp, rabin_karp
from src.similarity.classic.cosine import (
    compute_idf,
    cosine_similarity,
    tfidf_cosine_similarity,
)
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.classic.levenshtein import levenshtein_distance
from src.similarity.classic.needleman_wunsch import normalized_needleman_wunsch

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATOS = os.path.join(RAIZ, "data", "processed")
DIR_FIGURAS = os.path.join(RAIZ, "data", "figuras")
DIR_FUSIONES = os.path.join(RAIZ, "tests", "fixtures", "daniel", "corpus20")

app = Flask(__name__, static_folder=None)

# El corpus se carga una vez al arrancar: son 20 abstractos que no cambian.
with open(os.path.join(DIR_DATOS, "abstracts.json"), encoding="utf-8") as fh:
    CORPUS: list[dict] = json.load(fh)

POR_ID = {r["id"]: r for r in CORPUS}
TEXTOS = [r["texto"] for r in CORPUS]

# El IDF se calcula sobre los 20 abstractos y es el mismo para toda consulta:
# precalcularlo evita repetir ese recorrido en cada petición.
IDF = compute_idf(TEXTOS)

REPRESENTACIONES = {
    "minilm": ("Embeddings MiniLM", "fusiones_minilm.json", ""),
    "glove": ("Embeddings GloVe", "fusiones_glove.json", "_glove"),
    "glove_centrado": ("GloVe centrado", "fusiones_glove_centrado.json",
                       "_glove_centrado"),
    "jaccard": ("Coeficiente de Jaccard", None, "_jaccard"),
}


# --------------------------------------------------------------------------
# Cálculo de similitudes
# --------------------------------------------------------------------------

def _cronometrar(funcion, *args):
    """Ejecuta y devuelve (resultado, milisegundos)."""
    inicio = time.perf_counter()
    valor = funcion(*args)
    return valor, (time.perf_counter() - inicio) * 1000


def similitudes_clasicas(texto_a: str, texto_b: str) -> list[dict]:
    """
    Los cuatro algoritmos clásicos sobre un par de abstracts.

    Levenshtein devuelve una distancia en ediciones, no una similitud. Para
    poder comparar las cuatro medidas en la misma escala se normaliza:

        similitud = 1 − distancia / longitud_del_mas_largo

    Así un par idéntico da 1 y uno sin nada en común tiende a 0.
    """
    resultados = []

    # --- Levenshtein ------------------------------------------------------
    (distancia, _matriz), ms = _cronometrar(levenshtein_distance, texto_a, texto_b)
    mayor = max(len(texto_a), len(texto_b)) or 1
    resultados.append({
        "algoritmo": "Levenshtein",
        "familia": "Programación dinámica",
        "similitud": 1 - distancia / mayor,
        "detalle": f"{distancia} ediciones sobre {mayor} caracteres",
        "ms": ms,
        "complejidad": "O(m·n)",
    })

    # --- Needleman-Wunsch -------------------------------------------------
    valor, ms = _cronometrar(normalized_needleman_wunsch, texto_a, texto_b)
    resultados.append({
        "algoritmo": "Needleman-Wunsch",
        "familia": "Programación dinámica",
        "similitud": valor,
        "detalle": "alineamiento global normalizado",
        "ms": ms,
        "complejidad": "O(m·n)",
    })

    # --- Jaccard ----------------------------------------------------------
    (valor, conj_a, conj_b), ms = _cronometrar(jaccard_similarity, texto_a, texto_b)
    comunes = len(conj_a & conj_b)
    resultados.append({
        "algoritmo": "Jaccard",
        "familia": "Conjuntos",
        "similitud": valor,
        "detalle": f"{comunes} términos comunes de "
                   f"{len(conj_a | conj_b)} en la unión",
        "ms": ms,
        "complejidad": "O(m+n)",
    })

    # --- Coseno con frecuencias (TF) --------------------------------------
    (salida, ms) = _cronometrar(cosine_similarity, texto_a, texto_b)
    valor, _v1, _v2, vocabulario = salida
    resultados.append({
        "algoritmo": "Coseno (TF)",
        "familia": "Espacio vectorial",
        "similitud": valor,
        "detalle": f"vocabulario unificado de {len(vocabulario)} términos",
        "ms": ms,
        "complejidad": "O(m+n)",
    })

    # --- Coseno con TF-IDF ------------------------------------------------
    salida, ms = _cronometrar(tfidf_cosine_similarity, texto_a, texto_b, TEXTOS)
    resultados.append({
        "algoritmo": "Coseno (TF-IDF)",
        "familia": "Espacio vectorial",
        "similitud": salida["coseno"],
        "detalle": f"ángulo de {salida['angulo_grados']:.1f}° entre los vectores "
                   f"de {len(salida['vocabulario'])} términos",
        "ms": ms,
        "complejidad": "O(m+n)",
    })

    return resultados


def similitudes_ia(texto_a: str, texto_b: str) -> list[dict]:
    """
    Los dos enfoques de IA, si los modelos están disponibles.

    Se devuelve una lista vacía con un aviso cuando faltan: es preferible
    decirlo a mostrar un número que no corresponde al método anunciado.
    """
    resultados = []
    try:
        from src.similarity.ai.demo_modelos import (
            glove_embeddings,
            sentence_embeddings,
        )
        from src.similarity.metrics import cosine_similarity as coseno_vectores
    except ImportError:
        return resultados

    for nombre, funcion in (("GloVe promediado", glove_embeddings),
                            ("MiniLM", sentence_embeddings)):
        try:
            vectores, ms = _cronometrar(funcion, [texto_a, texto_b])
            valor = coseno_vectores(vectores[0], vectores[1])
            resultados.append({
                "algoritmo": nombre,
                "familia": "Embeddings",
                "similitud": valor,
                "detalle": f"vectores de {len(vectores[0])} dimensiones; "
                           "el coseno se calcula a mano",
                "ms": ms,
                "complejidad": "O(d) sobre los vectores",
            })
        except Exception as error:      # modelo ausente o sin memoria
            resultados.append({
                "algoritmo": nombre,
                "familia": "Embeddings",
                "similitud": None,
                "detalle": f"modelo no disponible: {type(error).__name__}",
                "ms": None,
                "complejidad": "O(d) sobre los vectores",
            })

    return resultados


# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------

@app.get("/")
def indice():
    return send_from_directory(os.path.join(RAIZ, "app", "static"), "index.html")


@app.get("/static/<path:nombre>")
def estaticos(nombre: str):
    return send_from_directory(os.path.join(RAIZ, "app", "static"), nombre)


@app.get("/figuras/<path:nombre>")
def figuras(nombre: str):
    return send_from_directory(DIR_FIGURAS, nombre)


@app.get("/api/articulos")
def articulos():
    """Catálogo del corpus para poblar el selector."""
    return jsonify({
        "total": len(CORPUS),
        "articulos": [
            {
                "id": r["id"],
                "titulo": r["titulo"],
                "autores": r["autores"],
                "palabras": r["_palabras"],
            }
            for r in CORPUS
        ],
    })


@app.post("/api/similitud")
def similitud():
    """
    Calcula las similitudes entre los artículos seleccionados.

    Con dos artículos devuelve el detalle de cada algoritmo. Con más de dos,
    devuelve además la matriz por pares de un algoritmo rápido, porque las
    medidas de programación dinámica sobre todos los pares de una selección
    grande tardarían demasiado para una petición web.
    """
    datos = request.get_json(silent=True) or {}
    ids = datos.get("ids", [])

    if len(ids) < 2:
        return jsonify({"error": "Elige al menos dos artículos."}), 400
    desconocidos = [i for i in ids if i not in POR_ID]
    if desconocidos:
        return jsonify({"error": f"Artículos no encontrados: {desconocidos}"}), 400

    inicio = time.perf_counter()

    if len(ids) == 2:
        a, b = POR_ID[ids[0]], POR_ID[ids[1]]
        algoritmos = similitudes_clasicas(a["texto"], b["texto"])
        algoritmos += similitudes_ia(a["texto"], b["texto"])
        return jsonify({
            "modo": "par",
            "articulos": [{"id": r["id"], "titulo": r["titulo"]} for r in (a, b)],
            "algoritmos": algoritmos,
            "ms_total": (time.perf_counter() - inicio) * 1000,
        })

    # --- selección múltiple: matriz por pares -----------------------------
    metrica = datos.get("metrica", "jaccard")
    matriz = []
    for i, id_a in enumerate(ids):
        fila = []
        for j, id_b in enumerate(ids):
            if i == j:
                fila.append(1.0)
                continue
            texto_a = POR_ID[id_a]["texto"]
            texto_b = POR_ID[id_b]["texto"]
            if metrica == "coseno":
                fila.append(cosine_similarity(texto_a, texto_b)[0])
            else:
                fila.append(jaccard_similarity(texto_a, texto_b)[0])
        matriz.append(fila)

    return jsonify({
        "modo": "matriz",
        "metrica": metrica,
        "ids": ids,
        "titulos": [POR_ID[i]["titulo"] for i in ids],
        "matriz": matriz,
        "ms_total": (time.perf_counter() - inicio) * 1000,
    })


@app.post("/api/buscar")
def buscar():
    """Busca un patrón en los 20 abstracts con los tres algoritmos."""
    datos = request.get_json(silent=True) or {}
    patron = (datos.get("patron") or "").strip()

    if not patron:
        return jsonify({"error": "Escribe un patrón para buscar."}), 400
    if len(patron) > 120:
        return jsonify({"error": "El patrón no puede superar 120 caracteres."}), 400

    objetivo = patron.lower()
    corpus_unido = "\n".join(r["texto"].lower() for r in CORPUS)

    algoritmos = []
    salidas = {}
    for nombre, modulo in (("KMP", kmp), ("Boyer-Moore", boyer_moore),
                           ("Rabin-Karp", rabin_karp)):
        posiciones, ms = _cronometrar(modulo.buscar, corpus_unido, objetivo)
        metricas = modulo.buscar_con_metricas(corpus_unido, objetivo)
        salidas[nombre] = posiciones
        extra = (f"{metricas['saltos']} saltos" if nombre == "KMP"
                 else f"{metricas['alineaciones']} alineaciones"
                 if nombre == "Boyer-Moore"
                 else f"{metricas['colisiones']} colisiones de hash")
        algoritmos.append({
            "algoritmo": nombre,
            "ms": ms,
            "comparaciones": metricas["comparaciones"],
            "extra": extra,
            "ocurrencias": len(posiciones),
        })

    # Los tres deben coincidir: si no, hay un error en alguna implementación.
    nombres = list(salidas)
    coinciden = all(salidas[nombres[0]] == salidas[o] for o in nombres[1:])

    # Detalle por artículo, con el contexto de cada coincidencia
    por_articulo = []
    for registro in CORPUS:
        texto = registro["texto"]
        posiciones = kmp.buscar(texto.lower(), objetivo)
        if not posiciones:
            continue
        por_articulo.append({
            "id": registro["id"],
            "titulo": registro["titulo"],
            "ocurrencias": len(posiciones),
            "contextos": [
                _contexto(texto, p, len(objetivo)) for p in posiciones[:3]
            ],
        })

    return jsonify({
        "patron": patron,
        "coinciden": coinciden,
        "total": len(salidas["KMP"]),
        "articulos_con_coincidencia": len(por_articulo),
        "algoritmos": algoritmos,
        "por_articulo": por_articulo,
    })


def _contexto(texto: str, pos: int, largo: int, margen: int = 55) -> dict:
    """Fragmento alrededor de una coincidencia, separado para resaltarla."""
    inicio = max(0, pos - margen)
    fin = min(len(texto), pos + largo + margen)
    return {
        "antes": ("…" if inicio > 0 else "") + texto[inicio:pos],
        "coincidencia": texto[pos:pos + largo],
        "despues": texto[pos + largo:fin] + ("…" if fin < len(texto) else ""),
    }


@app.get("/api/clustering")
def clustering():
    """Representaciones disponibles con su CPCC por enlace."""
    disponibles = []
    for clave, (etiqueta, archivo, sufijo) in REPRESENTACIONES.items():
        if archivo:
            ruta = os.path.join(DIR_FUSIONES, archivo)
        else:
            ruta = os.path.join(DIR_DATOS, "fusiones.json")
        if not os.path.exists(ruta):
            continue
        with open(ruta, encoding="utf-8") as fh:
            datos = json.load(fh)
        enlaces = {k: v["cpcc"] for k, v in datos["linkages"].items()}
        disponibles.append({
            "clave": clave,
            "etiqueta": etiqueta,
            "cpcc": enlaces,
            "mejor_enlace": max(enlaces, key=enlaces.get),
            "figuras": {
                enlace: f"/figuras/dendrograma_{enlace}{sufijo}.png"
                for enlace in enlaces
            },
            "comparacion": f"/figuras/dendrogramas_comparacion{sufijo}.png",
        })

    disponibles.sort(key=lambda d: max(d["cpcc"].values()), reverse=True)
    return jsonify({"representaciones": disponibles})


@app.get("/api/corpus")
def corpus_info():
    """Cifras del corpus para la ficha de cabecera."""
    caracteres = sum(len(r["texto"]) for r in CORPUS)
    return jsonify({
        "articulos": len(CORPUS),
        "caracteres": caracteres,
        "palabras": sum(r["_palabras"] for r in CORPUS),
        "revisar": sum(1 for r in CORPUS if r["_revisar"]),
    })


if __name__ == "__main__":
    print("Corpus cargado:", len(CORPUS), "abstracts")
    print("Abre http://127.0.0.1:5000")
    app.run(debug=False, port=5000)

"""
Aplicación web del proyecto (Requerimiento 4: despliegue)
==========================================================
Servidor Flask que expone los algoritmos del proyecto para consultarlos en
vivo desde el navegador.

QUÉ SE CALCULA EN VIVO Y QUÉ NO
-------------------------------
Al comparar artículos, Levenshtein y Needleman-Wunsch se leen de
data/processed/similitudes_pd.json, que tiene los 190 pares del corpus: en el
servidor publicado cada par tarda unos 10 segundos. Jaccard, el coseno TF-IDF y la
comparación de los vectores de IA se calculan en cada consulta.

Al oprimir "Ver demostración" se ejecuta en ese momento solo el algoritmo elegido,
incluida la matriz completa de los dos de programación dinámica (app/explicaciones.py).
Los vectores de MiniLM salen de data/processed/vectores.json, porque el modelo no cabe
en la memoria del servidor publicado. El promedio de GloVe se repite a mano con los
vectores de palabras de data/processed/glove_palabras.json.

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

from app import explicaciones_clustering
from app.explicaciones import ACEPTAN_TEXTOS, LADO_CORPUS, LADO_TEXTOS, NOMBRES, demostracion
from src.pattern import boyer_moore, kmp, rabin_karp
from src.similarity.classic.cosine import compute_idf, tfidf_cosine_similarity
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.metrics import cosine_steps, euclidean_distance

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIR_DATOS = os.path.join(RAIZ, "data", "processed")
DIR_FIGURAS = os.path.join(RAIZ, "data", "figuras")
DIR_FUSIONES = os.path.join(RAIZ, "data", "resultados")

app = Flask(__name__, static_folder=None)

# El corpus se carga una vez al arrancar: son 20 abstractos que no cambian.
with open(os.path.join(DIR_DATOS, "abstracts.json"), encoding="utf-8") as fh:
    CORPUS: list[dict] = json.load(fh)

POR_ID = {r["id"]: r for r in CORPUS}
TEXTOS = [r["texto"] for r in CORPUS]

with open(os.path.join(DIR_DATOS, "vectores.json"), encoding="utf-8") as fh:
    VECTORES: dict = json.load(fh)

with open(os.path.join(DIR_DATOS, "similitudes_pd.json"), encoding="utf-8") as fh:
    MATRICES_PD: dict = json.load(fh)

POSICION = {}
for indice, identificador in enumerate(VECTORES["ids"]):
    POSICION[identificador] = indice

# El IDF se calcula sobre los 20 abstractos y es el mismo para toda consulta:
# precalcularlo evita repetir ese recorrido en cada petición.
IDF = compute_idf(TEXTOS)

ALGORITMOS = (
    {"clave": "levenshtein", "algoritmo": "Levenshtein",
     "familia": "Programación dinámica", "complejidad": "O(m·n)", "origen": "precalculado"},
    {"clave": "needleman_wunsch", "algoritmo": "Needleman-Wunsch",
     "familia": "Programación dinámica", "complejidad": "O(m·n)", "origen": "precalculado"},
    {"clave": "jaccard", "algoritmo": "Jaccard",
     "familia": "Conjuntos", "complejidad": "O(T)", "origen": "en vivo"},
    {"clave": "coseno_tfidf", "algoritmo": "Coseno (TF-IDF)",
     "familia": "Espacio vectorial", "complejidad": "O(T + V)", "origen": "en vivo"},
    {"clave": "glove", "algoritmo": "GloVe promediado",
     "familia": "Embeddings", "complejidad": "O(p)", "origen": "en vivo"},
    {"clave": "minilm", "algoritmo": "MiniLM",
     "familia": "Embeddings", "complejidad": "O(p)", "origen": "en vivo"},
)

REPRESENTACIONES = {
    "minilm": ("Embeddings MiniLM", "fusiones_minilm.json", ""),
    "glove": ("Embeddings GloVe", "fusiones_glove.json", "_glove"),
    "glove_centrado": ("GloVe centrado", "fusiones_glove_centrado.json",
                       "_glove_centrado"),
    "jaccard": ("Coeficiente de Jaccard", "fusiones_jaccard.json", "_jaccard"),
}


# --------------------------------------------------------------------------
# Cálculo de similitudes
# --------------------------------------------------------------------------

def _cronometrar(funcion, *args):
    """Ejecuta y devuelve (resultado, milisegundos)."""
    inicio = time.perf_counter()
    valor = funcion(*args)
    return valor, (time.perf_counter() - inicio) * 1000


def _detalle_del_par(clave: str, id_a: str, id_b: str) -> str:
    """Una línea sobre cómo se obtuvo el valor del par."""
    texto_a, texto_b = POR_ID[id_a]["texto"], POR_ID[id_b]["texto"]
    if clave == "levenshtein" or clave == "needleman_wunsch":
        return "calculado una vez sobre los 190 pares del corpus"
    if clave == "jaccard":
        _, interseccion, union = jaccard_similarity(texto_a, texto_b)
        return f"{len(interseccion)} términos comunes de {len(union)} en la unión"
    if clave == "coseno_tfidf":
        pasos = tfidf_cosine_similarity(texto_a, texto_b, TEXTOS, IDF)
        return (f"ángulo de {pasos['angulo_grados']:.1f}° entre los vectores "
                f"de {len(pasos['vocabulario'])} términos")
    i, j = POSICION[id_a], POSICION[id_b]
    pasos = cosine_steps(VECTORES[clave][i], VECTORES[clave][j])
    return (f"vectores de {len(VECTORES[clave][i])} dimensiones; ángulo de "
            f"{pasos['angle_degrees']:.1f}°, distancia euclidiana de "
            f"{euclidean_distance(VECTORES[clave][i], VECTORES[clave][j]):.3f}")


def _similitud_del_par(clave: str, id_a: str, id_b: str) -> float:
    """Similitud de un par con el algoritmo indicado."""
    i, j = POSICION[id_a], POSICION[id_b]
    if clave == "levenshtein" or clave == "needleman_wunsch":
        return MATRICES_PD[clave][i][j]
    if clave == "jaccard":
        return jaccard_similarity(POR_ID[id_a]["texto"], POR_ID[id_b]["texto"])[0]
    if clave == "coseno_tfidf":
        return tfidf_cosine_similarity(
            POR_ID[id_a]["texto"], POR_ID[id_b]["texto"], TEXTOS, IDF
        )["coseno"]
    return cosine_steps(VECTORES[clave][i], VECTORES[clave][j])["cosine"]


def matriz_de_similitud(clave: str, ids: list[str]) -> list[list[float]]:
    """Matriz simétrica de los artículos elegidos: se calcula solo i < j y se copia."""
    n = len(ids)
    matriz = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            valor = _similitud_del_par(clave, ids[i], ids[j])
            matriz[i][j] = valor
            matriz[j][i] = valor
    return matriz


def _ficha(id_articulo: str) -> dict:
    """Metadatos y abstract de un artículo, para mostrarlos junto al resultado."""
    registro = POR_ID[id_articulo]
    return {
        "id": registro["id"],
        "titulo": registro["titulo"],
        "autores": registro["autores"],
        "texto": registro["texto"],
    }


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
    Compara los artículos seleccionados con los seis algoritmos.

    Con dos artículos devuelve una fila por algoritmo. Con más de dos devuelve una
    matriz por algoritmo, de modo que cada par queda medido con los seis. La
    demostración de cada uno se pide aparte, en /api/demostracion.
    """
    datos = request.get_json(silent=True) or {}
    ids = datos.get("ids", [])

    if not isinstance(ids, list) or len(ids) < 2:
        return jsonify({"error": "Elige al menos dos artículos."}), 400
    desconocidos = [i for i in ids if i not in POR_ID]
    if desconocidos:
        return jsonify({"error": f"Artículos no encontrados: {desconocidos}"}), 400
    vistos = {}
    for identificador in ids:
        if identificador in vistos:
            return jsonify({"error": f"Artículo repetido: {identificador}"}), 400
        vistos[identificador] = True

    inicio = time.perf_counter()
    articulos = [_ficha(i) for i in ids]

    if len(ids) == 2:
        algoritmos = []
        for algoritmo in ALGORITMOS:
            algoritmos.append({
                **algoritmo,
                "similitud": _similitud_del_par(algoritmo["clave"], ids[0], ids[1]),
                "detalle": _detalle_del_par(algoritmo["clave"], ids[0], ids[1]),
            })
        return jsonify({
            "modo": "par",
            "ids": ids,
            "articulos": articulos,
            "algoritmos": algoritmos,
            "ms_total": (time.perf_counter() - inicio) * 1000,
        })

    # --- selección múltiple: una matriz por algoritmo ---------------------
    matrices = {}
    for algoritmo in ALGORITMOS:
        matrices[algoritmo["clave"]] = matriz_de_similitud(algoritmo["clave"], ids)

    return jsonify({
        "modo": "matriz",
        "ids": ids,
        "articulos": articulos,
        "algoritmos": ALGORITMOS,
        "matrices": matrices,
        "ms_total": (time.perf_counter() - inicio) * 1000,
    })


_series: dict[str, list[float]] = {}


def _serie_del_corpus(clave: str) -> list[float]:
    """El valor de los 190 pares con un algoritmo. Se calcula una vez por algoritmo."""
    if clave not in _series:
        ids = VECTORES["ids"]
        matriz = matriz_de_similitud(clave, ids)
        valores = []
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                valores.append(matriz[i][j])
        _series[clave] = valores
    return _series[clave]


def _lectura_en_el_corpus(clave: str, valor: float) -> dict:
    """Dónde queda el par frente a los 190 del corpus: da contexto al número."""
    valores = _serie_del_corpus(clave)
    puesto = 1
    for otro in valores:
        if otro > valor + 1e-9:
            puesto += 1
    return {
        "puesto": puesto,
        "total": len(valores),
        "minimo": min(valores),
        "maximo": max(valores),
        "valores": valores,
    }


@app.post("/api/demostracion")
def ver_demostracion():
    """
    Ejecuta un solo algoritmo y devuelve su demostración paso a paso.

    Recibe `algoritmo` y, o bien `ids` con dos artículos del corpus, o bien `textos`
    con dos textos cortos escritos por quien consulta.
    """
    datos = request.get_json(silent=True) or {}
    clave = datos.get("algoritmo")
    if clave not in NOMBRES:
        return jsonify({"error": "Algoritmo desconocido."}), 400

    if "textos" in datos:
        return _demostracion_de_textos(clave, datos["textos"])

    ids = datos.get("ids", [])
    if (not isinstance(ids, list) or len(ids) != 2 or ids[0] == ids[1]
            or ids[0] not in POR_ID or ids[1] not in POR_ID):
        return jsonify({"error": "La demostración compara dos artículos distintos."}), 400

    i, j = POSICION[ids[0]], POSICION[ids[1]]
    resultado = demostracion(
        clave, POR_ID[ids[0]]["texto"], POR_ID[ids[1]]["texto"], TEXTOS, IDF, LADO_CORPUS,
        (VECTORES["minilm"][i], VECTORES["minilm"][j]),
    )
    resultado["articulos"] = [_ficha(ids[0]), _ficha(ids[1])]
    resultado["corpus"] = _lectura_en_el_corpus(clave, resultado["similitud"])
    return jsonify(resultado)


def _demostracion_de_textos(clave: str, textos):
    if clave not in ACEPTAN_TEXTOS:
        return jsonify({"error": "Este algoritmo solo trabaja con los abstracts del corpus, "
                                 "porque sus vectores salen de un modelo."}), 400
    limite = 40 if clave in ("levenshtein", "needleman_wunsch") else 300
    if (not isinstance(textos, list) or len(textos) != 2
            or not isinstance(textos[0], str) or not isinstance(textos[1], str)
            or not textos[0] or not textos[1]):
        return jsonify({"error": "Escribe los dos textos."}), 400
    if len(textos[0]) > limite or len(textos[1]) > limite:
        return jsonify({"error": f"Cada texto puede tener hasta {limite} caracteres."}), 400

    resultado = demostracion(clave, textos[0], textos[1], TEXTOS, None, LADO_TEXTOS)
    resultado["textos"] = textos
    return jsonify(resultado)


@app.post("/api/buscar")
def buscar():
    """
    Busca un patrón en los 20 abstracts con los tres algoritmos.

    La coincidencia es exacta: el patrón y el texto se comparan tal cual, así
    que "AI" y "ai" son cadenas distintas.
    """
    datos = request.get_json(silent=True) or {}
    patron = datos.get("patron") or ""

    if not patron:
        return jsonify({"error": "Escribe un patrón para buscar."}), 400
    if len(patron) > 120:
        return jsonify({"error": "El patrón no puede superar 120 caracteres."}), 400

    corpus_unido = "\n".join(r["texto"] for r in CORPUS)

    algoritmos = []
    salidas = {}
    for nombre, modulo in (("KMP", kmp), ("Boyer-Moore", boyer_moore),
                           ("Rabin-Karp", rabin_karp)):
        posiciones, ms = _cronometrar(modulo.buscar, corpus_unido, patron)
        metricas = modulo.buscar_con_metricas(corpus_unido, patron)
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
        posiciones = kmp.buscar(texto, patron)
        if not posiciones:
            continue
        por_articulo.append({
            "id": registro["id"],
            "titulo": registro["titulo"],
            "ocurrencias": len(posiciones),
            "contextos": [
                _contexto(texto, p, len(patron)) for p in posiciones[:3]
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
        ruta = os.path.join(DIR_FUSIONES, archivo)
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


@app.post("/api/demostracion-agrupamiento")
def demostracion_agrupamiento():
    """La demostración de un enlace con una representación, ejecutada en el momento."""
    datos = request.get_json(silent=True) or {}
    clave, enlace = datos.get("representacion"), datos.get("enlace")
    if clave not in explicaciones_clustering.REPRESENTACIONES or enlace not in explicaciones_clustering.ENLACES:
        return jsonify({"error": "Representación o enlace desconocidos."}), 400
    return jsonify(explicaciones_clustering.demostracion(clave, enlace, VECTORES, TEXTOS, VECTORES["ids"]))


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
    # Puerto y host por variable de entorno. Un servicio de hosting asigna el
    # puerto al arrancar y lo pasa en PORT, así que fijarlo en 5000 haría que
    # la aplicación escuchara donde nadie la busca. Y HOST tiene que ser
    # 0.0.0.0 fuera de la máquina propia: 127.0.0.1 solo acepta conexiones del
    # mismo equipo, que es justo lo que no queremos al publicarla.
    puerto = int(os.environ.get("PORT", "5000"))
    host = os.environ.get("HOST", "127.0.0.1")

    print("Corpus cargado:", len(CORPUS), "abstracts")
    print(f"Abre http://{host}:{puerto}")
    app.run(host=host, port=puerto, debug=False)

"""
Números intermedios de los cuatro algoritmos clásicos sobre un par de textos.
-----------------------------------------------------------------------------
Hay una función por algoritmo para poder calcular solo el que se pide: las de
programación dinámica llenan matrices de más de un millón de celdas. De aquí salen
la demostración paso a paso del documento técnico (`caso_estudio.py`) y la de la
aplicación web (`app/explicaciones.py`).

Las matrices no caben en un documento ni en una respuesta web: se guardan la esquina
inicial y la final, de `lado` x `lado`, y las operaciones de un camino óptimo. Con
textos cortos la esquina inicial es la matriz completa.
"""
from src.similarity.classic.cosine import tfidf_cosine_similarity
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.classic.levenshtein import levenshtein_distance
from src.similarity.classic.needleman_wunsch import needleman_wunsch, normalized_needleman_wunsch
from src.similarity.classic.tokenizador import terminos_unicos, tokenizar


def operaciones_levenshtein(a: str, b: str, matriz: list[list[int]]) -> dict:
    """
    Recorre la matriz desde D(m, n) hacia atrás y cuenta las operaciones de un
    camino óptimo. Complejidad temporal: O(m + n). Espacial: O(1).
    """
    i, j = len(a), len(b)
    conteo = {"coincidencias": 0, "sustituciones": 0, "eliminaciones": 0, "inserciones": 0}
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and matriz[i][j] == matriz[i - 1][j - 1]:
            conteo["coincidencias"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and j > 0 and matriz[i][j] == matriz[i - 1][j - 1] + 1:
            conteo["sustituciones"] += 1
            i, j = i - 1, j - 1
        elif i > 0 and matriz[i][j] == matriz[i - 1][j] + 1:
            conteo["eliminaciones"] += 1
            i -= 1
        else:
            conteo["inserciones"] += 1
            j -= 1
    return conteo


def operaciones_needleman(a: str, b: str, matriz: list[list[int]], gap: int = -1,
                          match: int = 1, mismatch: int = -1) -> dict:
    """Igual que `operaciones_levenshtein`, para el alineamiento de puntaje máximo."""
    i, j = len(a), len(b)
    conteo = {"coincidencias": 0, "discrepancias": 0, "huecos": 0}
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            s = match if a[i - 1] == b[j - 1] else mismatch
            if matriz[i][j] == matriz[i - 1][j - 1] + s:
                conteo["coincidencias" if s == match else "discrepancias"] += 1
                i, j = i - 1, j - 1
                continue
        if i > 0 and matriz[i][j] == matriz[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
        conteo["huecos"] += 1
    return conteo


def _esquina(matriz: list[list[int]], filas: range, columnas: range) -> list[list[int]]:
    return [[matriz[i][j] for j in columnas] for i in filas]


def _esquinas(a: str, b: str, matriz: list[list[int]], lado: int) -> dict:
    """Las dos esquinas de la matriz y los caracteres de cada fila y columna."""
    m, n = len(a), len(b)
    filas, columnas = min(lado, m + 1), min(lado, n + 1)
    return {
        "inicio_a": a[:filas - 1],
        "inicio_b": b[:columnas - 1],
        "esquina_inicial": _esquina(matriz, range(filas), range(columnas)),
        "fin_a": a[m - filas + 1:],
        "fin_b": b[n - columnas + 1:],
        "esquina_final": _esquina(matriz, range(m - filas + 1, m + 1),
                                  range(n - columnas + 1, n + 1)),
    }


def traza_levenshtein(a: str, b: str, lado: int = 8) -> dict:
    distancia, matriz = levenshtein_distance(a, b)
    largo = len(a) if len(a) > len(b) else len(b)
    return {
        "distancia": distancia,
        "similitud": 1.0 - distancia / largo if largo else 1.0,
        **_esquinas(a, b, matriz, lado),
        "operaciones": operaciones_levenshtein(a, b, matriz),
    }


def traza_needleman(a: str, b: str, lado: int = 8) -> dict:
    puntaje, matriz = needleman_wunsch(a, b)
    return {
        "puntaje": puntaje,
        "similitud": normalized_needleman_wunsch(a, b),
        **_esquinas(a, b, matriz, lado),
        "operaciones": operaciones_needleman(a, b, matriz),
    }


def traza_jaccard(a: str, b: str) -> dict:
    similitud, interseccion, union = jaccard_similarity(a, b)
    terminos_a = terminos_unicos(tokenizar(a))
    terminos_b = terminos_unicos(tokenizar(b))
    comunes = {}
    for termino in interseccion:
        comunes[termino] = True
    return {
        "similitud": similitud,
        "tam_a": len(terminos_a),
        "tam_b": len(terminos_b),
        "tam_interseccion": len(interseccion),
        "tam_union": len(union),
        "interseccion": interseccion,
        "solo_a": [t for t in terminos_a if t not in comunes],
        "solo_b": [t for t in terminos_b if t not in comunes],
    }


def traza_tfidf(a: str, b: str, corpus: list[str], idf_corpus: dict | None = None,
                cuantos: int = 12) -> dict:
    tfidf = tfidf_cosine_similarity(a, b, corpus, idf_corpus)
    terminos = []
    for i in range(len(tfidf["vocabulario"])):
        terminos.append({
            "termino": tfidf["vocabulario"][i],
            "tf_a": tfidf["tf_a"][i],
            "tf_b": tfidf["tf_b"][i],
            "idf": tfidf["idf"][i],
            "peso_a": tfidf["pesos_a"][i],
            "peso_b": tfidf["pesos_b"][i],
            "producto": tfidf["pesos_a"][i] * tfidf["pesos_b"][i],
        })
    terminos.sort(key=lambda t: t["producto"], reverse=True)
    return {
        "n_documentos": len(corpus),
        "dimensiones": len(tfidf["vocabulario"]),
        "producto_punto": tfidf["producto_punto"],
        "norma_a": tfidf["norma_a"],
        "norma_b": tfidf["norma_b"],
        "coseno": tfidf["coseno"],
        "angulo_grados": tfidf["angulo_grados"],
        "terminos_que_mas_aportan": terminos[:cuantos],
        "terminos_con_idf_cero": [t["termino"] for t in terminos if t["idf"] == 0.0],
    }


def traza_clasicos(texto_a: str, texto_b: str, corpus: list[str], lado: int = 8) -> dict:
    """Los cuatro algoritmos sobre el par, con el tamaño de las matrices."""
    m, n = len(texto_a), len(texto_b)
    return {
        "m": m,
        "n": n,
        "celdas": (m + 1) * (n + 1),
        "levenshtein": traza_levenshtein(texto_a, texto_b, lado),
        "needleman_wunsch": traza_needleman(texto_a, texto_b, lado),
        "jaccard": traza_jaccard(texto_a, texto_b),
        "coseno_tfidf": traza_tfidf(texto_a, texto_b, corpus),
    }

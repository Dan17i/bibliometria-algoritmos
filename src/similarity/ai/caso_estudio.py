"""
Caso de estudio P1-P3: clásicos de Samuel vs. enfoques de IA (Req. 1).
----------------------------------------------------------------------
Compara un par de abstracts con:
    - los clásicos de Samuel que ya funcionan, importados SIN modificarlos:
      Levenshtein, Jaccard y coseno de frecuencias (TF);
    - los 2 enfoques de IA: promedio de vectores GloVe (con y sin stopwords) y
      embeddings de oración all-MiniLM-L6-v2.
Needleman-Wunsch y coseno TF-IDF siguen pendientes de Samuel y se marcan así.

    .venv\\Scripts\\python.exe -m src.similarity.ai.caso_estudio

Genera dos archivos en tests/fixtures/daniel/:
    - matrices_p1_p4.json: matrices de similitud coseno de los 4 abstracts con
      cada enfoque de IA (GloVe con stopwords, GloVe sin stopwords y MiniLM).
    - caso_estudio_p1_p3.json: la tabla comparativa del par elegido.
"""
import json
from pathlib import Path

from src.similarity.ai.demo_modelos import (
    FIXTURE,
    format_matrix,
    glove_embeddings,
    load_abstracts,
    sentence_embeddings,
)
from src.similarity.ai.stopwords import STOPWORDS
from src.similarity.classic.cosine import cosine_similarity as classic_cosine
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.classic.levenshtein import levenshtein_distance
from src.similarity.metrics import cosine_steps, euclidean_distance, pairwise_matrix

SALIDA = FIXTURE.parent
PENDIENTE = "pendiente (Samuel)"


def classic_scores(texto_a: str, texto_b: str) -> list[dict]:
    """
    Corre los clásicos de Samuel que funcionan sobre el par de textos.

    Levenshtein trabaja sobre caracteres y devuelve una distancia; para ponerla
    en [0, 1] se normaliza (normalización nuestra, no de Samuel):
        sim = 1 - d / max(|A|, |B|)
    Jaccard y el coseno TF ya devuelven una similitud en [0, 1].

    Complejidad temporal: O(|A|*|B|) por Levenshtein (domina); O(T) por Jaccard
    y por el coseno de frecuencias, salvo que `cosine.py` busca cada token con
    `list.index`, lo que lo vuelve O(T*V).
    Complejidad espacial: O(|A|*|B|) por la matriz de programación dinámica.
    """
    dist, _matriz = levenshtein_distance(texto_a, texto_b)
    largo = max(len(texto_a), len(texto_b))
    jac, inter, union = jaccard_similarity(texto_a, texto_b)
    cos, _va, _vb, vocab = classic_cosine(texto_a, texto_b)

    return [
        {
            "algoritmo": "Levenshtein (caracteres)",
            "tipo": "clásico, PD",
            "valor": f"distancia {dist} de {largo}",
            "similitud": 1.0 - dist / largo,
            "nota": "normalizada: 1 - d / max(|A|, |B|)",
        },
        {
            "algoritmo": "Jaccard (palabras)",
            "tipo": "clásico, conjuntos",
            "valor": f"|∩| = {len(inter)}, |∪| = {len(union)}",
            "similitud": jac,
            "nota": "sin stopwords removidas",
        },
        {
            "algoritmo": "Coseno TF (frecuencias)",
            "tipo": "clásico, espacio vectorial",
            "valor": f"{len(vocab)} dimensiones",
            "similitud": cos,
            "nota": "solo TF, sin IDF",
        },
        {
            "algoritmo": "Needleman-Wunsch",
            "tipo": "clásico, PD",
            "valor": PENDIENTE,
            "similitud": None,
            "nota": "needleman_wunsch.py está vacío",
        },
        {
            "algoritmo": "Coseno TF-IDF",
            "tipo": "clásico, espacio vectorial",
            "valor": PENDIENTE,
            "similitud": None,
            "nota": "falta el IDF en cosine.py",
        },
    ]


def ai_scores(glove_con: tuple, glove_sin: tuple, minilm: tuple) -> list[dict]:
    """
    Filas de los enfoques de IA para un par. Cada argumento es la pareja
    `(vector_a, vector_b)` con los embeddings de los dos abstracts según ese
    enfoque (GloVe con stopwords, GloVe sin stopwords, MiniLM).

    Cada fila lleva el coseno (con los pasos de `metrics.cosine_steps`), el
    ángulo y la distancia euclidiana, todo calculado a mano.
    Complejidad temporal: O(n) por enfoque (n = dimensión). Espacial: O(1).
    """
    filas = []
    for nombre, tipo, (va, vb), nota in [
        ("GloVe-100 promedio", "IA, vectores de palabras", glove_con, "con stopwords"),
        ("GloVe-100 promedio", "IA, vectores de palabras", glove_sin, "sin stopwords"),
        ("all-MiniLM-L6-v2", "IA, embedding de oración", minilm, "384 dimensiones"),
    ]:
        pasos = cosine_steps(va, vb)
        filas.append(
            {
                "algoritmo": nombre,
                "tipo": tipo,
                "valor": f"ángulo {pasos['angle_degrees']:.1f}°, euclidiana {euclidean_distance(va, vb):.3f}",
                "similitud": pasos["cosine"],
                "nota": nota,
            }
        )
    return filas


def format_table(filas: list[dict]) -> str:
    """Tabla en markdown con las filas de `classic_scores` + `ai_scores`."""
    lineas = [
        "| Algoritmo | Tipo | Resultado | Similitud | Nota |",
        "|---|---|---|---|---|",
    ]
    for f in filas:
        sim = "—" if f["similitud"] is None else f"{f['similitud']:.4f}"
        lineas.append(f"| {f['algoritmo']} | {f['tipo']} | {f['valor']} | {sim} | {f['nota']} |")
    return "\n".join(lineas)


def main(par: tuple[str, str] = ("P1", "P3")) -> None:
    ids, textos = load_abstracts()
    glove_con = glove_embeddings(textos)["vectors"]
    glove_sin = glove_embeddings(textos, stopwords=STOPWORDS)["vectors"]
    minilm = sentence_embeddings(textos)

    # 1. matrices de los 4 abstracts, con y sin stopwords
    matrices = {
        "ids": ids,
        "glove_con_stopwords": pairwise_matrix(glove_con, lambda a, b: cosine_steps(a, b)["cosine"]),
        "glove_sin_stopwords": pairwise_matrix(glove_sin, lambda a, b: cosine_steps(a, b)["cosine"]),
        "minilm": pairwise_matrix(minilm, lambda a, b: cosine_steps(a, b)["cosine"]),
    }
    (SALIDA / "matrices_p1_p4.json").write_text(
        json.dumps(matrices, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    for clave, titulo in [
        ("glove_con_stopwords", "GloVe-100 promedio, CON stopwords"),
        ("glove_sin_stopwords", "GloVe-100 promedio, SIN stopwords"),
        ("minilm", "all-MiniLM-L6-v2"),
    ]:
        print(f"\n{titulo}\n{format_matrix(ids, matrices[clave])}")

    # 2. caso de estudio: un par
    i, j = ids.index(par[0]), ids.index(par[1])
    filas = classic_scores(textos[i], textos[j]) + ai_scores(
        (glove_con[i], glove_con[j]), (glove_sin[i], glove_sin[j]), (minilm[i], minilm[j])
    )
    (SALIDA / f"caso_estudio_{par[0].lower()}_{par[1].lower()}.json").write_text(
        json.dumps({"par": list(par), "filas": filas}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nCaso de estudio {par[0]} vs {par[1]}\n{format_table(filas)}")


if __name__ == "__main__":
    main()

"""
Caso de estudio P6-P9: clásicos de Samuel vs. enfoques de IA (Req. 1).
----------------------------------------------------------------------
P6-P9 es el par más parecido del corpus según MiniLM. Compara un par de abstracts con:
    - los 4 clásicos de Samuel (más el coseno TF), importados SIN modificarlos:
      Levenshtein, Needleman-Wunsch, Jaccard, coseno TF y coseno TF-IDF;
    - los 2 enfoques de IA: promedio de vectores GloVe (con stopwords, solo como
      referencia, y sobre el abstract transformado) y embeddings de oración
      all-MiniLM-L6-v2 sobre el abstract transformado.
El IDF del coseno TF-IDF se calcula sobre los 20 abstracts de
data/processed/abstracts.json (con solo 2 textos toda palabra compartida tendría
idf = 0).

    .venv\\Scripts\\python.exe -m src.similarity.ai.caso_estudio

Genera dos archivos en data/resultados/:
    - caso_estudio_p6_p9.json: la tabla comparativa del par elegido.
    - traza_req1_p6_p9.json: los números intermedios de cada algoritmo, para la
      demostración paso a paso del documento técnico.
"""
import json

from src.preprocessing.transformacion import transformar_abstract
from src.similarity.ai.demo_modelos import (
    RESULTADOS as SALIDA,
    glove_embeddings,
    load_abstracts,
    sentence_embeddings,
    vectores_del_corpus,
)
from src.similarity.ai.demostracion import average_steps, compare_vectors_steps
from src.similarity.ai.embeddings import load_gensim_vectors, tokenize
from src.similarity.ai.stopwords import STOPWORDS
from src.similarity.classic.cosine import cosine_similarity as classic_cosine
from src.similarity.classic.cosine import tfidf_cosine_similarity
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.classic.levenshtein import levenshtein_distance
from src.similarity.classic.needleman_wunsch import needleman_wunsch, normalized_needleman_wunsch
from src.similarity.traza_clasicos import operaciones_levenshtein, operaciones_needleman, traza_clasicos  # noqa: F401
from src.similarity.metrics import cosine_steps, euclidean_distance


def classic_scores(texto_a: str, texto_b: str, corpus: list[str] | None = None) -> list[dict]:
    """
    Corre los clásicos de Samuel sobre el par de textos.

    `corpus` es el conjunto de documentos del que sale el IDF del coseno TF-IDF
    (los 20 abstracts). Si se omite se usa solo el par, lo que anula el IDF de
    toda palabra compartida: sirve para pruebas pequeñas, no para resultados.

    Levenshtein trabaja sobre caracteres y devuelve una distancia; para ponerla
    en [0, 1] se normaliza (normalización nuestra, no de Samuel):
        sim = 1 - d / max(|A|, |B|)
    Needleman-Wunsch se toma de `normalized_needleman_wunsch` (pesos +1/-1/-1).
    Jaccard y los cosenos ya devuelven una similitud en [0, 1].

    Complejidad temporal: O(|A|*|B|) por Levenshtein y por Needleman-Wunsch
    (dominan); O(T) por Jaccard; O(T*V) por el coseno TF, que busca cada token
    con `list.index`; O(T_corpus + V) por el TF-IDF (T_corpus = tokens del corpus).
    Complejidad espacial: O(|A|*|B|) por las matrices de programación dinámica.
    """
    dist, _matriz = levenshtein_distance(texto_a, texto_b)
    largo = max(len(texto_a), len(texto_b))
    jac, inter, union = jaccard_similarity(texto_a, texto_b)
    cos, _va, _vb, vocab = classic_cosine(texto_a, texto_b)
    nw_puntaje, _matriz_nw = needleman_wunsch(texto_a, texto_b)
    tfidf = tfidf_cosine_similarity(texto_a, texto_b, corpus if corpus is not None else [texto_a, texto_b])

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
            "valor": f"puntaje {nw_puntaje} (L = {largo})",
            "similitud": normalized_needleman_wunsch(texto_a, texto_b),
            "nota": "normalizada: (s - L*gap) / (L*match - L*gap), pesos +1/-1/-1",
        },
        {
            "algoritmo": "Coseno TF-IDF",
            "tipo": "clásico, espacio vectorial",
            "valor": f"{len(tfidf['vocabulario'])} dimensiones, ángulo {tfidf['angulo_grados']:.1f}°",
            "similitud": tfidf["coseno"],
            "nota": "IDF = ln(N/df) sobre el corpus",
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


def traza_ia(texto_a: str, texto_b: str, dims: int = 6) -> dict:
    """Todos los números intermedios de los dos enfoques de IA sobre el par."""
    transformado_a = transformar_abstract(texto_a)
    transformado_b = transformar_abstract(texto_b)

    vocabulario = {}
    for token in tokenize(transformado_a) + tokenize(transformado_b):
        vocabulario[token] = True
    palabras = load_gensim_vectors("glove-wiki-gigaword-100", list(vocabulario))

    promedio_a = average_steps(transformado_a, palabras, STOPWORDS, dims)
    promedio_b = average_steps(transformado_b, palabras, STOPWORDS, dims)
    minilm = sentence_embeddings([transformado_a, transformado_b])

    return {
        "transformado_a": transformado_a,
        "transformado_b": transformado_b,
        "glove": {
            "k_a": promedio_a["k"],
            "k_b": promedio_b["k"],
            "fuera_de_vocabulario_a": promedio_a["fuera_de_vocabulario"],
            "fuera_de_vocabulario_b": promedio_b["fuera_de_vocabulario"],
            "suma_a": promedio_a["suma"],
            "suma_b": promedio_b["suma"],
            "promedio_a": promedio_a["promedio"],
            "promedio_b": promedio_b["promedio"],
            "comparacion": compare_vectors_steps(promedio_a["vector"], promedio_b["vector"], dims),
        },
        "minilm": {
            "comparacion": compare_vectors_steps(minilm[0], minilm[1], dims),
        },
    }


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


def main(par: tuple[str, str] = ("P6", "P9")) -> None:
    ids, textos = load_abstracts(cuantos=20)
    i, j = ids.index(par[0]), ids.index(par[1])
    glove_con = glove_embeddings([textos[i], textos[j]])["vectors"]
    vectores = vectores_del_corpus(textos)
    glove_sin = vectores["glove"]
    minilm = vectores["minilm"]

    SALIDA.mkdir(parents=True, exist_ok=True)
    filas = classic_scores(textos[i], textos[j], textos) + ai_scores(
        (glove_con[0], glove_con[1]), (glove_sin[i], glove_sin[j]), (minilm[i], minilm[j])
    )
    nombre = f"{par[0].lower()}_{par[1].lower()}"
    (SALIDA / f"caso_estudio_{nombre}.json").write_text(
        json.dumps({"par": list(par), "filas": filas}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nCaso de estudio {par[0]} vs {par[1]}\n{format_table(filas)}")

    traza = {
        "par": list(par),
        "texto_a": textos[i],
        "texto_b": textos[j],
        "clasicos": traza_clasicos(textos[i], textos[j], textos),
        "ia": traza_ia(textos[i], textos[j]),
    }
    (SALIDA / f"traza_req1_{nombre}.json").write_text(
        json.dumps(traza, indent=1, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

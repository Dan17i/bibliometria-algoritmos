"""
Clásicos de Samuel vs. enfoques de IA sobre los 190 pares de los 20 abstracts.
------------------------------------------------------------------------------
Cierra el pendiente del análisis crítico: ¿ordenan igual los pares los
algoritmos clásicos y los embeddings? Se mide con el tau de Kendall.

    .venv\\Scripts\\python.exe -m src.similarity.ai.demo_clasicos20

Los clásicos se importan de `caso_estudio.classic_scores` (que a su vez usa los
módulos de Samuel). Las dos representaciones de IA son GloVe y MiniLM sobre el
abstract transformado, con el coseno calculado a mano (`metrics.py`).

Salida: data/resultados/clasicos_resumen.json
    - similitudes de cada par (i < j) por algoritmo,
    - rango (máx - mín) de cada algoritmo,
    - matriz de tau de Kendall entre todos los algoritmos.
"""
import json
import time
from pathlib import Path

from src.similarity.ai.caso_estudio import classic_scores
from src.similarity.ai.demo_corpus20 import SALIDA, load_corpus
from src.similarity.ai.demo_flujo import kendall_tau, value_range
from src.similarity.ai.demo_modelos import vectores_del_corpus
from src.similarity.metrics import cosine_similarity

CLASICOS = {
    "Levenshtein (caracteres)": "levenshtein",
    "Needleman-Wunsch": "needleman_wunsch",
    "Jaccard (palabras)": "jaccard",
    "Coseno TF (frecuencias)": "coseno_tf",
    "Coseno TF-IDF": "coseno_tfidf",
}


def pares(n: int) -> list[tuple[int, int]]:
    """Pares i < j en orden de fila: n(n-1)/2 elementos."""
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def similitudes_clasicas(textos: list[str], corpus: list[str]) -> dict[str, list[float]]:
    """
    Una serie por algoritmo clásico, con un valor por cada par i < j.

    Complejidad temporal: O(n^2 * L^2) con n documentos y L caracteres por
    documento, dominada por las dos programaciones dinámicas (Levenshtein y
    Needleman-Wunsch, O(L^2) por par). Espacial: O(L^2) por la matriz de un par.
    """
    series = {clave: [] for clave in CLASICOS.values()}
    for i, j in pares(len(textos)):
        for fila in classic_scores(textos[i], textos[j], corpus):
            if fila["algoritmo"] in CLASICOS:
                series[CLASICOS[fila["algoritmo"]]].append(fila["similitud"])
    return series


def similitudes_ia(textos: list[str]) -> dict[str, list[float]]:
    """Series de GloVe y MiniLM: coseno a mano por cada par i < j."""
    vectores = vectores_del_corpus(textos)
    return {
        nombre: [cosine_similarity(vecs[i], vecs[j]) for i, j in pares(len(vecs))]
        for nombre, vecs in vectores.items()
    }


def matriz_tau(series: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    """
    Tau de Kendall entre cada par de series (1 = mismo orden de los pares,
    -1 = orden inverso). Complejidad O(k^2 * m^2) con k series y m pares.
    """
    return {a: {b: kendall_tau(series[a], series[b]) for b in series} for a in series}


def main() -> None:
    ids, textos = load_corpus()
    t0 = time.perf_counter()
    series = similitudes_clasicas(textos, textos)
    print(f"clásicos: {len(pares(len(ids)))} pares en {time.perf_counter() - t0:.1f} s")
    series.update(similitudes_ia(textos))

    resumen = {
        "n": len(ids),
        "ids": ids,
        "pares": [[ids[i], ids[j]] for i, j in pares(len(ids))],
        "similitudes": series,
        "rango": {k: value_range(v) for k, v in series.items()},
        "tau": matriz_tau(series),
    }
    SALIDA.mkdir(parents=True, exist_ok=True)
    ruta = SALIDA / "clasicos_resumen.json"
    ruta.write_text(json.dumps(resumen, indent=2), encoding="utf-8")

    print(f"\n{'algoritmo':<18}{'rango':>8}{'tau vs minilm':>16}{'tau vs glove':>14}")
    for k in series:
        print(f"{k:<18}{resumen['rango'][k]:>8.4f}{resumen['tau'][k]['minilm']:>16.3f}{resumen['tau'][k]['glove']:>14.3f}")
    print(f"\nescrito: {ruta}")


if __name__ == "__main__":
    main()

"""
Datos precalculados para la aplicación web.
-------------------------------------------
El corpus es estático, así que hay dos cálculos que se pueden hacer una sola vez:

    - Los embeddings. Los modelos ocupan más memoria de la que tiene el servidor
      publicado; pero solo hacen falta para obtener los vectores, y comparar dos
      vectores es un recorrido a mano en `metrics.py`.
    - Levenshtein y Needleman-Wunsch sobre los 190 pares. Cada par llena dos
      matrices de cerca de 2 millones de celdas, demasiado para una petición web
      cuando se eligen muchos artículos.

    .venv\\Scripts\\python.exe -m src.similarity.ai.precalculo

Salidas (en data/processed/):
    - vectores.json      : un vector por abstract con GloVe (sin stopwords) y MiniLM.
    - similitudes_pd.json: matrices n x n de similitud normalizada de los dos
                           algoritmos de programación dinámica.
"""
import json
import time
from pathlib import Path

from src.similarity.ai.demo_modelos import glove_embeddings, sentence_embeddings
from src.similarity.ai.stopwords import STOPWORDS
from src.similarity.classic.levenshtein import levenshtein_distance
from src.similarity.classic.needleman_wunsch import normalized_needleman_wunsch

RAIZ = Path(__file__).resolve().parents[3]
DATOS = RAIZ / "data" / "processed"


def similitud_levenshtein(texto_a: str, texto_b: str) -> float:
    """Normalizada a [0, 1]: sim = 1 - d / max(|A|, |B|)."""
    distancia, _ = levenshtein_distance(texto_a, texto_b)
    largo = len(texto_a)
    if len(texto_b) > largo:
        largo = len(texto_b)
    return 1.0 - distancia / largo


def matrices_programacion_dinamica(textos: list[str]) -> dict[str, list[list[float]]]:
    """
    Similitud de cada par i < j con los dos algoritmos, copiada al triángulo inferior.

    Complejidad temporal: O(n^2 * L^2) con n documentos y L caracteres por documento.
    Complejidad espacial: O(n^2) por las matrices de resultados.
    """
    n = len(textos)
    levenshtein = [[1.0] * n for _ in range(n)]
    needleman = [[1.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            levenshtein[i][j] = levenshtein[j][i] = similitud_levenshtein(textos[i], textos[j])
            needleman[i][j] = needleman[j][i] = normalized_needleman_wunsch(textos[i], textos[j])
        print(f"  fila {i + 1} de {n}", flush=True)
    return {"levenshtein": levenshtein, "needleman_wunsch": needleman}


def main() -> None:
    registros = json.loads((DATOS / "abstracts.json").read_text(encoding="utf-8"))
    ids = [r["id"] for r in registros]
    textos = [r["texto"] for r in registros]

    vectores = {
        "ids": ids,
        "glove": glove_embeddings(textos, stopwords=STOPWORDS)["vectors"],
        "minilm": sentence_embeddings(textos),
    }
    (DATOS / "vectores.json").write_text(json.dumps(vectores), encoding="utf-8")
    print(f"vectores: {len(vectores['glove'][0])} y {len(vectores['minilm'][0])} dimensiones", flush=True)

    inicio = time.perf_counter()
    matrices = matrices_programacion_dinamica(textos)
    (DATOS / "similitudes_pd.json").write_text(
        json.dumps({"ids": ids, **matrices}), encoding="utf-8"
    )
    print(f"programación dinámica: {time.perf_counter() - inicio:.0f} s")


if __name__ == "__main__":
    main()

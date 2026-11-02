"""
Demo con modelos preentrenados reales (descarga modelos la primera vez).

    .venv\\Scripts\\python.exe -m src.similarity.ai.demo_modelos

Requiere `pip install -r requirements-ai.txt`. Calcula, para los 4 primeros
abstracts del corpus, la matriz de similitud coseno con los dos enfoques:
    1. promedio de vectores GloVe (gensim: glove-wiki-gigaword-100)
    2. embeddings de oración all-MiniLM-L6-v2 (sentence-transformers)
Los modelos solo aportan vectores; el coseno sale de `src.similarity.metrics`.
"""
import json
from pathlib import Path

from src.preprocessing.transformacion import transformar_abstract
from src.similarity.ai.embeddings import (
    average_word_vectors,
    encode_documents,
    load_gensim_vectors,
    load_sentence_encoder,
    tokenize,
)
from src.similarity.ai.stopwords import STOPWORDS
from src.similarity.metrics import cosine_similarity, pairwise_matrix

RAIZ = Path(__file__).resolve().parents[3]
ABSTRACTS = RAIZ / "data" / "processed" / "abstracts.json"
RESULTADOS = RAIZ / "data" / "resultados"


def load_abstracts(path: Path = ABSTRACTS, cuantos: int = 4) -> tuple[list[str], list[str]]:
    """Devuelve (ids, textos) de los primeros abstracts del corpus."""
    registros = json.loads(path.read_text(encoding="utf-8"))[:cuantos]
    return [r["id"] for r in registros], [r["texto"] for r in registros]


def glove_embeddings(
    textos: list[str],
    model_name: str = "glove-wiki-gigaword-100",
    stopwords=None,
) -> dict:
    """
    Enfoque 1: promedio de vectores de palabras. Devuelve vectores y cobertura.
    Con `stopwords` se eliminan esos tokens antes de promediar.
    """
    vocabulario = set()
    for t in textos:
        vocabulario.update(tokenize(t))
    vectores = load_gensim_vectors(model_name, vocabulario)
    pasos = [average_word_vectors(t, vectores, stopwords) for t in textos]
    return {
        "vectors": [p["vector"] for p in pasos],
        "missing": [p["missing_tokens"] for p in pasos],
        "n_tokens": [len(p["tokens"]) for p in pasos],
    }


def sentence_embeddings(
    textos: list[str], model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
) -> list[list[float]]:
    """Enfoque 2: un vector por abstract con un modelo de oraciones."""
    return encode_documents(textos, load_sentence_encoder(model_name))


def vectores_del_corpus(textos: list[str]) -> dict[str, list[list[float]]]:
    """
    Los vectores oficiales de los dos enfoques: cada abstract se transforma y
    luego se vectoriza. Costo dominado por el modelo.
    """
    transformados = [transformar_abstract(t) for t in textos]
    return {
        "glove": glove_embeddings(transformados, stopwords=STOPWORDS)["vectors"],
        "minilm": sentence_embeddings(transformados),
    }


def format_matrix(ids: list[str], matrix: list[list[float]]) -> str:
    """Tabla de texto con la matriz (4 decimales)."""
    lineas = ["      " + "".join(f"{i:>8}" for i in ids)]
    for i, fila in zip(ids, matrix):
        lineas.append(f"{i:>5} " + "".join(f"{v:8.4f}" for v in fila))
    return "\n".join(lineas)


def main() -> None:
    ids, textos = load_abstracts()
    transformados = [transformar_abstract(t) for t in textos]

    glove = glove_embeddings(transformados, stopwords=STOPWORDS)
    print("Enfoque 1: promedio de vectores GloVe (glove-wiki-gigaword-100)")
    for i, miss, n in zip(ids, glove["missing"], glove["n_tokens"]):
        print(f"  {i}: {n} tokens, {len(miss)} fuera de vocabulario {sorted(set(miss))}")
    print(format_matrix(ids, pairwise_matrix(glove["vectors"], cosine_similarity)))

    sent = sentence_embeddings(transformados)
    print("\nEnfoque 2: all-MiniLM-L6-v2 (dimensión %d)" % len(sent[0]))
    print(format_matrix(ids, pairwise_matrix(sent, cosine_similarity)))


if __name__ == "__main__":
    main()

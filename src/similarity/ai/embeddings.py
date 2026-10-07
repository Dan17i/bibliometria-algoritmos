"""
Enfoques de similitud con IA: obtención de embeddings (Req. 1, Daniel).
----------------------------------------------------------------------
Los modelos preentrenados se usan SOLO para obtener vectores. Todo lo demás
(promedio, producto punto, norma, coseno, euclidiana) se calcula a mano con
`src.similarity.metrics`.

Dos enfoques:
    1. Promedio de vectores de palabras (Word2Vec / FastText):
       cada abstract se tokeniza y su embedding es el promedio de los vectores
       de sus palabras (`average_word_vectors`).
    2. Embeddings de oración/documento (sentence-transformers):
       el modelo devuelve directamente un vector por abstract
       (`encode_documents`).

Las funciones puras reciben el modelo "ya reducido" a un diccionario de
vectores (enfoque 1) o a una función `encode` (enfoque 2). Así se pueden probar
sin descargar modelos. Los cargadores `load_*` importan las librerías de forma
perezosa y convierten todo a `list[float]` con `.tolist()`.
"""
import re
from typing import Callable, Mapping

from src.similarity.metrics import mean_vector

_TOKEN_RE = re.compile(r"[a-záéíóúüñ0-9]+(?:[-'][a-záéíóúüñ0-9]+)*")


def tokenize(text: str) -> list[str]:
    """
    Pasa a minúsculas y extrae palabras (conserva guiones y apóstrofes internos,
    p. ej. "state-of-the-art"). No usa stopwords: los modelos de embeddings ya
    ponderan implícitamente las palabras comunes.

    Complejidad temporal: O(L) con L la longitud del texto en caracteres.
    Complejidad espacial: O(T) con T el número de tokens.
    """
    return _TOKEN_RE.findall(text.lower())


def average_word_vectors(
    text: str, word_vectors: Mapping[str, list[float]]
) -> dict:
    """
    Enfoque 1: embedding del documento = promedio de los vectores de sus palabras.

    Fórmula: d = (1/k) * sum_{w in T'} v_w, donde T' son los tokens del texto
    presentes en el vocabulario del modelo (k = |T'|, contando repeticiones).

    Los tokens fuera del vocabulario se descartan y se reportan en
    `missing_tokens`. Si un token compuesto ("problem-solving", "AI's") no está,
    se prueban sus partes separadas por guion o apóstrofe y se usan las que sí
    estén. Si ningún token está en el vocabulario, lanza ValueError.

    Returns:
        dict con: tokens, used_tokens, missing_tokens, token_vectors, vector.

    Complejidad temporal: O(T + k*n): T tokens consultados en el diccionario
    (O(1) cada uno) más el promedio de k vectores de dimensión n.
    Complejidad espacial: O(k*n) por los vectores de palabras que se conservan
    para mostrar los pasos; el resultado es O(n).
    """
    tokens = tokenize(text)
    used = []
    missing = []
    token_vectors = []
    for tok in tokens:
        vec = word_vectors.get(tok)
        if vec is not None:
            used.append(tok)
            token_vectors.append(list(vec))
            continue
        # palabra compuesta ("problem-solving"): se prueban sus partes
        parts = [p for p in re.split(r"[-']", tok) if p] if ("-" in tok or "'" in tok) else []
        found = [p for p in parts if p in word_vectors]
        if found:
            for p in found:
                used.append(p)
                token_vectors.append(list(word_vectors[p]))
        else:
            missing.append(tok)

    if not token_vectors:
        raise ValueError("Ningún token del texto está en el vocabulario del modelo")

    return {
        "tokens": tokens,
        "used_tokens": used,
        "missing_tokens": missing,
        "token_vectors": token_vectors,
        "vector": mean_vector(token_vectors),
    }


def encode_documents(
    texts: list[str], encode: Callable[[list[str]], list[list[float]]]
) -> list[list[float]]:
    """
    Enfoque 2: un vector por documento producido por un modelo de oraciones.

    `encode` recibe la lista de textos y devuelve una lista de vectores
    (`list[list[float]]`). Se valida que haya un vector por texto y que todos
    tengan la misma dimensión.

    Complejidad temporal: O(m*n) para validar m vectores de dimensión n (el
    costo de inferir el modelo es externo y no forma parte del algoritmo).
    Complejidad espacial: O(m*n) por los embeddings.
    """
    vectors = encode(texts)
    if len(vectors) != len(texts):
        raise ValueError(
            f"El modelo devolvió {len(vectors)} vectores para {len(texts)} textos"
        )
    if vectors:
        n = len(vectors[0])
        for v in vectors:
            if len(v) != n:
                raise ValueError("El modelo devolvió vectores de dimensión distinta")
    return [list(v) for v in vectors]


def load_word_vectors(path: str, binary: bool = False) -> dict[str, list[float]]:
    """
    Carga vectores Word2Vec/FastText (formato word2vec) a un dict {palabra: vector}.

    gensim se usa únicamente para leer el archivo; nunca se llama a
    `.similarity()` ni a `.wmdistance()`. Ojo: con modelos grandes esto carga
    todo el vocabulario en listas de Python y consume mucha memoria.
    """
    from gensim.models import KeyedVectors  # import perezoso

    kv = KeyedVectors.load_word2vec_format(path, binary=binary)
    return {word: kv[word].tolist() for word in kv.index_to_key}


def load_gensim_vectors(
    model_name: str = "glove-wiki-gigaword-100", vocabulary: set[str] | None = None
) -> dict[str, list[float]]:
    """
    Descarga (con `gensim.downloader`) un modelo de vectores de palabras y lo
    devuelve como {palabra: vector}. Si se pasa `vocabulary`, solo se convierten
    esas palabras (y las partes de las compuestas), para no volcar cientos de
    miles de vectores a listas de Python.

    Solo se obtienen vectores (`kv[word].tolist()`); nunca se usa `.similarity()`.
    Complejidad: O(|vocabulary| * n), con n la dimensión de los vectores.
    """
    import gensim.downloader as api  # import perezoso

    kv = api.load(model_name)
    if vocabulary is None:
        return {w: kv[w].tolist() for w in kv.index_to_key}

    needed = set()
    for w in vocabulary:
        needed.add(w)
        for part in re.split(r"[-']", w):
            if part:
                needed.add(part)
    return {w: kv[w].tolist() for w in needed if w in kv.key_to_index}


def load_sentence_encoder(
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> Callable[[list[str]], list[list[float]]]:
    """
    Devuelve una función `encode(textos) -> list[list[float]]` basada en
    sentence-transformers. Solo se usa `model.encode`; la similitud la calcula
    `src.similarity.metrics` (nunca `util.cos_sim`).
    """
    from sentence_transformers import SentenceTransformer  # import perezoso

    model = SentenceTransformer(model_name)

    def encode(texts: list[str]) -> list[list[float]]:
        return model.encode(texts).tolist()

    return encode

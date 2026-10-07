"""
Pruebas con modelos preentrenados REALES (descargan cientos de MB).

NO corren por defecto. Para ejecutarlas:

    set RUN_MODEL_TESTS=1            (PowerShell: $env:RUN_MODEL_TESTS = "1")
    .venv\\Scripts\\python.exe -m pytest tests/test_ai_modelos.py -v

Requieren `pip install -r requirements-ai.txt`.
"""
import os

import pytest

from src.similarity.ai.demo_modelos import (
    glove_embeddings,
    load_abstracts,
    sentence_embeddings,
)
from src.similarity.metrics import cosine_similarity, pairwise_matrix

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_MODEL_TESTS") != "1",
    reason="descarga modelos; activar con RUN_MODEL_TESTS=1",
)


@pytest.fixture(scope="module")
def corpus():
    return load_abstracts()


def _comprobar_matriz(vectores, n):
    m = pairwise_matrix(vectores, cosine_similarity)
    assert len(m) == n
    for i in range(n):
        assert m[i][i] == pytest.approx(1.0)
        for j in range(n):
            assert m[i][j] == pytest.approx(m[j][i])
            assert -1.0 <= m[i][j] <= 1.0
    return m


def test_glove_promedio_de_palabras(corpus):
    ids, textos = corpus
    res = glove_embeddings(textos)
    assert len(res["vectors"]) == len(ids)
    assert all(len(v) == 100 for v in res["vectors"])
    # la cobertura del vocabulario debe ser alta en textos en inglés
    for faltantes, total in zip(res["missing"], res["n_tokens"]):
        assert len(faltantes) / total < 0.1
    m = _comprobar_matriz(res["vectors"], len(ids))
    # abstracts del mismo dominio: GloVe promediado da similitudes altas
    assert all(m[i][j] > 0.8 for i in range(len(ids)) for j in range(len(ids)))


def test_sentence_transformer_minilm(corpus):
    ids, textos = corpus
    vectores = sentence_embeddings(textos)
    assert all(len(v) == 384 for v in vectores)
    m = _comprobar_matriz(vectores, len(ids))
    assert all(m[i][j] > 0.0 for i in range(len(ids)) for j in range(len(ids)))

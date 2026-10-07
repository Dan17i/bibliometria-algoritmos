"""Pruebas de src/similarity/ai (con modelos simulados, sin descargas)."""
import pytest

from src.similarity.ai.compare import compare_embeddings
from src.similarity.ai.embeddings import (
    average_word_vectors,
    encode_documents,
    tokenize,
)
from src.similarity.metrics import (
    cosine_distance,
    euclidean_distance,
    pairwise_matrix,
)

VECTORES = {
    "gato": [1.0, 0.0],
    "perro": [0.0, 1.0],
    "casa": [1.0, 1.0],
}


def test_tokenize():
    assert tokenize("Hola, Mundo! State-of-the-art 2024") == [
        "hola", "mundo", "state-of-the-art", "2024",
    ]
    assert tokenize("") == []


def test_promedio_a_mano():
    # (gato + perro) / 2 = (0.5, 0.5)
    res = average_word_vectors("Gato perro", VECTORES)
    assert res["vector"] == [0.5, 0.5]
    assert res["used_tokens"] == ["gato", "perro"]
    assert res["missing_tokens"] == []


def test_promedio_cuenta_repeticiones_y_reporta_faltantes():
    # gato, gato, casa -> ((1+1+1)/3, (0+0+1)/3) = (1, 1/3); 'xyz' no está
    res = average_word_vectors("gato gato casa xyz", VECTORES)
    assert res["vector"] == pytest.approx([1.0, 1 / 3])
    assert res["missing_tokens"] == ["xyz"]


def test_promedio_sin_vocabulario_lanza_error():
    with pytest.raises(ValueError):
        average_word_vectors("desconocida", VECTORES)
    with pytest.raises(ValueError):
        average_word_vectors("", VECTORES)


def test_encode_documents_valida_salida():
    fake = lambda textos: [[float(len(t)), 1.0] for t in textos]
    assert encode_documents(["ab", "abc"], fake) == [[2.0, 1.0], [3.0, 1.0]]
    with pytest.raises(ValueError):
        encode_documents(["a", "b"], lambda t: [[1.0]])
    with pytest.raises(ValueError):
        encode_documents(["a", "b"], lambda t: [[1.0], [1.0, 2.0]])


def test_compare_embeddings_pares():
    res = compare_embeddings(["A", "B", "C"], [[1, 0], [0, 1], [1, 0]])
    assert [(r["a"], r["b"]) for r in res] == [("A", "B"), ("A", "C"), ("B", "C")]
    assert res[0]["cosine"] == 0.0 and res[0]["angle_degrees"] == pytest.approx(90.0)
    assert res[1]["cosine"] == pytest.approx(1.0) and res[1]["euclidean"] == 0.0
    assert res[0]["euclidean"] == pytest.approx(2 ** 0.5)


def test_compare_embeddings_errores():
    with pytest.raises(ValueError):
        compare_embeddings(["A"], [[1, 0]])
    with pytest.raises(ValueError):
        compare_embeddings(["A", "B"], [[1, 0]])


def test_pairwise_matrix():
    vecs = [[1, 0], [0, 1], [1, 0]]
    m = pairwise_matrix(vecs, cosine_distance)
    assert m == [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    e = pairwise_matrix([[0, 0], [3, 4]], euclidean_distance)
    assert e == [[0.0, 5.0], [5.0, 0.0]]

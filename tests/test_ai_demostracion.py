"""Pruebas sin descargas: demostración paso a paso, centrado y utilidades del demo de punta a punta."""
import pytest

from src.similarity.ai.demo_flujo import kendall_tau, upper_triangle, value_range
from src.similarity.ai.demostracion import (
    average_steps,
    compare_vectors_steps,
    demostracion_markdown,
)
from src.similarity.ai.stopwords import STOPWORDS
from src.similarity.metrics import center_vectors, cosine_similarity, pairwise_matrix

VEC = {"cat": [1.0, 0.0], "dog": [0.0, 1.0], "pet": [1.0, 1.0]}


def test_compare_vectors_steps_a_mano():
    # A=(3,4), B=(4,3): A·B=24, ||A||=||B||=5, cos=24/25, euclidiana sqrt(2)
    p = compare_vectors_steps([3.0, 4.0], [4.0, 3.0], dims=5)
    assert p["dot"] == 24.0 and p["norm_a"] == 5.0 and p["norm_b"] == 5.0
    assert p["cosine"] == pytest.approx(24 / 25)
    assert p["euclidean"] == pytest.approx(2 ** 0.5)
    assert p["first"] == [(1, 3.0, 4.0, 12.0), (2, 4.0, 3.0, 12.0)]  # solo 2 dimensiones existen
    assert p["dim"] == 2


def test_average_steps():
    s = average_steps("The cat and the dog and a pet", VEC, STOPWORDS, dims=2)
    assert s["tokens_totales"] == 8
    assert s["stopwords_eliminadas"] == 5  # the, and, the, and, a
    assert s["tokens_que_quedan"] == ["cat", "dog", "pet"]
    assert s["k"] == 3
    assert s["suma"] == [2.0, 2.0]
    assert s["promedio"] == pytest.approx([2 / 3, 2 / 3])


def test_demostracion_markdown_contiene_los_pasos():
    md = demostracion_markdown(
        "A", "B", "cat dog the pet", "dog pet the pet", VEC, STOPWORDS,
        [0.6, 0.8], [1.0, 0.0],
    )
    for fragmento in ("Enfoque 1", "Enfoque 2", "Tokens que quedan", "Vector promedio",
                      "Producto punto", "Normas", "Coseno", "Ángulo", "Euclidiana", "caja negra"):
        assert fragmento in md
    # coseno del enfoque 2: (0.6*1 + 0.8*0) / (1*1) = 0.6
    assert "0.600000" in md


def test_center_vectors():
    centrados, media = center_vectors([[1.0, 2.0], [3.0, 6.0]])
    assert media == [2.0, 4.0]
    assert centrados == [[-1.0, -2.0], [1.0, 2.0]]
    # tras centrar, la suma por dimensión es 0
    assert [sum(v[j] for v in centrados) for j in range(2)] == [0.0, 0.0]
    with pytest.raises(ValueError):
        center_vectors([])


def test_centrado_abre_el_rango_de_vectores_con_direccion_comun():
    # tres vectores con una gran componente común (10, 10) y una pequeña diferencia
    vecs = [[10.0, 10.5], [10.5, 10.0], [10.2, 10.2]]
    antes = upper_triangle(pairwise_matrix(vecs, cosine_similarity))
    centrados, _ = center_vectors(vecs)
    despues = upper_triangle(pairwise_matrix(centrados, cosine_similarity))
    assert value_range(antes) < 0.01
    assert value_range(despues) > 0.5


def test_utilidades_del_flujo():
    assert upper_triangle([[1, 2, 3], [2, 1, 4], [3, 4, 1]]) == [2, 3, 4]
    assert value_range([0.2, 0.9, 0.5]) == pytest.approx(0.7)
    assert kendall_tau([1, 2, 3], [10, 20, 30]) == 1.0
    assert kendall_tau([1, 2, 3], [30, 20, 10]) == -1.0
    assert kendall_tau([1, 2, 3, 4], [1, 3, 2, 4]) == pytest.approx(4 / 6)

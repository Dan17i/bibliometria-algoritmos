"""Pruebas de src/similarity/metrics.py, con casos verificables a mano."""
import pytest

from src.similarity.metrics import (
    angle_degrees,
    cosine_distance,
    cosine_similarity,
    cosine_steps,
    dot_product,
    euclidean_distance,
    euclidean_steps,
    mean_vector,
    norm,
)


def test_dot_product_y_norma():
    assert dot_product([1, 2, 3], [4, 5, 6]) == 32  # 4 + 10 + 18
    assert norm([3, 4]) == 5.0


def test_coseno_vectores_identicos():
    assert cosine_similarity([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert cosine_distance([1, 2, 3], [1, 2, 3]) == pytest.approx(0.0)


def test_coseno_vectores_proporcionales():
    assert cosine_similarity([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)


def test_coseno_vectores_ortogonales():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0
    assert cosine_distance([1, 0], [0, 1]) == 1.0
    assert cosine_steps([1, 0], [0, 1])["angle_degrees"] == pytest.approx(90.0)


def test_coseno_vectores_opuestos():
    assert cosine_similarity([1, 2], [-1, -2]) == pytest.approx(-1.0)
    assert cosine_distance([1, 2], [-1, -2]) == pytest.approx(2.0)


def test_coseno_a_mano():
    # A=(1,1), B=(1,0): A·B=1, ||A||=sqrt2, ||B||=1 -> cos = 1/sqrt2, ángulo 45°
    steps = cosine_steps([1, 1], [1, 0])
    assert steps["dot"] == 1
    assert steps["norm_a"] == pytest.approx(2 ** 0.5)
    assert steps["norm_b"] == 1.0
    assert steps["cosine"] == pytest.approx(0.7071067811865475)
    assert steps["angle_degrees"] == pytest.approx(45.0)


def test_coseno_vector_cero():
    assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0
    assert cosine_distance([0, 0, 0], [0, 0, 0]) == 1.0


def test_dimensiones_distintas():
    with pytest.raises(ValueError):
        cosine_similarity([1, 2], [1, 2, 3])
    with pytest.raises(ValueError):
        euclidean_distance([1], [1, 2])
    with pytest.raises(ValueError):
        dot_product([1, 2], [1])


def test_euclidiana_a_mano():
    # (0,0) a (3,4): sqrt(9 + 16) = 5
    assert euclidean_distance([0, 0], [3, 4]) == 5.0
    steps = euclidean_steps([0, 0], [3, 4])
    assert steps["diffs"] == [-3, -4]
    assert steps["squared_diffs"] == [9, 16]
    assert steps["sum_squares"] == 25
    assert steps["distance"] == 5.0


def test_euclidiana_identicos_y_cero():
    assert euclidean_distance([1, 2, 3], [1, 2, 3]) == 0.0
    assert euclidean_distance([0, 0], [0, 0]) == 0.0


def test_euclidiana_simetrica():
    a, b = [1.5, -2.0, 0.5], [0.0, 1.0, 4.0]
    assert euclidean_distance(a, b) == euclidean_distance(b, a)


def test_mean_vector():
    assert mean_vector([[1, 2], [3, 4], [5, 6]]) == [3.0, 4.0]
    assert mean_vector([[2.0, 4.0]]) == [2.0, 4.0]


def test_mean_vector_errores():
    with pytest.raises(ValueError):
        mean_vector([])
    with pytest.raises(ValueError):
        mean_vector([[1, 2], [1]])


def test_angle_degrees_valores_notables():
    assert angle_degrees(1.0) == 0.0
    assert angle_degrees(-1.0) == 180.0
    assert angle_degrees(0.5) == pytest.approx(60.0)
    assert angle_degrees(-0.5) == pytest.approx(120.0)

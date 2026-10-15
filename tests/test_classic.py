"""Pruebas de los algoritmos clásicos (src/similarity/classic), con casos resueltos a mano."""
import pytest

from src.similarity.classic.cosine import cosine_similarity
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.classic.levenshtein import levenshtein_distance
from src.similarity.classic.needleman_wunsch import (
    needleman_wunsch,
    normalized_needleman_wunsch,
)


# --- Levenshtein -------------------------------------------------------------

def test_levenshtein_ejemplo_clasico():
    # kitten -> sitting: 2 sustituciones (k->s, e->i) y 1 inserción (g)
    distancia, _ = levenshtein_distance("kitten", "sitting")
    assert distancia == 3


def test_levenshtein_matriz_gato_pato():
    distancia, matriz = levenshtein_distance("gato", "pato")
    assert distancia == 1
    assert matriz == [
        [0, 1, 2, 3, 4],
        [1, 1, 2, 3, 4],
        [2, 2, 1, 2, 3],
        [3, 3, 2, 1, 2],
        [4, 4, 3, 2, 1],
    ]


def test_levenshtein_casos_borde():
    assert levenshtein_distance("", "")[0] == 0
    assert levenshtein_distance("", "abc")[0] == 3
    assert levenshtein_distance("abc", "abc")[0] == 0


# --- Needleman-Wunsch --------------------------------------------------------

def test_needleman_wunsch_ejemplo_de_libro():
    # Ejemplo clásico (match +1, mismatch -1, gap -1): puntaje óptimo 0
    puntaje, _ = needleman_wunsch("GATTACA", "GCATGCU")
    assert puntaje == 0


def test_needleman_wunsch_matriz_atcg_acg():
    puntaje, matriz = needleman_wunsch("ATCG", "ACG")
    # ATCG / A-CG: 3 matches y 1 gap
    assert puntaje == 2
    assert matriz == [
        [0, -1, -2, -3],
        [-1, 1, 0, -1],
        [-2, 0, 0, -1],
        [-3, -1, 1, 0],
        [-4, -2, 0, 2],
    ]


def test_needleman_wunsch_casos_borde():
    assert needleman_wunsch("", "")[0] == 0
    assert needleman_wunsch("", "abc")[0] == -3  # tres gaps
    assert needleman_wunsch("abc", "abc")[0] == 3


def test_needleman_wunsch_pesos_personalizados():
    # Un match vale 2 y un gap -2: "ab" vs "a" -> 2 - 2 = 0
    puntaje, _ = needleman_wunsch("ab", "a", match_score=2, mismatch_penalty=-1, gap_penalty=-2)
    assert puntaje == 0


def test_needleman_wunsch_normalizado():
    assert normalized_needleman_wunsch("abc", "abc") == 1.0
    assert normalized_needleman_wunsch("abc", "xyz") == 0.0
    assert normalized_needleman_wunsch("", "") == 1.0
    # ATCG vs ACG: (2 - (-4)) / (4 - (-4)) = 6/8
    assert normalized_needleman_wunsch("ATCG", "ACG") == pytest.approx(0.75)


# --- Jaccard -----------------------------------------------------------------

def test_jaccard_a_mano():
    sim, inter, union = jaccard_similarity("the cat sat", "the cat ran")
    assert inter == {"the", "cat"}
    assert union == {"the", "cat", "sat", "ran"}
    assert sim == pytest.approx(2 / 4)


def test_jaccard_ignora_mayusculas_y_puntuacion():
    sim, _, _ = jaccard_similarity("Hola, Mundo.", "hola mundo")
    assert sim == 1.0


def test_jaccard_casos_borde():
    assert jaccard_similarity("", "")[0] == 1.0
    assert jaccard_similarity("a b", "c d")[0] == 0.0


# --- Coseno (frecuencias) ----------------------------------------------------

def test_coseno_a_mano():
    # vocabulario [a, b, c]: A = [2, 1, 0], B = [1, 0, 1]
    # A·B = 2, ||A|| = sqrt(5), ||B|| = sqrt(2) -> 2 / sqrt(10)
    sim, vec_a, vec_b, vocab = cosine_similarity("a a b", "a c")
    assert vocab == ["a", "b", "c"]
    assert vec_a == [2, 1, 0]
    assert vec_b == [1, 0, 1]
    assert sim == pytest.approx(2 / 10 ** 0.5)


def test_coseno_textos_identicos_y_disjuntos():
    assert cosine_similarity("x y z", "x y z")[0] == pytest.approx(1.0)
    assert cosine_similarity("x y", "z w")[0] == 0.0


def test_coseno_texto_vacio():
    assert cosine_similarity("", "algo")[0] == 0.0

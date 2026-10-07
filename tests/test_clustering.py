"""
Pruebas de clustering jerárquico, resueltas a mano.

4 puntos en una recta: x = 0, 1, 5, 7 -> distancias
    d01=1  d02=5  d03=7  d12=4  d13=6  d23=2
"""
import pytest

from src.clustering.evaluation import (
    best_k_by_silhouette,
    cophenetic_correlation,
    cophenetic_matrix,
    cut_tree,
    pearson,
    silhouette,
)
from src.clustering.linkage import hierarchical_clustering

DIST = [
    [0, 1, 5, 7],
    [1, 0, 4, 6],
    [5, 4, 0, 2],
    [7, 6, 2, 0],
]


def test_single_a_mano():
    # (0,1)@1 -> id4 ; (2,3)@2 -> id5 ; min(5,7,4,6)=4
    assert hierarchical_clustering(DIST, "single") == [
        (0, 1, 1, 2), (2, 3, 2, 2), (4, 5, 4, 4),
    ]


def test_complete_a_mano():
    # la última fusión usa max(5,7,4,6)=7
    assert hierarchical_clustering(DIST, "complete") == [
        (0, 1, 1, 2), (2, 3, 2, 2), (4, 5, 7, 4),
    ]


def test_average_a_mano():
    # la última fusión usa (5+7+4+6)/4 = 5.5
    assert hierarchical_clustering(DIST, "average") == [
        (0, 1, 1, 2), (2, 3, 2, 2), (4, 5, 5.5, 4),
    ]


def test_average_con_clusters_de_distinto_tamano():
    # puntos 0,1,2 (d01=1, d02=3, d12=2) y 10 (d03=10, d13=9, d23=7)
    dist = [[0, 1, 3, 10], [1, 0, 2, 9], [3, 2, 0, 7], [10, 9, 7, 0]]
    merges = hierarchical_clustering(dist, "average")
    # (0,1)@1 -> id4, D(4,2) = (3+2)/2 = 2.5 ; D(4,3) = (10+9)/2 = 9.5
    assert merges[1] == (2, 4, 2.5, 3)
    # D({0,1,2},3) = (10+9+7)/3
    assert merges[2][:2] == (3, 5)
    assert merges[2][2] == pytest.approx(26 / 3)
    assert merges[2][3] == 4


def test_steps():
    merges, steps = hierarchical_clustering(DIST, "single", return_steps=True)
    assert len(steps) == 3
    assert steps[0]["merged"] == (0, 1) and steps[0]["new_id"] == 4
    assert steps[0]["active_before"] == [0, 1, 2, 3]
    assert steps[0]["distances_to_new"] == {2: 4, 3: 6}
    assert [s["distance"] for s in steps] == [m[2] for m in merges]


def test_desempate_determinista():
    dist = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
    assert hierarchical_clustering(dist, "single")[0] == (0, 1, 1, 2)


def test_dos_elementos():
    assert hierarchical_clustering([[0, 3], [3, 0]], "complete") == [(0, 1, 3, 2)]


def test_entradas_invalidas():
    with pytest.raises(ValueError):
        hierarchical_clustering(DIST, "ward")
    with pytest.raises(ValueError):
        hierarchical_clustering([[0]], "single")
    with pytest.raises(ValueError):
        hierarchical_clustering([[0, 1], [2, 0]], "single")  # asimétrica
    with pytest.raises(ValueError):
        hierarchical_clustering([[0, 1, 2], [1, 0]], "single")  # no cuadrada
    with pytest.raises(ValueError):
        hierarchical_clustering([[1, 1], [1, 0]], "single")  # diagonal != 0


def test_matriz_cofenetica_single():
    merges = hierarchical_clustering(DIST, "single")
    assert cophenetic_matrix(merges, 4) == [
        [0, 1, 4, 4],
        [1, 0, 4, 4],
        [4, 4, 0, 2],
        [4, 4, 2, 0],
    ]


def test_cophenetic_matrix_valida_numero_de_fusiones():
    with pytest.raises(ValueError):
        cophenetic_matrix([(0, 1, 1, 2)], 4)


def test_cpcc_a_mano():
    # originales [1,5,7,4,6,2] vs cofenéticas [1,4,4,4,4,2] -> r = 13.8333/sqrt(26.8333*8.8333)
    merges = hierarchical_clustering(DIST, "single")
    res = cophenetic_correlation(DIST, merges)
    assert res["original"] == [1, 5, 7, 4, 6, 2]
    assert res["cophenetic"] == [1, 4, 4, 4, 4, 2]
    expected = (83 / 6) / ((161 / 6 * 53 / 6) ** 0.5)
    assert res["cpcc"] == pytest.approx(expected)
    assert res["cpcc"] == pytest.approx(0.8985, abs=1e-3)


def test_cpcc_perfecto_si_distancias_ya_son_ultrametricas():
    ultra = [[0, 1, 3, 3], [1, 0, 3, 3], [3, 3, 0, 2], [3, 3, 2, 0]]
    for linkage in ("single", "complete", "average"):
        merges = hierarchical_clustering(ultra, linkage)
        assert cophenetic_correlation(ultra, merges)["cpcc"] == pytest.approx(1.0)


def test_pearson():
    assert pearson([1, 2, 3], [2, 4, 6]) == pytest.approx(1.0)
    assert pearson([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)
    assert pearson([1, 1, 1], [1, 2, 3]) == 0.0
    with pytest.raises(ValueError):
        pearson([1, 2], [1])


def test_cut_tree():
    merges = hierarchical_clustering(DIST, "single")
    assert cut_tree(merges, 4, 4) == [0, 1, 2, 3]
    assert cut_tree(merges, 4, 3) == [0, 0, 1, 2]
    assert cut_tree(merges, 4, 2) == [0, 0, 1, 1]
    assert cut_tree(merges, 4, 1) == [0, 0, 0, 0]
    with pytest.raises(ValueError):
        cut_tree(merges, 4, 5)


def test_silueta_a_mano():
    # etiquetas [0,0,1,1]: s = 5/6, 4/5, 5/9, 9/13
    res = silhouette(DIST, [0, 0, 1, 1])
    esperado = [5 / 6, 4 / 5, 5 / 9, 9 / 13]
    assert res["per_item"] == pytest.approx(esperado)
    assert res["silhouette"] == pytest.approx(sum(esperado) / 4)


def test_silueta_cluster_unitario_vale_cero():
    res = silhouette(DIST, [0, 1, 1, 1])
    assert res["per_item"][0] == 0.0


def test_silueta_requiere_dos_clusters():
    with pytest.raises(ValueError):
        silhouette(DIST, [0, 0, 0, 0])
    with pytest.raises(ValueError):
        silhouette(DIST, [0, 1])


def test_best_k_por_silueta():
    merges = hierarchical_clustering(DIST, "single")
    res = best_k_by_silhouette(DIST, merges)
    assert res["best_k"] == 2
    assert set(res["by_k"]) == {2, 3}

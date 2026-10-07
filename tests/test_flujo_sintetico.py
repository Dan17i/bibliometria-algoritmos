"""
Flujo completo con datos sintéticos (tests/fixtures/daniel/):
abstracts -> embeddings (promedio de vectores) -> matriz de distancias coseno
-> 3 enlaces -> CPCC y silueta. No requiere modelos reales ni los PDFs.

El corpus tiene 6 abstracts "puros" (V1 V2 | L1 L2 | A1 A2) y 2 abstracts
"puente" (B1 = visión + lenguaje, B2 = lenguaje + audio) que sirven para ver en
qué se diferencian los enlaces.
"""
import json
from pathlib import Path

import pytest

from src.clustering.evaluation import (
    best_k_by_silhouette,
    cophenetic_correlation,
    cut_tree,
    silhouette,
)
from src.clustering.linkage import LINKAGES, hierarchical_clustering
from src.similarity.ai.compare import compare_embeddings
from src.similarity.ai.embeddings import average_word_vectors
from src.similarity.metrics import cosine_distance, pairwise_matrix

FIXTURES = Path(__file__).parent / "fixtures" / "daniel"
N = 8
# V1 V2 | L1 L2 | A1 A2 | B1 (puente V-L) -> visión | B2 (puente L-A) -> lenguaje
GRUPOS_ESPERADOS = [0, 0, 1, 1, 2, 2, 0, 1]


@pytest.fixture(scope="module")
def datos():
    abstracts = json.loads((FIXTURES / "abstracts_sinteticos.json").read_text(encoding="utf-8"))["abstracts"]
    vectores = json.loads((FIXTURES / "vectores_sinteticos.json").read_text(encoding="utf-8"))["vectores"]
    ids = [a["id"] for a in abstracts]
    pasos = [average_word_vectors(a["texto"], vectores) for a in abstracts]
    embeddings = [p["vector"] for p in pasos]
    dist = pairwise_matrix(embeddings, cosine_distance)
    return {"ids": ids, "pasos": pasos, "embeddings": embeddings, "dist": dist}


@pytest.fixture(scope="module")
def resultados(datos):
    """Fusiones y CPCC por enlace."""
    out = {}
    for linkage in LINKAGES:
        merges = hierarchical_clustering(datos["dist"], linkage)
        out[linkage] = {
            "merges": merges,
            "cpcc": cophenetic_correlation(datos["dist"], merges)["cpcc"],
        }
    return out


def test_embeddings_y_tokens_fuera_de_vocabulario(datos):
    assert len(datos["embeddings"]) == N
    assert all(len(v) == 3 for v in datos["embeddings"])
    # hay palabras que el "modelo" no conoce y se reportan en vez de fallar
    assert any(p["missing_tokens"] for p in datos["pasos"])


def test_matriz_de_distancias_valida(datos):
    d = datos["dist"]
    for i in range(N):
        assert d[i][i] == pytest.approx(0.0, abs=1e-12)
        for j in range(N):
            assert d[i][j] == pytest.approx(d[j][i])
            assert -1e-12 <= d[i][j] <= 1.0 + 1e-12


def test_pares_del_mismo_tema_son_mas_cercanos(datos):
    d = datos["dist"]
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        for otro in range(6):  # sin contar los puentes
            if otro not in (a, b):
                assert d[a][b] < d[a][otro]
                assert d[a][b] < d[b][otro]


def test_los_puentes_estan_entre_dos_temas(datos):
    d = datos["dist"]
    # B1 (6) está más cerca de V1 que de A1; B2 (7) más cerca de L1 que de A1 y de V1
    assert d[6][0] < d[6][4]
    assert d[7][2] < d[7][4]
    assert d[7][2] < d[7][0]


@pytest.mark.parametrize("linkage", LINKAGES)
def test_estructura_de_las_fusiones(resultados, linkage):
    merges = resultados[linkage]["merges"]
    assert len(merges) == N - 1
    alturas = [m[2] for m in merges]
    assert alturas == sorted(alturas)  # single, complete y average son monótonos
    assert merges[-1][3] == N


@pytest.mark.parametrize("linkage", LINKAGES)
def test_corte_en_tres_clusters_asigna_los_puentes(datos, resultados, linkage):
    labels = cut_tree(resultados[linkage]["merges"], N, 3)
    assert labels == GRUPOS_ESPERADOS
    assert silhouette(datos["dist"], labels)["silhouette"] > 0.5
    assert best_k_by_silhouette(datos["dist"], resultados[linkage]["merges"])["best_k"] == 3


def test_single_encadena_y_comprime_las_alturas(resultados):
    """
    Efecto de encadenamiento: con los puentes, single une los temas a alturas
    mucho más bajas que complete y average (basta el par más cercano, que pasa
    por el puente), así que su dendrograma pierde contraste en las últimas fusiones.
    """
    alturas = {l: [m[2] for m in resultados[l]["merges"]] for l in LINKAGES}
    assert alturas["single"][-1] < 0.5 * alturas["complete"][-1]
    assert alturas["single"][-1] < 0.5 * alturas["average"][-1]
    assert alturas["average"][-1] < alturas["complete"][-1]
    # el salto entre las fusiones de abajo y las finales es mucho menor en single
    assert alturas["single"][-1] - alturas["single"][-3] < 0.5 * (
        alturas["complete"][-1] - alturas["complete"][-3]
    )


def test_cpcc_distingue_los_enlaces(resultados):
    """Con los puentes single deja de conservar las distancias originales."""
    cpcc = {l: resultados[l]["cpcc"] for l in LINKAGES}
    for valor in cpcc.values():
        assert 0.5 < valor <= 1.0
    assert cpcc["single"] < cpcc["complete"] < cpcc["average"]


def test_comparacion_por_pares_usa_metrics(datos):
    res = compare_embeddings(datos["ids"], datos["embeddings"])
    assert len(res) == N * (N - 1) // 2
    par = next(r for r in res if (r["a"], r["b"]) == ("V1", "V2"))
    # el coseno de compare.py coincide con 1 - distancia de la matriz
    assert par["cosine"] == pytest.approx(1.0 - datos["dist"][0][1])
    assert par["cosine"] > 0.9

"""
Flujo completo con datos sintéticos (tests/fixtures/daniel/):
abstracts -> embeddings (promedio de vectores) -> matriz de distancias coseno
-> 3 enlaces -> CPCC y silueta. No requiere modelos reales ni los PDFs.
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
GRUPOS_ESPERADOS = [0, 0, 1, 1, 2, 2]  # V1 V2 | L1 L2 | A1 A2


@pytest.fixture(scope="module")
def datos():
    abstracts = json.loads((FIXTURES / "abstracts_sinteticos.json").read_text(encoding="utf-8"))["abstracts"]
    vectores = json.loads((FIXTURES / "vectores_sinteticos.json").read_text(encoding="utf-8"))["vectores"]
    ids = [a["id"] for a in abstracts]
    pasos = [average_word_vectors(a["texto"], vectores) for a in abstracts]
    embeddings = [p["vector"] for p in pasos]
    dist = pairwise_matrix(embeddings, cosine_distance)
    return {"ids": ids, "pasos": pasos, "embeddings": embeddings, "dist": dist}


def test_embeddings_y_tokens_fuera_de_vocabulario(datos):
    assert len(datos["embeddings"]) == 6
    assert all(len(v) == 3 for v in datos["embeddings"])
    # hay palabras que el "modelo" no conoce y se reportan en vez de fallar
    assert any(p["missing_tokens"] for p in datos["pasos"])


def test_matriz_de_distancias_valida(datos):
    d = datos["dist"]
    for i in range(6):
        assert d[i][i] == pytest.approx(0.0, abs=1e-12)
        for j in range(6):
            assert d[i][j] == pytest.approx(d[j][i])
            assert -1e-12 <= d[i][j] <= 1.0 + 1e-12


def test_pares_del_mismo_tema_son_mas_cercanos(datos):
    d = datos["dist"]
    for a, b in [(0, 1), (2, 3), (4, 5)]:
        for otro in range(6):
            if otro not in (a, b):
                assert d[a][b] < d[a][otro]
                assert d[a][b] < d[b][otro]


@pytest.mark.parametrize("linkage", LINKAGES)
def test_los_tres_enlaces_recuperan_los_tres_temas(datos, linkage):
    merges = hierarchical_clustering(datos["dist"], linkage)
    assert len(merges) == 5
    # las alturas no decrecen (single, complete y average son monótonos)
    alturas = [m[2] for m in merges]
    assert alturas == sorted(alturas)
    assert merges[-1][3] == 6
    assert cut_tree(merges, 6, 3) == GRUPOS_ESPERADOS


@pytest.mark.parametrize("linkage", LINKAGES)
def test_metricas_internas_sobre_el_flujo(datos, linkage):
    merges = hierarchical_clustering(datos["dist"], linkage)
    cpcc = cophenetic_correlation(datos["dist"], merges)["cpcc"]
    assert 0.9 < cpcc <= 1.0
    assert silhouette(datos["dist"], GRUPOS_ESPERADOS)["silhouette"] > 0.5
    assert best_k_by_silhouette(datos["dist"], merges)["best_k"] == 3


def test_comparacion_por_pares_usa_metrics(datos):
    res = compare_embeddings(datos["ids"], datos["embeddings"])
    assert len(res) == 15
    par = next(r for r in res if (r["a"], r["b"]) == ("V1", "V2"))
    # el coseno de compare.py coincide con 1 - distancia de la matriz
    assert par["cosine"] == pytest.approx(1.0 - datos["dist"][0][1])
    assert par["cosine"] > 0.9

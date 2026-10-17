"""Pruebas de demo_corpus20 sin descargar modelos."""
from src.similarity.ai.demo_corpus20 import evaluar, load_corpus


def test_corpus_tiene_20_abstracts_con_ids_unicos():
    ids, textos = load_corpus()
    assert len(ids) == 20 and len(set(ids)) == 20
    assert all(len(t.split()) > 50 for t in textos)


def test_evaluar_matriz_pequena_resuelta_a_mano():
    # Dos pares muy cercanos (0-1 y 2-3) separados por 1.0: cualquier enlace los une primero.
    d = [[0, .1, 1, 1], [.1, 0, 1, 1], [1, 1, 0, .1], [1, 1, .1, 0]]
    r = evaluar(d)
    assert set(r) == {"single", "complete", "average"}
    for e in r.values():
        assert e["mejor_k"] == 2
        assert e["corte_mejor_k"][0] == e["corte_mejor_k"][1] != e["corte_mejor_k"][2] == e["corte_mejor_k"][3]
        assert e["cpcc"] > 0.99

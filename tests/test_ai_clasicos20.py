"""Pruebas sin descargas: pares, series de los clásicos y matriz de tau de Kendall."""
import pytest

from src.similarity.ai.demo_clasicos20 import CLASICOS, matriz_tau, pares, similitudes_clasicas

TEXTOS = ["the cat sat", "the cat ran", "a dog barks loudly"]


def test_pares_son_n_por_n_menos_1_sobre_2():
    assert pares(3) == [(0, 1), (0, 2), (1, 2)]
    assert len(pares(20)) == 190
    assert pares(1) == []


def test_una_serie_por_clasico_con_un_valor_por_par():
    series = similitudes_clasicas(TEXTOS, TEXTOS)
    assert set(series) == set(CLASICOS.values())
    assert all(len(v) == 3 for v in series.values())
    # Jaccard a mano: {the,cat,sat}/{the,cat,ran} = 2/4 ; con "a dog barks loudly" no comparten nada
    assert series["jaccard"] == [0.5, 0.0, 0.0]
    assert all(0.0 <= x <= 1.0 for v in series.values() for x in v)


def test_tau_misma_serie_inversa_y_mixta():
    series = {"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0], "c": [3.0, 2.0, 1.0], "d": [1.0, 3.0, 2.0]}
    tau = matriz_tau(series)
    assert tau["a"]["a"] == 1.0
    assert tau["a"]["b"] == 1.0          # mismo orden aunque distinta escala
    assert tau["a"]["c"] == -1.0         # orden inverso
    assert tau["a"]["d"] == pytest.approx(1 / 3)  # 2 concordantes, 1 discordante, de 3 pares
    assert tau["a"]["d"] == tau["d"]["a"]

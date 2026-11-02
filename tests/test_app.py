"""
Pruebas de la aplicación web sobre el corpus real. Se saltan si Flask no está
instalado, porque el resto de la suite no lo necesita.
"""
import pytest

pytest.importorskip("flask")

from app.servidor import CORPUS, MATRICES_PD, POSICION, app
from src.pattern import kmp


@pytest.fixture
def cliente():
    return app.test_client()


def _buscar(cliente, patron):
    return cliente.post("/api/buscar", json={"patron": patron}).get_json()


@pytest.mark.parametrize("patron", ["AI", "ai", "artificial intelligence", "ChatGPT"])
def test_busqueda_exacta_sobre_el_corpus(cliente, patron):
    datos = _buscar(cliente, patron)
    esperado = 0
    for registro in CORPUS:
        esperado += len(kmp.buscar(registro["texto"], patron))
    assert datos["coinciden"] is True
    assert datos["total"] == esperado


def test_mayusculas_y_minusculas_dan_resultados_distintos(cliente):
    assert _buscar(cliente, "ChatGPT")["total"] > 0
    assert _buscar(cliente, "chatgpt")["total"] == 0


def test_el_contexto_resalta_la_cadena_exacta(cliente):
    for articulo in _buscar(cliente, "AI")["por_articulo"]:
        for contexto in articulo["contextos"]:
            assert contexto["coincidencia"] == "AI"


def test_patron_vacio_da_error(cliente):
    assert cliente.post("/api/buscar", json={"patron": ""}).status_code == 400


# --- Requerimiento 1: selección de dos o más artículos ------------------------

SEIS = ["Levenshtein", "Needleman-Wunsch", "Jaccard", "Coseno (TF-IDF)",
        "GloVe promediado", "MiniLM"]


def _comparar(cliente, ids):
    return cliente.post("/api/similitud", json={"ids": ids})


def test_dos_articulos_con_los_seis_algoritmos(cliente):
    datos = _comparar(cliente, ["P1", "P3"]).get_json()
    assert datos["modo"] == "par"
    assert [a["algoritmo"] for a in datos["algoritmos"]] == SEIS
    for algoritmo in datos["algoritmos"]:
        assert 0.0 <= algoritmo["similitud"] <= 1.0
    assert datos["articulos"][0]["texto"] and datos["articulos"][0]["autores"]


def test_lo_guardado_coincide_con_el_calculo_en_vivo(cliente):
    filas = _comparar(cliente, ["P1", "P3"]).get_json()["algoritmos"]
    i, j = POSICION["P1"], POSICION["P3"]
    assert filas[0]["similitud"] == pytest.approx(MATRICES_PD["levenshtein"][i][j])
    assert filas[1]["similitud"] == pytest.approx(MATRICES_PD["needleman_wunsch"][i][j])


@pytest.mark.parametrize("ids", [["P1", "P3", "P4"], [f"P{i}" for i in range(1, 21)]])
def test_mas_de_dos_articulos_con_los_seis_algoritmos(cliente, ids):
    datos = _comparar(cliente, ids).get_json()
    assert datos["modo"] == "matriz"
    assert [a["algoritmo"] for a in datos["algoritmos"]] == SEIS
    assert len(datos["articulos"]) == len(ids)
    for algoritmo in datos["algoritmos"]:
        matriz = datos["matrices"][algoritmo["clave"]]
        assert len(matriz) == len(ids)
        for i in range(len(ids)):
            assert matriz[i][i] == 1.0
            for j in range(i + 1, len(ids)):
                assert matriz[i][j] == matriz[j][i]
                assert 0.0 <= matriz[i][j] <= 1.0


def test_el_par_da_lo_mismo_solo_o_dentro_de_una_seleccion(cliente):
    par = _comparar(cliente, ["P1", "P3"]).get_json()["algoritmos"]
    grupo = _comparar(cliente, ["P1", "P3", "P4"]).get_json()
    for fila, algoritmo in zip(par, grupo["algoritmos"]):
        assert grupo["matrices"][algoritmo["clave"]][0][1] == pytest.approx(fila["similitud"])


@pytest.mark.parametrize("ids", [[], ["P1"], ["P1", "P99"], ["P1", "P1"], "P1"])
def test_seleccion_invalida_da_error(cliente, ids):
    assert _comparar(cliente, ids).status_code == 400

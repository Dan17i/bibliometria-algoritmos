"""
Pruebas de la aplicación web sobre el corpus real. Se saltan si Flask no está
instalado, porque el resto de la suite no lo necesita.
"""
import pytest

pytest.importorskip("flask")

from app.servidor import CORPUS, app
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

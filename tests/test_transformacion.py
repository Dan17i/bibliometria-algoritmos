"""Pruebas de la transformación del abstract, con casos resueltos a mano."""
import json
from pathlib import Path

from src.preprocessing.transformacion import transformar_abstract

ABSTRACTS = Path(__file__).resolve().parent.parent / "data" / "processed" / "abstracts.json"


def test_minusculas_sin_puntuacion_ni_stopwords():
    assert transformar_abstract("The Use of AI, in Education.") == "use ai education"


def test_compuestos_y_posesivos():
    assert transformar_abstract("AI's problem-solving") == "ai problem solving"


def test_conserva_orden_y_repeticiones():
    assert transformar_abstract("learning and learning") == "learning learning"


def test_texto_vacio_o_solo_stopwords():
    assert transformar_abstract("") == ""
    assert transformar_abstract("the of and") == ""


def test_el_corpus_transformado_es_mas_corto_y_no_queda_vacio():
    for registro in json.loads(ABSTRACTS.read_text(encoding="utf-8")):
        transformado = transformar_abstract(registro["texto"])
        assert 50 < len(transformado.split()) < registro["_palabras"], registro["id"]

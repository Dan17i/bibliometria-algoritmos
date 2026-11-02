"""
Pruebas del preprocesamiento sobre los 20 PDF reales del corpus.

Los autores esperados se copiaron de la portada de cada PDF. Los fragmentos de
los abstracts son los que estaban dañados en versiones anteriores del extractor.
"""
import os
import re

import pytest

from src.preprocessing.extractor import procesar_corpus, unir_cortes

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

AUTORES = {
    "P1": "Matthias Carl Laupichler, Alexandra Aster, Jana Schirch, Tobias Raupach",
    "P2": "Thomas K.F. Chiu, Qi Xia, Xinyan Zhou, Ching Sing Chai, Miaoting Cheng",
    "P3": "Arne Bewersdorff, Xiaoming Zhai, Jessica Roberts, Claudia Nerdel",
    "P4": "Srecko Joksimovic, Dirk Ifenthaler, Rebecca Marrone, Maarten De Laat, George Siemens",
    "P5": "Xieling Chen, Haoran Xie, Di Zou, Gwo-Jen Hwang",
    "P6": "Xieling Chen, Haoran Xie, Gwo-Jen Hwang",
    "P7": "Riordan Alfredo, Vanessa Echeverria, Yueqiao Jin, Lixiang Yan, Zachari Swiecki, "
          "Dragan Gašević, Roberto Martinez-Maldonado",
    "P8": "Seng Chee Tan, Alwyn Vwen Yen Lee, Min Lee",
    "P9": "Ke Zhang, Ayse Begum Aslan",
    "P10": "Chih-Pu Dai, Fengfeng Ke",
    "P11": "Qi Xia, Xiaojing Weng, Fan Ouyang, Tzung Jin Lin, Thomas K.F. Chiu",
    "P12": "Melissa Bond, Hassan Khosravi, Maarten De Laat, Nina Bergdahl, Violeta Negrea, "
           "Emily Oxley, Phuong Pham, Sin Wang Chong, George Siemens",
    "P13": "Jyoti Mudkanna Gavhane, Reena Pagare",
    "P14": "Tamar Mikeladze, Paulien C. Meijer, Roald P. Verhoeff",
    "P15": "Bayode Ogunleye, Kudirat Ibilola Zakariyyah, Oluwaseun Ajao, Olakunle Olayinka, "
           "Hemlata Sharma",
    "P16": "Che Yee Lye, Lyndon Lim",
    "P17": "Fan Ouyang, Tuan Anh Dinh, Weiqi Xu",
    "P18": "Zied Bahroun, Chiraz Anane, Vian Ahmed, Andrew Zacca",
    "P19": "Chung Kwan Lo",
    "P20": "Olaf Zawacki-Richter, Victoria I. Marín, Melissa Bond, Franziska Gouverneur",
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    salida = tmp_path_factory.mktemp("corpus")
    registros = procesar_corpus(os.path.join(RAIZ, "docs", "articulos"), str(salida))
    return {r["id"]: r for r in registros}


def test_son_20_sin_incidencias(corpus):
    assert len(corpus) == 20
    assert [i for i, r in corpus.items() if r["_revisar"]] == []


@pytest.mark.parametrize("ident", list(AUTORES))
def test_autores(corpus, ident):
    assert corpus[ident]["autores"] == AUTORES[ident]


def test_no_quedan_palabras_cortadas(corpus):
    # "pre- and misconceptions" (P3) es un guion suspensivo real, no un corte
    for ident, registro in corpus.items():
        cortes = re.findall(r"\b[A-Za-z]+- [a-z]+\b", registro["texto"])
        assert [c for c in cortes if c != "pre- and"] == [], ident


def test_orden_de_lectura_de_p3(corpus):
    assert "effective educational programs in the field of AI" in corpus["P3"]["texto"]


def test_p7_salta_el_bloque_de_keywords(corpus):
    assert corpus["P7"]["texto"].startswith("The rapid expansion of Learning Analytics")


def test_p9_termina_antes_de_la_introduccion(corpus):
    assert corpus["P9"]["texto"].endswith("research and development efforts.")


def test_p15_esta_completo(corpus):
    assert "document sources/authors, keywords, and co-authorship." in corpus["P15"]["texto"]
    assert corpus["P15"]["texto"].endswith("policies for GenAI usage.")


def test_p20_sin_palabras_pegadas(corpus):
    assert "research on AI applications in higher education through" in corpus["P20"]["texto"]
    assert "inhighereducationthrough" not in corpus["P20"]["texto"]


def test_titulos_corregidos(corpus):
    assert "transforms assessment" in corpus["P11"]["titulo"]
    assert corpus["P13"]["titulo"].endswith("quotient: a review")


def test_unir_cortes_usa_el_vocabulario():
    vocabulario = {"education", "well", "defined"}
    texto, decisiones = unir_cortes("edu-\ncation is well-\ndefined and holisti-\ncally", vocabulario)
    assert texto == "education is well-defined and holistically"
    assert len(decisiones) == 3

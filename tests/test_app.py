"""
Pruebas de la aplicación web sobre el corpus real. Se saltan si Flask no está
instalado, porque el resto de la suite no lo necesita.
"""
import subprocess
import sys
from pathlib import Path

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


# --- Requerimiento 1: selección de dos o más artículos ------------------------

SEIS = ["Levenshtein", "Needleman-Wunsch", "Jaccard", "Coseno (TF-IDF)",
        "GloVe promediado", "MiniLM"]


def _comparar(cliente, ids):
    return cliente.post("/api/similitud", json={"ids": ids})


def _demostrar(cliente, clave, ids):
    return cliente.post("/api/demostracion", json={"algoritmo": clave, "ids": ids})


def test_dos_articulos_con_los_seis_algoritmos(cliente):
    datos = _comparar(cliente, ["P1", "P3"]).get_json()
    assert datos["modo"] == "par"
    assert [a["algoritmo"] for a in datos["algoritmos"]] == SEIS
    for algoritmo in datos["algoritmos"]:
        assert 0.0 <= algoritmo["similitud"] <= 1.0
    assert datos["articulos"][0]["texto"] and datos["articulos"][0]["autores"]


def test_lo_guardado_coincide_con_la_demostracion_en_vivo(cliente):
    filas = _comparar(cliente, ["P1", "P3"]).get_json()["algoritmos"]
    for fila in filas:
        en_vivo = _demostrar(cliente, fila["clave"], ["P1", "P3"]).get_json()
        assert en_vivo["similitud"] == pytest.approx(fila["similitud"], abs=1e-5), fila["clave"]


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


# --- Demostraciones -----------------------------------------------------------

CLAVES = ["levenshtein", "needleman_wunsch", "jaccard", "coseno_tfidf", "glove", "minilm"]


@pytest.mark.parametrize("clave", CLAVES)
def test_demostracion_trae_todas_sus_partes(cliente, clave):
    d = _demostrar(cliente, clave, ["P6", "P9"]).get_json()
    assert d["clave"] == clave and 0.0 <= d["similitud"] <= 1.0
    assert d["idea"] and d["en_una_frase"] and d["interpretacion"]
    assert d["formulas"] and d["pasos"] and d["visual"]
    assert d["codigo"]["archivo"].startswith("src/")
    assert d["corpus"]["total"] == 190 and 1 <= d["corpus"]["puesto"] <= 190
    assert d["corpus"]["minimo"] <= d["similitud"] <= d["corpus"]["maximo"]


def test_p6_p9_es_el_par_mas_parecido_segun_minilm(cliente):
    assert _demostrar(cliente, "minilm", ["P6", "P9"]).get_json()["corpus"]["puesto"] == 1


def test_los_valores_coinciden_con_el_documento(cliente):
    esperado = {"levenshtein": 0.2454, "needleman_wunsch": 0.3146, "jaccard": 0.1552,
                "coseno_tfidf": 0.1600, "glove": 0.9792, "minilm": 0.8500}
    for clave, valor in esperado.items():
        d = _demostrar(cliente, clave, ["P6", "P9"]).get_json()
        assert round(d["similitud"], 4) == valor, clave


def test_tus_palabras_matriz_completa_a_mano(cliente):
    d = cliente.post("/api/demostracion", json={"algoritmo": "levenshtein", "textos": ["gato", "pato"]}).get_json()
    assert d["similitud"] == pytest.approx(0.75)
    assert d["visual"][0]["completa"] is True
    assert d["visual"][0]["celdas"][4][4] == 1
    assert "corpus" not in d


def test_needleman_tus_palabras_a_mano(cliente):
    d = cliente.post("/api/demostracion", json={"algoritmo": "needleman_wunsch", "textos": ["ATCG", "ACG"]}).get_json()
    assert d["visual"][0]["celdas"][-1][-1] == 2
    assert d["similitud"] == pytest.approx(0.75)


@pytest.mark.parametrize("cuerpo", [
    {"algoritmo": "nada", "ids": ["P1", "P2"]},
    {"algoritmo": "jaccard", "ids": ["P1"]},
    {"algoritmo": "jaccard", "ids": ["P1", "P1"]},
    {"algoritmo": "jaccard", "ids": ["P1", "P99"]},
    {"algoritmo": "glove", "textos": ["a", "b"]},
    {"algoritmo": "levenshtein", "textos": ["a", ""]},
    {"algoritmo": "levenshtein", "textos": ["a" * 41, "b"]},
])
def test_peticion_invalida_da_error(cliente, cuerpo):
    assert cliente.post("/api/demostracion", json=cuerpo).status_code == 400


def test_el_servidor_publicado_no_necesita_los_modelos():
    """requirements-web.txt solo trae Flask y waitress: las seis demostraciones deben funcionar así."""
    codigo = (
        "import sys\n"
        "from app.servidor import app\n"
        "c = app.test_client()\n"
        "for clave in ('levenshtein', 'needleman_wunsch', 'jaccard', 'coseno_tfidf', 'glove', 'minilm'):\n"
        "    assert c.post('/api/demostracion', json={'algoritmo': clave, 'ids': ['P6', 'P9']}).status_code == 200\n"
        "assert c.post('/api/demostracion-agrupamiento', json={'representacion': 'minilm', 'enlace': 'average'}).status_code == 200\n"
        "assert c.post('/api/demostracion-patron', json={'algoritmo': 'kmp', 'patron': 'learning'}).status_code == 200\n"
        "pesadas = [m for m in ('numpy', 'torch', 'gensim', 'sentence_transformers', 'fitz', 'matplotlib') if m in sys.modules]\n"
        "assert pesadas == [], pesadas\n"
    )
    raiz = Path(__file__).resolve().parent.parent
    resultado = subprocess.run([sys.executable, "-c", codigo], cwd=raiz, capture_output=True, text=True)
    assert resultado.returncode == 0, resultado.stderr


def test_tus_palabras_redacta_bien_una_sola_edicion(cliente):
    d = cliente.post("/api/demostracion", json={"algoritmo": "levenshtein", "textos": ["gato", "pato"]}).get_json()
    assert d["en_una_frase"].startswith("Se necesita 1 edición")


def test_las_formulas_de_la_pagina_estan_en_el_proyecto(cliente):
    assert cliente.get("/static/vendor/katex/katex.min.js").status_code == 200
    assert cliente.get("/static/demostraciones.js").status_code == 200


# --- Agrupamiento -------------------------------------------------------------

@pytest.mark.parametrize("representacion", ["minilm", "glove", "glove_centrado", "jaccard"])
@pytest.mark.parametrize("enlace", ["single", "complete", "average"])
def test_demostracion_del_agrupamiento_coincide_con_el_dendrograma(cliente, representacion, enlace):
    d = cliente.post("/api/demostracion-agrupamiento", json={"representacion": representacion, "enlace": enlace}).get_json()
    publicado = {r["clave"]: r["cpcc"][enlace] for r in cliente.get("/api/clustering").get_json()["representaciones"]}
    assert d["similitud"] == pytest.approx(publicado[representacion], abs=1e-9)
    assert d["conclusiones"] and d["pasos"] and d["codigo"]["archivo"].startswith("src/")


def test_agrupamiento_peticion_invalida(cliente):
    assert cliente.post("/api/demostracion-agrupamiento", json={"representacion": "x", "enlace": "single"}).status_code == 400


def test_agrupamiento_nombra_los_grupos_por_sus_miembros(cliente):
    d = cliente.post("/api/demostracion-agrupamiento", json={"representacion": "minilm", "enlace": "average"}).get_json()
    titulos = [p["titulo"] for p in d["pasos"]]
    assert "Fusión 1: se unen P6 y P9" in titulos
    assert any("{P6, P9}" in t for t in titulos)
    assert not any("grupo 2" in t for t in titulos)


def test_agrupamiento_la_tabla_de_silueta_incluye_el_mejor_corte(cliente):
    d = cliente.post("/api/demostracion-agrupamiento", json={"representacion": "glove_centrado", "enlace": "single"}).get_json()
    assert 13 in [f[0] for f in d["visual"][1]["filas"]]


def test_agrupamiento_usa_la_escala_de_la_silueta(cliente):
    d = cliente.post("/api/demostracion-agrupamiento", json={"representacion": "glove", "enlace": "single"}).get_json()
    assert "una estructura débil" in d["conclusiones"][2]     # 0.291
    d = cliente.post("/api/demostracion-agrupamiento", json={"representacion": "minilm", "enlace": "average"}).get_json()
    assert "sin estructura sustancial" in d["conclusiones"][2]   # 0.220


@pytest.mark.parametrize("ruta, cuerpo", [
    ("/api/similitud", {"ids": [["P1"], "P2"]}),
    ("/api/demostracion", {"algoritmo": ["jaccard"], "ids": ["P1", "P2"]}),
    ("/api/demostracion", {"algoritmo": "jaccard", "ids": [["P1"], "P2"]}),
    ("/api/demostracion-agrupamiento", {"representacion": ["minilm"], "enlace": "single"}),
    ("/api/demostracion-patron", {"algoritmo": ["kmp"], "patron": "AI"}),
    ("/api/buscar", {"patron": ["AI"]}),
])
def test_entradas_de_otro_tipo_dan_error_y_no_una_pagina_html(cliente, ruta, cuerpo):
    respuesta = cliente.post(ruta, json=cuerpo)
    assert respuesta.status_code == 400 and respuesta.is_json


# --- Demostración de la búsqueda de patrones ------------------------------------

PATRONES = ["kmp", "boyer_moore", "rabin_karp"]


def _patron(cliente, clave, **datos):
    return cliente.post("/api/demostracion-patron", json={"algoritmo": clave, **datos})


@pytest.mark.parametrize("clave, comparaciones", [("kmp", 126), ("boyer_moore", 31), ("rabin_karp", 133)])
def test_caso_del_documento(cliente, clave, comparaciones):
    d = _patron(cliente, clave, patron="learning").get_json()
    assert d["ocurrencias"] == [49, 104]
    assert d["comparaciones"] == comparaciones
    assert d["texto_usado"].startswith("Trend Two posits")
    assert d["conclusiones"] and d["pasos"] and d["formulas"]
    assert d["codigo"]["archivo"].startswith("src/pattern/")


@pytest.mark.parametrize("clave", PATRONES)
def test_cada_paso_alinea_el_patron_dentro_del_texto(cliente, clave):
    d = _patron(cliente, clave, patron="learning").get_json()
    busqueda = [v for v in d["visual"] if v["tipo"] == "busqueda"][0]
    m, n = len(d["patron"]), len(d["texto_usado"])
    assert 0 <= busqueda["inicial"] < len(busqueda["pasos"])
    # KMP compara hasta el último carácter del texto, aunque el patrón ya no quepa
    tope = n - 1 if clave == "kmp" else n - m
    for paso in busqueda["pasos"]:
        assert 0 <= paso["alineacion"] <= tope
        assert len(paso["estados"]) == m
        assert paso["latex"] and paso["texto"]


@pytest.mark.parametrize("clave", PATRONES)
def test_patron_con_bordes_en_texto_propio(cliente, clave):
    d = _patron(cliente, clave, patron="ABABCABAB", texto="ABABDABACDABABCABAB").get_json()
    assert d["ocurrencias"] == [10]
    assert d["textos"] == ["ABABDABACDABABCABAB", "ABABCABAB"]
    assert "conclusiones" not in d


def test_lps_con_bordes(cliente):
    d = _patron(cliente, "kmp", patron="ABABC", texto="ABABABC").get_json()
    assert d["ocurrencias"] == [2]
    assert "[0, 0, 1, 2, 0]".replace(" ", "") in d["pasos"][0]["latex"][0].replace(" ", "")


@pytest.mark.parametrize("patron", ["ChatGPT", "AI", "systematic review", "zzzzzz", "100%", "a_b{c}"])
@pytest.mark.parametrize("clave", PATRONES)
def test_cualquier_patron_del_corpus(cliente, clave, patron):
    respuesta = _patron(cliente, clave, patron=patron)
    assert respuesta.status_code == 200
    d = respuesta.get_json()
    assert isinstance(d["en_una_frase"], str) and d["visual"]


def test_la_busqueda_es_exacta_en_la_demostracion(cliente):
    assert _patron(cliente, "kmp", patron="chatgpt").get_json()["ocurrencias"] == []
    assert len(_patron(cliente, "kmp", patron="ChatGPT").get_json()["ocurrencias"]) >= 1


@pytest.mark.parametrize("cuerpo", [
    {"algoritmo": "nada", "patron": "a"},
    {"algoritmo": "kmp", "patron": ""},
    {"algoritmo": "kmp", "patron": "a" * 121},
    {"algoritmo": "kmp", "patron": "abc", "texto": "ab"},
    {"algoritmo": "kmp", "patron": "a", "texto": ""},
    {"algoritmo": "kmp", "patron": "a", "texto": "b" * 301},
])
def test_demostracion_patron_invalida(cliente, cuerpo):
    respuesta = cliente.post("/api/demostracion-patron", json=cuerpo)
    assert respuesta.status_code == 400 and respuesta.is_json


def test_las_filas_de_la_busqueda_traen_su_clave(cliente):
    filas = _buscar(cliente, "AI")["algoritmos"]
    assert [f["clave"] for f in filas] == PATRONES

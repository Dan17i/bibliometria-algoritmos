"""
Los diagramas de secuencia se escriben a mano: estas pruebas verifican que digan la
verdad. Cada llamada dibujada debe existir en el módulo de destino y aparecer como
llamada real en el módulo de origen.
"""
import glob
import os

import pytest

pytest.importorskip("matplotlib")

from python_viz.diagramas import (
    GRUPOS_DE_MODULOS,
    PARTICIPANTES,
    RAIZ,
    SECUENCIAS,
    dependencias,
    funciones_de,
    imports_entre_modulos,
    llamadas_de,
)

LLAMADAS = [
    (clave, origen, destino, firma)
    for clave, (_, _, mensajes) in SECUENCIAS.items()
    for tipo, origen, destino, firma in mensajes
    if tipo == "llamada"
]


def _nombre(firma: str) -> str:
    return firma.split("(")[0]


def test_hay_un_diagrama_por_algoritmo():
    for clave in ("levenshtein", "needleman_wunsch", "jaccard", "coseno_tfidf", "glove", "minilm",
                  "clustering", "kmp", "boyer_moore", "rabin_karp"):
        assert clave in SECUENCIAS


@pytest.mark.parametrize("clave", list(SECUENCIAS))
def test_los_mensajes_usan_participantes_del_diagrama(clave):
    _, participantes, mensajes = SECUENCIAS[clave]
    for _, origen, destino, _ in mensajes:
        assert origen in participantes and destino in participantes


@pytest.mark.parametrize("clave", list(PARTICIPANTES))
def test_los_archivos_de_los_participantes_existen(clave):
    ruta = PARTICIPANTES[clave][1]
    if ruta is not None:
        assert os.path.exists(os.path.join(RAIZ, ruta)), ruta


@pytest.mark.parametrize("clave, origen, destino, firma", LLAMADAS)
def test_la_funcion_existe_en_el_destino(clave, origen, destino, firma):
    assert _nombre(firma) in funciones_de(PARTICIPANTES[destino][1]), (clave, firma)


@pytest.mark.parametrize("clave, origen, destino, firma", LLAMADAS)
def test_el_origen_llama_de_verdad_a_la_funcion(clave, origen, destino, firma):
    assert _nombre(firma) in llamadas_de(PARTICIPANTES[origen][1]), (clave, firma)


def test_los_diagramas_de_modulos_cubren_todo_el_codigo():
    incluidos = {r for _, modulos in GRUPOS_DE_MODULOS.values() for r in modulos}
    for carpeta in ("src", "app", "python_viz"):
        for ruta in glob.glob(os.path.join(RAIZ, carpeta, "**", "*.py"), recursive=True):
            relativa = os.path.relpath(ruta, RAIZ).replace(os.sep, "/")
            if not relativa.endswith("__init__.py"):
                assert relativa in incluidos, relativa


def test_no_hay_ciclos_entre_modulos():
    """preprocessing y ai se importan entre sí como paquetes, pero no como módulos."""
    grafo = imports_entre_modulos()
    visitando, terminados = set(), set()

    def recorrer(modulo):
        assert modulo not in visitando, f"ciclo que pasa por {modulo}"
        if modulo in terminados:
            return
        visitando.add(modulo)
        for siguiente in grafo.get(modulo, ()):
            recorrer(siguiente)
        visitando.discard(modulo)
        terminados.add(modulo)

    for modulo in grafo:
        recorrer(modulo)


def test_las_dependencias_se_leen_del_codigo():
    pares = dependencias()
    assert ("app", "pattern") in pares
    assert ("clustering", "metrics") not in pares   # el clustering recibe la matriz ya calculada
    assert ("classic", "metrics") in pares

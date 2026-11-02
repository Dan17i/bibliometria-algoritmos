"""
Pruebas de los algoritmos de búsqueda de patrones (Requerimiento 3)
=====================================================================
La propiedad central es que los tres algoritmos deben devolver EXACTAMENTE
las mismas posiciones: en una búsqueda de coincidencia exacta no hay margen
de interpretación.

La referencia se construye con una búsqueda ingenua escrita aquí mismo, no
con `str.find`: el proyecto no usa funciones de alto nivel del lenguaje para
buscar, y una prueba que las usara dejaría la puerta abierta a que alguien las
copiara al código de producción.
"""

from __future__ import annotations

import pytest

from src.pattern import boyer_moore, buscador, kmp, rabin_karp

ALGORITMOS = [
    pytest.param(kmp, id="kmp"),
    pytest.param(boyer_moore, id="boyer-moore"),
    pytest.param(rabin_karp, id="rabin-karp"),
]


def busqueda_ingenua(texto: str, patron: str) -> list[int]:
    """Referencia O(n*m): compara el patrón en cada posición posible."""
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return []
    posiciones = []
    for i in range(n - m + 1):
        j = 0
        while j < m and texto[i + j] == patron[j]:
            j += 1
        if j == m:
            posiciones.append(i)
    return posiciones


CASOS = [
    ("abracadabra", "abra", [0, 7]),
    ("aaaa", "aa", [0, 1, 2]),                 # ocurrencias solapadas
    ("abcabcabc", "abc", [0, 3, 6]),
    ("hola mundo", "mundo", [5]),
    ("hola mundo", "xyz", []),
    ("a", "a", [0]),
    ("ABABDABACDABABCABAB", "ABABCABAB", [10]),  # caso clásico de KMP
    ("texto sin el patron", "zzz", []),
]


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize("texto,patron,esperado", CASOS)
def test_encuentra_las_posiciones_correctas(algoritmo, texto, patron, esperado):
    assert algoritmo.buscar(texto, patron) == esperado


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
@pytest.mark.parametrize("texto,patron", [
    ("abc", ""),            # patrón vacío
    ("ab", "abc"),          # patrón más largo que el texto
    ("", "abc"),            # texto vacío
    ("", ""),               # ambos vacíos
])
def test_casos_borde_devuelven_lista_vacia(algoritmo, texto, patron):
    assert algoritmo.buscar(texto, patron) == []


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_coincide_con_la_busqueda_ingenua(algoritmo):
    """Sobre un texto repetitivo, donde los tres algoritmos más se tensionan."""
    texto = "abababcababcabcababab" * 12
    for patron in ("ab", "abc", "ababc", "cab", "bbb", "abababc"):
        assert algoritmo.buscar(texto, patron) == busqueda_ingenua(texto, patron), \
            f"discrepancia con el patron {patron!r}"


def test_los_tres_algoritmos_coinciden_entre_si():
    texto = "the quick brown fox jumps over the lazy dog. " * 25
    for patron in ("the", "quick", "o", "dog.", "zebra", "the lazy"):
        resultados = [m.buscar(texto, patron)
                      for m in (kmp, boyer_moore, rabin_karp)]
        assert resultados[0] == resultados[1] == resultados[2], \
            f"los algoritmos difieren con el patron {patron!r}"


# --- pruebas de las estructuras auxiliares -------------------------------

def test_tabla_lps_conocida():
    """La tabla LPS de 'ABABC': solo 'ABA' y 'ABAB' tienen borde."""
    assert kmp.construir_lps("ABABC") == [0, 0, 1, 2, 0]
    assert kmp.construir_lps("AAAA") == [0, 1, 2, 3]
    assert kmp.construir_lps("ABCDE") == [0, 0, 0, 0, 0]


def test_tabla_caracter_malo_guarda_la_ultima_aparicion():
    """Con 'EJEMPLO', la E aparece en 0 y 2; debe quedar la posición 2."""
    tabla = boyer_moore.tabla_caracter_malo("EJEMPLO")
    assert tabla["E"] == 2
    assert tabla["O"] == 6
    assert "Z" not in tabla


def test_rabin_karp_resuelve_las_colisiones():
    """
    Una colisión de hash no debe producir un falso positivo.

    Se verifica indirectamente: sobre un texto con muchas ventanas similares,
    el resultado sigue coincidiendo con la referencia.
    """
    texto = "".join(chr(97 + (i * 7) % 26) for i in range(900))
    for patron in ("abc", "hqz", texto[100:106]):
        assert rabin_karp.buscar(texto, patron) == busqueda_ingenua(texto, patron)


def test_las_metricas_son_coherentes():
    texto = "abracadabra " * 40
    for modulo in (kmp, boyer_moore, rabin_karp):
        m = modulo.buscar_con_metricas(texto, "abra")
        assert m["ocurrencias"] == modulo.buscar(texto, "abra")
        assert m["comparaciones"] > 0


def test_boyer_moore_peor_caso_con_muchas_ocurrencias():
    """Con ocurrencias en todas las posiciones, las comparaciones crecen como n·m."""
    n, m = 2000, 100
    metricas = boyer_moore.buscar_con_metricas("a" * n, "a" * m)
    assert len(metricas["ocurrencias"]) == n - m + 1
    assert metricas["comparaciones"] == (n - m + 1) * m


@pytest.mark.parametrize("algoritmo", ALGORITMOS)
def test_la_coincidencia_es_exacta(algoritmo):
    """Mayúsculas y minúsculas son caracteres distintos: "AI" no está en "training"."""
    texto = "AI training and ai tools"
    assert algoritmo.buscar(texto, "AI") == [0]
    assert algoritmo.buscar(texto, "ai") == [5, 16]


def test_el_buscador_no_transforma_el_texto():
    registros = [{"id": "P1", "titulo": "t", "texto": "AI and training"}]
    assert buscador.comparar(registros, "AI")["total_ocurrencias"] == 1
    assert buscador.comparar(registros, "ai")["total_ocurrencias"] == 1
    assert buscador.comparar(registros, "ai")["por_articulo"][0]["posiciones"] == [9]
    assert buscador.comparar(registros, "Training")["total_ocurrencias"] == 0


def test_boyer_moore_salta_parte_del_texto():
    """
    La ventaja de Boyer-Moore es no leer todo el texto.

    Con un patrón largo cuyos caracteres escasean, debe hacer bastantes menos
    comparaciones que la longitud del texto.
    """
    texto = "x" * 4000
    metricas = boyer_moore.buscar_con_metricas(texto, "zqjkvw")
    assert metricas["ocurrencias"] == []
    assert metricas["comparaciones"] < len(texto), \
        "deberia saltar caracteres en vez de recorrerlos todos"

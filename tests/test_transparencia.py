"""
Guardia de la regla de transparencia algorítmica. Se revisa el AST, así que los
comentarios y docstrings que mencionan estos nombres no cuentan.

    - Ningún módulo de algoritmos puede importar librerías que los resuelvan ni
      llamar a las similitudes que traen los modelos de embeddings.
    - El núcleo de los algoritmos tampoco puede usar funciones nativas del
      lenguaje que hagan parte del cálculo: min, max, sorted, lower, split,
      find, intersection, math.log, etc.
    - Los tres algoritmos del Req. 3 no pueden usar el operador `in` sobre el texto.
"""
import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent / "src"
ARCHIVOS = sorted(
    list((RAIZ / "similarity" / "classic").glob("*.py"))
    + list((RAIZ / "similarity" / "ai").glob("*.py"))
    + list((RAIZ / "clustering").glob("*.py"))
    + [RAIZ / "similarity" / "metrics.py"]
)
BUSQUEDA = [RAIZ / "pattern" / "kmp.py", RAIZ / "pattern" / "boyer_moore.py",
            RAIZ / "pattern" / "rabin_karp.py"]
NUCLEO = sorted(
    list((RAIZ / "similarity" / "classic").glob("*.py"))
    + [RAIZ / "similarity" / "metrics.py",
       RAIZ / "similarity" / "ai" / "embeddings.py",
       RAIZ / "preprocessing" / "transformacion.py",
       RAIZ / "clustering" / "linkage.py",
       RAIZ / "clustering" / "evaluation.py"]
    + BUSQUEDA
)

MODULOS_PROHIBIDOS = {"sklearn", "scipy", "numpy", "difflib", "Levenshtein", "rapidfuzz",
                      "textdistance", "jellyfish"}
LLAMADAS_PROHIBIDAS = {"cos_sim", "similarity", "n_similarity", "wmdistance",
                       "most_similar", "pairwise_distances", "cosine_similarity_matrix"}
MODULOS_NATIVOS = {"math", "re", "string", "statistics", "cmath", "functools", "itertools"}
LLAMADAS_NATIVAS = {
    "min", "max", "sum", "abs", "sorted", "sort", "set", "frozenset", "pow",
    "lower", "upper", "casefold", "split", "strip", "replace", "join",
    "find", "rfind", "index", "rindex", "count", "startswith", "endswith",
    "intersection", "union", "difference",
}


def _importados(arbol):
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                yield alias.name.split(".")[0]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            yield nodo.module.split(".")[0]


def _llamadas(arbol):
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call):
            f = nodo.func
            yield nodo.lineno, f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)


def violaciones(codigo: str) -> list[str]:
    """Devuelve la lista de infracciones de transparencia que hay en `codigo`."""
    arbol = ast.parse(codigo)
    encontradas = []
    for modulo in sorted(set(_importados(arbol)) & MODULOS_PROHIBIDOS):
        encontradas.append(f"importa {modulo}")
    for linea, nombre in _llamadas(arbol):
        if nombre in LLAMADAS_PROHIBIDAS:
            encontradas.append(f"línea {linea}: llama a {nombre}")
    return encontradas


def nativas(codigo: str) -> list[str]:
    """Funciones nativas usadas en el código de un módulo del núcleo."""
    arbol = ast.parse(codigo)
    encontradas = [f"importa {m}" for m in sorted(set(_importados(arbol)) & MODULOS_NATIVOS)]
    for linea, nombre in _llamadas(arbol):
        if nombre in LLAMADAS_NATIVAS:
            encontradas.append(f"línea {linea}: usa {nombre}")
    return encontradas


def usos_de_in(codigo: str) -> list[int]:
    """Líneas con el operador `in` como comparación (no cuenta `for x in ...`)."""
    lineas = []
    for nodo in ast.walk(ast.parse(codigo)):
        if isinstance(nodo, ast.Compare):
            for operador in nodo.ops:
                if isinstance(operador, (ast.In, ast.NotIn)):
                    lineas.append(nodo.lineno)
    return lineas


def test_hay_archivos_que_revisar():
    assert len(ARCHIVOS) >= 5
    assert len(NUCLEO) >= 10


@pytest.mark.parametrize("ruta", ARCHIVOS + BUSQUEDA, ids=lambda p: p.name)
def test_sin_librerias_ni_similitudes_prohibidas(ruta):
    assert violaciones(ruta.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("ruta", NUCLEO, ids=lambda p: p.name)
def test_nucleo_sin_funciones_nativas(ruta):
    assert nativas(ruta.read_text(encoding="utf-8")) == []


@pytest.mark.parametrize("ruta", BUSQUEDA, ids=lambda p: p.name)
def test_busqueda_sin_operador_in(ruta):
    assert usos_de_in(ruta.read_text(encoding="utf-8")) == []


# --- La guardia debe detectar código que sí infringe la regla ---------------

@pytest.mark.parametrize(
    "codigo, esperado",
    [
        ("import sklearn\n", "importa sklearn"),
        ("from sklearn.metrics.pairwise import cosine_similarity\n", "importa sklearn"),
        ("from scipy.spatial import distance\n", "importa scipy"),
        ("import numpy as np\n", "importa numpy"),
        ("def f():\n    import sklearn.cluster\n", "importa sklearn"),
        ("s = x.similarity(y)\n", "llama a similarity"),
        ("s = model.wv.n_similarity(a, b)\n", "llama a n_similarity"),
        ("s = util.cos_sim(a, b)\n", "llama a cos_sim"),
        ("d = kv.wmdistance(a, b)\n", "llama a wmdistance"),
        ("s = cos_sim(a, b)\n", "llama a cos_sim"),  # llamada directa, sin objeto
    ],
)
def test_la_guardia_detecta_codigo_prohibido(codigo, esperado):
    encontradas = violaciones(codigo)
    assert any(esperado in e for e in encontradas), encontradas


@pytest.mark.parametrize(
    "codigo, esperado",
    [
        ("x = min(a, b)\n", "usa min"),
        ("x = texto.lower()\n", "usa lower"),
        ("x = a.intersection(b)\n", "usa intersection"),
        ("x = texto.find('a')\n", "usa find"),
        ("import math\n", "importa math"),
    ],
)
def test_la_guardia_detecta_funciones_nativas(codigo, esperado):
    encontradas = nativas(codigo)
    assert any(esperado in e for e in encontradas), encontradas


def test_la_guardia_detecta_el_operador_in():
    assert usos_de_in("if patron in texto:\n    pass\n") == [1]
    assert usos_de_in("for c in texto:\n    pass\n") == []


def test_la_guardia_no_da_falsos_positivos():
    limpio = (
        '"""Menciona sklearn y x.similarity(y) solo en el docstring."""\n'
        "# util.cos_sim(a, b) en un comentario\n"
        "from src.similarity.metrics import cosine_similarity\n"
        "def f(a, b):\n"
        "    return cosine_similarity(a, b)\n"
    )
    assert violaciones(limpio) == []
    assert nativas(limpio) == []

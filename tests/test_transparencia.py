"""
Guardia de la regla de transparencia algorítmica: el código de Daniel
(src/similarity/ai, src/similarity/metrics.py, src/clustering) no puede importar
librerías que resuelvan los algoritmos ni llamar a las similitudes que traen los
modelos de embeddings. Se revisa el AST, así que los comentarios y docstrings
que mencionan esos nombres no cuentan.
"""
import ast
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent / "src"
ARCHIVOS = sorted(
    list((RAIZ / "similarity" / "ai").glob("*.py"))
    + list((RAIZ / "clustering").glob("*.py"))
    + [RAIZ / "similarity" / "metrics.py"]
)

MODULOS_PROHIBIDOS = {"sklearn", "scipy", "numpy", "difflib", "Levenshtein", "rapidfuzz",
                      "textdistance", "jellyfish"}
LLAMADAS_PROHIBIDAS = {"cos_sim", "similarity", "n_similarity", "wmdistance",
                       "most_similar", "pairwise_distances", "cosine_similarity_matrix"}


def _importados(arbol):
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Import):
            for alias in nodo.names:
                yield alias.name.split(".")[0]
        elif isinstance(nodo, ast.ImportFrom) and nodo.module:
            yield nodo.module.split(".")[0]


def violaciones(codigo: str) -> list[str]:
    """Devuelve la lista de infracciones de transparencia que hay en `codigo`."""
    arbol = ast.parse(codigo)
    encontradas = []
    for modulo in sorted(set(_importados(arbol)) & MODULOS_PROHIBIDOS):
        encontradas.append(f"importa {modulo}")
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call):
            f = nodo.func
            nombre = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if nombre in LLAMADAS_PROHIBIDAS:
                encontradas.append(f"línea {nodo.lineno}: llama a {nombre}")
    return encontradas


def test_hay_archivos_que_revisar():
    assert len(ARCHIVOS) >= 5


@pytest.mark.parametrize("ruta", ARCHIVOS, ids=lambda p: p.name)
def test_sin_librerias_ni_similitudes_prohibidas(ruta):
    assert violaciones(ruta.read_text(encoding="utf-8")) == []


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


def test_la_guardia_no_da_falsos_positivos():
    limpio = (
        '"""Menciona sklearn y x.similarity(y) solo en el docstring."""\n'
        "# util.cos_sim(a, b) en un comentario\n"
        "from src.similarity.metrics import cosine_similarity\n"
        "def f(a, b):\n"
        "    return cosine_similarity(a, b)\n"
    )
    assert violaciones(limpio) == []

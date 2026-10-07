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


def test_hay_archivos_que_revisar():
    assert len(ARCHIVOS) >= 5


@pytest.mark.parametrize("ruta", ARCHIVOS, ids=lambda p: p.name)
def test_sin_librerias_ni_similitudes_prohibidas(ruta):
    arbol = ast.parse(ruta.read_text(encoding="utf-8"))
    assert not (set(_importados(arbol)) & MODULOS_PROHIBIDOS)
    for nodo in ast.walk(arbol):
        if isinstance(nodo, ast.Call):
            f = nodo.func
            nombre = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            assert nombre not in LLAMADAS_PROHIBIDAS, f"{ruta.name}:{nodo.lineno} llama a {nombre}"

"""
Exportación de las fusiones del clustering a JSON (para el dendrograma de Cayón).
--------------------------------------------------------------------------------
Se calcula el clustering jerárquico con cada enlace (single, complete, average)
a partir de una misma matriz de distancias y se guarda todo en un único JSON,
usando solo el módulo estándar `json`.

FORMATO DEL ARCHIVO (versión 1)
===============================

    {
      "version": 1,
      "n": 8,
      "labels": ["V1", "V2", "L1", ...],
      "ids": "hojas 0..n-1 (posición en 'labels'); la fusión k (desde 0) crea el cluster n+k",
      "linkages": {
        "single":   { "cpcc": 0.671, "merges": [[0, 1, 0.001, 2], [2, 3, 0.001, 2], ...] },
        "complete": { "cpcc": 0.749, "merges": [...] },
        "average":  { "cpcc": 0.751, "merges": [...] }
      }
    }

- `n`: número de abstracts (hojas del dendrograma).
- `labels`: etiqueta de cada abstract; `labels[i]` es la hoja con id `i`.
- `linkages[enlace]["merges"]`: lista de n-1 fusiones en orden de ejecución. Cada
  fusión es `[a, b, distancia, tamaño]`:
    * `a`, `b` (enteros, a < b): ids de los dos clusters que se unen. Un id < n es
      una hoja original; un id >= n es el cluster creado en la fusión k = id - n.
    * `distancia` (número): altura a la que se unen (eje Y del dendrograma).
      Es no decreciente a lo largo de la lista (no hay inversiones).
    * `tamaño` (entero): cantidad de hojas del cluster resultante.
  Es el mismo formato de la "linkage matrix" clásica, pero ya es JSON plano:
  Cayón puede dibujar con matplotlib recorriendo la lista, sin necesidad de scipy.
- `linkages[enlace]["cpcc"]`: correlación cofenética de ese enlace (para rotular
  o comparar los tres dendrogramas).

Cómo leerlo:

    import json
    datos = json.load(open("fusiones.json", encoding="utf-8"))
    etiquetas = datos["labels"]
    for a, b, altura, tam in datos["linkages"]["average"]["merges"]:
        ...

`load_export` lee el archivo y valida su estructura.
"""
import json

from src.clustering.evaluation import cophenetic_correlation
from src.clustering.linkage import LINKAGES, hierarchical_clustering

FORMAT_VERSION = 1
IDS_DESCRIPTION = (
    "hojas 0..n-1 (posición en 'labels'); la fusión k (desde 0) crea el cluster n+k"
)


def build_export(
    dist: list[list[float]], labels: list[str], linkages: tuple[str, ...] = LINKAGES
) -> dict:
    """
    Construye el diccionario que se guarda en JSON (ver formato en el docstring
    del módulo). Calcula las fusiones y el CPCC de cada enlace.

    Complejidad temporal: O(L * n^3), con L enlaces: el clustering de cada uno es
    O(n^3) y domina sobre el CPCC (O(n^2)).
    Complejidad espacial: O(L * n + n^2): L listas de n-1 fusiones y la matriz
    cofenética temporal.
    """
    n = len(dist)
    if len(labels) != n:
        raise ValueError(f"Se esperaban {n} etiquetas y llegaron {len(labels)}")

    result = {}
    for linkage in linkages:
        merges = hierarchical_clustering(dist, linkage)
        result[linkage] = {
            "cpcc": cophenetic_correlation(dist, merges)["cpcc"],
            "merges": [[a, b, h, size] for a, b, h, size in merges],
        }

    return {
        "version": FORMAT_VERSION,
        "n": n,
        "labels": list(labels),
        "ids": IDS_DESCRIPTION,
        "linkages": result,
    }


def export_merges_json(
    path: str,
    dist: list[list[float]],
    labels: list[str],
    linkages: tuple[str, ...] = LINKAGES,
) -> dict:
    """
    Calcula las fusiones de cada enlace y las guarda en `path` como JSON UTF-8
    (formato en el docstring del módulo). Devuelve el diccionario escrito.

    Se rechazan NaN/Infinity (`allow_nan=False`) porque no son JSON válido.
    Complejidad: la de `build_export` más O(L * n) para escribir el archivo.
    """
    data = build_export(dist, labels, linkages)
    # se serializa antes de abrir el archivo: si hay NaN no queda un JSON a medias
    text = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return data


def load_export(path: str) -> dict:
    """
    Lee un archivo exportado y valida su estructura: versión, n y etiquetas
    coherentes, y para cada enlace n-1 fusiones `[a, b, distancia, tamaño]` con
    ids válidos y alturas no decrecientes. Lanza ValueError si algo no cuadra.

    Complejidad temporal: O(L * n). Complejidad espacial: O(L * n).
    """
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    if data.get("version") != FORMAT_VERSION:
        raise ValueError(f"Versión no soportada: {data.get('version')}")
    n = data["n"]
    if len(data["labels"]) != n:
        raise ValueError("labels no coincide con n")

    for name, info in data["linkages"].items():
        merges = info["merges"]
        if len(merges) != n - 1:
            raise ValueError(f"{name}: se esperaban {n - 1} fusiones")
        alive = set(range(n))
        previous = None
        for k, merge in enumerate(merges):
            if len(merge) != 4:
                raise ValueError(f"{name}: la fusión {k} no tiene 4 campos")
            a, b, height, size = merge
            if a not in alive or b not in alive or a == b:
                raise ValueError(f"{name}: la fusión {k} usa ids no válidos")
            if previous is not None and height < previous:
                raise ValueError(f"{name}: alturas decrecientes en la fusión {k}")
            alive.discard(a)
            alive.discard(b)
            alive.add(n + k)
            previous = height
    return data

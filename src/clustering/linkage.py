"""
Clustering jerárquico aglomerativo con 3 criterios de enlace (Req. 2, Daniel).
------------------------------------------------------------------------------
Implementación desde cero: Single, Complete y Average (UPGMA). Entrada: matriz
de distancias simétrica (lista de listas), p. ej. de `pairwise_matrix`.

Identificadores de cluster (convención tipo "linkage matrix"):
    - los n elementos originales son 0 .. n-1
    - la k-ésima fusión (k = 0, 1, ...) crea el cluster con id n + k

Cada fusión es una tupla `(a, b, distancia, tamaño)` con a < b: los ids de los
dos clusters unidos, la distancia de enlace a la que se unen y el número de
elementos del cluster resultante. Con esa lista Cayón dibuja el dendrograma.

Criterios de enlace, para clusters A y B con distancia d entre elementos:
    single   : D(A, B) = min_{x in A, y in B} d(x, y)
    complete : D(A, B) = max_{x in A, y in B} d(x, y)
    average  : D(A, B) = (1 / |A||B|) * sum_{x in A, y in B} d(x, y)

En lugar de recalcular D sobre todos los pares tras cada fusión, se actualiza
con la fórmula de Lance-Williams. Al unir A y B en M, para cualquier otro C:
    single   : D(M, C) = min(D(A, C), D(B, C))
    complete : D(M, C) = max(D(A, C), D(B, C))
    average  : D(M, C) = (|A| D(A, C) + |B| D(B, C)) / (|A| + |B|)
Para average esta actualización es exactamente igual al promedio sobre todos
los pares: sum_{M x C} d = sum_{A x C} d + sum_{B x C} d.

Desempates: ante igual distancia mínima se fusiona el par con ids menores
(orden lexicográfico (a, b)), de modo que el resultado es determinista.
"""

LINKAGES = ("single", "complete", "average")


def _validate_matrix(dist: list[list[float]]) -> int:
    """Comprueba que la matriz sea cuadrada, simétrica y con diagonal cero."""
    n = len(dist)
    if n < 2:
        raise ValueError("Se requieren al menos 2 elementos para hacer clustering")
    for i in range(n):
        if len(dist[i]) != n:
            raise ValueError("La matriz de distancias debe ser cuadrada")
    for i in range(n):
        if abs(dist[i][i]) > 1e-9:
            raise ValueError(f"La diagonal debe ser 0 (dist[{i}][{i}] = {dist[i][i]})")
        for j in range(i + 1, n):
            if abs(dist[i][j] - dist[j][i]) > 1e-9:
                raise ValueError(f"La matriz no es simétrica en ({i}, {j})")
    return n


def _update_distance(linkage: str, d_ac: float, d_bc: float, size_a: int, size_b: int) -> float:
    """Fórmula de Lance-Williams para cada criterio. Complejidad O(1)."""
    if linkage == "single":
        return d_ac if d_ac < d_bc else d_bc
    if linkage == "complete":
        return d_ac if d_ac > d_bc else d_bc
    return (size_a * d_ac + size_b * d_bc) / (size_a + size_b)


def hierarchical_clustering(
    dist: list[list[float]], linkage: str = "average", return_steps: bool = False
):
    """
    Clustering jerárquico aglomerativo.

    Algoritmo:
        1. Cada elemento es un cluster (ids 0..n-1) y D(i, j) = dist[i][j].
        2. Mientras haya más de un cluster activo:
           a. buscar el par (a, b) de clusters activos con D mínima;
           b. crear el cluster n+k = a ∪ b, con tamaño |a| + |b|;
           c. calcular D(n+k, c) para todo c activo con Lance-Williams;
           d. desactivar a y b; registrar la fusión (a, b, D(a, b), tamaño).

    Args:
        dist: matriz de distancias n x n (simétrica, diagonal 0).
        linkage: "single", "complete" o "average".
        return_steps: si es True devuelve además el historial paso a paso.

    Returns:
        merges: lista de n-1 tuplas (a, b, distancia, tamaño).
        Si return_steps: (merges, steps), donde cada paso es un dict con
        `step`, `merged` (a, b), `new_id`, `distance`, `size`, `active_before`
        (ids activos antes de fusionar) y `distances_to_new` ({c: D(new, c)}).

    Complejidad temporal: O(n^3). Hay n-1 iteraciones; en cada una la búsqueda
    del mínimo recorre O(k^2) pares de los k clusters activos y la actualización
    cuesta O(k). Sumando k = n..2: sum k^2 = O(n^3).
    Complejidad espacial: O(n^2) por la tabla de distancias entre clusters
    (2n-1 ids como máximo, pero solo se conservan las filas de los activos).
    """
    if linkage not in LINKAGES:
        raise ValueError(f"Enlace desconocido '{linkage}'; usar uno de {LINKAGES}")
    n = _validate_matrix(dist)

    # D[(i, j)] con i < j: distancia entre clusters activos
    D = {}
    for i in range(n):
        for j in range(i + 1, n):
            D[(i, j)] = dist[i][j]
    sizes = {i: 1 for i in range(n)}
    active = list(range(n))

    merges = []
    steps = []
    for k in range(n - 1):
        # a. par de distancia mínima (empates: menor (a, b) por ids)
        best_pair = None
        best_d = None
        for x in range(len(active)):
            for y in range(x + 1, len(active)):
                pair = (active[x], active[y])
                d = D[pair]
                if best_d is None or d < best_d:
                    best_d = d
                    best_pair = pair
        a, b = best_pair
        new_id = n + k
        size_a, size_b = sizes[a], sizes[b]

        # c. Lance-Williams contra los demás clusters activos
        active_before = list(active)
        distances_to_new = {}
        for c in active:
            if c == a or c == b:
                continue
            d_ac = D[(a, c) if a < c else (c, a)]
            d_bc = D[(b, c) if b < c else (c, b)]
            nd = _update_distance(linkage, d_ac, d_bc, size_a, size_b)
            D[(c, new_id)] = nd  # c < new_id siempre
            distances_to_new[c] = nd

        # d. actualizar estado
        active = [c for c in active if c != a and c != b] + [new_id]
        sizes[new_id] = size_a + size_b
        merges.append((a, b, best_d, size_a + size_b))
        steps.append(
            {
                "step": k + 1,
                "merged": (a, b),
                "new_id": new_id,
                "distance": best_d,
                "size": size_a + size_b,
                "active_before": active_before,
                "distances_to_new": distances_to_new,
            }
        )

    if return_steps:
        return merges, steps
    return merges

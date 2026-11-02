"""
Métricas de evaluación interna del clustering jerárquico (Req. 2, Daniel).
-------------------------------------------------------------------------
Implementadas a mano, sin numpy/scipy/sklearn:

    - Matriz cofenética y coeficiente de correlación cofenética (CPCC).
    - Corte del dendrograma en k clusters.
    - Coeficiente de silueta.

Las fusiones tienen el formato de `linkage.hierarchical_clustering`:
`(a, b, distancia, tamaño)`, con ids 0..n-1 para los elementos y n+k para el
cluster creado en la k-ésima fusión.
"""


def _members_after_merges(merges: list[tuple], n: int, upto: int) -> dict[int, list[int]]:
    """Aplica las primeras `upto` fusiones y devuelve {id_cluster_activo: miembros}."""
    members = {i: [i] for i in range(n)}
    for k in range(upto):
        a, b = merges[k][0], merges[k][1]
        members[n + k] = members.pop(a) + members.pop(b)
    return members


def cophenetic_matrix(merges: list[tuple], n: int) -> list[list[float]]:
    """
    Matriz cofenética C (n x n): C[x][y] es la distancia de enlace a la que
    x e y quedan por primera vez en el mismo cluster. Para cada fusión
    (a, b, h, _), todo par (x en a, y en b) recibe C[x][y] = h.

    Complejidad temporal: O(n^2): cada par (x, y) se asigna exactamente una vez
    (en la fusión que reúne a x e y) y el manejo de las listas de miembros suma
    O(n) por fusión.
    Complejidad espacial: O(n^2) por la matriz.
    """
    if len(merges) != n - 1:
        raise ValueError(f"Se esperaban {n - 1} fusiones y llegaron {len(merges)}")
    coph = [[0.0] * n for _ in range(n)]
    members = {i: [i] for i in range(n)}
    for k, (a, b, h, _) in enumerate(merges):
        for x in members[a]:
            for y in members[b]:
                coph[x][y] = h
                coph[y][x] = h
        members[n + k] = members.pop(a) + members.pop(b)
    return coph


def pearson(xs: list[float], ys: list[float]) -> float:
    """
    Correlación de Pearson: r = sum (x - x̄)(y - ȳ) / sqrt(sum (x - x̄)^2 * sum (y - ȳ)^2).
    Si alguna serie es constante (varianza 0) devuelve 0.0.

    Complejidad temporal: O(m) (cuatro recorridos de longitud m). Espacial: O(1).
    """
    if len(xs) != len(ys):
        raise ValueError("Las series deben tener la misma longitud")
    m = len(xs)
    if m == 0:
        raise ValueError("Series vacías")
    mean_x = 0.0
    mean_y = 0.0
    for i in range(m):
        mean_x += xs[i]
        mean_y += ys[i]
    mean_x /= m
    mean_y /= m
    cov = 0.0
    var_x = 0.0
    var_y = 0.0
    for i in range(m):
        dx = xs[i] - mean_x
        dy = ys[i] - mean_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    if var_x == 0.0 or var_y == 0.0:
        return 0.0
    return cov / ((var_x * var_y) ** 0.5)


def cophenetic_correlation(
    dist: list[list[float]], merges: list[tuple]
) -> dict:
    """
    Coeficiente de correlación cofenética (CPCC):
        c = Pearson( d(x, y), C(x, y) ) sobre todos los pares x < y,
    donde d es la distancia original y C la cofenética. Mide qué tan fielmente
    el dendrograma conserva las distancias originales: más cerca de 1 es mejor.

    Returns:
        dict con `cpcc`, `original` y `cophenetic` (las dos listas de pares,
        para poder mostrar la demostración).

    Complejidad temporal: O(n^2) (matriz cofenética + Pearson sobre n(n-1)/2 pares).
    Complejidad espacial: O(n^2).
    """
    n = len(dist)
    coph = cophenetic_matrix(merges, n)
    original = []
    cophenetic = []
    for i in range(n):
        for j in range(i + 1, n):
            original.append(dist[i][j])
            cophenetic.append(coph[i][j])
    return {
        "cpcc": pearson(original, cophenetic),
        "original": original,
        "cophenetic": cophenetic,
    }


def cut_tree(merges: list[tuple], n: int, k: int) -> list[int]:
    """
    Corta el dendrograma para obtener k clusters: se aplican las primeras
    n - k fusiones. Devuelve una etiqueta 0..k-1 por elemento (numeradas por
    el menor índice de elemento de cada cluster).

    Complejidad temporal: O(n^2) en el peor caso, por la concatenación de listas
    de miembros al aplicar las n-k fusiones; numerar los clusters cuesta O(n).
    Espacial: O(n).
    """
    if not 1 <= k <= n:
        raise ValueError(f"k debe estar entre 1 y {n}")
    members = _members_after_merges(merges, n, n - k)

    cluster_de = [0] * n
    for cluster in members:
        for x in members[cluster]:
            cluster_de[x] = cluster

    # Al recorrer los elementos en orden, cada cluster recibe su etiqueta cuando
    # aparece su menor elemento.
    etiquetas = {}
    labels = [0] * n
    for x in range(n):
        cluster = cluster_de[x]
        if cluster not in etiquetas:
            etiquetas[cluster] = len(etiquetas)
        labels[x] = etiquetas[cluster]
    return labels


def silhouette(dist: list[list[float]], labels: list[int]) -> dict:
    """
    Coeficiente de silueta sobre una matriz de distancias.

    Para cada elemento i:
        a(i) = distancia media a los demás de su cluster
        b(i) = min sobre los otros clusters de la distancia media de i a ese cluster
        s(i) = (b - a) / max(a, b)       (s = 0 si el cluster es unitario o max = 0)
    El valor global es la media de s(i). Rango [-1, 1]; más alto es mejor.
    Requiere al menos 2 clusters.

    Returns:
        dict con `silhouette` (media) y `per_item` (s(i) de cada elemento).

    Complejidad temporal: O(n^2): para cada i se recorren sus n-1 distancias.
    Complejidad espacial: O(n) (acumuladores por cluster y la lista de s(i)).
    """
    n = len(dist)
    if len(labels) != n:
        raise ValueError("labels debe tener un valor por elemento")
    sizes = {}
    for lab in labels:
        if lab in sizes:
            sizes[lab] += 1
        else:
            sizes[lab] = 1
    cluster_ids = list(sizes)
    if len(cluster_ids) < 2:
        raise ValueError("La silueta requiere al menos 2 clusters")

    per_item = []
    for i in range(n):
        sums = {c: 0.0 for c in cluster_ids}
        for j in range(n):
            if j != i:
                sums[labels[j]] += dist[i][j]
        own = labels[i]
        if sizes[own] == 1:
            per_item.append(0.0)
            continue
        a = sums[own] / (sizes[own] - 1)
        b = None
        for c in cluster_ids:
            if c != own:
                mean_c = sums[c] / sizes[c]
                if b is None or mean_c < b:
                    b = mean_c
        denom = a if a > b else b
        per_item.append(0.0 if denom == 0 else (b - a) / denom)

    total = 0.0
    for s in per_item:
        total += s
    return {"silhouette": total / n, "per_item": per_item}


def best_k_by_silhouette(
    dist: list[list[float]], merges: list[tuple], k_min: int = 2, k_max: int | None = None
) -> dict:
    """
    Evalúa la silueta para cada corte k = k_min..k_max y devuelve el mejor.

    Returns:
        dict con `best_k`, `best_silhouette` y `by_k` ({k: silueta}).

    Complejidad temporal: O((k_max - k_min + 1) * n^2).
    Complejidad espacial: O(n) más el diccionario de resultados.
    """
    n = len(dist)
    if k_max is None:
        k_max = n - 1
    by_k = {}
    best_k = None
    for k in range(k_min, k_max + 1):
        labels = cut_tree(merges, n, k)
        by_k[k] = silhouette(dist, labels)["silhouette"]
        if best_k is None or by_k[k] > by_k[best_k]:
            best_k = k
    return {"best_k": best_k, "best_silhouette": by_k[best_k], "by_k": by_k}

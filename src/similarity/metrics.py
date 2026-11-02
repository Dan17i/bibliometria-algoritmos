"""
Métricas de similitud y distancia entre vectores, implementadas a mano.
-----------------------------------------------------------------------
Se usan para comparar los embeddings (Req. 1 IA) y para construir la matriz de
distancias del clustering jerárquico (Req. 2). Solo se emplean listas, bucles y
aritmética básica: nada de numpy, scipy ni sklearn.

Los vectores son `list[float]` (si un modelo devuelve arrays, convertir antes
con `.tolist()`).

Cada métrica tiene dos niveles:
    - `*_steps(a, b)`: devuelve un diccionario con los pasos intermedios, para
      poder reproducir la demostración paso a paso en el documento.
    - función simple (`cosine_similarity`, ...): devuelve solo el valor final.

Convención con el vector cero: la similitud coseno no está definida (norma 0).
Aquí se devuelve 0.0 (sin relación) y la distancia coseno 1.0, igual que el
coseno clásico del equipo. Dimensiones distintas lanzan `ValueError`.
"""


def _check_dimensions(a: list[float], b: list[float]) -> int:
    """Valida que ambos vectores tengan la misma dimensión y devuelve n."""
    if len(a) != len(b):
        raise ValueError(
            f"Los vectores deben tener la misma dimensión ({len(a)} != {len(b)})"
        )
    return len(a)


def dot_product(a: list[float], b: list[float]) -> float:
    """
    Producto punto: A · B = sum_{i=1..n} A_i * B_i

    Complejidad temporal: O(n), un recorrido con una multiplicación y una suma
    por componente.
    Complejidad espacial: O(1), solo un acumulador.
    """
    n = _check_dimensions(a, b)
    total = 0.0
    for i in range(n):
        total += a[i] * b[i]
    return total


def norm(a: list[float]) -> float:
    """
    Norma euclidiana (L2): ||A|| = sqrt(sum_{i=1..n} A_i^2)

    Complejidad temporal: O(n), un recorrido acumulando cuadrados.
    Complejidad espacial: O(1).
    """
    suma_cuadrados = 0.0
    for x in a:
        suma_cuadrados += x * x
    return suma_cuadrados ** 0.5


def cosine_steps(a: list[float], b: list[float]) -> dict:
    """
    Similitud coseno con todos los pasos intermedios.

    Fórmula: cos(θ) = (A · B) / (||A|| * ||B||)

    Pasos:
        1. producto punto A · B
        2. normas ||A|| y ||B||
        3. división; si alguna norma es 0 el coseno queda en 0.0 por convención
        4. el valor se acota a [-1, 1] para absorber el error de punto flotante
        5. ángulo θ = arccos(cos) en grados (solo informativo, ver `angle_degrees`)

    Returns:
        dict con: dot, norm_a, norm_b, cosine, distance (1 - cos), angle_degrees.

    Complejidad temporal: O(n) (tres recorridos independientes de longitud n).
    Complejidad espacial: O(1) (solo escalares; el dict tiene tamaño constante).
    """
    _check_dimensions(a, b)
    dot = dot_product(a, b)
    norm_a = norm(a)
    norm_b = norm(b)

    if norm_a == 0.0 or norm_b == 0.0:
        cosine = 0.0
    else:
        cosine = dot / (norm_a * norm_b)
        if cosine > 1.0:
            cosine = 1.0
        elif cosine < -1.0:
            cosine = -1.0

    return {
        "dot": dot,
        "norm_a": norm_a,
        "norm_b": norm_b,
        "cosine": cosine,
        "distance": 1.0 - cosine,
        "angle_degrees": angle_degrees(cosine),
    }


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """
    Similitud coseno entre dos vectores: (A · B) / (||A|| * ||B||).

    Rango [-1, 1]; 1 = misma dirección, 0 = ortogonales, -1 = opuestos.
    Complejidad temporal: O(n). Complejidad espacial: O(1).
    """
    return cosine_steps(a, b)["cosine"]


def cosine_distance(a: list[float], b: list[float]) -> float:
    """
    Distancia coseno: d(A, B) = 1 - cos(θ). Rango [0, 2].

    Se usa como disimilitud para el clustering. Con embeddings de signo mixto
    puede superar 1; con TF/TF-IDF (componentes >= 0) queda en [0, 1].
    Complejidad temporal: O(n). Complejidad espacial: O(1).
    """
    return cosine_steps(a, b)["distance"]


def euclidean_steps(a: list[float], b: list[float]) -> dict:
    """
    Distancia euclidiana con los pasos intermedios.

    Fórmula: d(A, B) = sqrt( sum_{i=1..n} (A_i - B_i)^2 )

    Returns:
        dict con: diffs (A_i - B_i), squared_diffs, sum_squares, distance.

    Complejidad temporal: O(n), un recorrido que calcula la diferencia y su cuadrado.
    Complejidad espacial: O(n), se guardan las listas de diferencias para poder
    mostrarlas (la versión sin pasos, `euclidean_distance`, usa O(1)).
    """
    n = _check_dimensions(a, b)
    diffs = []
    squared = []
    sum_squares = 0.0
    for i in range(n):
        d = a[i] - b[i]
        diffs.append(d)
        squared.append(d * d)
        sum_squares += d * d

    return {
        "diffs": diffs,
        "squared_diffs": squared,
        "sum_squares": sum_squares,
        "distance": sum_squares ** 0.5,
    }


def euclidean_distance(a: list[float], b: list[float]) -> float:
    """
    Distancia euclidiana: sqrt(sum (A_i - B_i)^2). Rango [0, inf).

    Complejidad temporal: O(n). Complejidad espacial: O(1) (sin listas auxiliares).
    """
    n = _check_dimensions(a, b)
    sum_squares = 0.0
    for i in range(n):
        d = a[i] - b[i]
        sum_squares += d * d
    return sum_squares ** 0.5


def pairwise_matrix(vectors: list[list[float]], metric) -> list[list[float]]:
    """
    Matriz simétrica M (m x m) con M[i][j] = metric(V_i, V_j).

    `metric` es una función de dos vectores (p. ej. `cosine_distance` o
    `euclidean_distance`). Como las métricas son simétricas se evalúa solo el
    triángulo superior y se copia; la diagonal es metric(V_i, V_i).
    Es la matriz de distancias de entrada del clustering jerárquico.

    Complejidad temporal: O(m^2 * n) (m(m+1)/2 evaluaciones, cada una O(n)).
    Complejidad espacial: O(m^2) por la matriz resultado.
    """
    m = len(vectors)
    matrix = [[0.0] * m for _ in range(m)]
    for i in range(m):
        for j in range(i, m):
            value = metric(vectors[i], vectors[j])
            matrix[i][j] = value
            matrix[j][i] = value
    return matrix


def mean_vector(vectors: list[list[float]]) -> list[float]:
    """
    Promedio componente a componente de k vectores de dimensión n:
        m_j = (1/k) * sum_{i=1..k} V_i[j]

    Sirve para obtener el embedding de un documento como promedio de los
    vectores de sus palabras.

    Complejidad temporal: O(k * n). Complejidad espacial: O(n) (el vector resultado).
    """
    if not vectors:
        raise ValueError("Se requiere al menos un vector para promediar")
    n = len(vectors[0])
    sums = [0.0] * n
    for v in vectors:
        if len(v) != n:
            raise ValueError(
                f"Todos los vectores deben tener la misma dimensión ({len(v)} != {n})"
            )
        for j in range(n):
            sums[j] += v[j]
    k = len(vectors)
    return [s / k for s in sums]


def center_vectors(vectors: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """
    Centra un conjunto de vectores restando el vector medio del conjunto:
        V'_i = V_i - m,   con   m = (1/k) * sum_i V_i

    Quita la dirección común a todos los vectores (por ejemplo, la que domina en
    los promedios de GloVe), de modo que el coseno posterior refleje solo lo que
    distingue a cada vector. Ojo: el vector medio depende del conjunto, así que
    los valores solo son comparables dentro del mismo conjunto.

    Returns:
        (vectores centrados, vector medio).

    Complejidad temporal: O(k * n) (el promedio y la resta recorren k vectores de
    dimensión n). Complejidad espacial: O(k * n) por los vectores centrados.
    """
    m = mean_vector(vectors)
    n = len(m)
    centered = []
    for v in vectors:
        centered.append([v[j] - m[j] for j in range(n)])
    return centered, m


def natural_log(x: float) -> float:
    """
    Logaritmo natural a mano, para x > 0. Se usa en el IDF.

    Se reduce x a m en [0.75, 1.5) dividiendo o multiplicando por 2, de modo que
    ln(x) = k * ln(2) + ln(m), y ln(m) se obtiene con la serie
        ln(m) = 2 * sum_{i>=0} y^(2i+1) / (2i+1),   y = (m - 1) / (m + 1)
    que con |y| <= 1/5 converge en pocos términos.

    Complejidad temporal: O(log x) por la reducción, más 30 términos fijos.
    Complejidad espacial: O(1).
    """
    if x <= 0:
        raise ValueError("El logaritmo solo está definido para x > 0")

    ln2 = 0.6931471805599453
    k = 0
    m = x
    while m >= 1.5:
        m /= 2
        k += 1
    while m < 0.75:
        m *= 2
        k -= 1

    y = (m - 1) / (m + 1)
    potencia = y
    total = 0.0
    for i in range(30):
        total += potencia / (2 * i + 1)
        potencia *= y * y
    return k * ln2 + 2 * total


def angle_degrees(cosine: float) -> float:
    """
    Ángulo en grados a partir de un coseno: θ = arccos(cos) * 180 / π.

    arccos se obtiene a mano: se resuelve cos(θ) = c por bisección en [0, π],
    evaluando cos con su serie de Taylor. Basta para informar el ángulo en la
    demostración (precisión ~1e-12).

    Complejidad temporal: O(1) (número fijo de iteraciones, 60).
    Complejidad espacial: O(1).
    """
    if cosine >= 1.0:
        return 0.0
    if cosine <= -1.0:
        return 180.0

    pi = 3.141592653589793

    def cos_taylor(x: float) -> float:
        # cos(x) = sum (-1)^k x^(2k) / (2k)!, con 25 términos converge en [0, π].
        term = 1.0
        total = 1.0
        for k in range(1, 25):
            term *= -x * x / ((2 * k - 1) * (2 * k))
            total += term
        return total

    low, high = 0.0, pi
    for _ in range(60):
        mid = (low + high) / 2
        if cos_taylor(mid) > cosine:
            low = mid
        else:
            high = mid
    return (low + high) / 2 * 180.0 / pi

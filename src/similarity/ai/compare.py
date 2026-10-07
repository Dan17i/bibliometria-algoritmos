"""
Comparación de abstracts mediante embeddings (Req. 1 IA).
---------------------------------------------------------
Dados los embeddings de 2 o más artículos, calcula para cada par la similitud
coseno, el ángulo y la distancia euclidiana, todo con `src.similarity.metrics`.
"""
from src.similarity.metrics import cosine_steps, euclidean_steps


def compare_embeddings(
    names: list[str], vectors: list[list[float]]
) -> list[dict]:
    """
    Compara todos los pares (i < j) de embeddings.

    Returns:
        lista de dicts con: a, b, cosine, angle_degrees, euclidean, y los pasos
        completos en `cosine_steps` y `euclidean_steps` para la demostración.

    Complejidad temporal: O(m^2 * n) (m(m-1)/2 pares, cada uno O(n)).
    Complejidad espacial: O(m^2 * n) si se conservan los pasos de la euclidiana
    (cada par guarda dos listas de longitud n).
    """
    if len(names) != len(vectors):
        raise ValueError("names y vectors deben tener la misma longitud")
    if len(vectors) < 2:
        raise ValueError("Se requieren al menos 2 artículos para comparar")

    results = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            cos = cosine_steps(vectors[i], vectors[j])
            euc = euclidean_steps(vectors[i], vectors[j])
            results.append(
                {
                    "a": names[i],
                    "b": names[j],
                    "cosine": cos["cosine"],
                    "angle_degrees": cos["angle_degrees"],
                    "euclidean": euc["distance"],
                    "cosine_steps": cos,
                    "euclidean_steps": euc,
                }
            )
    return results

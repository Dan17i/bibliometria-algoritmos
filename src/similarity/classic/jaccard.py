"""
Módulo de Similitud de Jaccard
------------------------------
Este módulo implementa el Índice de Similitud de Jaccard desde cero.

Contexto Teórico:
A diferencia de los algoritmos de Programación Dinámica (Levenshtein, Needleman-Wunsch)
que evalúan la similitud basándose en la secuencia estricta y el orden de los caracteres,
el coeficiente de Jaccard evalúa la similitud basándose en la superposición de conjuntos.

En el contexto de análisis de 'abstracts' (textos científicos), este enfoque asume un modelo
de "Bolsa de Palabras" (Bag of Words) binario, donde no importa el orden de las palabras
ni su frecuencia, sino únicamente su presencia o ausencia en ambos textos.

Los conjuntos se representan con listas sin repetidos, y la intersección y la unión se
calculan recorriéndolas con un diccionario de presencia, sin las operaciones nativas de set.
"""

from src.similarity.classic.tokenizador import terminos_unicos, tokenizar


def jaccard_similarity(text1: str, text2: str) -> tuple[float, list[str], list[str]]:
    """
    Calcula el Índice de Similitud de Jaccard entre dos textos.

    Fórmula Matemática:
        J(A, B) = |A ∩ B| / |A ∪ B|
    Donde:
        A = Conjunto de palabras únicas del texto 1
        B = Conjunto de palabras únicas del texto 2
        |A ∩ B| = Cardinalidad de la intersección (palabras comunes)
        |A ∪ B| = Cardinalidad de la unión (total de palabras únicas en ambos textos)

    Complejidad temporal: O(T), con T el total de tokens de los dos textos.
    Complejidad espacial: O(T), por los conjuntos.

    Args:
        text1 (str): Primer texto a comparar.
        text2 (str): Segundo texto a comparar.

    Returns:
        tuple[float, list[str], list[str]]:
            - El valor de similitud (float entre 0.0 y 1.0, donde 1.0 es idéntico).
            - La intersección (para la demostración de análisis matemático).
            - La unión (para la demostración de análisis matemático).
    """
    # 1. Transformación del texto a representación de conjuntos matemáticos
    conjunto_a = terminos_unicos(tokenizar(text1))
    conjunto_b = terminos_unicos(tokenizar(text2))

    # Manejo de caso borde: si ambos conjuntos están vacíos, son idénticos matemáticamente
    if len(conjunto_a) == 0 and len(conjunto_b) == 0:
        return 1.0, [], []

    en_a = {}
    for termino in conjunto_a:
        en_a[termino] = True

    # 2. Intersección: los términos de B que también están en A
    interseccion = []
    for termino in conjunto_b:
        if termino in en_a:
            interseccion.append(termino)

    # 3. Unión: todos los de A más los de B que no estaban
    union = []
    for termino in conjunto_a:
        union.append(termino)
    for termino in conjunto_b:
        if termino not in en_a:
            union.append(termino)

    # 4. Cálculo final del índice
    # La similitud es la proporción de elementos compartidos frente al universo total de elementos.
    similitud = len(interseccion) / len(union)

    return similitud, interseccion, union

if __name__ == "__main__":
    # Prueba del algoritmo para verificar transparencia y corretitud
    abstract_mock_1 = "La inteligencia artificial avanza rápido."
    abstract_mock_2 = "La inteligencia artificial es el futuro rápido."

    sim, inter, un = jaccard_similarity(abstract_mock_1, abstract_mock_2)

    print(f"Texto 1: '{abstract_mock_1}'")
    print(f"Texto 2: '{abstract_mock_2}'\n")
    print(f"Intersección A intersect B ({len(inter)} palabras): {inter}")
    print(f"Unión A union B ({len(un)} palabras): {un}")
    print(f"\nSimilitud de Jaccard: {len(inter)} / {len(un)} = {sim:.4f}")

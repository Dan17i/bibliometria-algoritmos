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
"""

import string

def preprocess_text_to_set(text: str) -> set:
    """
    Función auxiliar para convertir un texto en un conjunto matemático de tokens (palabras).
    
    Pasos de la transparencia algorítmica:
    1. Conversión a minúsculas para evitar que 'El' y 'el' se cuenten como distintas.
    2. Eliminación de signos de puntuación básicos.
    3. Separación por espacios para extraer los tokens.
    4. Conversión a estructura de datos 'Set' (conjunto) para garantizar elementos únicos.
    """
    text_lower = text.lower()
    # Eliminación manual de puntuación (evitando librerías de NLP como NLTK o Spacy)
    for p in string.punctuation:
        text_lower = text_lower.replace(p, "")
    
    tokens = text_lower.split()
    return set(tokens)


def jaccard_similarity(text1: str, text2: str) -> tuple[float, set, set]:
    """
    Calcula el Índice de Similitud de Jaccard entre dos textos.
    
    Fórmula Matemática:
        J(A, B) = |A ∩ B| / |A ∪ B|
    Donde:
        A = Conjunto de palabras únicas del texto 1
        B = Conjunto de palabras únicas del texto 2
        |A ∩ B| = Cardinalidad de la intersección (palabras comunes)
        |A ∪ B| = Cardinalidad de la unión (total de palabras únicas en ambos textos)
        
    Args:
        text1 (str): Primer texto a comparar.
        text2 (str): Segundo texto a comparar.
        
    Returns:
        tuple[float, set, set]:
            - El valor de similitud (float entre 0.0 y 1.0, donde 1.0 es idéntico).
            - El conjunto intersección (para la demostración de análisis matemático).
            - El conjunto unión (para la demostración de análisis matemático).
    """
    # 1. Transformación del texto a representación de conjuntos matemáticos
    set_a = preprocess_text_to_set(text1)
    set_b = preprocess_text_to_set(text2)
    
    # Manejo de caso borde: si ambos conjuntos están vacíos, son idénticos matemáticamente
    if len(set_a) == 0 and len(set_b) == 0:
        return 1.0, set(), set()
        
    # 2. Cálculo explícito de la Intersección (elementos en común)
    interseccion = set_a.intersection(set_b)
    
    # 3. Cálculo explícito de la Unión (elementos totales sin repetir)
    union = set_a.union(set_b)
    
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

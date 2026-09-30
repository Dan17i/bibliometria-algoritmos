"""
Módulo de Similitud del Coseno (Clásico / Frecuencias)
------------------------------------------------------
Este módulo implementa la Similitud del Coseno desde cero, sin el uso de 
librerías de álgebra lineal como NumPy ni herramientas de Machine Learning.

Contexto Teórico:
La similitud del coseno evalúa qué tan "alineados" están dos vectores en un espacio
multidimensional calculando el coseno del ángulo entre ellos. 
En este enfoque clásico de NLP, los vectores se construyen usando la frecuencia de
términos (Term Frequency - TF) de un vocabulario unificado.

Fórmula Matemática:
    cos(θ) = (A · B) / (||A|| * ||B||)
"""

import string

def preprocess_and_tokenize(text: str) -> list[str]:
    """Tokeniza el texto en minúsculas y sin puntuación."""
    text_lower = text.lower()
    for p in string.punctuation:
        text_lower = text_lower.replace(p, "")
    return text_lower.split()

def get_frequency_vectors(text1: str, text2: str) -> tuple[list[int], list[int], list[str]]:
    """
    Convierte dos textos en vectores de frecuencias basados en un vocabulario unificado.
    
    Returns:
        tuple: vector1, vector2, vocabulario_unificado
    """
    tokens1 = preprocess_and_tokenize(text1)
    tokens2 = preprocess_and_tokenize(text2)
    
    # Construcción del vocabulario único (el espacio multidimensional)
    vocabulario = list(set(tokens1).union(set(tokens2)))
    # Ordenar para mantener un índice determinista
    vocabulario.sort()
    
    # Inicialización de vectores con ceros
    vec1 = [0] * len(vocabulario)
    vec2 = [0] * len(vocabulario)
    
    # Conteo de frecuencias (Term Frequency)
    for token in tokens1:
        idx = vocabulario.index(token)
        vec1[idx] += 1
        
    for token in tokens2:
        idx = vocabulario.index(token)
        vec2[idx] += 1
        
    return vec1, vec2, vocabulario

def cosine_similarity(text1: str, text2: str) -> tuple[float, list[int], list[int], list[str]]:
    """
    Calcula la Similitud del Coseno entre dos textos.
    
    Pasos algorítmicos transparentes:
    1. Generar vectores de frecuencia (A y B).
    2. Calcular el Producto Punto (Suma de A[i] * B[i]).
    3. Calcular la magnitud de A y la magnitud de B (Raíz cuadrada de la suma de sus cuadrados).
    4. Dividir producto punto entre el producto de las magnitudes.
    
    Returns:
        tuple[float, list[int], list[int], list[str]]:
            - El valor del coseno (float entre 0.0 y 1.0).
            - El vector de frecuencias A.
            - El vector de frecuencias B.
            - El vocabulario (dimensiones).
    """
    vec_a, vec_b, vocab = get_frequency_vectors(text1, text2)
    
    # 1. Producto Punto (A · B)
    producto_punto = 0
    for i in range(len(vocab)):
        producto_punto += vec_a[i] * vec_b[i]
        
    # 2. Magnitud de A (||A||)
    suma_cuadrados_a = 0
    for val in vec_a:
        suma_cuadrados_a += val ** 2
    magnitud_a = suma_cuadrados_a ** 0.5  # Equivalente a math.sqrt
    
    # 3. Magnitud de B (||B||)
    suma_cuadrados_b = 0
    for val in vec_b:
        suma_cuadrados_b += val ** 2
    magnitud_b = suma_cuadrados_b ** 0.5
    
    # Validación de división por cero (textos completamente vacíos)
    if magnitud_a == 0 or magnitud_b == 0:
        return 0.0, vec_a, vec_b, vocab
        
    # 4. Cálculo final del coseno
    coseno = producto_punto / (magnitud_a * magnitud_b)
    
    return coseno, vec_a, vec_b, vocab

if __name__ == "__main__":
    # Prueba del algoritmo
    text_A = "algoritmos de inteligencia artificial"
    text_B = "algoritmos en la inteligencia artificial"
    
    sim, vA, vB, vocab = cosine_similarity(text_A, text_B)
    
    print("Vocabulario (Dimensiones):", vocab)
    print("Vector A:", vA)
    print("Vector B:", vB)
    print(f"\nSimilitud del Coseno: {sim:.4f}")

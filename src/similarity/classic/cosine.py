"""
Módulo de Similitud del Coseno (Clásico / Frecuencias)
------------------------------------------------------
Este módulo implementa la Similitud del Coseno desde cero, sin el uso de 
librerías de álgebra lineal como NumPy ni herramientas de Machine Learning.

Contexto Teórico:
La similitud del coseno evalúa qué tan "alineados" están dos vectores en un espacio
multidimensional calculando el coseno del ángulo entre ellos. 
En este enfoque clásico de NLP, los vectores se construyen sobre un vocabulario
unificado de dos formas:
    - cosine_similarity: frecuencia de términos (Term Frequency - TF).
    - tfidf_cosine_similarity: pesos TF-IDF, donde el IDF se calcula sobre el
      corpus de abstracts y castiga las palabras que aparecen en casi todos.

Fórmula Matemática:
    cos(θ) = (A · B) / (||A|| * ||B||)
"""

import math
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

def compute_idf(corpus: list[str]) -> dict[str, float]:
    """
    Calcula el IDF (Inverse Document Frequency) de cada término del corpus.

    Fórmula:
        idf(t) = ln(N / df(t))
    Donde:
        N     = número de documentos (abstracts) del corpus
        df(t) = número de documentos en los que aparece t al menos una vez

    Un término que aparece en todos los documentos tiene idf = ln(1) = 0: no
    sirve para distinguirlos. Uno que aparece en un solo documento tiene el
    idf máximo, ln(N).

    Por eso el IDF debe calcularse sobre el corpus completo (los 20 abstracts)
    y no solo sobre los dos textos comparados: con N = 2, toda palabra que
    compartan tendría idf = ln(2/2) = 0 y el producto punto sería siempre 0.

    Complejidad: O(T) en tiempo, con T el total de tokens del corpus, y O(V)
    en espacio, con V el tamaño del vocabulario.
    """
    n_documentos = len(corpus)

    # df(t): se cuenta una vez por documento, por eso se recorre el conjunto
    # de tokens únicos de cada abstract
    df = {}
    for documento in corpus:
        for token in set(preprocess_and_tokenize(documento)):
            df[token] = df.get(token, 0) + 1

    idf = {}
    for token in df:
        idf[token] = math.log(n_documentos / df[token])
    return idf


def tfidf_cosine_similarity(text1: str, text2: str, corpus: list[str]) -> dict:
    """
    Calcula la Similitud del Coseno entre dos textos con pesos TF-IDF.

    Pasos algorítmicos transparentes:
    1. Calcular el IDF de cada término sobre el corpus (`compute_idf`).
       Si alguno de los dos textos no está en el corpus, se agrega.
    2. Construir el vocabulario unificado de los dos textos (las dimensiones).
    3. Contar la frecuencia de cada término en cada texto (TF).
    4. Peso de cada término: w(t) = tf(t) * idf(t).
    5. Producto punto: A · B = Σ wA(t) * wB(t).
    6. Normas: ||A|| = sqrt(Σ wA(t)^2), igual para B.
    7. cos(θ) = (A · B) / (||A|| * ||B||) y el ángulo θ = arccos(cos(θ)).

    Complejidad: O(T + V) en tiempo (T tokens del corpus, V vocabulario del
    par de textos) y O(V) en espacio para los vectores.

    Returns:
        dict con todos los pasos intermedios, para la demostración:
            vocabulario, tf_a, tf_b, idf (solo los términos del vocabulario),
            pesos_a, pesos_b, producto_punto, norma_a, norma_b, coseno y
            angulo_grados.
    """
    if text1 not in corpus:
        corpus = corpus + [text1]
    if text2 not in corpus:
        corpus = corpus + [text2]

    # 1. IDF sobre el corpus
    idf_corpus = compute_idf(corpus)

    # 2. Vocabulario unificado y ordenado (índice determinista)
    tokens_a = preprocess_and_tokenize(text1)
    tokens_b = preprocess_and_tokenize(text2)
    vocabulario = sorted(set(tokens_a) | set(tokens_b))
    posicion = {}
    for i in range(len(vocabulario)):
        posicion[vocabulario[i]] = i

    # 3. Frecuencias (TF)
    tf_a = [0] * len(vocabulario)
    tf_b = [0] * len(vocabulario)
    for token in tokens_a:
        tf_a[posicion[token]] += 1
    for token in tokens_b:
        tf_b[posicion[token]] += 1

    # 4. Pesos TF-IDF
    idf = [idf_corpus[t] for t in vocabulario]
    pesos_a = [0.0] * len(vocabulario)
    pesos_b = [0.0] * len(vocabulario)
    for i in range(len(vocabulario)):
        pesos_a[i] = tf_a[i] * idf[i]
        pesos_b[i] = tf_b[i] * idf[i]

    # 5. Producto punto
    producto_punto = 0.0
    for i in range(len(vocabulario)):
        producto_punto += pesos_a[i] * pesos_b[i]

    # 6. Normas
    suma_cuadrados_a = 0.0
    suma_cuadrados_b = 0.0
    for i in range(len(vocabulario)):
        suma_cuadrados_a += pesos_a[i] ** 2
        suma_cuadrados_b += pesos_b[i] ** 2
    norma_a = suma_cuadrados_a ** 0.5
    norma_b = suma_cuadrados_b ** 0.5

    # 7. Coseno y ángulo (si un vector es nulo, por ejemplo porque todos sus
    # términos aparecen en todo el corpus, la similitud se toma como 0)
    if norma_a == 0 or norma_b == 0:
        coseno = 0.0
    else:
        coseno = producto_punto / (norma_a * norma_b)
    # Se recorta a [-1, 1] por los errores de redondeo antes de arccos
    angulo_grados = math.degrees(math.acos(max(-1.0, min(1.0, coseno))))

    return {
        "vocabulario": vocabulario,
        "tf_a": tf_a,
        "tf_b": tf_b,
        "idf": idf,
        "pesos_a": pesos_a,
        "pesos_b": pesos_b,
        "producto_punto": producto_punto,
        "norma_a": norma_a,
        "norma_b": norma_b,
        "coseno": coseno,
        "angulo_grados": angulo_grados,
    }


if __name__ == "__main__":
    # Prueba del algoritmo
    text_A = "algoritmos de inteligencia artificial"
    text_B = "algoritmos en la inteligencia artificial"

    sim, vA, vB, vocab = cosine_similarity(text_A, text_B)

    print("Vocabulario (Dimensiones):", vocab)
    print("Vector A:", vA)
    print("Vector B:", vB)
    print(f"\nSimilitud del Coseno: {sim:.4f}")

    # TF-IDF: el IDF sale de un corpus pequeño de ejemplo
    corpus = [
        text_A,
        text_B,
        "redes neuronales y aprendizaje profundo",
        "inteligencia artificial generativa en la educación",
    ]
    pasos = tfidf_cosine_similarity(text_A, text_B, corpus)
    print("\nTF-IDF")
    for i in range(len(pasos["vocabulario"])):
        print(f"  {pasos['vocabulario'][i]:<12} idf={pasos['idf'][i]:.4f} "
              f"wA={pasos['pesos_a'][i]:.4f} wB={pasos['pesos_b'][i]:.4f}")
    print(f"Coseno TF-IDF: {pasos['coseno']:.4f} (ángulo {pasos['angulo_grados']:.2f}°)")

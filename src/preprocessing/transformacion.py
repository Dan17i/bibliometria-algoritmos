"""
Transformación del abstract antes de vectorizarlo
-------------------------------------------------
Los enfoques de IA y el agrupamiento no reciben el abstract tal como sale del
PDF, sino esta versión:

    1. Minúsculas y sin puntuación (`tokenizar`, recorriendo los caracteres).
    2. Sin stopwords: las palabras funcionales casi no aportan al tema y dominan
       el promedio de los vectores.
    3. Las palabras que quedan, separadas por un espacio.

El paso 2 también resuelve un límite de all-MiniLM-L6-v2: el modelo solo lee 256
tokens y descarta el resto sin avisar. De los 20 abstracts originales, 13 pasan
ese límite; transformados, el más largo tiene 212 y el modelo los lee completos.

Levenshtein y Needleman-Wunsch no usan esta transformación, porque comparan el
texto carácter por carácter.

Complejidad temporal: O(L), con L la longitud del abstract en caracteres.
Complejidad espacial: O(L).
"""

from src.similarity.ai.stopwords import remove_stopwords
from src.similarity.classic.tokenizador import tokenizar


def transformar_abstract(texto: str) -> str:
    """Abstract en minúsculas, sin puntuación ni stopwords."""
    resultado = ""
    for token in remove_stopwords(tokenizar(texto)):
        if resultado != "":
            resultado += " "
        resultado += token
    return resultado

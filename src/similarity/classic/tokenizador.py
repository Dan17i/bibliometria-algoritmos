"""
Módulo de Tokenización de los algoritmos clásicos
-------------------------------------------------
Convierte un abstract en la lista de palabras que usan Jaccard y el coseno. Se
recorre el texto carácter por carácter, sin lower(), split(), replace() ni
expresiones regulares, para que el preprocesamiento también sea transparente.

Reglas:
    1. Las mayúsculas pasan a minúsculas sumando en la tabla de caracteres.
    2. Una letra o un dígito extiende la palabra actual.
    3. Cualquier otro carácter la cierra: "problem-solving" da "problem" y
       "solving", y "1993–2020" da "1993" y "2020".
    4. Un apóstrofe la cierra y descarta lo que sigue: "AI's" da "ai" y
       "learners’" da "learners". Así el posesivo no crea una palabra distinta.

Complejidad temporal: O(L), con L la longitud del texto en caracteres.
Complejidad espacial: O(T), con T el número de tokens.
"""

APOSTROFES = {"'": True, "’": True}


def a_minuscula(caracter: str) -> str:
    """Pasa una letra a minúscula: en ASCII y Latin-1 la minúscula está 32 lugares después."""
    codigo = ord(caracter)
    if 65 <= codigo <= 90 or (192 <= codigo <= 222 and codigo != 215):
        return chr(codigo + 32)
    return caracter


def es_alfanumerico(caracter: str) -> bool:
    """Letra o dígito: rangos ASCII, Latin-1 y los alfabetos fuera de la puntuación Unicode."""
    codigo = ord(caracter)
    if 48 <= codigo <= 57 or 97 <= codigo <= 122:
        return True
    if 192 <= codigo <= 255:
        return codigo != 215 and codigo != 247
    # 0x2000-0x2BFF son guiones, comillas y símbolos tipográficos
    return codigo >= 256 and not 0x2000 <= codigo <= 0x2BFF


def tokenizar(texto: str) -> list[str]:
    """Lista de palabras del texto, en minúsculas y en orden de aparición."""
    tokens = []
    actual = ""
    descartando = False

    for caracter in texto:
        minuscula = a_minuscula(caracter)
        if es_alfanumerico(minuscula):
            if not descartando:
                actual += minuscula
            continue
        if actual != "":
            tokens.append(actual)
            actual = ""
        descartando = caracter in APOSTROFES

    if actual != "":
        tokens.append(actual)
    return tokens


def terminos_unicos(tokens: list[str]) -> list[str]:
    """Tokens sin repetir, en orden de primera aparición. O(T) con un diccionario."""
    vistos = {}
    unicos = []
    for token in tokens:
        if token not in vistos:
            vistos[token] = True
            unicos.append(token)
    return unicos

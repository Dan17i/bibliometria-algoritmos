"""
Demostración matemática de los seis algoritmos de similitud
===========================================================
Arma lo que muestra la aplicación cuando se oprime "Ver demostración": la idea, la
fórmula, los pasos con los valores reemplazados en LaTeX, la lectura del resultado y
dónde está cada cosa en el repositorio.

Los números salen de ejecutar el algoritmo en ese momento, solo el que se pidió. La
redacción sigue la sección 3 del documento técnico.

Los cuatro clásicos también aceptan textos escritos por quien está mirando, para
comparar contra un cálculo hecho en papel. Los dos de IA solo trabajan con los
abstracts del corpus, porque sus vectores están guardados.
"""

from __future__ import annotations

import json
import os
import time

from src.preprocessing.transformacion import transformar_abstract
from src.similarity.ai.demostracion import compare_vectors_steps
from src.similarity.ai.embeddings import average_word_vectors
from src.similarity.ai.stopwords import STOPWORDS
from src.similarity.traza_clasicos import (
    traza_jaccard,
    traza_levenshtein,
    traza_needleman,
    traza_tfidf,
)

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Con textos del corpus se muestra una esquina de la matriz; con textos escritos a
# mano, que son cortos, la matriz completa.
LADO_CORPUS = 8
LADO_TEXTOS = 41

_palabras_glove: dict | None = None


def _vectores_de_palabras() -> dict:
    """Se cargan al primer uso: son cerca de mil vectores que casi ninguna consulta necesita."""
    global _palabras_glove
    if _palabras_glove is None:
        ruta = os.path.join(RAIZ, "data", "processed", "glove_palabras.json")
        with open(ruta, encoding="utf-8") as fh:
            _palabras_glove = json.load(fh)
    return _palabras_glove


def _n(valor: float, decimales: int = 4) -> str:
    return f"{valor:.{decimales}f}"


def _miles(valor: int) -> str:
    return f"{valor:,}".replace(",", r"\,")


def _caracter(c: str) -> str:
    """Un carácter dentro de un texto LaTeX."""
    if c == " ":
        return "espacio"
    return c


def _pasos_vector(pasos: dict) -> list[dict]:
    """Producto punto, normas, coseno, ángulo y euclidiana, comunes a los dos enfoques de IA."""
    dot, na, nb = pasos["dot"], pasos["norm_a"], pasos["norm_b"]
    return [
        {
            "titulo": f"Producto punto sobre las {pasos['dim']} dimensiones",
            "texto": "Se multiplica cada dimensión de un vector por la misma dimensión del otro y se suma todo. "
                     "Si los dos vectores suelen tener el mismo signo en las mismas dimensiones, la suma crece.",
            "latex": [rf"A \cdot B = \sum_{{i=1}}^{{{pasos['dim']}}} A_i B_i = {_n(dot, 6)}"],
        },
        {
            "titulo": "Normas",
            "texto": "La norma es la longitud de cada vector. Se divide por ella para que solo cuente la dirección, no el tamaño.",
            "latex": [
                rf"\|A\| = \sqrt{{\sum A_i^2}} = {_n(na, 6)}",
                rf"\|B\| = \sqrt{{\sum B_i^2}} = {_n(nb, 6)}",
            ],
        },
        {
            "titulo": "Coseno y ángulo",
            "texto": "El coseno mide qué tan alineados están: 1 si apuntan igual, 0 si son perpendiculares.",
            "latex": [
                rf"\cos\theta = \frac{{A \cdot B}}{{\|A\|\,\|B\|}} = \frac{{{_n(dot, 6)}}}{{{_n(na, 6)} \times {_n(nb, 6)}}} = {_n(pasos['cosine'], 6)}",
                rf"\theta = \arccos({_n(pasos['cosine'], 6)}) = {_n(pasos['angle_degrees'], 2)}^\circ",
            ],
        },
        {
            "titulo": "Distancia euclidiana",
            "texto": "Es la otra forma de comparar: la distancia en línea recta entre las puntas de los dos vectores. Más cerca de 0 es más parecido.",
            "latex": [rf"d = \sqrt{{\sum (A_i - B_i)^2}} = \sqrt{{{_n(pasos['sum_squares'], 6)}}} = {_n(pasos['euclidean'], 6)}"],
        },
    ]


def _matriz_visual(trz: dict, simbolo: str, operacion: str, m: int, n: int) -> dict:
    """Los datos que necesita la matriz interactiva de la página."""
    visual = {
        "tipo": "matriz",
        "simbolo": simbolo,
        "operacion": operacion,
        "filas": list(trz["inicio_a"]),
        "columnas": list(trz["inicio_b"]),
        "celdas": trz["esquina_inicial"],
        "completa": len(trz["esquina_inicial"]) == m + 1 and len(trz["esquina_inicial"][0]) == n + 1,
    }
    if not visual["completa"]:
        visual["final"] = {
            "filas": list(trz["fin_a"]),
            "columnas": list(trz["fin_b"]),
            "celdas": trz["esquina_final"],
        }
    return visual


def _codigo(archivo: str, funciones: list[list[str]], tiempo: str, espacio: str, porque: str) -> dict:
    return {"archivo": archivo, "funciones": funciones, "tiempo": tiempo, "espacio": espacio, "porque": porque}


# --------------------------------------------------------------------------
# Levenshtein
# --------------------------------------------------------------------------

def _levenshtein(a: str, b: str, lado: int) -> dict:
    inicio = time.perf_counter()
    t = traza_levenshtein(a, b, lado)
    ms = (time.perf_counter() - inicio) * 1000
    m, n = len(a), len(b)
    o = t["operaciones"]
    largo = max(m, n)
    costo = o["sustituciones"] + o["eliminaciones"] + o["inserciones"]
    e = t["esquina_inicial"]

    pasos = [
        {
            "titulo": "Se crea la matriz",
            "texto": f"Una fila por cada carácter del primer texto y una columna por cada carácter del segundo, más la fila y la columna de la cadena vacía.",
            "latex": [rf"(m + 1) \times (n + 1) = {m + 1} \times {n + 1} = {_miles((m + 1) * (n + 1))} \text{{ celdas}}"],
        },
        {
            "titulo": "Se llenan los bordes",
            "texto": "Convertir un prefijo en la cadena vacía cuesta tantas eliminaciones como caracteres tenga.",
            "latex": [r"D(i, 0) = i \qquad D(0, j) = j"],
        },
    ]
    if m and n:
        c = 0 if a[0] == b[0] else 1
        pasos.append({
            "titulo": "Se llena cada celda con el mínimo de tres caminos",
            "texto": f"Primera celda: se compara «{_caracter(a[0])}» con «{_caracter(b[0])}». "
                     + ("Coinciden, así que sustituir no cuesta nada." if c == 0 else "No coinciden, así que sustituir cuesta 1."),
            "latex": [rf"D(1, 1) = \min({e[0][1]} + 1,\; {e[1][0]} + 1,\; {e[0][0]} + {c}) = {e[1][1]}"],
        })
    pasos.append({
        "titulo": "La distancia es la última celda",
        "texto": f"Un camino óptimo tiene {o['coincidencias']} coincidencias, {o['sustituciones']} sustituciones, "
                 f"{o['eliminaciones']} eliminaciones y {o['inserciones']} inserciones. Las coincidencias no cuestan.",
        "latex": [rf"D(m, n) = {o['sustituciones']} + {o['eliminaciones']} + {o['inserciones']} = {costo}"],
    })
    pasos.append({
        "titulo": "Se normaliza a [0, 1]",
        "texto": "Para compararla con las demás medidas: 1 si los textos son idénticos.",
        "latex": [rf"\text{{similitud}} = 1 - \frac{{d}}{{\max(m, n)}} = 1 - \frac{{{t['distancia']}}}{{{largo}}} = {_n(t['similitud'])}"],
    })

    return {
        "similitud": t["similitud"],
        "ms": ms,
        "que_es": "Es un algoritmo que compara dos textos letra por letra y se encarga de calcular cuántos cambios hacen falta para que uno quede igual al otro. Entre menos cambios, más se parecen. No entiende lo que dicen los textos: solo mira sus letras.",
        "que_significa": "Este número va de 0 a 1: 1 significa que los textos son idénticos, y entre más bajo, más letras hay que cambiar.",
        "formula_simple": "Se llena una tabla en la que cada casilla, D(i, j), guarda los cambios necesarios para convertir las primeras i letras de un texto en las primeras j del otro. Cada casilla se queda con la opción más barata entre quitar, poner o cambiar una letra. c vale 0 si las dos letras son iguales, porque no hay nada que cambiar, y 1 si son distintas.",
        "idea": "¿Cuántos cambios de una letra necesito para convertir un texto en el otro? Cada cambio puede ser "
                "quitar una letra, poner una o cambiarla por otra. Es como corregir un texto a mano y contar las correcciones.",
        "en_una_frase": ("Los dos textos son idénticos: no hace falta ninguna edición." if t["distancia"] == 0
                         else f"Se necesita {t['distancia']} edición de letras para convertir un texto en el otro." if t["distancia"] == 1
                         else f"Se necesitan {t['distancia']} ediciones de letras para convertir un texto en el otro."),
        "formulas": [
            r"D(i, j) = \min \begin{cases} D(i-1, j) + 1 & \text{eliminar} \\ D(i, j-1) + 1 & \text{insertar} \\ D(i-1, j-1) + c & \text{sustituir} \end{cases}",
            r"c = \begin{cases} 0 & \text{si } a_i = b_j \\ 1 & \text{si no} \end{cases}",
        ],
        "pasos": pasos,
        "visual": [_matriz_visual(t, "D", "min", m, n)],
        "interpretacion": "Mide la forma del texto, no su tema. Compara letra por letra y respeta el orden, así que dos abstracts "
                          "sobre lo mismo escritos con otras palabras salen lejos. Sirve para detectar textos casi iguales, como "
                          "una versión corregida de otro.",
        "codigo": _codigo(
            "src/similarity/classic/levenshtein.py",
            [["levenshtein_distance", "llena la matriz y devuelve la distancia"],
             ["operaciones_levenshtein (src/similarity/traza_clasicos.py)", "recorre la matriz hacia atrás y cuenta las operaciones"]],
            "O(m · n)", "O(m · n)", "una operación constante por celda; se guarda la matriz completa",
        ),
    }


# --------------------------------------------------------------------------
# Needleman-Wunsch
# --------------------------------------------------------------------------

def _needleman(a: str, b: str, lado: int) -> dict:
    inicio = time.perf_counter()
    t = traza_needleman(a, b, lado)
    ms = (time.perf_counter() - inicio) * 1000
    m, n = len(a), len(b)
    o = t["operaciones"]
    largo = max(m, n)
    e = t["esquina_inicial"]

    pasos = [
        {
            "titulo": "Se crea la matriz y se llenan los bordes",
            "texto": "Empezar un texto contra nada es abrir huecos seguidos, y cada hueco resta 1.",
            "latex": [
                rf"(m + 1) \times (n + 1) = {m + 1} \times {n + 1} = {_miles((m + 1) * (n + 1))} \text{{ celdas}}",
                r"F(i, 0) = i \cdot d \qquad F(0, j) = j \cdot d \qquad d = -1",
            ],
        },
    ]
    if m and n:
        s = 1 if a[0] == b[0] else -1
        pasos.append({
            "titulo": "Se llena cada celda con el máximo de tres caminos",
            "texto": f"Primera celda: se compara «{_caracter(a[0])}» con «{_caracter(b[0])}». "
                     + ("Coinciden: alinearlos suma 1." if s == 1 else "No coinciden: alinearlos resta 1."),
            "latex": [rf"F(1, 1) = \max({e[0][0]} + ({s}),\; {e[0][1]} + (-1),\; {e[1][0]} + (-1)) = {e[1][1]}"],
        })
    pasos.append({
        "titulo": "El puntaje es la última celda",
        "texto": f"El alineamiento tiene {o['coincidencias']} coincidencias, {o['discrepancias']} discrepancias y {o['huecos']} huecos.",
        "latex": [rf"F(m, n) = {o['coincidencias']} - {o['discrepancias']} - {o['huecos']} = {t['puntaje']}"],
    })
    pasos.append({
        "titulo": "Se normaliza a [0, 1]",
        "texto": "El peor puntaje posible es −L (todo falla) y el mejor es L (textos idénticos), con L la longitud del texto más largo.",
        "latex": [rf"\text{{sim}} = \frac{{F(m, n) - L \cdot d}}{{L \cdot \text{{match}} - L \cdot d}} = \frac{{{t['puntaje']} + {largo}}}{{{largo} + {largo}}} = {_n(t['similitud'])}"],
    })

    return {
        "similitud": t["similitud"],
        "ms": ms,
        "que_es": "Es un algoritmo que pone dos textos uno sobre el otro y se encarga de acomodarlos para que coincida la mayor cantidad posible de letras. Gana puntos por cada letra que coincide y pierde por cada letra distinta o cada hueco que tenga que dejar. Nació en la biología para comparar proteínas, y aquí se usa con texto.",
        "que_significa": "Este número va de 0 a 1: 1 significa que todas las letras coinciden sin dejar huecos, y entre más bajo, peor encajan los dos textos.",
        "formula_simple": "Cada casilla de la tabla, F(i, j), guarda el mejor puntaje posible al alinear las primeras i letras de un texto con las primeras j del otro. Se queda con la mejor de tres opciones: emparejar las dos letras o dejar un hueco en uno de los dos textos. S suma 1 si las letras coinciden y resta 1 si no, y d es lo que resta cada hueco.",
        "idea": "Pone los dos textos uno encima del otro y los alinea para que coincidan la mayor cantidad de letras, abriendo "
                "huecos donde haga falta. Cada coincidencia suma y cada fallo o hueco resta. A diferencia de Levenshtein, "
                "que minimiza un costo, este maximiza un puntaje.",
        "en_una_frase": f"Al alinear los dos textos completos, el mejor puntaje posible es {t['puntaje']}.",
        "formulas": [
            r"F(i, j) = \max \begin{cases} F(i-1, j-1) + S(a_i, b_j) & \text{alinear} \\ F(i-1, j) + d & \text{hueco en el segundo} \\ F(i, j-1) + d & \text{hueco en el primero} \end{cases}",
            r"S(a_i, b_j) = \begin{cases} +1 & \text{si } a_i = b_j \\ -1 & \text{si no} \end{cases} \qquad d = -1",
        ],
        "pasos": pasos,
        "visual": [_matriz_visual(t, "F", "max", m, n)],
        "interpretacion": "Igual que Levenshtein, mide coincidencia de caracteres en orden, no de tema. El alineamiento global obliga "
                          "a emparejar el texto completo, por eso el puntaje de abstracts distintos suele ser negativo. Su uso natural "
                          "es comparar secuencias parecidas, como en biología.",
        "codigo": _codigo(
            "src/similarity/classic/needleman_wunsch.py",
            [["needleman_wunsch", "llena la matriz y devuelve el puntaje"],
             ["normalized_needleman_wunsch", "lleva el puntaje a [0, 1]"],
             ["operaciones_needleman (src/similarity/traza_clasicos.py)", "recorre la matriz hacia atrás y cuenta las operaciones"]],
            "O(m · n)", "O(m · n)", "mismo llenado de matriz, con máximo en vez de mínimo",
        ),
    }


# --------------------------------------------------------------------------
# Jaccard
# --------------------------------------------------------------------------

def _jaccard(a: str, b: str, _lado: int) -> dict:
    inicio = time.perf_counter()
    t = traza_jaccard(a, b)
    ms = (time.perf_counter() - inicio) * 1000
    inter, union = t["tam_interseccion"], t["tam_union"]

    pasos = [
        {
            "titulo": "Se convierte cada texto en un conjunto de palabras",
            "texto": "Se pasa a minúsculas, se corta en todo lo que no sea letra o dígito y se quitan los repetidos.",
            "latex": [rf"|A| = {t['tam_a']} \qquad |B| = {t['tam_b']}"],
        },
        {
            "titulo": "Se cuentan las palabras que comparten",
            "texto": "La intersección son las palabras que están en los dos.",
            "latex": [rf"|A \cap B| = {inter}"],
        },
        {
            "titulo": "Se cuentan todas las palabras distintas",
            "texto": "La unión son todas, sin contar dos veces las compartidas.",
            "latex": [rf"|A \cup B| = |A| + |B| - |A \cap B| = {t['tam_a']} + {t['tam_b']} - {inter} = {union}"],
        },
        {
            "titulo": "Se dividen",
            "texto": "La proporción de palabras compartidas sobre todas las que aparecen.",
            "latex": [rf"J(A, B) = \frac{{|A \cap B|}}{{|A \cup B|}} = \frac{{{inter}}}{{{union}}} = {_n(t['similitud'])}"],
        },
    ]
    return {
        "similitud": t["similitud"],
        "ms": ms,
        "que_es": "Es un algoritmo que compara dos textos por las palabras que usan, sin importar en qué orden aparecen. Se encarga de responder una pregunta sencilla: de todas las palabras que aparecen entre los dos textos, ¿cuántas están en ambos?",
        "que_significa": "Este número va de 0 a 1: 0 significa que no comparten ninguna palabra, y 1 que usan exactamente las mismas.",
        "formula_simple": "A y B son las listas de palabras de cada texto, sin repetir. Arriba se cuentan las palabras que están en los dos (∩), y abajo todas las palabras distintas que hay entre ambos (∪).",
        "idea": "Mete las palabras de cada texto en una bolsa, sin orden y sin repetir, y pregunta qué proporción de todas "
                "las palabras está en las dos bolsas. No importa cuántas veces aparece una palabra, solo si aparece.",
        "en_una_frase": f"Comparten {inter} de las {union} palabras distintas que aparecen en los dos textos.",
        "formulas": [r"J(A, B) = \frac{|A \cap B|}{|A \cup B|}"],
        "pasos": pasos,
        "visual": [{
            "tipo": "conjuntos",
            "solo_a": t["solo_a"],
            "comunes": t["interseccion"],
            "solo_b": t["solo_b"],
        }],
        "interpretacion": "Mide cuánto vocabulario comparten. Es rápido y fácil de explicar, pero cuenta igual a «the» que a «journals» "
                          "y no reconoce sinónimos: «learning» y «education» cuentan como palabras sin relación.",
        "codigo": _codigo(
            "src/similarity/classic/jaccard.py",
            [["jaccard_similarity", "calcula la intersección, la unión y la similitud"],
             ["tokenizar, terminos_unicos (src/similarity/classic/tokenizador.py)", "minúsculas, corte y términos sin repetir"]],
            "O(T)", "O(T)", "T es el total de palabras: cada una se revisa una vez en un diccionario",
        ),
    }


# --------------------------------------------------------------------------
# Coseno TF-IDF
# --------------------------------------------------------------------------

def _tfidf(a: str, b: str, _lado: int, corpus: list[str], idf: dict | None) -> dict:
    inicio = time.perf_counter()
    t = traza_tfidf(a, b, corpus, idf)
    ms = (time.perf_counter() - inicio) * 1000
    terminos = t["terminos_que_mas_aportan"]
    n_docs = t["n_documentos"]
    ceros = t["terminos_con_idf_cero"]

    pasos = [
        {
            "titulo": "Se calcula el IDF sobre el corpus",
            "texto": f"N = {n_docs} abstracts. Una palabra presente en todos pesa 0: no sirve para distinguirlos. "
                     + (f"Aquí pesan 0: {', '.join(ceros)}." if ceros else "Aquí ninguna palabra del par pesa 0.")
                     + " Se usa todo el corpus porque con solo dos textos toda palabra compartida tendría idf = 0.",
            "latex": [r"\text{idf}(t) = \ln \frac{N}{\text{df}(t)}"],
        },
    ]
    if terminos and terminos[0]["producto"] > 0:
        x = terminos[0]
        df = round(n_docs / (2.718281828459045 ** x["idf"]))
        pasos.append({
            "titulo": f"Se calcula el peso de cada palabra. Por ejemplo, «{x['termino']}»",
            "texto": f"Aparece {x['tf_a']} y {x['tf_b']} veces en cada texto, y en {df} de los {n_docs} abstracts. El peso es la frecuencia por el IDF.",
            "latex": [
                rf"\text{{idf}} = \ln \frac{{{n_docs}}}{{{df}}} = {_n(x['idf'])}",
                rf"w_A = {x['tf_a']} \times {_n(x['idf'])} = {_n(x['peso_a'])} \qquad w_B = {x['tf_b']} \times {_n(x['idf'])} = {_n(x['peso_b'])}",
            ],
        })
    pasos += [
        {
            "titulo": "Producto punto",
            "texto": f"Solo suman las palabras que están en los dos textos y no pesan 0. Hay {t['dimensiones']} dimensiones, una por palabra del vocabulario.",
            "latex": [rf"A \cdot B = \sum_{{t}} w_A(t)\, w_B(t) = {_n(t['producto_punto'])}"],
        },
        {
            "titulo": "Normas",
            "texto": "Para que un texto largo no gane solo por tener más palabras.",
            "latex": [rf"\|A\| = \sqrt{{\sum w_A(t)^2}} = {_n(t['norma_a'])} \qquad \|B\| = {_n(t['norma_b'])}"],
        },
        {
            "titulo": "Coseno y ángulo",
            "texto": "",
            "latex": [
                rf"\cos\theta = \frac{{{_n(t['producto_punto'])}}}{{{_n(t['norma_a'])} \times {_n(t['norma_b'])}}} = {_n(t['coseno'])}",
                rf"\theta = \arccos({_n(t['coseno'])}) = {_n(t['angulo_grados'], 2)}^\circ",
            ],
        },
    ]
    return {
        "similitud": t["coseno"],
        "ms": ms,
        "que_es": "Es un algoritmo que compara dos textos por las palabras que usan, pero dándoles más importancia a las palabras poco comunes. Se encarga de que compartir «the», que está en todos los artículos, no cuente, y de que compartir una palabra especializada sí. Así, dos textos se parecen cuando usan el mismo vocabulario específico.",
        "que_significa": "Este número va de 0 a 1: 0 significa que no comparten ninguna palabra que aporte, y 1 que usan las mismas palabras en la misma proporción.",
        "formula_simple": "La primera fórmula le da un peso a cada palabra, w(t). Crece con las veces que aparece en el texto (tf) y baja si aparece en muchos de los N artículos (df): si está en todos, el logaritmo da 0 y no pesa. La segunda compara los dos textos con esos pesos: mide el ángulo θ entre sus flechas, y su coseno es el resultado.",
        "idea": "Cada texto se vuelve una flecha en un espacio con una dimensión por palabra. Las palabras raras alargan más la flecha "
                "que las comunes. Dos textos se parecen si sus flechas apuntan hacia el mismo lado: compartir una palabra poco "
                "usada dice más que compartir «the».",
        "en_una_frase": f"Sus flechas forman un ángulo de {_n(t['angulo_grados'], 1)}° después de quitarle peso a las palabras comunes.",
        "formulas": [
            r"w(t) = \text{tf}(t) \cdot \ln \frac{N}{\text{df}(t)}",
            r"\cos\theta = \frac{A \cdot B}{\|A\|\,\|B\|}",
        ],
        "pasos": pasos,
        "visual": [
            {"tipo": "terminos", "terminos": terminos},
            {"tipo": "angulo", "grados": t["angulo_grados"]},
        ],
        "interpretacion": "Mide vocabulario específico compartido. Un coseno bajo no significa que el tema sea distinto: dos textos sobre "
                          "lo mismo pueden usar palabras distintas, y en un corpus de un solo tema las palabras más comunes pesan 0. "
                          "Por eso sus valores son bajos aun en pares muy parecidos.",
        "codigo": _codigo(
            "src/similarity/classic/cosine.py",
            [["compute_idf", "calcula el IDF de cada término sobre el corpus"],
             ["tfidf_cosine_similarity", "pesos, producto punto, normas, coseno y ángulo"],
             ["natural_log, angle_degrees (src/similarity/metrics.py)", "logaritmo y arcocoseno calculados a mano"]],
            "O(T + V)", "O(V)", "contar frecuencias, ponderar y recorrer los vectores; T palabras y V vocabulario del par",
        ),
    }


# --------------------------------------------------------------------------
# GloVe
# --------------------------------------------------------------------------

def _glove(a: str, b: str) -> dict:
    inicio = time.perf_counter()
    palabras = _vectores_de_palabras()
    transformado_a, transformado_b = transformar_abstract(a), transformar_abstract(b)
    pa = average_word_vectors(transformado_a, palabras, STOPWORDS)
    pb = average_word_vectors(transformado_b, palabras, STOPWORDS)
    comparacion = compare_vectors_steps(pa["vector"], pb["vector"], 6)
    ms = (time.perf_counter() - inicio) * 1000

    def suma(res: dict, j: int) -> float:
        total = 0.0
        for v in res["token_vectors"]:
            total += v[j]
        return total

    ka, kb = len(pa["used_tokens"]), len(pb["used_tokens"])
    filas = []
    for j in range(6):
        filas.append([j + 1, suma(pa, j), pa["vector"][j], suma(pb, j), pb["vector"][j], pa["vector"][j] * pb["vector"][j]])

    pasos = [
        {
            "titulo": "Se transforma cada abstract",
            "texto": f"Minúsculas, sin puntuación ni stopwords. Quedan {ka} y {kb} palabras con vector. "
                     f"No tienen vector en GloVe {len(pa['missing_tokens'])} y {len(pb['missing_tokens'])} tokens.",
            "latex": [],
        },
        {
            "titulo": "Se busca el vector de cada palabra y se promedian",
            "texto": "Cada palabra tiene un vector de 100 dimensiones, entrenado por el modelo. "
                     "El vector del abstract es el promedio, dimensión por dimensión. Las repeticiones cuentan.",
            "latex": [
                r"d_j = \frac{1}{k} \sum_{w=1}^{k} v_{w, j}",
                rf"d^{{A}}_1 = \frac{{{_n(filas[0][1])}}}{{{ka}}} = {_n(filas[0][2], 5)} \qquad d^{{B}}_1 = \frac{{{_n(filas[0][3])}}}{{{kb}}} = {_n(filas[0][4], 5)}",
            ],
        },
    ] + _pasos_vector(comparacion)

    return {
        "similitud": comparacion["cosine"],
        "ms": ms,
        "que_es": "Es un método de inteligencia artificial que compara dos textos por el significado de sus palabras, no por cómo se escriben. Usa un diccionario ya entrenado que le da a cada palabra una lista de 100 números, de modo que palabras con significado parecido tienen números parecidos. Cada texto se resume en el promedio de sus palabras, y luego se comparan los dos promedios.",
        "que_significa": "Cerca de 1, los dos promedios apuntan casi al mismo lado. En estos artículos casi todos los pares dan más de 0.9, así que el número solo dice poco: hay que mirarlo frente a los demás pares.",
        "formula_simple": "La primera fórmula saca el promedio. Para cada una de las 100 posiciones (j), suma el número de cada una de las k palabras y lo divide entre k. La segunda compara los dos promedios, A y B: mide el ángulo θ entre ellos, y su coseno es el resultado.",
        "idea": "Cada palabra ya tiene una posición en un mapa de significados: «teacher» queda cerca de «student». El abstract se resume "
                "en el punto medio de todas sus palabras, y dos abstracts se comparan por el ángulo entre esos puntos. Las palabras "
                "cercanas en significado acercan los textos aunque no sean iguales.",
        "en_una_frase": f"El promedio de sus {ka} y {kb} palabras apunta casi al mismo lado: {_n(comparacion['angle_degrees'], 1)}° de diferencia.",
        "formulas": [
            r"d_j = \frac{1}{k} \sum_{w=1}^{k} v_{w, j}",
            r"\cos\theta = \frac{A \cdot B}{\|A\|\,\|B\|}",
        ],
        "pasos": pasos,
        "visual": [
            {
                "tipo": "palabras",
                "a": pa["used_tokens"][:40], "b": pb["used_tokens"][:40],
                "fuera_a": pa["missing_tokens"], "fuera_b": pb["missing_tokens"],
                "total_a": ka, "total_b": kb,
            },
            {
                "tipo": "tabla",
                "titulo": "Primeras dimensiones del promedio",
                "columnas": ["j", "Suma A", f"÷ {ka} = A[j]", "Suma B", f"÷ {kb} = B[j]", "A[j] · B[j]"],
                "filas": [[f[0], _n(f[1]), _n(f[2], 5), _n(f[3]), _n(f[4], 5), _n(f[5], 6)] for f in filas],
            },
            {"tipo": "angulo", "grados": comparacion["angle_degrees"]},
        ],
        "interpretacion": "Un valor cercano a 1 no significa que los textos sean casi iguales. Promediar muchas palabras borra las "
                          "diferencias: todos los abstracts terminan cerca del mismo punto, por eso sus valores casi no se separan. "
                          "Hay que mirar el orden entre pares, no el número solo.",
        "codigo": _codigo(
            "src/similarity/ai/embeddings.py",
            [["average_word_vectors", "busca los vectores y los promedia"],
             ["transformar_abstract (src/preprocessing/transformacion.py)", "deja el abstract listo para vectorizar"],
             ["cosine_steps, euclidean_steps (src/similarity/metrics.py)", "la comparación, calculada a mano"]],
            "O(L + k · p) por abstract; O(p) por par", "O(p)",
            "sumar k vectores de dimensión p; comparar es un recorrido lineal. El modelo solo entrega los vectores",
        ),
    }


# --------------------------------------------------------------------------
# MiniLM
# --------------------------------------------------------------------------

def _minilm(a: str, b: str, vector_a: list[float], vector_b: list[float]) -> dict:
    inicio = time.perf_counter()
    comparacion = compare_vectors_steps(vector_a, vector_b, 6)
    ms = (time.perf_counter() - inicio) * 1000

    pasos = [
        {
            "titulo": "Se transforma cada abstract",
            "texto": "El modelo solo lee 256 tokens y descarta el resto sin avisar. De los 20 abstracts originales, 13 pasan ese límite; "
                     "transformados, el más largo tiene 212 tokens y se lee completo.",
            "latex": [],
        },
        {
            "titulo": "El modelo entrega un vector de 384 dimensiones",
            "texto": "Lee la secuencia de palabras completa, en contexto, y resume el texto en un vector de longitud 1. Esta parte es una "
                     "caja negra: los vectores se calcularon una vez y están guardados. Todo lo que sigue se calcula a mano en este momento.",
            "latex": [],
        },
    ] + _pasos_vector(comparacion)

    return {
        "similitud": comparacion["cosine"],
        "ms": ms,
        "que_es": "Es un modelo de inteligencia artificial que lee un texto completo y lo resume en una lista de 384 números que representa de qué trata. Se encarga de que dos textos con un significado parecido reciban números parecidos, aunque usen palabras distintas. Después solo queda comparar las dos listas.",
        "que_significa": "Cerca de 1, el modelo considera que los dos textos hablan de algo muy parecido. En estos artículos, los valores van de 0.42 a 0.85.",
        "formula_simple": "A y B son las dos listas de 384 números. Arriba se multiplican posición por posición y se suma todo; abajo se divide entre el tamaño de cada lista. El resultado es el coseno del ángulo θ entre ellas: entre más cerca de 1, más apuntan hacia el mismo lado.",
        "idea": "Una red neuronal entrenada para que textos con significado parecido queden cerca lee el abstract completo y devuelve "
                "un solo vector. A diferencia de GloVe, no promedia palabras sueltas: la misma palabra aporta distinto según las que "
                "la rodean.",
        "en_una_frase": f"El modelo los ubica a {_n(comparacion['angle_degrees'], 1)}° de diferencia en su espacio de significados.",
        "formulas": [r"\cos\theta = \frac{A \cdot B}{\|A\|\,\|B\|}"],
        "pasos": pasos,
        "visual": [
            {
                "tipo": "tabla",
                "titulo": "Primeras dimensiones de los vectores",
                "columnas": ["i", "A[i]", "B[i]", "A[i] · B[i]"],
                "filas": [[i, _n(x, 5), _n(y, 5), _n(p, 6)] for i, x, y, p in comparacion["first"]],
            },
            {"tipo": "angulo", "grados": comparacion["angle_degrees"]},
        ],
        "interpretacion": "Es la medida que mejor separa los pares del corpus y la más cercana a lo que se entiende al leerlos. Su costo es "
                          "que no se puede demostrar por dentro: se puede verificar el coseno, pero no por qué el modelo eligió esos vectores. "
                          "Como las normas ya valen 1, el coseno coincide con el producto punto.",
        "codigo": _codigo(
            "src/similarity/ai/embeddings.py",
            [["load_sentence_encoder, encode_documents", "obtienen los vectores con el modelo (src/similarity/ai/precalculo.py los guarda)"],
             ["cosine_steps, euclidean_steps (src/similarity/metrics.py)", "la comparación, calculada a mano"]],
            "O(p) por par, sin contar la inferencia", "O(p)",
            "la inferencia del modelo es una caja negra; solo se analiza la comparación",
        ),
    }


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------

NOMBRES = {
    "levenshtein": ("Levenshtein", "Programación dinámica"),
    "needleman_wunsch": ("Needleman-Wunsch", "Programación dinámica"),
    "jaccard": ("Jaccard", "Conjuntos"),
    "coseno_tfidf": ("Coseno (TF-IDF)", "Espacio vectorial"),
    "glove": ("GloVe promediado", "Embeddings"),
    "minilm": ("MiniLM", "Embeddings"),
}

ACEPTAN_TEXTOS = ("levenshtein", "needleman_wunsch", "jaccard", "coseno_tfidf")


def demostracion(clave: str, texto_a: str, texto_b: str, corpus: list[str], idf: dict | None,
                 lado: int, vectores: tuple | None = None) -> dict:
    """La demostración de un solo algoritmo sobre un par de textos."""
    if clave == "levenshtein":
        resultado = _levenshtein(texto_a, texto_b, lado)
    elif clave == "needleman_wunsch":
        resultado = _needleman(texto_a, texto_b, lado)
    elif clave == "jaccard":
        resultado = _jaccard(texto_a, texto_b, lado)
    elif clave == "coseno_tfidf":
        resultado = _tfidf(texto_a, texto_b, lado, corpus, idf)
    elif clave == "glove":
        resultado = _glove(texto_a, texto_b)
    else:
        resultado = _minilm(texto_a, texto_b, vectores[0], vectores[1])

    nombre, familia = NOMBRES[clave]
    resultado.update({
        "clave": clave,
        "algoritmo": nombre,
        "familia": familia,
        "acepta_textos": clave in ACEPTAN_TEXTOS,
    })
    return resultado

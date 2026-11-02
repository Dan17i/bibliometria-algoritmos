"""
Algoritmo Rabin-Karp
====================
Busca ocurrencias exactas comparando **números** en vez de cadenas, usando una
función de hash que se puede actualizar en tiempo constante al deslizarse.

LA IDEA: COMPARAR HUELLAS, NO TEXTO
------------------------------------
Comparar dos cadenas de m caracteres cuesta hasta m operaciones. Comparar dos
números cuesta una. Si a cada trozo de texto se le asigna un número —su hash—,
basta comparar números.

El problema obvio: calcular el hash de cada ventana también cuesta m. Si hay
n ventanas, volvemos a O(n·m) y no se ganó nada.

EL ROLLING HASH
---------------
La solución es no recalcular: **actualizar**. Si ya se tiene el hash de una
ventana, el de la siguiente se obtiene quitando el carácter que sale y
añadiendo el que entra.

    Texto "ABCD", ventana de 3, base 10 (simplificado con A=1, B=2, ...):

    hash("ABC") = 1·10² + 2·10¹ + 3·10⁰ = 123

    Para pasar a "BCD":
      1. quitar la 'A':   123 − 1·10² = 23
      2. correr un lugar:  23 × 10     = 230
      3. añadir la 'D':   230 + 4      = 234

    hash("BCD") = 234, en tres operaciones aritméticas en vez de tres
    multiplicaciones y dos sumas desde cero.

Es la misma mecánica que mover una ventana sobre un número: se tira el dígito
de la izquierda y se mete uno por la derecha.

LA ARITMÉTICA MODULAR
---------------------
Con textos largos esos números se vuelven gigantescos. Por eso todo se calcula
módulo un primo q: los valores quedan acotados y las operaciones son rápidas.

    hash(s) = (s[0]·b^(m-1) + s[1]·b^(m-2) + ... + s[m-1]) mod q

Se usa un primo porque reparte los valores de forma más uniforme y reduce las
colisiones. Aquí q = 1.000.003 y la base b = 256 (el rango de un byte).

LAS COLISIONES: POR QUÉ HAY QUE VERIFICAR
------------------------------------------
Dos cadenas distintas pueden dar el mismo hash. Es inevitable: hay infinitas
cadenas y solo q valores posibles.

Por eso, cuando los hashes coinciden, **hay que comparar los caracteres uno a
uno** para confirmar. Sin esa verificación el algoritmo daría falsos positivos.

Ese es el precio del método: el hash descarta rápido, pero no confirma.

COMPLEJIDAD
-----------
    Preprocesamiento: O(m)   — hash del patrón y de la primera ventana
    Caso promedio:    O(n + m)
    Peor caso:        O(n · m)  — si todas las ventanas colisionan

El peor caso es teórico: requiere un texto construido adversarialmente contra
la función de hash elegida. Con un primo grande sobre texto natural, las
colisiones son raras y el algoritmo se comporta como lineal.

Su ventaja real no está en buscar un patrón, sino **varios a la vez**: con el
mismo recorrido se comparan los hashes de todos los patrones buscados.
"""

from __future__ import annotations

# Base = 256: cada carácter se trata como un byte, así que el hash se comporta
# como un número en base 256.
BASE = 256

# Primo grande para el módulo. Reduce colisiones y mantiene los valores dentro
# de un rango manejable.
PRIMO = 1_000_003


def calcular_hash(cadena: str, longitud: int, primo: int = PRIMO) -> int:
    """
    Hash polinómico de los primeros `longitud` caracteres.

        h = (s[0]·b^(L-1) + s[1]·b^(L-2) + ... + s[L-1]) mod q

    Se aplica el módulo en cada paso, no al final: así los valores
    intermedios nunca crecen sin control.

    Complejidad: O(longitud).
    """
    h = 0
    for i in range(longitud):
        h = (h * BASE + ord(cadena[i])) % primo
    return h


def buscar(texto: str, patron: str) -> list[int]:
    """
    Devuelve todas las posiciones del texto donde empieza el patrón.

    Recorre el texto manteniendo el hash de la ventana actual. Cuando ese hash
    coincide con el del patrón, verifica carácter a carácter (por las
    colisiones) y solo entonces registra la ocurrencia.

    Complejidad: O(n + m) en promedio.
    """
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return []
    return buscar_con_hashes(texto, patron, *preprocesar(texto, patron))


def preprocesar(texto: str, patron: str) -> tuple[int, int, int]:
    """
    El peso del carácter que sale de la ventana, b^(m-1) mod q, y los hashes del
    patrón y de la primera ventana.

    Complejidad: O(m).
    """
    m = len(patron)
    factor = 1
    for _ in range(m - 1):
        factor = (factor * BASE) % PRIMO
    return factor, calcular_hash(patron, m), calcular_hash(texto, m)


def buscar_con_hashes(texto: str, patron: str, factor: int, hash_patron: int,
                      hash_ventana: int) -> list[int]:
    """La fase de búsqueda, con los hashes ya calculados. Permite medirla por separado."""
    n, m = len(texto), len(patron)
    ocurrencias: list[int] = []

    for i in range(n - m + 1):
        if hash_ventana == hash_patron:
            # Los hashes coinciden, pero puede ser una colisión:
            # hay que confirmar comparando los caracteres.
            igual = True
            for j in range(m):
                if texto[i + j] != patron[j]:
                    igual = False
                    break
            if igual:
                ocurrencias.append(i)

        # Rolling: quitar el carácter que sale, correr, añadir el que entra
        if i < n - m:
            hash_ventana = (
                (hash_ventana - ord(texto[i]) * factor) * BASE
                + ord(texto[i + m])
            ) % PRIMO
            # Python ya normaliza el módulo de negativos, pero se deja
            # explícito para que la traducción a otros lenguajes sea directa.
            if hash_ventana < 0:
                hash_ventana += PRIMO

    return ocurrencias


# --------------------------------------------------------------------------
# Instrumentación para el documento técnico
# --------------------------------------------------------------------------

def buscar_con_pasos(texto: str, patron: str, max_pasos: int = 12, primo: int = PRIMO
                     ) -> tuple[list[int], list[dict]]:
    """
    Búsqueda que registra el hash de cada ventana y si hubo colisión.

    Permite mostrar el cálculo numérico iterativo del hash y las colisiones. Con
    un primo pequeño las colisiones aparecen aun en textos cortos.
    """
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return [], []

    factor = 1
    for _ in range(m - 1):
        factor = (factor * BASE) % primo

    hash_patron = calcular_hash(patron, m, primo)
    hash_ventana = calcular_hash(texto, m, primo)
    ocurrencias: list[int] = []
    pasos: list[dict] = []

    for i in range(n - m + 1):
        registro = {
            "posicion": i,
            "ventana": texto[i:i + m],
            "hash_ventana": hash_ventana,
            "hash_patron": hash_patron,
            "coincide_hash": hash_ventana == hash_patron,
            "colision": False,
            "ocurrencia": False,
        }

        if hash_ventana == hash_patron:
            igual = True
            for j in range(m):
                if texto[i + j] != patron[j]:
                    igual = False
                    break
            if igual:
                ocurrencias.append(i)
                registro["ocurrencia"] = True
            else:
                # Mismo hash, cadenas distintas: colisión real
                registro["colision"] = True

        if len(pasos) < max_pasos:
            pasos.append(registro)

        if i < n - m:
            hash_ventana = (
                (hash_ventana - ord(texto[i]) * factor) * BASE
                + ord(texto[i + m])
            ) % primo
            if hash_ventana < 0:
                hash_ventana += primo

    return ocurrencias, pasos


def buscar_con_metricas(texto: str, patron: str) -> dict:
    """Búsqueda que cuenta verificaciones, colisiones y comparaciones."""
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return {"ocurrencias": [], "comparaciones": 0, "verificaciones": 0,
                "colisiones": 0, "preprocesamiento": 0}

    factor = 1
    for _ in range(m - 1):
        factor = (factor * BASE) % PRIMO

    hash_patron = calcular_hash(patron, m)
    hash_ventana = calcular_hash(texto, m)
    ocurrencias: list[int] = []
    comparaciones = verificaciones = colisiones = 0

    for i in range(n - m + 1):
        comparaciones += 1          # la comparación de hashes
        if hash_ventana == hash_patron:
            verificaciones += 1
            igual = True
            for j in range(m):
                comparaciones += 1
                if texto[i + j] != patron[j]:
                    igual = False
                    break
            if igual:
                ocurrencias.append(i)
            else:
                colisiones += 1

        if i < n - m:
            hash_ventana = (
                (hash_ventana - ord(texto[i]) * factor) * BASE
                + ord(texto[i + m])
            ) % PRIMO
            if hash_ventana < 0:
                hash_ventana += PRIMO

    return {
        "ocurrencias": ocurrencias,
        "comparaciones": comparaciones,
        "verificaciones": verificaciones,
        "colisiones": colisiones,
        "preprocesamiento": 2 * m,      # hash del patrón + primera ventana
    }

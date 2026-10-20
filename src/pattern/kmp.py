"""
Algoritmo Knuth-Morris-Pratt (KMP)
===================================
Busca todas las ocurrencias exactas de un patrón dentro de un texto sin
retroceder nunca sobre el texto.

EL PROBLEMA QUE RESUELVE
------------------------
La búsqueda ingenua compara el patrón contra cada posición del texto y, al
fallar, retrocede el puntero del texto a la siguiente posición:

    Texto:   A B A B A B C
    Patrón:  A B A B C
             ✓ ✓ ✓ ✓ ✗        falla en la posición 4

    La búsqueda ingenua reinicia en la posición 1 del texto:
    Texto:   A B A B A B C
    Patrón:    A B A B C
               ✗              vuelve a comparar lo que ya sabía

Ese retroceso desperdicia información. Cuando el patrón falló en la posición
4, ya se sabía que las cuatro letras anteriores eran "ABAB". KMP usa ese
conocimiento para saltar sin volver atrás.

LA IDEA CENTRAL: PREFIJOS QUE SON TAMBIÉN SUFIJOS
--------------------------------------------------
Si el patrón falla tras haber emparejado k caracteres, esos k caracteres ya
están confirmados en el texto. La pregunta es: ¿cuánto del patrón puedo dar
por emparejado todavía, sin volver a mirar el texto?

La respuesta depende de si el trozo emparejado termina igual que empieza.

    Patrón "ABAB", emparejados los 4 caracteres y falla el quinto.
    El trozo "ABAB" empieza por "AB" y termina por "AB".
    Entonces ese "AB" final del texto ya es el "AB" inicial del patrón:
    se puede continuar desde la posición 2 del patrón, sin tocar el texto.

A ese "AB" se le llama el **borde** del trozo: el prefijo propio más largo que
es a la vez sufijo. La tabla LPS (Longest Prefix that is also Suffix) guarda,
para cada posición del patrón, la longitud de ese borde.

POR QUÉ ES O(n + m)
-------------------
La construcción de la tabla recorre el patrón una vez: O(m).
La búsqueda recorre el texto una vez: O(n). El índice del texto NUNCA
retrocede; solo puede avanzar o quedarse mientras el del patrón baja. Y el
índice del patrón solo baja tantas veces como subió, así que el total sigue
siendo lineal.

    Preprocesamiento: O(m)
    Búsqueda:         O(n)
    Total:            O(n + m)
    Espacio:          O(m) por la tabla LPS
"""

from __future__ import annotations


def construir_lps(patron: str) -> list[int]:
    """
    Tabla LPS: para cada posición i, la longitud del borde de `patron[0..i]`.

    Un **borde** es el prefijo propio más largo que también es sufijo.
    "Propio" significa que no puede ser la cadena entera.

    Ejemplo con "ABABC":

        i  prefijo    borde más largo        lps[i]
        0  "A"        —  (no hay propio)     0
        1  "AB"       —                      0
        2  "ABA"      "A"                    1
        3  "ABAB"     "AB"                   2
        4  "ABABC"    —                      0

    CÓMO SE CONSTRUYE

    Se avanza con dos punteros: `i` recorre el patrón y `longitud` lleva el
    tamaño del borde actual.

      - Si `patron[i] == patron[longitud]`, el borde crece en uno.
      - Si no coinciden y el borde no es cero, se retrocede al borde DEL BORDE
        (`lps[longitud - 1]`), que es el siguiente candidato más largo. Esto
        es recursión disfrazada de bucle: el borde de un borde también es un
        borde.
      - Si no coinciden y el borde ya es cero, no hay nada que preservar.

    Complejidad: O(m). `longitud` solo decrece cuando antes creció, así que el
    número total de decrementos está acotado por el de incrementos.
    """
    m = len(patron)
    lps = [0] * m
    longitud = 0        # tamaño del borde que se viene arrastrando
    i = 1               # lps[0] siempre es 0: una cadena de 1 no tiene borde propio

    while i < m:
        if patron[i] == patron[longitud]:
            longitud += 1
            lps[i] = longitud
            i += 1
        elif longitud != 0:
            # No coincide, pero había un borde: se prueba con el borde del borde.
            # Importante: aquí NO se avanza i, solo se reduce el candidato.
            longitud = lps[longitud - 1]
        else:
            lps[i] = 0
            i += 1

    return lps


def buscar(texto: str, patron: str) -> list[int]:
    """
    Devuelve todas las posiciones del texto donde empieza el patrón.

    Dos punteros: `i` sobre el texto (nunca retrocede) y `j` sobre el patrón.

      - Si los caracteres coinciden, ambos avanzan.
      - Si se completó el patrón (`j == m`), se registra la ocurrencia y se
        continúa desde `lps[j-1]`, lo que permite encontrar solapamientos.
      - Si fallan y `j > 0`, se baja `j` a `lps[j-1]` sin mover `i`: el salto
        que da el algoritmo su nombre.
      - Si fallan y `j == 0`, se avanza `i`: no había nada emparejado.

    Complejidad: O(n). El puntero del texto solo avanza.
    """
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return []

    lps = construir_lps(patron)
    ocurrencias: list[int] = []
    i = j = 0

    while i < n:
        if texto[i] == patron[j]:
            i += 1
            j += 1
            if j == m:
                ocurrencias.append(i - m)
                # Continuar desde el borde permite detectar ocurrencias
                # solapadas (buscar "AA" en "AAA" da 0 y 1, no solo 0).
                j = lps[j - 1]
        elif j > 0:
            j = lps[j - 1]      # el texto NO retrocede
        else:
            i += 1

    return ocurrencias


# --------------------------------------------------------------------------
# Instrumentación para el documento técnico
# --------------------------------------------------------------------------

def construir_lps_con_pasos(patron: str) -> tuple[list[int], list[dict]]:
    """Igual que `construir_lps`, registrando cada decisión para la traza."""
    m = len(patron)
    lps = [0] * m
    longitud, i = 0, 1
    pasos: list[dict] = []

    while i < m:
        if patron[i] == patron[longitud]:
            longitud += 1
            lps[i] = longitud
            pasos.append({
                "i": i, "comparacion": f"patron[{i}]='{patron[i]}' == patron[{longitud-1}]='{patron[longitud-1]}'",
                "accion": f"coinciden: borde crece a {longitud}",
                "lps": list(lps),
            })
            i += 1
        elif longitud != 0:
            anterior = longitud
            longitud = lps[longitud - 1]
            pasos.append({
                "i": i, "comparacion": f"patron[{i}]='{patron[i]}' != patron[{anterior}]='{patron[anterior]}'",
                "accion": f"retrocede al borde del borde: {anterior} -> {longitud}",
                "lps": list(lps),
            })
        else:
            lps[i] = 0
            pasos.append({
                "i": i, "comparacion": f"patron[{i}]='{patron[i]}' != patron[0]='{patron[0]}'",
                "accion": "sin borde que preservar: lps[i]=0",
                "lps": list(lps),
            })
            i += 1

    return lps, pasos


def buscar_con_metricas(texto: str, patron: str) -> dict:
    """
    Búsqueda que además cuenta comparaciones y saltos.

    Las métricas permiten contrastar el comportamiento real de los tres
    algoritmos sobre el mismo texto, más allá del tiempo de reloj.
    """
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return {"ocurrencias": [], "comparaciones": 0, "saltos": 0,
                "preprocesamiento": 0}

    lps = construir_lps(patron)
    ocurrencias: list[int] = []
    comparaciones = saltos = 0
    i = j = 0

    while i < n:
        comparaciones += 1
        if texto[i] == patron[j]:
            i += 1
            j += 1
            if j == m:
                ocurrencias.append(i - m)
                j = lps[j - 1]
                saltos += 1
        elif j > 0:
            j = lps[j - 1]
            saltos += 1
        else:
            i += 1

    return {
        "ocurrencias": ocurrencias,
        "comparaciones": comparaciones,
        "saltos": saltos,
        "preprocesamiento": m,      # coste de construir la tabla LPS
    }

"""
Algoritmo Boyer-Moore
=====================
Busca ocurrencias exactas comparando el patrón **de derecha a izquierda**, lo
que le permite saltar bloques enteros del texto sin mirarlos.

LA INTUICIÓN
------------
Los otros dos algoritmos leen el texto entero. Boyer-Moore se salta partes, y
es más rápido cuanto más largo sea el patrón. Suena contradictorio hasta que
se ve el mecanismo.

La clave es comparar empezando por el ÚLTIMO carácter del patrón:

    Texto:   E S T O   E S   U N   E J E M P L O
    Patrón:  E J E M P L O
             . . . . . . ✗

             Compara primero la 'O' final del patrón contra la ' ' del texto.
             No coinciden. Y además el espacio NO APARECE en el patrón.

             Si el espacio no está en el patrón, ninguna alineación que lo
             incluya puede funcionar. Se puede saltar el patrón entero:

    Texto:   E S T O   E S   U N   E J E M P L O
    Patrón:          E J E M P L O
                     ↑ siete posiciones de un golpe, sin mirarlas

Una sola comparación descartó siete posiciones. Ese es el truco.

LA REGLA DEL CARÁCTER MALO
--------------------------
El "carácter malo" es el del texto que provocó el fallo. Hay dos casos:

  1. **No está en el patrón** → se desplaza el patrón hasta pasarlo por
     completo. El salto es grande.

  2. **Sí está en el patrón** → se desplaza hasta alinear su aparición más a
     la derecha con esa posición del texto. Si esa aparición queda a la
     derecha del punto de fallo, alinearla retrocedería, así que se avanza
     solo una posición.

Se precalcula una tabla con la última posición de cada carácter del patrón.

    Patrón "EJEMPLO":
        E → 2   (aparece en 0 y 2; se guarda la mayor)
        J → 1
        M → 3
        P → 4
        L → 5
        O → 6

COMPLEJIDAD: EL MATIZ IMPORTANTE
---------------------------------
    Preprocesamiento: O(m + σ), con σ el tamaño del alfabeto
    Mejor caso:       O(n / m)   ← sublineal: no lee todo el texto
    Peor caso:        O(n · m)   ← solo con la regla del carácter malo
    Espacio:          O(σ)

Es el único de los tres que puede ser **sublineal**: con un patrón largo sobre
texto natural, salta la mayoría de los caracteres. Por eso en la práctica suele
ganar, pese a tener peor cota teórica que KMP.

El peor caso O(n·m) aparece con textos muy repetitivos ("AAAA..." buscando
"BAAA"). La implementación completa de Boyer-Moore añade la **regla del sufijo
bueno**, que lo baja a O(n); aquí se implementan ambas reglas y se toma el
mayor de los dos saltos, que es lo que hace el algoritmo original.
"""

from __future__ import annotations


def tabla_caracter_malo(patron: str) -> dict[str, int]:
    """
    Última posición de cada carácter dentro del patrón.

    Se guarda la posición MÁS A LA DERECHA porque al desplazar se busca la
    alineación más conservadora: saltar hasta la aparición más lejana podría
    pasarse de largo una ocurrencia válida.

    Complejidad: O(m). Espacio: O(σ), con σ los caracteres distintos.
    """
    tabla: dict[str, int] = {}
    for i, c in enumerate(patron):
        tabla[c] = i        # una aparición posterior sobrescribe la anterior
    return tabla


def tabla_sufijo_bueno(patron: str) -> list[int]:
    """
    Desplazamientos por la regla del sufijo bueno.

    Mientras la regla del carácter malo mira el carácter que FALLÓ, esta mira
    el trozo que SÍ coincidió (el "sufijo bueno") y se pregunta dónde más
    aparece ese trozo dentro del patrón.

        Patrón "ABCAB", falla tras emparejar el sufijo "AB".
        Ese "AB" también aparece al principio del patrón, así que se desplaza
        hasta alinear ambas apariciones, en vez de avanzar a ciegas.

    `desplazamiento[j]` indica cuánto mover el patrón cuando el fallo ocurre
    en la posición j. Se construye en dos recorridos:

      1. Casos donde el sufijo bueno reaparece completo en otro punto.
      2. Casos donde solo un PREFIJO del patrón coincide con el final del
         sufijo bueno — el desplazamiento debe alinear ese prefijo.

    Complejidad: O(m). Espacio: O(m).
    """
    m = len(patron)
    desplazamiento = [0] * (m + 1)
    borde = [0] * (m + 1)       # borde[i]: inicio del borde ancho del sufijo i

    # --- Recorrido 1: el sufijo bueno reaparece dentro del patrón ---------
    i, j = m, m + 1
    borde[i] = j
    while i > 0:
        while j <= m and patron[i - 1] != patron[j - 1]:
            if desplazamiento[j] == 0:
                desplazamiento[j] = j - i
            j = borde[j]
        i -= 1
        j -= 1
        borde[i] = j

    # --- Recorrido 2: solo un prefijo del patrón coincide -----------------
    j = borde[0]
    for i in range(m + 1):
        if desplazamiento[i] == 0:
            desplazamiento[i] = j
        if i == j:
            j = borde[j]

    return desplazamiento


def buscar(texto: str, patron: str) -> list[int]:
    """
    Devuelve todas las posiciones del texto donde empieza el patrón.

    El bucle principal alinea el patrón en la posición `desplazado` y compara
    de derecha a izquierda. Al fallar, toma el MAYOR de los dos saltos que
    proponen las dos reglas: ambas son seguras, así que el mayor también lo es
    y avanza más rápido.

    Complejidad: O(n/m) en el mejor caso, O(n) en el peor con ambas reglas.
    """
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return []

    malo = tabla_caracter_malo(patron)
    bueno = tabla_sufijo_bueno(patron)

    ocurrencias: list[int] = []
    desplazado = 0

    while desplazado <= n - m:
        j = m - 1

        # Comparación de derecha a izquierda
        while j >= 0 and patron[j] == texto[desplazado + j]:
            j -= 1

        if j < 0:
            # Se emparejó todo el patrón
            ocurrencias.append(desplazado)
            desplazado += bueno[0]
        else:
            caracter_malo = texto[desplazado + j]
            ultima = malo.get(caracter_malo, -1)
            # max(1, ...) evita retroceder cuando la última aparición del
            # carácter queda a la derecha del punto de fallo.
            salto_malo = max(1, j - ultima)
            salto_bueno = bueno[j + 1]
            desplazado += max(salto_malo, salto_bueno)

    return ocurrencias


# --------------------------------------------------------------------------
# Instrumentación para el documento técnico
# --------------------------------------------------------------------------

def buscar_con_pasos(texto: str, patron: str, max_pasos: int = 12
                     ) -> tuple[list[int], list[dict]]:
    """
    Búsqueda que registra cada alineación y el salto que la siguió.

    Sirve para graficar las iteraciones y desplazamientos sobre el texto, que
    es justo lo que pide el análisis del requerimiento.
    """
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return [], []

    malo = tabla_caracter_malo(patron)
    bueno = tabla_sufijo_bueno(patron)
    ocurrencias: list[int] = []
    pasos: list[dict] = []
    desplazado = 0

    while desplazado <= n - m:
        j = m - 1
        comparaciones = 0
        while j >= 0 and patron[j] == texto[desplazado + j]:
            j -= 1
            comparaciones += 1

        if j < 0:
            comparaciones += 0
            salto = bueno[0]
            registro = {
                "posicion": desplazado,
                "ventana": texto[desplazado:desplazado + m],
                "comparaciones": comparaciones,
                "resultado": "OCURRENCIA",
                "salto": salto,
            }
            ocurrencias.append(desplazado)
        else:
            comparaciones += 1
            c = texto[desplazado + j]
            ultima = malo.get(c, -1)
            salto_malo = max(1, j - ultima)
            salto_bueno = bueno[j + 1]
            salto = max(salto_malo, salto_bueno)
            registro = {
                "posicion": desplazado,
                "ventana": texto[desplazado:desplazado + m],
                "comparaciones": comparaciones,
                "resultado": f"falla en j={j}, caracter malo '{c}'"
                             + ("" if ultima >= 0 else " (no esta en el patron)"),
                "salto_malo": salto_malo,
                "salto_bueno": salto_bueno,
                "salto": salto,
            }

        if len(pasos) < max_pasos:
            pasos.append(registro)
        desplazado += salto

    return ocurrencias, pasos


def buscar_con_metricas(texto: str, patron: str) -> dict:
    """Búsqueda que cuenta comparaciones, alineaciones probadas y saltos."""
    n, m = len(texto), len(patron)
    if m == 0 or m > n:
        return {"ocurrencias": [], "comparaciones": 0, "alineaciones": 0,
                "caracteres_saltados": 0, "preprocesamiento": 0}

    malo = tabla_caracter_malo(patron)
    bueno = tabla_sufijo_bueno(patron)
    ocurrencias: list[int] = []
    comparaciones = alineaciones = 0
    desplazado = 0

    while desplazado <= n - m:
        alineaciones += 1
        j = m - 1
        while j >= 0 and patron[j] == texto[desplazado + j]:
            comparaciones += 1
            j -= 1

        if j < 0:
            ocurrencias.append(desplazado)
            desplazado += bueno[0]
        else:
            comparaciones += 1
            ultima = malo.get(texto[desplazado + j], -1)
            desplazado += max(max(1, j - ultima), bueno[j + 1])

    return {
        "ocurrencias": ocurrencias,
        "comparaciones": comparaciones,
        "alineaciones": alineaciones,
        # Cuántos caracteres del texto NUNCA se miraron: la ventaja del método
        "caracteres_saltados": max(0, n - comparaciones),
        "preprocesamiento": m + len(malo),
    }

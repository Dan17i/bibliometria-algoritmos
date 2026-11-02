"""
Demostración de la búsqueda de patrones (Requerimiento 3)
=========================================================
Arma lo que muestra la aplicación al oprimir "Ver demostración" en una fila de la
búsqueda: la idea, las estructuras auxiliares con sus valores, cada iteración sobre el
texto y la comparación de los tres algoritmos en el corpus.

La traza se hace sobre un fragmento del abstract donde aparece el patrón, porque sobre
los 28 mil caracteres del corpus no se podría seguir a mano. Con «learning» se usa la
oración del caso de estudio del documento técnico. También se aceptan un texto y un
patrón escritos por quien está mirando, para comparar con un cálculo en papel.
"""

from __future__ import annotations

from src.benchmark.runner import fases_de_busqueda, medir
from src.pattern import boyer_moore, kmp, rabin_karp, traza

MODULOS = {"kmp": kmp, "boyer_moore": boyer_moore, "rabin_karp": rabin_karp}
NOMBRES = {"kmp": "KMP", "boyer_moore": "Boyer-Moore", "rabin_karp": "Rabin-Karp"}

LIMITE_PATRON = 120
LIMITE_TEXTO = 300

# Cinco repeticiones bastan para mostrar el orden de los tiempos en una consulta. El
# análisis del documento técnico usa 101 (src/benchmark/runner.py).
REPETICIONES_EN_VIVO = 5


ESCAPES = {
    "\\": r"\backslash", "{": r"\{", "}": r"\}", "%": r"\%", "#": r"\#", "&": r"\&",
    "$": r"\$", "_": r"\_", "^": r"\hat{}", "~": r"\sim",
}


def _tex(c: str) -> str:
    """Un carácter del texto dentro de una fórmula."""
    if c == " ":
        return r"\text{espacio}"
    if c == "\n":
        return r"\text{salto de línea}"
    if c in ESCAPES:
        return ESCAPES[c]
    return rf"\text{{{c}}}"


def _ver(c: str) -> str:
    """Un carácter del texto dentro de una frase."""
    if c == " ":
        return "espacio"
    if c == "\n":
        return "salto de línea"
    return f"«{c}»"


def fragmento(registros: list[dict], patron: str) -> tuple[str, str, str]:
    """
    (id del artículo, fragmento, nota) sobre el que se hace la traza.

    Se recorta alrededor de la primera ocurrencia, en límites de palabra para que se
    pueda leer. Si el patrón no aparece, se usa el inicio del primer abstract.
    """
    if patron == traza.PATRON:
        for r in registros:
            if r["id"] == traza.ID_ARTICULO:
                return r["id"], traza.extraer_oracion(r["texto"]), "la oración del caso de estudio del documento técnico"

    for r in registros:
        posiciones = kmp.buscar(r["texto"], patron)
        if posiciones:
            texto, pos = r["texto"], posiciones[0]
            inicio = pos - 50 if pos > 50 else 0
            while 0 < inicio < pos and texto[inicio - 1] != " ":
                inicio += 1
            fin = pos + len(patron) + 60
            if fin >= len(texto):
                fin = len(texto)
            else:
                while fin > pos + len(patron) and texto[fin - 1] != " ":
                    fin -= 1
            return r["id"], texto[inicio:fin], "el fragmento donde aparece por primera vez"

    primero = registros[0]
    fin = 140
    while fin > 0 and primero["texto"][fin - 1] != " ":
        fin -= 1
    return primero["id"], primero["texto"][:fin], "el inicio de este abstract, porque el patrón no aparece en el corpus"


# --------------------------------------------------------------------------
# KMP
# --------------------------------------------------------------------------

def _kmp(texto: str, patron: str) -> dict:
    lps, pasos_lps = kmp.construir_lps_con_pasos(patron)
    ocurrencias, pasos = kmp.buscar_con_pasos(texto, patron)
    m = len(patron)

    visual_pasos = []
    inicial = None
    for k, p in enumerate(pasos):
        estados = ["si"] * p["j"] + ["si" if p["coincide"] else "no"] + ["-"] * (m - p["j"] - 1)
        if inicial is None and p["j"] > 0:
            inicial = k
        relacion = "=" if p["coincide"] else r"\neq"
        visual_pasos.append({
            "alineacion": p["alineacion"],
            "cursor": p["i"],
            "estados": estados,
            "latex": rf"i = {p['i']},\; j = {p['j']}: \quad {_tex(p['caracter_texto'])} {relacion} {_tex(p['caracter_patron'])}",
            "texto": f"Se compara {_ver(p['caracter_texto'])} del texto con {_ver(p['caracter_patron'])} del patrón: {p['accion']}.",
        })

    filas_lps = [["i"] + list(range(m)), ["patron[i]"] + [c if c != " " else "␣" for c in patron], ["lps[i]"] + lps]
    con_borde = [p for p in pasos_lps if "borde crece" in p["accion"]]
    if con_borde:
        ejemplo = con_borde[0]
        lectura_lps = (f"En i = {ejemplo['i']}, {ejemplo['comparacion'].replace('==', 'es igual a')}: el borde crece. "
                       "Al fallar más adelante, ese prefijo no se vuelve a comparar.")
    else:
        lectura_lps = ("Ningún prefijo de este patrón es también sufijo, así que todos los valores son 0: al fallar no se conserva "
                       "nada de lo emparejado. Con ABABC, en cambio, la tabla es [0, 0, 1, 2, 0].")

    resultado = kmp.buscar_con_metricas(texto, patron)
    return {
        "que_es": "Es un algoritmo que busca una palabra o frase dentro de un texto largo. Se encarga de recorrer el texto de izquierda a derecha sin devolverse nunca, "
                  "y aprovecha lo que ya leyó para no comparar dos veces lo mismo. A lo que se busca se le llama patrón.",
        "formula_simple": "El puntero i marca por dónde va el texto, T, y j por dónde va el patrón, P. La tabla lps guarda, para cada posición del patrón, cuántas letras de su inicio se repiten justo "
                          "al final de lo que lleva hasta ahí. Cuando una letra falla, j retrocede a lo que dice esa tabla, e i se queda donde está.",
        "idea": "Cuando falla, KMP ya sabe qué parte del patrón coincidió, y usa eso para seguir sin volver atrás en el texto. "
                "La búsqueda ingenua retrocede y vuelve a comparar lo que ya conocía; KMP no.",
        "formulas": [
            r"\text{lps}[i] = \text{longitud del prefijo propio más largo de } P[0..i] \text{ que también es sufijo}",
            r"j \leftarrow \text{lps}[j - 1] \quad \text{cuando } T[i] \neq P[j] \text{ y } j > 0",
        ],
        "pasos": [
            {
                "titulo": "Se construye la tabla LPS",
                "texto": f"Para cada posición del patrón, la longitud del borde más largo. {lectura_lps}",
                "latex": [rf"\text{{lps}} = [{', '.join(str(x) for x in lps)}]"],
            },
            {
                "titulo": "Se recorre el texto con dos punteros",
                "texto": "i avanza sobre el texto y nunca retrocede; j avanza sobre el patrón. Si coinciden, avanzan los dos. "
                         "Si fallan con j > 0, j baja según la tabla e i se queda quieto. Si fallan con j = 0, avanza i.",
                "latex": [],
            },
            {
                "titulo": "Resultado",
                "texto": f"{resultado['comparaciones']} comparaciones sobre {len(texto)} caracteres. El puntero i solo avanza, y j solo "
                         "baja tantas veces como subió, así que el total queda acotado por 2n.",
                "latex": [rf"\text{{comparaciones}} = {resultado['comparaciones']} \leq 2n = {2 * len(texto)}"],
            },
        ],
        "visual": [
            {"tipo": "tabla", "titulo": "Tabla LPS", "columnas": filas_lps[0], "filas": filas_lps[1:]},
            {"tipo": "busqueda", "texto": texto, "patron": patron, "pasos": visual_pasos,
             "inicial": inicial if inicial is not None else 0,
             "aviso": "Cada paso es una comparación. Observa que el patrón se desplaza, pero el texto nunca se vuelve a leer hacia atrás."},
        ],
        "ocurrencias": ocurrencias,
        "comparaciones": resultado["comparaciones"],
        "codigo": {
            "archivo": "src/pattern/kmp.py",
            "funciones": [["construir_lps", "construye la tabla, en O(m)"], ["buscar, buscar_con_tabla", "recorren el texto, en O(n)"]],
            "tiempo": "O(n + m)", "espacio": "O(m)",
            "porque": "el puntero del texto nunca retrocede y el del patrón solo baja tantas veces como subió",
        },
        "interpretacion": "KMP trabaja más o menos lo mismo sin importar qué se busque: alrededor de una comparación por cada carácter del texto, y nunca más del doble. "
                          "No tiene un caso malo, pero tampoco puede saltarse texto.",
    }


# --------------------------------------------------------------------------
# Boyer-Moore
# --------------------------------------------------------------------------

def _boyer_moore(texto: str, patron: str) -> dict:
    malo = boyer_moore.tabla_caracter_malo(patron)
    bueno = boyer_moore.tabla_sufijo_bueno(patron)
    ocurrencias, pasos = boyer_moore.buscar_con_pasos(texto, patron, max_pasos=len(texto))
    m = len(patron)

    visual_pasos = []
    for p in pasos:
        posicion = p["posicion"]
        estados = ["-"] * m
        j = m - 1
        while j >= 0 and patron[j] == texto[posicion + j]:
            estados[j] = "si"
            j -= 1
        if j < 0:
            visual_pasos.append({
                "alineacion": posicion,
                "cursor": posicion,
                "estados": estados,
                "latex": rf"\text{{salto}} = \text{{bueno}}[0] = {p['salto']}",
                "texto": f"Coinciden los {m} caracteres: hay una ocurrencia en la posición {posicion}.",
            })
            continue

        estados[j] = "no"
        c = texto[posicion + j]
        ultima = malo.get(c, -1)
        aparece = f"su última aparición en el patrón es la posición {ultima}" if ultima >= 0 else "no aparece en el patrón"
        visual_pasos.append({
            "alineacion": posicion,
            "cursor": posicion + j,
            "estados": estados,
            "latex": rf"\text{{salto}} = \max(\max(1,\; {j} - ({ultima})),\; \text{{bueno}}[{j + 1}]) = \max({p['salto_malo']},\; {p['salto_bueno']}) = {p['salto']}",
            "texto": f"De derecha a izquierda, falla en j = {j} contra {_ver(c)}, que {aparece}.",
        })

    orden = sorted(malo, key=lambda k: malo[k])
    resultado = boyer_moore.buscar_con_metricas(texto, patron)
    primer_fallo = next((v for v in visual_pasos if "no" in v["estados"]), None)
    return {
        "que_es": "Es un algoritmo que busca una palabra o frase dentro de un texto largo, pero empieza a comparar por el final de lo que busca. Se encarga de saltarse "
                  "pedazos enteros del texto cuando ve que ahí no puede estar, así que no necesita leerlo todo. A lo que se busca se le llama patrón.",
        "formula_simple": "Cada vez que algo falla, el patrón se mueve a la derecha. La fórmula calcula dos saltos posibles y toma el mayor: uno según la letra que falló, c, y su última "
                          "posición en el patrón; y otro, bueno, según el pedazo que sí coincidió. j es la posición donde ocurrió el fallo. Ninguno de los dos saltos se pasa de largo una aparición.",
        "idea": "Compara el patrón de derecha a izquierda. Si el carácter del texto que falló no está en el patrón, ninguna alineación que lo "
                "incluya puede funcionar y el patrón salta completo. Por eso puede terminar sin leer todo el texto.",
        "formulas": [
            r"\text{salto} = \max\big(\max(1,\; j - \text{última}(c)),\; \text{bueno}[j + 1]\big)",
        ],
        "pasos": [
            {
                "titulo": "Se construye la tabla del carácter malo",
                "texto": "Guarda la última posición de cada carácter del patrón. Un carácter que no está cuenta como −1.",
                "latex": [],
            },
            {
                "titulo": "Se construye la tabla del sufijo bueno",
                "texto": "Dice cuánto mover el patrón según el trozo que sí coincidió, buscando dónde más aparece ese trozo dentro del patrón.",
                "latex": [rf"\text{{bueno}} = [{', '.join(str(x) for x in bueno)}]"],
            },
            {
                "titulo": "En cada alineación se toma el mayor de los dos saltos",
                "texto": "Las dos reglas son seguras: ninguna se salta una ocurrencia. Por eso el mayor también lo es."
                         + (f" Primera alineación: {primer_fallo['texto']}" if primer_fallo else ""),
                "latex": [primer_fallo["latex"]] if primer_fallo else [],
            },
            {
                "titulo": "Resultado",
                "texto": f"{len(pasos)} alineaciones y {resultado['comparaciones']} comparaciones sobre {len(texto)} caracteres.",
                "latex": [],
            },
        ],
        "visual": [
            {"tipo": "tabla", "titulo": "Tabla del carácter malo", "columnas": ["Carácter"] + [c if c != " " else "␣" for c in orden],
             "filas": [["Última posición"] + [malo[c] for c in orden]]},
            {"tipo": "busqueda", "texto": texto, "patron": patron, "pasos": visual_pasos, "inicial": 0,
             "aviso": "Cada paso es una alineación. Observa cuánto salta el patrón: lo que queda en medio nunca se lee."},
        ],
        "ocurrencias": ocurrencias,
        "comparaciones": resultado["comparaciones"],
        "codigo": {
            "archivo": "src/pattern/boyer_moore.py",
            "funciones": [["tabla_caracter_malo, tabla_sufijo_bueno", "construyen las tablas, en O(m + σ)"], ["buscar, buscar_con_tablas", "recorren el texto"]],
            "tiempo": "O(n / m) en el mejor caso y O(n · m) en el peor", "espacio": "O(m + σ)",
            "porque": "cada alineación compara hasta m caracteres y salta hasta m posiciones. El peor caso aparece cuando el patrón se repite muchas veces",
        },
        "interpretacion": "Entre más largo es lo que se busca, más puede saltar, porque cada salto llega como máximo al largo del patrón. Por eso en texto normal, con patrones de varias letras, "
                          "suele ser el más rápido. Su punto débil es un texto muy repetitivo: buscando «aaa» en «aaaaaa…» compara el patrón completo en cada posición.",
    }


# --------------------------------------------------------------------------
# Rabin-Karp
# --------------------------------------------------------------------------

def _rabin_karp(texto: str, patron: str) -> dict:
    b, q = rabin_karp.BASE, rabin_karp.PRIMO
    factor, hash_patron, _ = rabin_karp.preprocesar(texto, patron)
    ocurrencias, ventanas = rabin_karp.buscar_con_pasos(texto, patron, max_pasos=len(texto))
    m = len(patron)

    filas_hash = []
    h = 0
    for c in patron:
        crudo = h * b + ord(c)
        h = crudo % q
        filas_hash.append([c if c != " " else "␣", ord(c), crudo, h])

    visual_pasos = []
    for k, v in enumerate(ventanas):
        i = v["posicion"]
        estados = ["-"] * m
        if v["coincide_hash"]:
            j = 0
            while j < m and texto[i + j] == patron[j]:
                estados[j] = "si"
                j += 1
            if j < m:
                estados[j] = "no"

        if k == 0:
            latex = rf"h = \text{{hash}}(\text{{primera ventana}}) = {v['hash_ventana']}"
        else:
            previa = ventanas[k - 1]["hash_ventana"]
            sale, entra = ord(texto[i - 1]), ord(texto[i + m - 1])
            latex = rf"h = (({previa} - {sale} \cdot {factor}) \cdot {b} + {entra}) \bmod {q} = {v['hash_ventana']}"

        if v["ocurrencia"]:
            explica = f"El hash es igual al del patrón ({hash_patron}) y al comparar los caracteres se confirma: ocurrencia en {i}."
        elif v["colision"]:
            explica = f"El hash es igual al del patrón, pero los caracteres no coinciden: es una colisión y se descarta."
        else:
            explica = f"El hash es distinto del del patrón ({hash_patron}): se descarta sin comparar ningún carácter."
        visual_pasos.append({"alineacion": i, "cursor": i, "estados": estados, "latex": latex, "texto": explica})

    q_pequeno = traza.PRIMO_PEQUENO
    _, con_q_pequeno = rabin_karp.buscar_con_pasos(texto, patron, max_pasos=len(texto), primo=q_pequeno)
    colisiones = [v for v in con_q_pequeno if v["colision"]]
    if colisiones:
        ejemplo = colisiones[0]
        texto_colision = (f"Con q = {_miles(q)} no hay colisiones en este texto. Si se repite con q = {q_pequeno}, la ventana «{ejemplo['ventana']}» "
                          f"en la posición {ejemplo['posicion']} da el mismo hash que el patrón ({ejemplo['hash_patron']}) sin ser igual. "
                          "Por eso, cuando dos hashes coinciden, siempre se comparan los caracteres.")
    else:
        texto_colision = (f"Con q = {_miles(q)} no hay colisiones en este texto, y tampoco con q = {q_pequeno}. Pueden existir: dos cadenas distintas "
                          "pueden dar el mismo hash, por eso cuando coinciden siempre se comparan los caracteres.")

    resultado = rabin_karp.buscar_con_metricas(texto, patron)
    n_ventanas = len(ventanas)
    ej_sale, ej_entra = (ord(texto[0]), ord(texto[m])) if n_ventanas > 1 else (0, 0)
    pasos = [
        {
            "titulo": "Se calcula el hash del patrón, carácter por carácter",
            "texto": f"Con base b = {b} y módulo q = {_miles(q)}. Se aplica el módulo en cada paso para que el número no crezca.",
            "latex": [rf"h \leftarrow (h \cdot {b} + \text{{código}}(c)) \bmod q", rf"\text{{hash}}(P) = {hash_patron}"],
        },
        {
            "titulo": "Se calcula el peso del carácter que sale de la ventana",
            "texto": "Es lo que hay que restar para quitar el primer carácter de la ventana.",
            "latex": [rf"f = b^{{m - 1}} \bmod q = {b}^{{{m - 1}}} \bmod {q} = {factor}"],
        },
    ]
    if n_ventanas > 1:
        pasos.append({
            "titulo": "Cada ventana se calcula a partir de la anterior",
            "texto": "Quita el carácter que sale, desplaza un lugar y añade el que entra. Es una operación constante, sin recorrer la ventana.",
            "latex": [rf"h_{{1}} = (({ventanas[0]['hash_ventana']} - {ej_sale} \cdot {factor}) \cdot {b} + {ej_entra}) \bmod {q} = {ventanas[1]['hash_ventana']}"],
        })
    pasos += [
        {"titulo": "El hash descarta, pero no confirma", "texto": texto_colision, "latex": []},
        {
            "titulo": "Resultado",
            "texto": f"{resultado['comparaciones']} comparaciones: {n_ventanas} de hashes, una por ventana, y "
                     f"{resultado['comparaciones'] - n_ventanas} de caracteres para confirmar.",
            "latex": [],
        },
    ]
    return {
        "que_es": "Es un algoritmo que busca una palabra o frase dentro de un texto largo convirtiendo las letras en números. Se encarga de calcular un número para lo que se busca "
                  "y otro para cada pedazo del texto del mismo tamaño. Solo cuando los dos números coinciden revisa letra por letra si de verdad es lo que se busca. A lo que se busca se le llama patrón.",
        "formula_simple": "La primera fórmula convierte un pedazo de texto en un número, como si sus letras fueran las cifras de un número en base b. Con mod q se queda solo con el residuo de dividir "
                          "entre q, para que el número no crezca demasiado. La segunda calcula el número del pedazo siguiente a partir del anterior: quita la letra que sale y agrega la que entra.",
        "idea": "En vez de comparar letras, convierte cada pedazo de texto en un número y compara números. El número se actualiza al deslizar "
                "la ventana (el pedazo del texto que se está mirando), sin recalcularlo. Es como una huella: si no coincide, las cadenas son distintas; si coincide, hay que confirmar.",
        "formulas": [
            r"\text{hash}(s) = \left(s_0 b^{m-1} + s_1 b^{m-2} + \dots + s_{m-1}\right) \bmod q",
            r"h_{i+1} = \big((h_i - T[i] \cdot b^{m-1}) \cdot b + T[i + m]\big) \bmod q",
        ],
        "pasos": pasos,
        "visual": [
            {"tipo": "tabla", "titulo": "Hash del patrón, paso a paso", "columnas": ["Carácter", "Código", "h · b + código", "mod q"], "filas": filas_hash},
            {"tipo": "busqueda", "texto": texto, "patron": patron, "pasos": visual_pasos,
             "inicial": ocurrencias[0] if ocurrencias else 0,
             "aviso": "Cada paso es una ventana, y su hash es el número que la representa. Solo se comparan las letras cuando ese número es igual al del patrón."},
        ],
        "ocurrencias": ocurrencias,
        "comparaciones": resultado["comparaciones"],
        "codigo": {
            "archivo": "src/pattern/rabin_karp.py",
            "funciones": [["calcular_hash, preprocesar", "el hash del patrón y de la primera ventana, en O(m)"], ["buscar, buscar_con_hashes", "el rolling hash y la verificación"]],
            "tiempo": "O(n + m) en promedio y O(n · m) en el peor caso", "espacio": "O(1)",
            "porque": "el rolling hash es constante por ventana, pero cada coincidencia de hash verifica hasta m caracteres",
        },
        "interpretacion": "En cada ventana hace una operación aritmética, y eso cuesta más que comparar dos letras, por eso en la práctica es el "
                          "más lento de los tres. Su ventaja real está en buscar varios patrones a la vez, porque el mismo recorrido sirve para todos.",
    }


def _miles(valor: int) -> str:
    return f"{valor:,}".replace(",", " ")


# --------------------------------------------------------------------------
# Comparación en el corpus
# --------------------------------------------------------------------------

def comparacion_en_corpus(corpus_unido: str, patron: str) -> dict:
    """Los tres algoritmos sobre el corpus: tiempos de cada fase, comparaciones y conclusiones."""
    fases = fases_de_busqueda(corpus_unido, patron)
    filas = {}
    for clave, (preparar, buscar) in fases.items():
        metricas = MODULOS[clave].buscar_con_metricas(corpus_unido, patron)
        pre = medir(preparar, REPETICIONES_EN_VIVO)["mediana"] * 1000
        bus = medir(buscar, REPETICIONES_EN_VIVO)["mediana"] * 1000
        filas[clave] = {"pre": pre, "bus": bus, "comparaciones": metricas["comparaciones"], "ocurrencias": len(metricas["ocurrencias"])}

    mas_rapido = min(filas, key=lambda k: filas[k]["bus"])
    menos_comparaciones = min(filas, key=lambda k: filas[k]["comparaciones"])
    mas_lento = max(filas, key=lambda k: filas[k]["bus"])
    pre_maximo = max(100 * f["pre"] / (f["pre"] + f["bus"]) for f in filas.values())
    m = len(patron)

    conclusiones = [
        f"Los tres encuentran las mismas {filas['kmp']['ocurrencias']} ocurrencias de un patrón de {m} caracteres en {_miles(len(corpus_unido))} caracteres.",
        f"El más rápido en esta consulta es {NOMBRES[mas_rapido]} ({filas[mas_rapido]['bus']:.2f} ms) y el más lento {NOMBRES[mas_lento]} ({filas[mas_lento]['bus']:.2f} ms). "
        f"El que menos compara es {NOMBRES[menos_comparaciones]} ({_miles(filas[menos_comparaciones]['comparaciones'])}).",
    ]
    if mas_rapido != menos_comparaciones:
        conclusiones.append("Comparar menos no siempre es tardar menos: con patrones muy cortos, Boyer-Moore compara menos que KMP, "
                            "pero cada alineación le cuesta más trabajo que un paso de KMP.")
    conclusiones.append(f"El preprocesamiento pesa como máximo el {pre_maximo:.2f} % del tiempo: en estos patrones casi todo el costo está en recorrer el texto.")
    conclusiones.append("Estos tiempos son la mediana de 5 ejecuciones en este servidor y sirven para ver el orden. Las mediciones del documento técnico usan 101 repeticiones.")

    return {
        "visual": {
            "tipo": "tabla",
            "titulo": f"Los tres algoritmos sobre el corpus completo (n = {_miles(len(corpus_unido))})",
            "columnas": ["Algoritmo", "Preproceso (ms)", "Búsqueda (ms)", "Comparaciones"],
            "filas": [[NOMBRES[k], f"{filas[k]['pre']:.4f}", f"{filas[k]['bus']:.3f}", _miles(filas[k]["comparaciones"])] for k in filas],
        },
        "conclusiones": conclusiones,
    }


# --------------------------------------------------------------------------
# Punto de entrada
# --------------------------------------------------------------------------

def demostracion(clave: str, texto: str, patron: str) -> dict:
    """La demostración de un solo algoritmo sobre un texto y un patrón."""
    resultado = {"kmp": _kmp, "boyer_moore": _boyer_moore, "rabin_karp": _rabin_karp}[clave](texto, patron)
    ocurrencias = resultado["ocurrencias"]
    n_oc = len(ocurrencias)
    donde = f"en las posiciones {', '.join(str(x) for x in ocurrencias)}" if n_oc > 1 else f"en la posición {ocurrencias[0]}" if n_oc == 1 else ""
    if n_oc == 0:
        frase = f"El patrón no aparece en este texto. {NOMBRES[clave]} lo determina con {resultado['comparaciones']} comparaciones."
    else:
        frase = f"Encuentra el patrón {donde}, con {resultado['comparaciones']} comparaciones sobre {len(texto)} caracteres."
    resultado.update({
        "tipo": "patron",
        "clave": clave,
        "algoritmo": NOMBRES[clave],
        "familia": "Búsqueda de patrones",
        "valor_texto": str(n_oc),
        "etiqueta_valor": "ocurrencia" if n_oc == 1 else "ocurrencias",
        "que_significa": "Es el número de veces que aparece lo que se busca en este texto.",
        "en_una_frase": frase,
        "patron": patron,
        "texto_usado": texto,
        "ms": 0.0,
        "acepta_textos": True,
    })
    return resultado

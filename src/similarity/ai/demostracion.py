"""
Demostración paso a paso de los 2 enfoques de IA sobre un par de abstracts.
---------------------------------------------------------------------------
Reúne, en un diccionario y en markdown, todos los números intermedios que el
documento necesita (igual que Samuel muestra la matriz de PD o las frecuencias):

    Enfoque 1 (promedio de vectores de palabras):
        tokens -> stopwords eliminadas -> tokens que quedan -> vector promedio
        (suma y división, primeras dimensiones) -> producto punto -> normas ->
        coseno -> ángulo -> euclidiana.
    Enfoque 2 (embedding de oración): vectores dados por el modelo -> producto
        punto -> normas -> coseno -> ángulo -> euclidiana.

Las funciones son puras (reciben los vectores de palabras o de oración ya
obtenidos del modelo), así que se prueban sin descargar nada.
"""
from src.similarity.ai.embeddings import average_word_vectors
from src.similarity.metrics import cosine_steps, euclidean_steps


def compare_vectors_steps(va: list[float], vb: list[float], dims: int = 5) -> dict:
    """
    Pasos de la comparación de dos vectores del mismo enfoque.

    Returns:
        dict con `dim`, `first` (primeras `dims` dimensiones: a_i, b_i, a_i*b_i),
        y los resultados completos: dot, norm_a, norm_b, cosine, angle_degrees,
        euclidean, sum_squares.

    Complejidad temporal: O(n) (producto punto, normas y euclidiana, cada uno un
    recorrido). Complejidad espacial: O(n) por las diferencias de `euclidean_steps`.
    """
    cos = cosine_steps(va, vb)
    euc = euclidean_steps(va, vb)
    return {
        "dim": len(va),
        "first": [(i + 1, va[i], vb[i], va[i] * vb[i]) for i in range(min(dims, len(va)))],
        "dot": cos["dot"],
        "norm_a": cos["norm_a"],
        "norm_b": cos["norm_b"],
        "cosine": cos["cosine"],
        "angle_degrees": cos["angle_degrees"],
        "euclidean": euc["distance"],
        "sum_squares": euc["sum_squares"],
    }


def average_steps(texto: str, word_vectors, stopwords, dims: int = 5) -> dict:
    """
    Pasos del promedio de vectores de un texto: tokens, stopwords eliminadas,
    tokens que quedan, suma y vector promedio (primeras `dims` dimensiones).

    Complejidad temporal: O(T + k*n). Complejidad espacial: O(k*n) (se conservan
    los vectores de palabras para mostrar los pasos).
    """
    res = average_word_vectors(texto, word_vectors, stopwords)
    k = len(res["token_vectors"])
    sums = [0.0] * len(res["vector"])
    for v in res["token_vectors"]:
        for j in range(len(sums)):
            sums[j] += v[j]
    return {
        "tokens_totales": len(res["tokens"]) + len(res["removed_stopwords"]),
        "stopwords_eliminadas": len(res["removed_stopwords"]),
        "fuera_de_vocabulario": res["missing_tokens"],
        "tokens_que_quedan": res["used_tokens"],
        "k": k,
        "suma": sums[:dims],
        "promedio": res["vector"][:dims],
        "vector": res["vector"],
    }


def _fmt(x: float, nd: int = 4) -> str:
    return f"{x:.{nd}f}"


def _tabla_dimensiones(pasos: dict, nombre_a: str, nombre_b: str) -> list[str]:
    lineas = [f"| i | {nombre_a}[i] | {nombre_b}[i] | producto |", "|---|---|---|---|"]
    for i, a, b, p in pasos["first"]:
        lineas.append(f"| {i} | {_fmt(a, 5)} | {_fmt(b, 5)} | {_fmt(p, 6)} |")
    lineas.append(f"| … | … | … | … (hasta i = {pasos['dim']}) |")
    return lineas


def _bloque_comparacion(pasos: dict, nombre_a: str, nombre_b: str) -> list[str]:
    n = pasos["dim"]
    return [
        *_tabla_dimensiones(pasos, nombre_a, nombre_b),
        "",
        f"- Producto punto: $A\\cdot B=\\sum_{{i=1}}^{{{n}}}A_iB_i = {_fmt(pasos['dot'], 6)}$",
        f"- Normas: $\\|A\\|=\\sqrt{{\\sum A_i^2}}={_fmt(pasos['norm_a'], 6)}$, "
        f"$\\|B\\|=\\sqrt{{\\sum B_i^2}}={_fmt(pasos['norm_b'], 6)}$",
        f"- Coseno: $\\cos\\theta=\\dfrac{{A\\cdot B}}{{\\|A\\|\\,\\|B\\|}}="
        f"\\dfrac{{{_fmt(pasos['dot'], 6)}}}{{{_fmt(pasos['norm_a'], 6)}\\times{_fmt(pasos['norm_b'], 6)}}}"
        f"={_fmt(pasos['cosine'], 6)}$",
        f"- Ángulo: $\\theta=\\arccos({_fmt(pasos['cosine'], 6)})={_fmt(pasos['angle_degrees'], 2)}^\\circ$",
        f"- Euclidiana: $d=\\sqrt{{\\sum(A_i-B_i)^2}}=\\sqrt{{{_fmt(pasos['sum_squares'], 6)}}}"
        f"={_fmt(pasos['euclidean'], 6)}$",
    ]


def demostracion_markdown(
    id_a: str,
    id_b: str,
    texto_a: str,
    texto_b: str,
    word_vectors,
    stopwords,
    minilm_a: list[float],
    minilm_b: list[float],
    dims: int = 5,
) -> str:
    """
    Markdown de la demostración completa del par (id_a, id_b) con los 2 enfoques.
    El enfoque 1 se muestra sin stopwords (la variante que se propone usar).
    """
    sa = average_steps(texto_a, word_vectors, stopwords, dims)
    sb = average_steps(texto_b, word_vectors, stopwords, dims)
    comp1 = compare_vectors_steps(sa["vector"], sb["vector"], dims)
    comp2 = compare_vectors_steps(minilm_a, minilm_b, dims)

    out = [f"#### Demostración paso a paso: {id_a} vs {id_b}", ""]
    out += [
        "##### Enfoque 1: promedio de vectores GloVe-100 (sin stopwords)",
        "",
        "**Paso 1. Tokens que quedan.** Se pasa a minúsculas, se extraen las palabras con una expresión "
        "regular y se eliminan las stopwords de `src/similarity/ai/stopwords.py`.",
        "",
        "| Abstract | Tokens totales | Stopwords eliminadas | Tokens que quedan (k) | Fuera de vocabulario |",
        "|---|---|---|---|---|",
        f"| {id_a} | {sa['tokens_totales']} | {sa['stopwords_eliminadas']} | {sa['k']} | {sa['fuera_de_vocabulario'] or '—'} |",
        f"| {id_b} | {sb['tokens_totales']} | {sb['stopwords_eliminadas']} | {sb['k']} | {sb['fuera_de_vocabulario'] or '—'} |",
        "",
    ]
    for ident, s in ((id_a, sa), (id_b, sb)):
        out += [
            f"<details><summary>Los {s['k']} tokens que quedan de {ident}</summary>",
            "",
            " ".join(s["tokens_que_quedan"]),
            "",
            "</details>",
            "",
        ]
    out += [
        "**Paso 2. Vector promedio.** Para cada dimensión $j$: "
        "$d_j=\\dfrac{1}{k}\\sum_{w=1}^{k}v_{w,j}$, donde $v_w$ es el vector GloVe de la palabra $w$ "
        "(las repeticiones cuentan). Primeras dimensiones:",
        "",
        f"| j | suma {id_a} | ÷ k={sa['k']} → d{id_a}[j] | suma {id_b} | ÷ k={sb['k']} → d{id_b}[j] |",
        "|---|---|---|---|---|",
    ]
    for j in range(len(sa["suma"])):
        out.append(
            f"| {j + 1} | {_fmt(sa['suma'][j], 4)} | {_fmt(sa['promedio'][j], 5)} | "
            f"{_fmt(sb['suma'][j], 4)} | {_fmt(sb['promedio'][j], 5)} |"
        )
    out += ["", f"**Paso 3. Comparación** ({comp1['dim']} dimensiones):", ""]
    out += _bloque_comparacion(comp1, f"d{id_a}", f"d{id_b}")

    out += [
        "",
        "##### Enfoque 2: all-MiniLM-L6-v2 (embedding de oración)",
        "",
        "**Paso 1.** El modelo recibe cada abstract completo y devuelve un vector de "
        f"{comp2['dim']} dimensiones (la inferencia del modelo es una caja negra). "
        "**Paso 2. Comparación**, con la misma fórmula:",
        "",
    ]
    out += _bloque_comparacion(comp2, f"e{id_a}", f"e{id_b}")
    return "\n".join(out) + "\n"

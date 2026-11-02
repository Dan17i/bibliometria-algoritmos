"""
Demostración del agrupamiento jerárquico (Requerimiento 2)
==========================================================
Arma lo que muestra la aplicación al oprimir "Ver demostración" en un dendrograma: el
procedimiento con los valores reales, el cálculo del CPCC y de la silueta y las
conclusiones. Se ejecuta en ese momento el agrupamiento de la representación elegida,
con los tres enlaces, porque con 20 abstracts cuesta unos milisegundos.

Las conclusiones se redactan con reglas sobre los números de esa corrida, así que
siempre coinciden con lo que se ve y no dependen de ningún servicio externo.
"""

from __future__ import annotations

import json
import os
import time

from src.clustering.evaluation import best_k_by_silhouette, cophenetic_correlation, cophenetic_matrix
from src.clustering.linkage import hierarchical_clustering
from src.preprocessing.transformacion import transformar_abstract
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.metrics import center_vectors, cosine_distance, pairwise_matrix

ENLACES = ("single", "complete", "average")
NOMBRE_ENLACE = {"single": "Single", "complete": "Complete", "average": "Average"}
REPRESENTACIONES = {
    "minilm": "all-MiniLM-L6-v2",
    "glove": "GloVe promediado",
    "glove_centrado": "GloVe centrado",
    "jaccard": "1 − Jaccard",
}


def _n(valor: float, decimales: int = 4) -> str:
    return f"{valor:.{decimales}f}"


def matriz_de_distancias(clave: str, vectores: dict, textos: list[str]) -> list[list[float]]:
    """La misma matriz con la que se dibujó el dendrograma de esa representación."""
    if clave == "jaccard":
        transformados = [transformar_abstract(t) for t in textos]
        n = len(transformados)
        dist = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                dist[i][j] = dist[j][i] = 1.0 - jaccard_similarity(transformados[i], transformados[j])[0]
        return dist
    if clave == "glove_centrado":
        centrados, _ = center_vectors(vectores["glove"])
        return pairwise_matrix(centrados, cosine_distance)
    return pairwise_matrix(vectores[clave], cosine_distance)


def _corrida(dist: list[list[float]], enlace: str) -> dict:
    fusiones, pasos = hierarchical_clustering(dist, enlace, return_steps=True)
    cpcc = cophenetic_correlation(dist, fusiones)
    silueta = best_k_by_silhouette(dist, fusiones)
    return {"fusiones": fusiones, "pasos": pasos, "cpcc": cpcc, "silueta": silueta}


def _pearson_a_mano(xs: list[float], ys: list[float]) -> dict:
    m = len(xs)
    media_x = _suma(xs) / m
    media_y = _suma(ys) / m
    cov = var_x = var_y = 0.0
    for i in range(m):
        dx, dy = xs[i] - media_x, ys[i] - media_y
        cov += dx * dy
        var_x += dx * dx
        var_y += dy * dy
    return {"m": m, "media_x": media_x, "media_y": media_y, "cov": cov, "var_x": var_x, "var_y": var_y}


def _suma(valores: list[float]) -> float:
    total = 0.0
    for v in valores:
        total += v
    return total


def _conclusiones(clave: str, enlace: str, corridas: dict) -> list[str]:
    """Texto sobre qué significa esta corrida, escrito con sus propios números."""
    orden = sorted(ENLACES, key=lambda e: corridas[e]["cpcc"]["cpcc"], reverse=True)
    mejor, peor = orden[0], orden[-1]
    c = {e: corridas[e]["cpcc"]["cpcc"] for e in ENLACES}
    altura = {e: corridas[e]["fusiones"][-1][2] for e in ENLACES}
    texto = []

    texto.append(
        f"Con {REPRESENTACIONES[clave]}, el enlace más fiel a las distancias originales es "
        f"{NOMBRE_ENLACE[mejor]} (CPCC {_n(c[mejor])}), seguido de {NOMBRE_ENLACE[orden[1]]} ({_n(c[orden[1]])}) "
        f"y {NOMBRE_ENLACE[peor]} ({_n(c[peor])}). La diferencia entre el primero y el último es de {_n(c[mejor] - c[peor])}."
    )

    if enlace == "single":
        texto.append(
            f"Single une por el par más cercano, así que un solo abstract «puente» basta para juntar dos temas. "
            f"Su última fusión ocurre a altura {_n(altura['single'], 3)}, frente a {_n(altura['complete'], 3)} de Complete: "
            f"ese encadenamiento aplana el árbol y le quita contraste."
        )
    elif enlace == "complete":
        texto.append(
            f"Complete une por el par más lejano, así que solo junta grupos cuando todos sus miembros están razonablemente cerca. "
            f"Por eso su última fusión es la más alta ({_n(altura['complete'], 3)}) y sus grupos son compactos, a costa de ser "
            f"el más sensible a un abstract aislado."
        )
    else:
        texto.append(
            f"Average usa la media de todas las distancias entre dos grupos, un punto intermedio entre los otros dos: su última "
            f"fusión es a altura {_n(altura['average'], 3)}, entre {_n(altura['single'], 3)} (Single) y {_n(altura['complete'], 3)} (Complete)."
        )

    k = corridas[enlace]["silueta"]["best_k"]
    s = corridas[enlace]["silueta"]["best_silhouette"]
    # Escala de Kaufman y Rousseeuw para la silueta
    fuerza = ("sin estructura sustancial" if s <= 0.25 else "una estructura débil" if s <= 0.5
              else "una estructura razonable" if s <= 0.7 else "una estructura fuerte")
    texto.append(
        f"La silueta de {NOMBRE_ENLACE[enlace]} es mejor con k = {k} grupos ({_n(s, 3)}): {fuerza}. "
        + ("Todo el corpus trata de inteligencia artificial en educación, así que no hay grupos bien separados y la silueta, "
           "que mira un solo corte, aporta poca información. Por eso el CPCC, que evalúa el árbol completo, es la métrica principal."
           if s <= 0.5 else "Hay estructura de grupos, aunque el CPCC sigue siendo la métrica que evalúa el árbol completo.")
    )

    if mejor == "average":
        texto.append(
            "Una advertencia: el CPCC favorece a Average por construcción, porque la altura de cada fusión es la media de las "
            "distancias originales. Que Average gane no prueba que sus grupos sean mejores temáticamente, solo que su árbol "
            "distorsiona menos las distancias."
        )
    return texto


def demostracion(clave: str, enlace: str, vectores: dict, textos: list[str], ids: list[str]) -> dict:
    inicio = time.perf_counter()
    dist = matriz_de_distancias(clave, vectores, textos)
    corridas = {e: _corrida(dist, e) for e in ENLACES}
    mia = corridas[enlace]
    n = len(ids)
    pares = n * (n - 1) // 2

    # Distancias originales y cofenéticas de todos los pares (i < j)
    cof = cophenetic_matrix(mia["fusiones"], n)
    originales, cofeneticas = mia["cpcc"]["original"], mia["cpcc"]["cophenetic"]
    momentos = _pearson_a_mano(originales, cofeneticas)

    # Un grupo se nombra por sus miembros: su número interno no le dice nada a quien lee
    miembros = {i: [ids[i]] for i in range(n)}
    for p in mia["pasos"]:
        a, b = p["merged"]
        miembros[p["new_id"]] = miembros[a] + miembros[b]

    def nombre_grupo(i: int) -> str:
        return ids[i] if i < n else "{" + ", ".join(miembros[i]) + "}"

    primeras = mia["pasos"][:3]
    pasos = [
        {
            "titulo": "Cada abstract empieza como su propio grupo",
            "texto": f"Se parte de la matriz de distancias {n} × {n} ({REPRESENTACIONES[clave]}, distancia = 1 − coseno). "
                     f"Hay {n} grupos y se harán {n - 1} fusiones.",
            "latex": [r"d(x, y) = 1 - \cos\theta_{xy}"],
        },
    ]
    for p in primeras:
        a, b = p["merged"]
        pasos.append({
            "titulo": f"Fusión {p['step']}: se unen {nombre_grupo(a)} y {nombre_grupo(b)}",
            "texto": f"Es el par de grupos con menor distancia entre los {len(p['active_before'])} que quedaban. "
                     f"El grupo nuevo tiene {p['size']} abstracts.",
            "latex": [rf"D = {_n(p['distance'])}"],
        })
    pasos.append({
        "titulo": "Cómo se actualizan las distancias al unir dos grupos",
        "texto": {
            "single": "Single toma la menor de las dos distancias: el nuevo grupo queda tan cerca de otro como su miembro más cercano.",
            "complete": "Complete toma la mayor: el nuevo grupo queda tan lejos como su miembro más lejano.",
            "average": "Average hace la media ponderada por el tamaño de cada grupo, que equivale a la media de todos los pares.",
        }[enlace],
        "latex": [{
            "single": r"D(M, C) = \min\big(D(A, C),\, D(B, C)\big)",
            "complete": r"D(M, C) = \max\big(D(A, C),\, D(B, C)\big)",
            "average": r"D(M, C) = \frac{|A|\,D(A, C) + |B|\,D(B, C)}{|A| + |B|}",
        }[enlace]],
    })
    ultima = mia["pasos"][-1]
    pasos.append({
        "titulo": f"Después de {n - 1} fusiones queda un solo grupo",
        "texto": "La última fusión ocurre a la mayor altura del árbol: es la distancia a la que los dos grandes grupos se unen.",
        "latex": [rf"D_{{\text{{final}}}} = {_n(ultima['distance'])}"],
    })

    # El par con mayor diferencia entre distancia real y distancia en el árbol
    mayor = (0.0, 0, 1)
    for i in range(n):
        for j in range(i + 1, n):
            dif = abs(dist[i][j] - cof[i][j])
            if dif > mayor[0]:
                mayor = (dif, i, j)
    _, mi, mj = mayor

    pasos_cpcc = [
        {
            "titulo": "Distancia cofenética",
            "texto": "Para cada par de abstracts, es la altura de la fusión en la que quedan juntos por primera vez. "
                     "Si el árbol fuera perfecto, coincidiría con la distancia original.",
            "latex": [rf"C({ids[mi]}, {ids[mj]}) = {_n(cof[mi][mj])} \qquad d({ids[mi]}, {ids[mj]}) = {_n(dist[mi][mj])}"],
        },
        {
            "titulo": f"Se repite para los {pares} pares",
            "texto": f"El par anterior es el más distorsionado: el árbol reúne a {ids[mi]} y {ids[mj]} a {_n(cof[mi][mj])}, cuando en realidad están a {_n(dist[mi][mj])}. "
                     f"Las medias de las {pares} distancias son {_n(momentos['media_x'])} (originales) y {_n(momentos['media_y'])} (cofenéticas).",
            "latex": [rf"\bar d = {_n(momentos['media_x'])} \qquad \bar C = {_n(momentos['media_y'])}"],
        },
        {
            "titulo": "Correlación de Pearson entre las dos listas",
            "texto": "Cerca de 1 significa que el árbol conserva las distancias; cerca de 0, que las deforma mucho.",
            "latex": [
                rf"\text{{CPCC}} = \frac{{\sum (d - \bar d)(C - \bar C)}}{{\sqrt{{\sum (d - \bar d)^2 \; \sum (C - \bar C)^2}}}} "
                rf"= \frac{{{_n(momentos['cov'], 5)}}}{{\sqrt{{{_n(momentos['var_x'], 5)} \times {_n(momentos['var_y'], 5)}}}}} = {_n(mia['cpcc']['cpcc'])}"
            ],
        },
    ]

    por_k = mia["silueta"]["by_k"]
    pasos_silueta = [
        {
            "titulo": "Se corta el árbol en k grupos",
            "texto": "Un corte aplica las primeras n − k fusiones. Para cada abstract se mide qué tan cerca está de su grupo "
                     "frente al grupo vecino más cercano.",
            "latex": [r"s(i) = \frac{b(i) - a(i)}{\max\{a(i),\, b(i)\}}"],
        },
        {
            "titulo": f"Mejor corte: k = {mia['silueta']['best_k']}",
            "texto": "a(i) es la distancia media a su propio grupo y b(i) la distancia media al grupo vecino más cercano. "
                     "Cerca de 1 está bien asignado; cerca de 0, en la frontera; negativo, probablemente en el grupo equivocado.",
            "latex": [rf"S = \frac{{1}}{{n}} \sum s(i) = {_n(mia['silueta']['best_silhouette'], 4)}"],
        },
    ]

    # Hasta k = 8, y además el mejor corte si queda por fuera
    mejor_k = mia["silueta"]["best_k"]
    filas_k = [[k, _n(v, 4)] for k, v in sorted(por_k.items()) if k <= 8 or k == mejor_k]
    comparacion = [[NOMBRE_ENLACE[e], _n(corridas[e]["cpcc"]["cpcc"]),
                    _n(corridas[e]["fusiones"][-1][2], 3),
                    f"{_n(corridas[e]['silueta']['best_silhouette'], 3)} (k = {corridas[e]['silueta']['best_k']})"]
                   for e in ENLACES]

    return {
        "clave": f"{clave}:{enlace}",
        "algoritmo": f"Enlace {NOMBRE_ENLACE[enlace]}",
        "familia": f"Agrupamiento jerárquico · {REPRESENTACIONES[clave]}",
        "similitud": mia["cpcc"]["cpcc"],
        "etiqueta_valor": "CPCC",
        "subtitulo": f"{REPRESENTACIONES[clave]}, {n} abstracts",
        "ms": (time.perf_counter() - inicio) * 1000,
        "acepta_textos": False,
        "que_es": "Es un algoritmo que organiza los 20 artículos en grupos según qué tanto se parecen, sin que nadie le diga cuáles son los grupos. "
                  "Empieza con cada artículo solo y va juntando los dos grupos más parecidos, hasta que todos quedan en uno. El resultado es un árbol, "
                  "llamado dendrograma, que muestra qué se unió con qué y en qué orden.",
        "que_significa": "Este número, llamado CPCC, es la nota del árbol: va de −1 a 1, y entre más cerca de 1, mejor respeta el dibujo los parecidos reales entre los artículos.",
        "formula_simple": {
            "single": "D(A, B) es la distancia entre dos grupos, y d(x, y) la distancia entre dos artículos. Single mira todas las parejas posibles entre los dos grupos y se queda con la más cercana (min).",
            "complete": "D(A, B) es la distancia entre dos grupos, y d(x, y) la distancia entre dos artículos. Complete mira todas las parejas posibles entre los dos grupos y se queda con la más lejana (max).",
            "average": "D(A, B) es la distancia entre dos grupos, y d(x, y) la distancia entre dos artículos. Average suma las distancias de todas las parejas posibles entre los dos grupos y saca el promedio.",
        }[enlace] + " La segunda fórmula es la nota del árbol: compara la distancia real entre cada par de artículos, d(x, y), con la distancia a la que el árbol los une, C(x, y). Si las dos listas se parecen, el árbol es fiel.",
        "idea": {
            "single": "Va uniendo, de a dos, los grupos más cercanos, y mide la cercanía entre grupos por sus dos miembros más próximos. Es como unir amigos de amigos: basta una persona en común para juntar dos círculos.",
            "complete": "Va uniendo los grupos más cercanos, pero mide la cercanía entre grupos por sus dos miembros más lejanos. Solo junta dos grupos si todos sus miembros están razonablemente cerca.",
            "average": "Va uniendo los grupos más cercanos y mide la cercanía entre grupos por el promedio de todas las distancias entre sus miembros. Es el punto medio entre Single y Complete.",
        }[enlace],
        "en_una_frase": f"Este árbol obtuvo una nota de {_n(mia['cpcc']['cpcc'])}, "
                        + ("la más alta de los tres enlaces con esta forma de medir el parecido."
                           if mia["cpcc"]["cpcc"] >= max(corridas[e]["cpcc"]["cpcc"] for e in ENLACES)
                           else f"frente a {_n(max(corridas[e]['cpcc']['cpcc'] for e in ENLACES))} del mejor enlace con esta forma de medir el parecido."),
        "formulas": [
            {
                "single": r"D(A, B) = \min_{x \in A,\, y \in B} d(x, y)",
                "complete": r"D(A, B) = \max_{x \in A,\, y \in B} d(x, y)",
                "average": r"D(A, B) = \frac{1}{|A|\,|B|} \sum_{x \in A,\, y \in B} d(x, y)",
            }[enlace],
            r"\text{CPCC} = \operatorname{Pearson}\big(d(x, y),\, C(x, y)\big)",
        ],
        "pasos": pasos + pasos_cpcc + pasos_silueta,
        "visual": [
            {"tipo": "tabla", "titulo": "Los tres enlaces con esta representación",
             "columnas": ["Enlace", "CPCC", "Altura de la última fusión", "Silueta del mejor corte"], "filas": comparacion},
            {"tipo": "tabla", "titulo": f"Silueta de {NOMBRE_ENLACE[enlace]} según el número de grupos",
             "columnas": ["Número de grupos (k)", "Silueta"], "filas": filas_k},
        ],
        "interpretacion": "El CPCC mide fidelidad: cerca de 1, el árbol casi no deforma las distancias entre abstracts. "
                          "No mide si los grupos tienen sentido temático, solo si el dibujo es honesto con los datos. La silueta es otra nota: "
                          "si el árbol se corta en cierto número de grupos, dice qué tan separados quedan. Cerca de 1 hay grupos bien separados, de 0.25 a 0.5 la estructura es débil, y por "
                          "debajo de 0.25 no hay una estructura sustancial. "
                          "Las alturas del dendrograma son distancias reales: cuanto más abajo se unen dos abstracts, más se parecen.",
        "conclusiones": _conclusiones(clave, enlace, corridas),
        "codigo": {
            "archivo": "src/clustering/linkage.py",
            "funciones": [
                ["hierarchical_clustering", "las fusiones y la actualización de distancias (Lance-Williams)"],
                ["cophenetic_correlation, cophenetic_matrix (src/clustering/evaluation.py)", "el CPCC"],
                ["silhouette, best_k_by_silhouette (src/clustering/evaluation.py)", "la silueta y el mejor corte"],
            ],
            "tiempo": "O(n³)", "espacio": "O(n²)",
            "porque": "n − 1 fusiones; cada una busca el mínimo entre O(n²) pares. El CPCC y la silueta de un corte son O(n²)",
        },
    }

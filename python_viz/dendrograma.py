"""
Renderizado de dendrogramas (Requerimiento 2)
----------------------------------------------
Dibuja el árbol de agrupamiento jerárquico a partir del JSON que produce
`src.clustering.export`, sin usar scipy ni ninguna función de alto nivel que
dibuje dendrogramas: la geometría se calcula aquí.

Entrada: el formato versión 1 de `export.py`, donde cada fusión es
`[a, b, altura, tamaño]`, las hojas son los ids `0..n-1` y la fusión k crea el
cluster con id `n+k`.

CÓMO SE CONSTRUYE LA FIGURA
===========================

Un dendrograma es un conjunto de "U" invertidas. Cada fusión aporta una.

El trabajo real no es dibujar, sino decidir la coordenada X de cada nodo. Se
hace en tres pasos:

  1. Reconstruir el árbol: qué dos hijos tiene cada cluster.
  2. Recorrerlo en profundidad desde la raíz para fijar el ORDEN de las hojas.
     Este paso es el que evita que las ramas se crucen: hojas que pertenecen al
     mismo subárbol quedan contiguas en el eje X.
  3. Asignar x = posición en ese orden a cada hoja, y a cada cluster interno el
     punto medio de sus dos hijos.

El recorrido es iterativo y no recursivo a propósito: con 20 abstracts la
recursión sobraría, pero un corpus grande desbordaría la pila de Python.

Complejidad: O(n) para el recorrido y O(n) para el dibujo, sobre n-1 fusiones.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt

from python_viz.estilo import (
    COLOR_ENLACE,
    REJILLA,
    TINTA_SECUNDARIA,
    aplicar_estilo,
)


def _construir_hijos(merges: list, n: int) -> dict[int, tuple[int, int]]:
    """Mapa id_cluster -> (hijo_a, hijo_b). La fusión k crea el id n+k."""
    return {n + k: (int(a), int(b)) for k, (a, b, _, _) in enumerate(merges)}


def _orden_de_hojas(hijos: dict[int, tuple[int, int]], raiz: int, n: int) -> list[int]:
    """
    Devuelve las hojas en el orden en que deben aparecer en el eje X.

    Recorrido en profundidad iterativo. Al apilar primero el hijo derecho, el
    izquierdo se visita antes y las hojas salen de izquierda a derecha.
    """
    orden: list[int] = []
    pila = [raiz]
    while pila:
        nodo = pila.pop()
        if nodo < n:                       # es una hoja
            orden.append(nodo)
            continue
        a, b = hijos[nodo]
        pila.append(b)
        pila.append(a)
    return orden


def _coordenadas_x(
    merges: list, hijos: dict[int, tuple[int, int]], orden: list[int], n: int
) -> dict[int, float]:
    """
    Coordenada X de cada nodo.

    Las hojas se reparten uniformemente según `orden`; cada cluster interno se
    coloca en el punto medio de sus dos hijos. Las fusiones vienen en orden de
    ejecución, así que al procesarlas en secuencia los hijos de un cluster ya
    tienen su X calculada cuando le toca el turno.
    """
    x = {hoja: float(i) for i, hoja in enumerate(orden)}
    for k in range(len(merges)):
        a, b = hijos[n + k]
        x[n + k] = (x[a] + x[b]) / 2.0
    return x


def rango_util(merges: list, umbral: float = 0.45) -> tuple[float, float]:
    """
    Decide el rango del eje Y.

    Con distancias de texto el problema es casi siempre el mismo: dos abstracts
    distintos comparten pocas palabras, así que todas las fusiones caen en una
    franja estrecha y alta (p. ej. entre 0.80 y 0.89 con Jaccard). Anclar el eje
    en 0 deja el 90% del lienzo vacío y aplasta la estructura hasta hacerla
    ilegible.

    Por eso el eje arranca en 0 solo cuando la primera fusión ocurre
    razonablemente cerca del origen (min/max < umbral). Si no, se recorta a la
    franja donde de verdad pasa algo y los montantes de las hojas entran desde
    el borde inferior, que es la convención habitual en dendrogramas con
    distancias comprimidas.
    """
    alturas = [float(m[2]) for m in merges]
    h_min, h_max = min(alturas), max(alturas)
    if h_max <= 0 or h_min / h_max < umbral:
        return 0.0, h_max * 1.06

    margen = (h_max - h_min) * 0.18 or h_max * 0.02
    return max(0.0, h_min - margen), h_max + margen * 0.5


def dibujar_dendrograma(
    ax,
    merges: list,
    labels: list[str],
    color: str,
    titulo: str = "",
    ylim: tuple[float, float] | None = None,
) -> None:
    """Dibuja en `ax` el dendrograma de una lista de fusiones."""
    n = len(labels)
    if len(merges) != n - 1:
        raise ValueError(
            f"Se esperaban {n - 1} fusiones para {n} hojas y llegaron {len(merges)}"
        )

    hijos = _construir_hijos(merges, n)
    raiz = n + len(merges) - 1
    orden = _orden_de_hojas(hijos, raiz, n)
    x = _coordenadas_x(merges, hijos, orden, n)

    # Altura a la que cuelga cada nodo: las hojas nacen en 0; un cluster, en la
    # altura de la fusión que lo creó.
    altura = {hoja: 0.0 for hoja in range(n)}
    for k, (_, _, h, _) in enumerate(merges):
        altura[n + k] = float(h)

    for k, (a, b, h, _) in enumerate(merges):
        a, b, h = int(a), int(b), float(h)
        xa, xb = x[a], x[b]
        # Dos montantes desde donde cuelga cada hijo hasta la altura de fusión,
        # y un travesaño que los une: la "U" invertida.
        ax.plot([xa, xa], [altura[a], h], color=color, linewidth=1.6,
                solid_capstyle="round")
        ax.plot([xb, xb], [altura[b], h], color=color, linewidth=1.6,
                solid_capstyle="round")
        ax.plot([xa, xb], [h, h], color=color, linewidth=1.6,
                solid_capstyle="round")

    ax.set_xticks(range(n))
    ax.set_xticklabels([labels[i] for i in orden], fontsize=8, rotation=0)
    ax.set_xlim(-0.6, n - 0.4)

    y0, y1 = ylim if ylim is not None else rango_util(merges)
    ax.set_ylim(y0, y1)
    etiqueta_y = "Distancia de fusión"
    if y0 > 0:
        etiqueta_y += "  (eje recortado)"
    ax.set_ylabel(etiqueta_y)
    ax.grid(axis="y", linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)
    for lado in ("top", "right", "bottom"):
        ax.spines[lado].set_visible(False)
    if titulo:
        ax.set_title(titulo, loc="left", pad=12)


def renderizar(ruta_json: str, dir_salida: str, sufijo: str = "",
               fuente: str = "") -> list[str]:
    """
    Dibuja un dendrograma por enlace y una figura comparativa con los tres.

    `sufijo` distingue los archivos cuando se renderiza más de una
    representación; `fuente` se rotula en la figura para que nunca haya duda
    de qué distancia produjo ese árbol. Rotularlo importa: los mismos 20
    abstracts dan árboles distintos según se midan con Jaccard o con
    embeddings, y una figura sin esa etiqueta es ambigua.

    Devuelve las rutas de los archivos generados.
    """
    aplicar_estilo()

    with open(ruta_json, encoding="utf-8") as fh:
        datos = json.load(fh)

    labels = datos["labels"]
    enlaces = datos["linkages"]
    os.makedirs(dir_salida, exist_ok=True)
    generados: list[str] = []

    # --- una figura por enlace -------------------------------------------
    for nombre, info in enlaces.items():
        fig, ax = plt.subplots(figsize=(11, 5.5))
        color = COLOR_ENLACE.get(nombre, COLOR_ENLACE["average"])
        cpcc = info.get("cpcc")

        dibujar_dendrograma(ax, info["merges"], labels, color)
        ax.set_title(f"Agrupamiento jerárquico — enlace {nombre}",
                     loc="left", pad=14)

        pie = []
        if cpcc is not None:
            # El CPCC rotulado junto a la figura evita tener que buscarlo en el
            # texto para comparar la calidad de los tres enlaces.
            pie.append(f"Correlación cofenética (CPCC) = {cpcc:.4f}")
        if fuente:
            pie.append(f"distancia: {fuente}")
        if pie:
            ax.text(0.0, 1.015, "  ·  ".join(pie), transform=ax.transAxes,
                    fontsize=9, color=TINTA_SECUNDARIA)

        ruta = os.path.join(dir_salida, f"dendrograma_{nombre}{sufijo}.png")
        fig.savefig(ruta)
        plt.close(fig)
        generados.append(ruta)

    # --- figura comparativa ----------------------------------------------
    # Paneles apilados y con el MISMO eje Y: comparar alturas entre enlaces solo
    # tiene sentido si la escala es común.
    orden_enlaces = [e for e in ("single", "complete", "average") if e in enlaces]
    orden_enlaces += [e for e in enlaces if e not in orden_enlaces]

    # Un único rango para los tres paneles, calculado sobre todas las fusiones:
    # comparar alturas entre enlaces solo es válido con la misma escala.
    todas = [m for info in enlaces.values() for m in info["merges"]]
    ylim_comun = rango_util(todas)

    # sharex=False a propósito: cada criterio de enlace produce un orden de
    # hojas distinto, así que compartir el eje X pondría bajo un panel las
    # etiquetas de otro. Cada panel rotula sus propias hojas.
    fig, axes = plt.subplots(
        len(orden_enlaces), 1,
        figsize=(11, 3.9 * len(orden_enlaces)),
        sharex=False,
    )
    if len(orden_enlaces) == 1:
        axes = [axes]

    for ax, nombre in zip(axes, orden_enlaces):
        info = enlaces[nombre]
        color = COLOR_ENLACE.get(nombre, COLOR_ENLACE["average"])
        dibujar_dendrograma(ax, info["merges"], labels, color, ylim=ylim_comun)
        cpcc = info.get("cpcc")
        etiqueta = f"{nombre}" + (f"  ·  CPCC {cpcc:.4f}" if cpcc is not None else "")
        ax.set_title(etiqueta, loc="left", pad=8)

    titulo = "Comparación de criterios de enlace"
    if fuente:
        titulo += f"  —  distancia: {fuente}"
    fig.suptitle(titulo, x=0.125, ha="left", fontsize=14, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    ruta = os.path.join(dir_salida, f"dendrogramas_comparacion{sufijo}.png")
    fig.savefig(ruta)
    plt.close(fig)
    generados.append(ruta)

    return generados


def generar_fusiones(ruta_abstracts: str, ruta_json: str) -> None:
    """
    Produce el JSON de fusiones a partir de los abstracts del corpus.

    Es solo el pegamento entre módulos existentes —no reimplementa nada—:
    calcula la matriz de distancias con el Jaccard de similarity/classic y la
    pasa al export de clustering. Sirve para que la figura sea reproducible de
    punta a punta sin depender de un paso manual intermedio.
    """
    from src.clustering.export import export_merges_json
    from src.similarity.classic.jaccard import jaccard_similarity

    with open(ruta_abstracts, encoding="utf-8") as fh:
        registros = json.load(fh)

    textos = [r["texto"] for r in registros]
    labels = [r["id"] for r in registros]
    n = len(textos)

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            similitud = jaccard_similarity(textos[i], textos[j])[0]
            dist[i][j] = dist[j][i] = 1.0 - similitud

    export_merges_json(ruta_json, dist, labels)
    print(f"  fusiones generadas: {ruta_json}")


# Fuentes de distancia disponibles, en orden de preferencia.
#
# Los dendrogramas deben dibujarse con la MISMA distancia que se evalúa en el
# Requerimiento 2. Evaluar con embeddings y dibujar con Jaccard produce dos
# clusterings distintos presentados como si fueran uno, y los CPCC de la
# memoria no corresponderían a las figuras.
#
# Las fusiones de embeddings las exporta src/similarity/ai/demo_corpus20.py.
REPRESENTACIONES = (
    ("minilm", "coseno sobre embeddings MiniLM",
     ("tests", "fixtures", "daniel", "corpus20", "fusiones_minilm.json")),
    ("glove", "coseno sobre embeddings GloVe promediados",
     ("tests", "fixtures", "daniel", "corpus20", "fusiones_glove.json")),
    ("glove_centrado", "coseno sobre GloVe centrado",
     ("tests", "fixtures", "daniel", "corpus20", "fusiones_glove_centrado.json")),
    ("jaccard", "1 − coeficiente de Jaccard",
     ("data", "processed", "fusiones.json")),
)


def fuentes_disponibles(raiz: str) -> list[tuple[str, str, str]]:
    """(clave, descripción, ruta) de las representaciones presentes en disco."""
    encontradas = []
    for clave, descripcion, partes in REPRESENTACIONES:
        ruta = os.path.join(raiz, *partes)
        if os.path.exists(ruta):
            encontradas.append((clave, descripcion, ruta))
    return encontradas


def main() -> None:
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_abstracts = os.path.join(raiz, "data", "processed", "abstracts.json")
    ruta_jaccard = os.path.join(raiz, "data", "processed", "fusiones.json")
    dir_salida = os.path.join(raiz, "data", "figuras")

    # Jaccard se genera si falta: sirve de contraste frente a los embeddings.
    if not os.path.exists(ruta_jaccard) and os.path.exists(ruta_abstracts):
        generar_fusiones(ruta_abstracts, ruta_jaccard)

    fuentes = fuentes_disponibles(raiz)
    if not fuentes:
        print("No se encontraron fusiones que renderizar.")
        print("Genera las de embeddings con:")
        print("    python -m src.similarity.ai.demo_corpus20")
        print("o las de Jaccard ejecutando antes el extractor de abstracts.")
        return

    for clave, descripcion, ruta in fuentes:
        # La representación principal conserva los nombres sin sufijo, para no
        # romper las referencias que ya existan en la memoria del proyecto.
        sufijo = "" if clave == fuentes[0][0] else f"_{clave}"
        print(f"\n{clave}  ({descripcion})")
        for generado in renderizar(ruta, dir_salida, sufijo, descripcion):
            print(f"  generado: {generado}")

    print(f"\nRepresentacion principal: {fuentes[0][0]} — {fuentes[0][1]}")


if __name__ == "__main__":
    main()

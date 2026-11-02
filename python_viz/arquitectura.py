"""
Diagrama de módulos y flujo general del proyecto
------------------------------------------------
Genera data/figuras/arquitectura.png, la figura de la sección 2.1 del documento
técnico. Se dibuja con matplotlib para poder actualizarla junto con el código.

    python -m python_viz.arquitectura
"""

from __future__ import annotations

import os

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from python_viz.estilo import TINTA_SECUNDARIA, aplicar_estilo

ANCHO, ALTO = 2.9, 1.05
BORDE = "#c8cabb"
AZUL = "#1f4e79"
FONDO_AZUL = "#e4edf7"

# (x, y, título, primera línea, segunda línea, resaltado)
CAJAS = {
    "pdfs": (0.0, 6.0, "PDFs del docente", "20 artículos", "docs/articulos", False),
    "extractor": (3.5, 6.0, "Preprocesamiento", "src/preprocessing/extractor.py", "abstract, título y autores", False),
    "abstracts": (7.0, 6.0, "Abstracts limpios", "data/processed", "entrada única de los módulos", False),
    "clasica": (0.0, 4.5, "Similitud clásica", "similarity/classic", "Levenshtein, NW, Jaccard, TF-IDF", False),
    "transformacion": (3.5, 4.5, "Transformación", "preprocessing/transformacion.py", "minúsculas, sin stopwords", False),
    "ia": (7.0, 4.5, "Similitud con IA", "similarity/ai + metrics.py", "el modelo solo da vectores", True),
    "patrones": (0.0, 3.0, "Búsqueda de patrones", "src/pattern", "KMP, Boyer-Moore, Rabin-Karp", False),
    "matrices": (3.5, 3.0, "Matriz de distancias", "metrics.py", "1 − coseno, n × n", False),
    "enlaces": (7.0, 3.0, "Enlaces jerárquicos", "clustering/linkage.py", "Single, Complete, Average", False),
    "medicion": (0.0, 1.5, "Medición de tiempos", "src/benchmark/runner.py", "101 repeticiones por punto", False),
    "visual": (3.5, 1.5, "Visualización", "python_viz/", "dendrogramas y gráficas", False),
    "evaluacion": (7.0, 1.5, "Evaluación interna", "clustering/evaluation.py", "CPCC y silueta", False),
    "app": (3.5, 0.0, "Aplicación desplegada", "app/ (Flask y waitress, en Render)", "los tres requerimientos en vivo", True),
}

# (origen, destino, punto de salida, punto de llegada)
FLECHAS = (
    ("pdfs", "extractor", "der", "izq"),
    ("extractor", "abstracts", "der", "izq"),
    ("transformacion", "ia", "der", "izq"),
    ("transformacion", "matrices", "abajo", "arriba"),
    ("matrices", "enlaces", "der", "izq"),
    ("enlaces", "evaluacion", "abajo", "arriba"),
    ("evaluacion", "visual", "izq", "der"),
    ("medicion", "visual", "der", "izq"),
    ("visual", "app", "abajo", "arriba"),
    ("patrones", "medicion", "abajo", "arriba"),
)


def _linea(ax, xs: list[float], ys: list[float], con_flecha: bool = True) -> None:
    """Línea con quiebres; la punta de flecha va en su último tramo."""
    ax.plot(xs, ys, color=BORDE, linewidth=1.3, solid_capstyle="round")
    if con_flecha:
        ax.add_patch(FancyArrowPatch((xs[-2], ys[-2]), (xs[-1], ys[-1]), arrowstyle="-|>",
                                     mutation_scale=11, color=BORDE, linewidth=1.3,
                                     shrinkA=0, shrinkB=0))


def _punto(clave: str, lado: str) -> tuple[float, float]:
    x, y = CAJAS[clave][0], CAJAS[clave][1]
    return {
        "izq": (x, y + ALTO / 2),
        "der": (x + ANCHO, y + ALTO / 2),
        "der_alto": (x + ANCHO - 0.25, y + ALTO),
        "arriba": (x + ANCHO / 2, y + ALTO),
        "abajo": (x + ANCHO / 2, y),
    }[lado]


def main() -> None:
    aplicar_estilo()
    fig, ax = plt.subplots(figsize=(10, 6.9))

    for x, y, titulo, linea_1, linea_2, resaltado in CAJAS.values():
        ax.add_patch(FancyBboxPatch(
            (x, y), ANCHO, ALTO, boxstyle="round,pad=0,rounding_size=0.09",
            facecolor=FONDO_AZUL if resaltado else "white",
            edgecolor=AZUL if resaltado else BORDE, linewidth=1.8 if resaltado else 1.2,
        ))
        ax.text(x + ANCHO / 2, y + 0.78, titulo, ha="center", va="center",
                fontsize=10.5, fontweight="bold")
        ax.text(x + ANCHO / 2, y + 0.47, linea_1, ha="center", va="center",
                fontsize=8, color=TINTA_SECUNDARIA)
        ax.text(x + ANCHO / 2, y + 0.22, linea_2, ha="center", va="center",
                fontsize=8, color=TINTA_SECUNDARIA)

    for origen, destino, salida, llegada in FLECHAS:
        ax.add_patch(FancyArrowPatch(
            _punto(origen, salida), _punto(destino, llegada), arrowstyle="-|>",
            mutation_scale=11, color=BORDE, linewidth=1.3, shrinkA=0, shrinkB=0,
        ))

    # Los abstracts limpios alimentan la similitud clásica, la transformación y la búsqueda
    _linea(ax, [8.45, 8.45, 1.45, 1.45], [6.0, 5.8, 5.8, 5.55])
    _linea(ax, [4.95, 4.95], [5.8, 5.55])
    _linea(ax, [3.2, 3.2, 2.9], [5.8, 3.52, 3.52])

    # Los vectores de la IA van a la matriz de distancias
    _linea(ax, [8.45, 8.45, 5.6, 5.6], [4.5, 4.28, 4.28, 4.05])

    # La aplicación usa directamente los clásicos, la IA y la búsqueda
    _linea(ax, [0.0, -0.2, -0.2, 3.5], [5.02, 5.02, 0.52, 0.52])
    _linea(ax, [0.0, -0.2], [3.52, 3.52], con_flecha=False)
    _linea(ax, [9.9, 10.1, 10.1, 6.4], [5.02, 5.02, 0.52, 0.52])

    ax.set_xlim(-0.35, 10.25)
    ax.set_ylim(-0.1, 7.5)
    ax.axis("off")
    ax.text(-0.2, 7.25, "Los modelos de IA solo entregan vectores; todo el cálculo posterior es propio",
            fontsize=12, fontweight="bold")

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta = os.path.join(raiz, "data", "figuras", "arquitectura.png")
    fig.savefig(ruta)
    plt.close(fig)
    print(f"Figura: {ruta}")


if __name__ == "__main__":
    main()

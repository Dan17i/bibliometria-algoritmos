"""
Estilo visual compartido de las figuras del proyecto
----------------------------------------------------
Centraliza la paleta y los parámetros de matplotlib para que los dendrogramas
y las gráficas de rendimiento se lean como un mismo sistema.

Criterios aplicados:

  - Paleta categórica en orden FIJO, nunca ciclada. Las series siempre reciben
    el mismo color, de modo que un enlace o un algoritmo conserva su identidad
    entre figuras distintas.
  - Los hues elegidos mantienen separación suficiente bajo daltonismo, por lo
    que ninguna figura depende solo del color: además se rotula cada serie.
  - Rejilla y ejes recesivos: el dato es lo que debe resaltar, no el andamiaje.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")       # backend sin ventana: imprescindible para exportar PNG
import matplotlib.pyplot as plt   # noqa: E402

# Orden fijo de la paleta categórica. No se cicla: si hiciera falta una novena
# serie, se agrupa o se separa en varias figuras.
PALETA: tuple[str, ...] = (
    "#2a78d6",   # 1 azul
    "#eb6834",   # 2 naranja
    "#1baf7a",   # 3 aqua
    "#eda100",   # 4 amarillo
    "#e87ba4",   # 5 magenta
    "#008300",   # 6 verde
    "#6250d6",   # 7 violeta
    "#e34948",   # 8 rojo
)

TINTA_PRIMARIA = "#1a1a19"
TINTA_SECUNDARIA = "#5c5b55"
TINTA_TENUE = "#8d8c85"
REJILLA = "#e4e3de"
SUPERFICIE = "#ffffff"

# Color fijo por tipo de enlace: el mismo enlace se ve igual en toda la memoria
COLOR_ENLACE = {
    "single": PALETA[0],
    "complete": PALETA[1],
    "average": PALETA[2],
    "ward": PALETA[6],
}


def color_serie(indice: int) -> str:
    """Color de la serie n-ésima en el orden fijo de la paleta."""
    if indice >= len(PALETA):
        raise ValueError(
            f"La paleta tiene {len(PALETA)} posiciones y se pidio la {indice}. "
            "Agrupa series o separa la figura en varias; no se ciclan colores."
        )
    return PALETA[indice]


def aplicar_estilo() -> None:
    """Aplica los parámetros globales de matplotlib."""
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",

        "font.size": 10,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.titlecolor": TINTA_PRIMARIA,
        "axes.labelsize": 10,
        "axes.labelcolor": TINTA_SECUNDARIA,
        "text.color": TINTA_PRIMARIA,

        # Ejes y rejilla recesivos
        "axes.edgecolor": REJILLA,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": REJILLA,
        "grid.linewidth": 0.8,
        "xtick.color": TINTA_TENUE,
        "ytick.color": TINTA_TENUE,
        "xtick.labelcolor": TINTA_SECUNDARIA,
        "ytick.labelcolor": TINTA_SECUNDARIA,
        "xtick.major.size": 0,
        "ytick.major.size": 3,

        "lines.linewidth": 2.0,
        "lines.markersize": 8,

        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": TINTA_SECUNDARIA,
    })

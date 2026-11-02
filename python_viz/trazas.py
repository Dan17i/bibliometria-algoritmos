"""
Figuras del caso de estudio del Requerimiento 3
-----------------------------------------------
Dibuja las iteraciones y los desplazamientos de cada algoritmo sobre el texto, a
partir de data/resultados/traza_req3.json (`python -m src.pattern.traza`).

Cada fila es una alineación del patrón bajo el texto. Verde es un carácter que
coincidió, rojo el que falló y gris lo que el algoritmo no llegó a comparar. El
texto se parte en dos renglones porque una oración completa no cabe a un tamaño
legible en el ancho de una página.
"""

from __future__ import annotations

import json
import os

import matplotlib.pyplot as plt

from python_viz.estilo import TINTA_SECUNDARIA, aplicar_estilo, color_serie

VERDE = "#2e8b57"
ROJO = "#d1495b"
GRIS = "#e6e8ec"


def _celda(ax, x: float, y: float, caracter: str, fondo: str, tinta: str = "black") -> None:
    ax.add_patch(plt.Rectangle((x, y), 1, 1, facecolor=fondo, edgecolor="white", linewidth=0.6))
    visible = "␣" if caracter == " " else caracter
    ax.text(x + 0.5, y + 0.5, visible, ha="center", va="center", fontsize=7.5,
            family="monospace", color=tinta)


def _panel(ax, texto: str, patron: str, filas: list[dict], inicio: int, fin: int) -> None:
    """Un renglón del texto con las alineaciones que tienen alguna celda en ese tramo."""
    alto = len(filas) + 1
    ax.text(inicio - 0.3, alto + 0.5, "texto", ha="right", va="center", fontsize=7,
            color=TINTA_SECUNDARIA)
    for i in range(inicio, fin):
        _celda(ax, i, alto, texto[i], "#dfe7f2")
        if (i - inicio) % 10 == 0:
            ax.text(i + 0.5, alto + 1.35, str(i), ha="center", fontsize=6,
                    color=TINTA_SECUNDARIA)

    for fila, datos in enumerate(filas):
        y = alto - 1 - fila
        ax.text(inicio - 0.3, y + 0.5, datos["etiqueta"], ha="right", va="center",
                fontsize=6.5, color=TINTA_SECUNDARIA)
        for j in range(len(patron)):
            x = datos["alineacion"] + j
            if x < inicio or x >= fin:
                continue
            estado = datos["estados"][j]
            fondo = VERDE if estado == "si" else ROJO if estado == "no" else GRIS
            tinta = "white" if estado in ("si", "no") else "#5b6472"
            _celda(ax, x, y, patron[j], fondo, tinta)
        if datos.get("salto") and inicio < datos["alineacion"] + len(patron) <= fin:
            ax.text(datos["alineacion"] + len(patron) + 0.3, y + 0.5,
                    f"salto {datos['salto']}", va="center", fontsize=6.5,
                    color=TINTA_SECUNDARIA)

    ax.set_xlim(inicio - 7, fin + 6)
    ax.set_ylim(0, alto + 2)
    ax.axis("off")


def _figura_alineaciones(texto: str, patron: str, filas: list[dict], titulo: str,
                         nota: str, ruta: str) -> None:
    mitad = len(texto) // 2
    tramos = [(0, mitad), (mitad, len(texto))]
    # Una alineación que cruza el corte aparece en los dos renglones
    m = len(patron)
    grupos = [[f for f in filas if f["alineacion"] < b and f["alineacion"] + m > a]
              for a, b in tramos]

    alturas = [len(g) + 3 for g in grupos]
    fig, axes = plt.subplots(2, 1, figsize=(11, 0.21 * sum(alturas) + 1.3),
                             gridspec_kw={"height_ratios": alturas})
    for ax, (a, b), grupo in zip(axes, tramos, grupos):
        _panel(ax, texto, patron, grupo, a, b)
    fig.suptitle(titulo, x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.02, 0.012, nota, fontsize=8, color=TINTA_SECUNDARIA)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.05, hspace=0.05)
    fig.savefig(ruta)
    plt.close(fig)


def filas_kmp(traza: dict) -> list[dict]:
    """
    Una fila por alineación con algún carácter coincidente. Las alineaciones que
    fallan en la primera comparación solo avanzan el texto y se omiten.
    """
    m = traza["m"]
    por_alineacion: dict[int, list[str]] = {}
    for paso in traza["kmp"]["pasos"]:
        estados = por_alineacion.setdefault(paso["alineacion"], ["-"] * m)
        estados[paso["j"]] = "si" if paso["coincide"] else "no"

    filas = []
    for alineacion, estados in por_alineacion.items():
        if estados[0] == "si":
            filas.append({"alineacion": alineacion, "estados": estados,
                          "etiqueta": f"p = {alineacion}"})
    return filas


def filas_boyer_moore(traza: dict) -> list[dict]:
    """Una fila por alineación, con la comparación de derecha a izquierda."""
    m = traza["m"]
    patron, texto = traza["patron"], traza["texto"]
    filas = []
    for paso in traza["boyer_moore"]["pasos"]:
        p = paso["posicion"]
        estados = ["-"] * m
        j = m - 1
        while j >= 0 and patron[j] == texto[p + j]:
            estados[j] = "si"
            j -= 1
        if j >= 0:
            estados[j] = "no"
        filas.append({"alineacion": p, "estados": estados, "salto": paso["salto"],
                      "etiqueta": f"p = {p}"})
    return filas


def figura_rabin_karp(traza: dict, ruta: str) -> None:
    """El hash de cada ventana frente al del patrón, con los dos primos."""
    rk = traza["rabin_karp"]
    casos = (
        (rk["ventanas"], rk["primo"], rk["hash_patron"]),
        (rk["ventanas_primo_pequeno"], rk["primo_pequeno"],
         rk["ventanas_primo_pequeno"][0]["hash_patron"]),
    )
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2))
    for ax, (ventanas, primo, hash_patron) in zip(axes, casos):
        posiciones = [v["posicion"] for v in ventanas]
        ax.plot(posiciones, [v["hash_ventana"] for v in ventanas], marker="o", markersize=3,
                linewidth=0.8, color=color_serie(0), label="Hash de la ventana")
        ax.axhline(hash_patron, color=color_serie(1), linewidth=1.2,
                   label=f"Hash del patrón = {hash_patron}")
        for v in ventanas:
            if v["ocurrencia"]:
                ax.plot(v["posicion"], v["hash_ventana"], marker="o", markersize=9,
                        color=VERDE, linestyle="none")
                ax.annotate(f"ocurrencia en {v['posicion']}", (v["posicion"], v["hash_ventana"]),
                            textcoords="offset points", xytext=(6, 8), fontsize=8, color=VERDE)
            elif v["colision"]:
                ax.plot(v["posicion"], v["hash_ventana"], marker="X", markersize=10,
                        color=ROJO, linestyle="none")
                ax.annotate(f"colisión en {v['posicion']}: «{v['ventana']}»",
                            (v["posicion"], v["hash_ventana"]), textcoords="offset points",
                            xytext=(-8, 34), fontsize=8, color=ROJO, ha="right",
                            arrowprops={"arrowstyle": "-", "color": ROJO, "linewidth": 0.8})
        ax.set_title(f"q = {primo}", loc="left", fontsize=10)
        ax.set_xlabel("Posición de la ventana en el texto")
        ax.set_ylabel("Hash")
        ax.grid(True, linestyle="-", alpha=0.5)
        ax.set_axisbelow(True)
        ax.legend(loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2, fontsize=8, frameon=False)
    fig.suptitle(f"Rabin-Karp: rolling hash de cada ventana (patrón «{traza['patron']}»)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(ruta)
    plt.close(fig)


def main() -> None:
    aplicar_estilo()
    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(raiz, "data", "resultados", "traza_req3.json"), encoding="utf-8") as fh:
        traza = json.load(fh)
    dir_salida = os.path.join(raiz, "data", "figuras")

    nota = "Verde: coincide. Rojo: falla. Gris: no se compara."
    _figura_alineaciones(
        traza["texto"], traza["patron"], filas_kmp(traza),
        f"KMP: alineaciones con coincidencia parcial (patrón «{traza['patron']}»)",
        nota + " Se omiten las alineaciones que fallan en la primera comparación.",
        os.path.join(dir_salida, "traza_kmp.png"),
    )
    _figura_alineaciones(
        traza["texto"], traza["patron"], filas_boyer_moore(traza),
        f"Boyer-Moore: iteraciones y desplazamientos (patrón «{traza['patron']}»)",
        nota + " Compara de derecha a izquierda; a la derecha de cada fila, el salto.",
        os.path.join(dir_salida, "traza_boyer_moore.png"),
    )
    figura_rabin_karp(traza, os.path.join(dir_salida, "traza_rabin_karp.png"))
    print(f"Figuras en: {dir_salida}")


if __name__ == "__main__":
    main()

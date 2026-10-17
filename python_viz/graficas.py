"""
Gráficas comparativas de rendimiento
-------------------------------------
Lee el CSV que produce `src.benchmark.runner` y genera las figuras con las que
se contrasta la complejidad asintótica teórica de cada algoritmo frente a su
rendimiento observado.

LA IDEA CENTRAL: EL EXPONENTE EMPÍRICO
======================================

Decir "Levenshtein tardó 145 ms" no demuestra nada sobre su complejidad. Lo que
sí la demuestra es CÓMO CRECE ese tiempo al crecer la entrada.

Si el coste real sigue t = c · n^p, al tomar logaritmos queda:

        log(t) = log(c) + p · log(n)

es decir, una RECTA cuya pendiente es exactamente el exponente p. Por eso las
figuras usan ejes log-log: una complejidad polinómica se ve como una recta, y
ajustar esa recta por mínimos cuadrados devuelve el exponente que el algoritmo
exhibe en la práctica.

Ese exponente empírico es el que se compara con el teórico:

        O(n)   -> pendiente ~ 1
        O(n²)  -> pendiente ~ 2
        O(n³)  -> pendiente ~ 3

No se usa numpy: el ajuste de mínimos cuadrados sobre una recta se resuelve con
las fórmulas cerradas de la pendiente y la ordenada, que son cuatro sumatorias.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from python_viz.estilo import (
    TINTA_SECUNDARIA,
    TINTA_TENUE,
    aplicar_estilo,
    color_serie,
)

# Nombres legibles para las leyendas
ETIQUETAS = {
    "levenshtein": "Levenshtein",
    "coseno_clasico": "Similitud del coseno",
    "jaccard": "Coeficiente de Jaccard",
    "clustering_single": "Single linkage",
    "clustering_complete": "Complete linkage",
    "clustering_average": "Average linkage",
}


def leer_csv(ruta: str) -> list[dict]:
    with open(ruta, encoding="utf-8") as fh:
        filas = list(csv.DictReader(fh))
    for f in filas:
        f["tamano"] = int(f["tamano"])
        for k in ("mediana", "minimo", "maximo"):
            f[k] = float(f[k])
    return filas


def pendiente_loglog(tamanos: list[float], tiempos: list[float]) -> float | None:
    """
    Exponente empírico: pendiente de la recta de mínimos cuadrados en log-log.

    Fórmula cerrada de la regresión lineal simple:

        p = [ N·Σ(xy) − Σx·Σy ] / [ N·Σ(x²) − (Σx)² ]

    donde x = log(tamaño) e y = log(tiempo). Se descartan los puntos con tiempo
    cero, que aparecen cuando una medición cae por debajo de la resolución del
    reloj y cuyo logaritmo no está definido.
    """
    pares = [(math.log(t), math.log(s))
             for t, s in zip(tamanos, tiempos) if t > 0 and s > 0]
    if len(pares) < 2:
        return None

    n = len(pares)
    sx = sum(x for x, _ in pares)
    sy = sum(y for _, y in pares)
    sxy = sum(x * y for x, y in pares)
    sxx = sum(x * x for x, _ in pares)

    denominador = n * sxx - sx * sx
    if abs(denominador) < 1e-12:
        return None
    return (n * sxy - sx * sy) / denominador


def _agrupar(filas: list[dict], magnitud: str) -> dict[str, list[dict]]:
    grupos: dict[str, list[dict]] = defaultdict(list)
    for f in filas:
        if f["magnitud"] == magnitud:
            grupos[f["algoritmo"]].append(f)
    for puntos in grupos.values():
        puntos.sort(key=lambda p: p["tamano"])
    return dict(grupos)


def grafica_escalado(
    filas: list[dict],
    magnitud: str,
    titulo: str,
    etiqueta_x: str,
    ruta_salida: str,
) -> dict[str, float]:
    """
    Tiempo frente a tamaño de entrada, en ejes log-log.

    Devuelve el exponente empírico de cada algoritmo.
    """
    grupos = _agrupar(filas, magnitud)
    if not grupos:
        return {}

    fig, ax = plt.subplots(figsize=(9, 5.6))
    exponentes: dict[str, float] = {}

    for i, (algoritmo, puntos) in enumerate(sorted(grupos.items())):
        tamanos = [p["tamano"] for p in puntos]
        # Se grafica en milisegundos: los segundos obligarían a leer
        # exponentes negativos en el eje.
        tiempos = [p["mediana"] * 1000 for p in puntos]
        color = color_serie(i)

        p = pendiente_loglog(tamanos, tiempos)
        teorica = puntos[0]["complejidad_teorica"]
        etiqueta = ETIQUETAS.get(algoritmo, algoritmo)
        if p is not None:
            exponentes[algoritmo] = p
            etiqueta += f"  —  {teorica}, pendiente {p:.2f}"

        ax.plot(tamanos, tiempos, marker="o", color=color,
                label=etiqueta, markeredgecolor="white", markeredgewidth=1.2)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(etiqueta_x)
    ax.set_ylabel("Tiempo (ms, mediana de 5 ejecuciones)")
    ax.set_title(titulo, loc="left", pad=14)
    ax.grid(True, which="major", linestyle="-", alpha=0.8)
    ax.grid(True, which="minor", linestyle="-", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9)

    ax.text(
        0.0, -0.17,
        "Ejes logarítmicos: una complejidad polinómica aparece como recta y su "
        "pendiente es el exponente.",
        transform=ax.transAxes, fontsize=8.5, color=TINTA_TENUE,
    )

    fig.savefig(ruta_salida)
    plt.close(fig)
    return exponentes


def grafica_teorico_vs_observado(
    exponentes: dict[str, float],
    esperados: dict[str, float],
    ruta_salida: str,
) -> None:
    """
    Barras del exponente teórico frente al medido.

    Es la figura que responde directamente al análisis crítico: si las dos
    barras de un algoritmo coinciden, su comportamiento real confirma la
    complejidad deducida en el papel.
    """
    algoritmos = [a for a in exponentes if a in esperados]
    if not algoritmos:
        return
    algoritmos.sort(key=lambda a: esperados[a])

    fig, ax = plt.subplots(figsize=(9, 0.62 * len(algoritmos) + 2.4))
    posiciones = range(len(algoritmos))
    alto = 0.36

    # Dos series: identidad por color Y por etiqueta directa, nunca color solo.
    ax.barh([p + alto / 2 + 0.02 for p in posiciones],
            [esperados[a] for a in algoritmos],
            height=alto, color=color_serie(0), label="Exponente teórico")
    ax.barh([p - alto / 2 - 0.02 for p in posiciones],
            [exponentes[a] for a in algoritmos],
            height=alto, color=color_serie(1), label="Exponente medido")

    for i, a in enumerate(algoritmos):
        ax.text(esperados[a] + 0.05, i + alto / 2 + 0.02, f"{esperados[a]:.0f}",
                va="center", fontsize=9, color=TINTA_SECUNDARIA)
        ax.text(exponentes[a] + 0.05, i - alto / 2 - 0.02, f"{exponentes[a]:.2f}",
                va="center", fontsize=9, color=TINTA_SECUNDARIA)

    ax.set_yticks(list(posiciones))
    ax.set_yticklabels([ETIQUETAS.get(a, a) for a in algoritmos], fontsize=9)
    ax.set_xlabel("Exponente p en  t ≈ c · nᵖ")
    ax.set_title("Complejidad teórica frente a la observada", loc="left", pad=14)
    ax.set_xlim(0, max(max(esperados.values()), max(exponentes.values())) + 0.6)
    ax.grid(axis="x", linestyle="-", alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9)

    fig.savefig(ruta_salida)
    plt.close(fig)


def main() -> None:
    aplicar_estilo()

    raiz = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ruta_csv = os.path.join(raiz, "data", "benchmark", "rendimiento.csv")
    dir_salida = os.path.join(raiz, "data", "figuras")

    if not os.path.exists(ruta_csv):
        print(f"No se encontro {ruta_csv}")
        print("Ejecuta primero: python -m src.benchmark.runner")
        return

    os.makedirs(dir_salida, exist_ok=True)
    filas = leer_csv(ruta_csv)

    exp_similitud = grafica_escalado(
        filas, "caracteres",
        "Algoritmos de similitud: escalado con la longitud del texto",
        "Longitud del texto (caracteres)",
        os.path.join(dir_salida, "rendimiento_similitud.png"),
    )
    grafica_escalado(
        filas, "documentos",
        "Clustering jerárquico sobre el corpus real (20 documentos)",
        "Documentos del corpus",
        os.path.join(dir_salida, "rendimiento_clustering_corpus.png"),
    )
    # El exponente asintótico se toma del rango sintético, no del corpus: con
    # n <= 20 los costes constantes del intérprete pesan más que el término n³
    # y la pendiente medida subestima la complejidad real.
    exp_clustering = grafica_escalado(
        filas, "documentos_sinteticos",
        "Clustering jerárquico: escalado asintótico (matrices sintéticas)",
        "Número de documentos",
        os.path.join(dir_salida, "rendimiento_clustering.png"),
    )

    # Exponente que predice la teoría para cada algoritmo
    esperados = {
        "levenshtein": 2.0,
        "coseno_clasico": 1.0,
        "jaccard": 1.0,
        "clustering_single": 3.0,
        "clustering_complete": 3.0,
        "clustering_average": 3.0,
    }
    todos = {**exp_similitud, **exp_clustering}
    grafica_teorico_vs_observado(
        todos, esperados,
        os.path.join(dir_salida, "complejidad_teorica_vs_observada.png"),
    )

    print(f"{'ALGORITMO':<24}{'TEORICO':<10}{'MEDIDO':<10}DIFERENCIA")
    print("-" * 62)
    for algoritmo in sorted(todos):
        medido = todos[algoritmo]
        teorico = esperados.get(algoritmo)
        if teorico is None:
            continue
        print(f"{ETIQUETAS.get(algoritmo, algoritmo):<24}"
              f"{teorico:<10.1f}{medido:<10.2f}{medido - teorico:+.2f}")
    print(f"\nFiguras en: {dir_salida}")


if __name__ == "__main__":
    main()

"""
Medición de tiempos de ejecución de los algoritmos del proyecto
----------------------------------------------------------------
Produce la evidencia empírica con la que se contrasta la complejidad asintótica
teórica de cada algoritmo.

QUÉ SE MIDE Y POR QUÉ
=====================

Los algoritmos del proyecto escalan respecto a dos magnitudes distintas, así
que medirlos en un solo eje no diría nada:

  - Similitud (Levenshtein, coseno, Jaccard): escalan con la LONGITUD del texto.
    Se miden sobre prefijos crecientes de dos abstracts reales del corpus.
  - Clustering jerárquico: escala con el NÚMERO DE DOCUMENTOS. Se mide sobre
    submatrices de tamaño creciente.

CÓMO SE MIDE
============

  - `time.perf_counter`, que es el reloj monótono de mayor resolución
    disponible; `time.time` puede retroceder si el sistema ajusta la hora.
  - Varias repeticiones por punto y se reporta la MEDIANA, no la media: una
    sola pausa del recolector de basura o del planificador del sistema
    operativo dispara un valor atípico que arrastraría la media.
  - El recolector de basura se desactiva durante cada medición, por el mismo
    motivo.
  - Se descarta una ejecución de calentamiento antes de medir, para no incluir
    el coste de la primera resolución de nombres y de los cachés fríos.

El CSV resultante es la entrada de `python_viz.graficas`; este módulo no dibuja
nada, solo produce datos.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import statistics
import time
from typing import Callable

from src.clustering.linkage import hierarchical_clustering
from src.similarity.classic.cosine import cosine_similarity as coseno_clasico
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.classic.levenshtein import levenshtein_distance

# Complejidad teórica de cada algoritmo, para contrastarla con lo observado.
# m y n son las longitudes de los dos textos; para el clustering, n documentos.
COMPLEJIDAD_TEORICA = {
    "levenshtein": "O(m·n)",
    "coseno_clasico": "O(m + n)",
    "jaccard": "O(m + n)",
    "clustering_single": "O(n³)",
    "clustering_complete": "O(n³)",
    "clustering_average": "O(n³)",
}


def medir(funcion: Callable, repeticiones: int = 5) -> dict:
    """
    Ejecuta `funcion` varias veces y devuelve estadísticos del tiempo.

    Devuelve mediana, mínimo y máximo en segundos. El mínimo es el valor menos
    contaminado por ruido del sistema; la mediana es el que se grafica.
    """
    funcion()                       # calentamiento: no se mide

    tiempos: list[float] = []
    for _ in range(repeticiones):
        gc.collect()
        gc.disable()
        try:
            inicio = time.perf_counter()
            funcion()
            tiempos.append(time.perf_counter() - inicio)
        finally:
            gc.enable()

    return {
        "mediana": statistics.median(tiempos),
        "minimo": min(tiempos),
        "maximo": max(tiempos),
        "repeticiones": repeticiones,
    }


def benchmark_similitud(
    texto_a: str,
    texto_b: str,
    tamanos: list[int],
    repeticiones: int = 5,
) -> list[dict]:
    """
    Mide los algoritmos de similitud sobre prefijos crecientes de dos textos.

    Usar prefijos de abstracts reales, y no cadenas sintéticas, mantiene la
    distribución de caracteres y la densidad de vocabulario del corpus, que es
    justo lo que determina el coste de Jaccard y del coseno.
    """
    algoritmos: dict[str, Callable[[str, str], object]] = {
        "levenshtein": levenshtein_distance,
        "coseno_clasico": coseno_clasico,
        "jaccard": jaccard_similarity,
    }

    filas: list[dict] = []
    for nombre, funcion in algoritmos.items():
        for tam in tamanos:
            a, b = texto_a[:tam], texto_b[:tam]
            if len(a) < tam or len(b) < tam:
                continue            # no hay texto suficiente para este tamaño
            stats = medir(lambda f=funcion, x=a, y=b: f(x, y), repeticiones)
            filas.append({
                "algoritmo": nombre,
                "magnitud": "caracteres",
                "tamano": tam,
                "complejidad_teorica": COMPLEJIDAD_TEORICA[nombre],
                **stats,
            })
            print(f"  {nombre:<16} {tam:>6} car.  "
                  f"{stats['mediana'] * 1000:>9.3f} ms")
    return filas


def benchmark_clustering(
    dist: list[list[float]],
    tamanos: list[int],
    repeticiones: int = 5,
) -> list[dict]:
    """Mide los tres criterios de enlace sobre submatrices de tamaño creciente."""
    filas: list[dict] = []
    for enlace in ("single", "complete", "average"):
        nombre = f"clustering_{enlace}"
        for n in tamanos:
            if n > len(dist):
                continue
            sub = [fila[:n] for fila in dist[:n]]
            stats = medir(
                lambda s=sub, e=enlace: hierarchical_clustering(s, e),
                repeticiones,
            )
            filas.append({
                "algoritmo": nombre,
                "magnitud": "documentos",
                "tamano": n,
                "complejidad_teorica": COMPLEJIDAD_TEORICA[nombre],
                **stats,
            })
            print(f"  {nombre:<22} {n:>4} doc.  "
                  f"{stats['mediana'] * 1000:>9.3f} ms")
    return filas


def matriz_sintetica(n: int, semilla: int = 42) -> list[list[float]]:
    """
    Matriz de distancias pseudoaleatoria simétrica de n x n.

    El corpus real tiene 20 documentos, un rango demasiado corto para que el
    término n³ domine sobre los costes constantes del intérprete. Para observar
    el comportamiento asintótico de verdad hace falta estirar n más allá del
    corpus, y para eso sirven estas matrices.

    Se usa un generador lineal congruencial propio en vez de `random` para que
    la secuencia sea idéntica en cualquier máquina y versión de Python: la
    medición debe ser reproducible por quien revise el trabajo.
    """
    estado = semilla
    def siguiente() -> float:
        nonlocal estado
        estado = (1103515245 * estado + 12345) % (2 ** 31)
        return estado / (2 ** 31)

    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            dist[i][j] = dist[j][i] = siguiente()
    return dist


def benchmark_clustering_sintetico(
    tamanos: list[int],
    repeticiones: int = 3,
) -> list[dict]:
    """Mide el clustering sobre matrices sintéticas grandes (escalado asintótico)."""
    filas: list[dict] = []
    for enlace in ("single", "complete", "average"):
        nombre = f"clustering_{enlace}"
        for n in tamanos:
            dist = matriz_sintetica(n)
            stats = medir(
                lambda s=dist, e=enlace: hierarchical_clustering(s, e),
                repeticiones,
            )
            filas.append({
                "algoritmo": nombre,
                "magnitud": "documentos_sinteticos",
                "tamano": n,
                "complejidad_teorica": COMPLEJIDAD_TEORICA[nombre],
                **stats,
            })
            print(f"  {nombre:<22} {n:>4} doc.  "
                  f"{stats['mediana'] * 1000:>9.3f} ms")
    return filas


def matriz_distancias(textos: list[str]) -> list[list[float]]:
    """Matriz de distancias (1 - Jaccard) entre todos los pares de abstracts."""
    n = len(textos)
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            similitud = jaccard_similarity(textos[i], textos[j])[0]
            dist[i][j] = dist[j][i] = 1.0 - similitud
    return dist


def escribir_csv(filas: list[dict], ruta: str) -> None:
    columnas = ["algoritmo", "magnitud", "tamano", "complejidad_teorica",
                "mediana", "minimo", "maximo", "repeticiones"]
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def main() -> None:
    raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ruta_abstracts = os.path.join(raiz, "data", "processed", "abstracts.json")
    dir_salida = os.path.join(raiz, "data", "benchmark")

    if not os.path.exists(ruta_abstracts):
        print(f"No se encontro {ruta_abstracts}")
        print("Ejecuta primero: python -m src.preprocessing.extractor")
        return

    with open(ruta_abstracts, encoding="utf-8") as fh:
        registros = json.load(fh)
    textos = [r["texto"] for r in registros]

    print(f"Corpus: {len(textos)} abstracts, "
          f"{sum(len(t) for t in textos)} caracteres en total\n")

    print("Similitud (escala con la longitud del texto)")
    tamanos_texto = [100, 200, 400, 600, 800, 1000, 1200, 1400]
    filas = benchmark_similitud(textos[0], textos[1], tamanos_texto)

    print("\nClustering (escala con el numero de documentos)")
    dist = matriz_distancias(textos)
    tamanos_docs = [4, 6, 8, 10, 12, 14, 16, 18, 20]
    filas += benchmark_clustering(dist, tamanos_docs)

    print("\nClustering sobre matrices sinteticas (escalado asintotico)")
    filas += benchmark_clustering_sintetico([20, 30, 40, 60, 80, 110, 150])

    ruta_csv = os.path.join(dir_salida, "rendimiento.csv")
    escribir_csv(filas, ruta_csv)
    print(f"\n{len(filas)} mediciones escritas en: {ruta_csv}")


if __name__ == "__main__":
    main()

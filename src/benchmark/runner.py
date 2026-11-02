"""
Medición de tiempos de ejecución de los algoritmos del proyecto
----------------------------------------------------------------
Produce la evidencia empírica con la que se contrasta la complejidad asintótica
teórica de cada algoritmo.

QUÉ SE MIDE Y POR QUÉ
=====================

Los algoritmos del proyecto escalan respecto a magnitudes distintas, así que
medirlos en un solo eje no diría nada:

  - Similitud clásica (Levenshtein, Needleman-Wunsch, Jaccard, coseno TF-IDF):
    escala con la LONGITUD del texto. Se mide sobre prefijos crecientes de los
    dos abstracts más largos del corpus.
  - Comparación de embeddings: escala con la DIMENSIÓN del vector. La inferencia
    del modelo es una caja negra y no se mide.
  - Clustering jerárquico: escala con el NÚMERO DE DOCUMENTOS. Se mide sobre
    submatrices de tamaño creciente.
  - Búsqueda de patrones: se separan el PREPROCESAMIENTO y la BÚSQUEDA, se mide
    el escalado con la longitud del texto y se fuerza el peor caso de cada uno.

CÓMO SE MIDE
============

  - `time.perf_counter`, que es el reloj monótono de mayor resolución
    disponible; `time.time` puede retroceder si el sistema ajusta la hora.
  - 101 repeticiones por punto y se reporta la MEDIANA, no la media: una sola
    pausa del recolector de basura o del planificador del sistema operativo
    dispara un valor atípico que arrastraría la media. Con un número impar, la
    mediana es una medición real y no el promedio de dos.
  - El recolector de basura se desactiva durante cada medición, por el mismo
    motivo.
  - Se descarta una ejecución de calentamiento antes de medir, para no incluir
    el coste de la primera resolución de nombres y de los cachés fríos.

Los CSV resultantes son la entrada de `python_viz.graficas`; este módulo no
dibuja nada, solo produce datos.
"""

from __future__ import annotations

import csv
import gc
import json
import os
import platform
import statistics
import time
from typing import Callable

from src.clustering.linkage import hierarchical_clustering
from src.pattern import boyer_moore, kmp, rabin_karp
from src.preprocessing.transformacion import transformar_abstract
from src.similarity.classic.cosine import compute_idf, tfidf_cosine_similarity
from src.similarity.classic.jaccard import jaccard_similarity
from src.similarity.classic.levenshtein import levenshtein_distance
from src.similarity.classic.needleman_wunsch import normalized_needleman_wunsch
from src.similarity.metrics import dot_product, norm

REPETICIONES = 101

# Complejidad teórica de cada algoritmo, para contrastarla con lo observado.
# m y n son las longitudes de los dos textos; para el clustering, n documentos.
COMPLEJIDAD_TEORICA = {
    "levenshtein": "O(m·n)",
    "needleman_wunsch": "O(m·n)",
    "jaccard": "O(T)",
    "coseno_tfidf": "O(T + V)",
    "coseno_embeddings": "O(p)",
    "clustering_single": "O(n³)",
    "clustering_complete": "O(n³)",
    "clustering_average": "O(n³)",
    "kmp": "O(n + m)",
    "boyer_moore": "O(n·m) peor caso",
    "rabin_karp": "O(n·m) peor caso",
}

COLUMNAS = ["algoritmo", "magnitud", "tamano", "complejidad_teorica",
            "mediana", "minimo", "maximo", "repeticiones"]

COLUMNAS_PATRONES = ["algoritmo", "patron", "longitud_patron", "longitud_texto",
                     "ocurrencias", "comparaciones",
                     "preprocesamiento_mediana", "preprocesamiento_minimo",
                     "preprocesamiento_maximo", "busqueda_mediana",
                     "busqueda_minimo", "busqueda_maximo",
                     "porcentaje_preprocesamiento", "repeticiones"]

PATRONES = ["AI", "learning", "assessment", "artificial intelligence",
            "Artificial Intelligence in Education"]


def medir(funcion: Callable, repeticiones: int = REPETICIONES) -> dict:
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


def _fila(algoritmo: str, magnitud: str, tamano: int, stats: dict, unidad: str) -> dict:
    print(f"  {algoritmo:<22} {tamano:>7} {unidad:<5} "
          f"{stats['mediana'] * 1000:>10.4f} ms", flush=True)
    return {
        "algoritmo": algoritmo,
        "magnitud": magnitud,
        "tamano": tamano,
        "complejidad_teorica": COMPLEJIDAD_TEORICA[algoritmo],
        **stats,
    }


def benchmark_similitud(
    texto_a: str,
    texto_b: str,
    corpus: list[str],
    tamanos: list[int],
    repeticiones: int = REPETICIONES,
) -> list[dict]:
    """
    Mide los cuatro algoritmos clásicos sobre prefijos crecientes de dos textos.

    Usar prefijos de abstracts reales, y no cadenas sintéticas, mantiene la
    distribución de caracteres y la densidad de vocabulario del corpus, que es
    justo lo que determina el coste de Jaccard y del coseno.

    El IDF se calcula antes de medir: es un paso previo por corpus, no por par.
    """
    filas: list[dict] = []
    for tam in tamanos:
        a, b = texto_a[:tam], texto_b[:tam]
        if len(a) < tam or len(b) < tam:
            continue                # no hay texto suficiente para este tamaño
        # El prefijo puede cortar una palabra: se incluye para que tenga IDF
        idf = compute_idf(corpus + [a, b])
        for nombre, funcion in (
            ("levenshtein", lambda x=a, y=b: levenshtein_distance(x, y)),
            ("needleman_wunsch", lambda x=a, y=b: normalized_needleman_wunsch(x, y)),
            ("jaccard", lambda x=a, y=b: jaccard_similarity(x, y)),
            ("coseno_tfidf", lambda x=a, y=b, z=idf: tfidf_cosine_similarity(
                x, y, corpus, z, con_angulo=False)),
        ):
            filas.append(_fila(nombre, "caracteres", tam, medir(funcion, repeticiones), "car."))
    return filas


def benchmark_embeddings(
    vector_a: list[float],
    vector_b: list[float],
    dimensiones: list[int],
    repeticiones: int = REPETICIONES,
) -> list[dict]:
    """
    Mide el coseno de dos vectores, truncándolos a dimensiones crecientes.

    No incluye el ángulo: cuesta O(1), pero sus 60 iteraciones de bisección tardan
    más que el coseno de 384 dimensiones y taparían el crecimiento con p.
    """
    filas: list[dict] = []
    for dim in dimensiones:
        a, b = vector_a[:dim], vector_b[:dim]
        stats = medir(lambda x=a, y=b: dot_product(x, y) / (norm(x) * norm(y)), repeticiones)
        filas.append(_fila("coseno_embeddings", "dimensiones", dim, stats, "dim."))
    return filas


def benchmark_clustering(
    dist: list[list[float]],
    tamanos: list[int],
    repeticiones: int = REPETICIONES,
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
            filas.append(_fila(nombre, "documentos", n, stats, "doc."))
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
    repeticiones: int = REPETICIONES,
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
            filas.append(_fila(nombre, "documentos_sinteticos", n, stats, "doc."))
    return filas


def _fases_de_busqueda(texto: str, patron: str) -> dict[str, tuple[Callable, Callable]]:
    """Por algoritmo, la función de preprocesamiento y la de búsqueda ya preparada."""
    lps = kmp.construir_lps(patron)
    malo = boyer_moore.tabla_caracter_malo(patron)
    bueno = boyer_moore.tabla_sufijo_bueno(patron)
    hashes = rabin_karp.preprocesar(texto, patron)
    return {
        "kmp": (
            lambda: kmp.construir_lps(patron),
            lambda: kmp.buscar_con_tabla(texto, patron, lps),
        ),
        "boyer_moore": (
            lambda: (boyer_moore.tabla_caracter_malo(patron),
                     boyer_moore.tabla_sufijo_bueno(patron)),
            lambda: boyer_moore.buscar_con_tablas(texto, patron, malo, bueno),
        ),
        "rabin_karp": (
            lambda: rabin_karp.preprocesar(texto, patron),
            lambda: rabin_karp.buscar_con_hashes(texto, patron, *hashes),
        ),
    }


MODULOS_DE_BUSQUEDA = {"kmp": kmp, "boyer_moore": boyer_moore, "rabin_karp": rabin_karp}


def benchmark_fases_patrones(
    texto: str,
    patrones: list[str],
    repeticiones: int = REPETICIONES,
) -> list[dict]:
    """Preprocesamiento frente a búsqueda, por algoritmo y patrón, sobre el corpus."""
    filas: list[dict] = []
    for patron in patrones:
        for nombre, (preparar, buscar) in _fases_de_busqueda(texto, patron).items():
            pre = medir(preparar, repeticiones)
            bus = medir(buscar, repeticiones)
            metricas = MODULOS_DE_BUSQUEDA[nombre].buscar_con_metricas(texto, patron)
            filas.append({
                "algoritmo": nombre,
                "patron": patron,
                "longitud_patron": len(patron),
                "longitud_texto": len(texto),
                "ocurrencias": len(metricas["ocurrencias"]),
                "comparaciones": metricas["comparaciones"],
                "preprocesamiento_mediana": pre["mediana"],
                "preprocesamiento_minimo": pre["minimo"],
                "preprocesamiento_maximo": pre["maximo"],
                "busqueda_mediana": bus["mediana"],
                "busqueda_minimo": bus["minimo"],
                "busqueda_maximo": bus["maximo"],
                "porcentaje_preprocesamiento":
                    100 * pre["mediana"] / (pre["mediana"] + bus["mediana"]),
                "repeticiones": repeticiones,
            })
            print(f"  {nombre:<12} {patron[:24]:<25} preproceso "
                  f"{pre['mediana'] * 1e6:>8.2f} us   búsqueda "
                  f"{bus['mediana'] * 1e6:>10.2f} us", flush=True)
    return filas


def benchmark_busqueda_texto(
    texto: str,
    patron: str,
    tamanos: list[int],
    repeticiones: int = REPETICIONES,
) -> list[dict]:
    """Fase de búsqueda sobre prefijos crecientes del corpus: escalado con n."""
    filas: list[dict] = []
    for tam in tamanos:
        prefijo = texto[:tam]
        for nombre, (_, buscar) in _fases_de_busqueda(prefijo, patron).items():
            filas.append(_fila(nombre, "texto_corpus", tam, medir(buscar, repeticiones), "car."))
    return filas


def benchmark_peor_caso(
    n: int,
    longitudes: list[int],
    repeticiones: int = REPETICIONES,
) -> list[dict]:
    """
    Búsqueda de "aaa...a" en "aaa...a": el patrón aparece en todas las posiciones.

    Es la entrada que separa las cotas teóricas: Boyer-Moore y Rabin-Karp tienen
    que comparar los m caracteres en cada alineación, O(n·m); KMP no vuelve atrás
    sobre el texto y se mantiene en O(n + m). Se fija n y se hace crecer m.
    """
    texto = "a" * n
    filas: list[dict] = []
    for m in longitudes:
        for nombre, (_, buscar) in _fases_de_busqueda(texto, "a" * m).items():
            filas.append(_fila(nombre, "patron_peor_caso", m, medir(buscar, repeticiones), "m"))
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


def escribir_csv(filas: list[dict], ruta: str, columnas: list[str] = COLUMNAS) -> None:
    os.makedirs(os.path.dirname(ruta), exist_ok=True)
    with open(ruta, "w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def main() -> None:
    raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ruta_abstracts = os.path.join(raiz, "data", "processed", "abstracts.json")
    ruta_vectores = os.path.join(raiz, "data", "processed", "vectores.json")
    dir_salida = os.path.join(raiz, "data", "benchmark")

    if not os.path.exists(ruta_abstracts):
        print(f"No se encontro {ruta_abstracts}")
        print("Ejecuta primero: python -m src.preprocessing.extractor")
        return

    with open(ruta_abstracts, encoding="utf-8") as fh:
        registros = json.load(fh)
    textos = [r["texto"] for r in registros]

    with open(ruta_vectores, encoding="utf-8") as fh:
        vectores = json.load(fh)

    print(f"Corpus: {len(textos)} abstracts, "
          f"{sum(len(t) for t in textos)} caracteres en total, "
          f"{REPETICIONES} repeticiones por punto\n")

    print("Similitud clasica (escala con la longitud del texto)", flush=True)
    por_longitud = sorted(textos, key=len, reverse=True)
    tamanos_texto = [100, 200, 400, 800, 1200, 1600]
    filas = benchmark_similitud(por_longitud[0], por_longitud[1], textos, tamanos_texto)

    print("\nComparacion de embeddings (escala con la dimension)", flush=True)
    filas += benchmark_embeddings(vectores["minilm"][0], vectores["minilm"][1],
                                  [24, 48, 96, 192, 384])

    print("\nClustering (escala con el numero de documentos)", flush=True)
    dist = matriz_distancias([transformar_abstract(t) for t in textos])
    filas += benchmark_clustering(dist, [4, 6, 8, 10, 12, 14, 16, 18, 20])

    print("\nClustering sobre matrices sinteticas (escalado asintotico)", flush=True)
    filas += benchmark_clustering_sintetico([20, 30, 40, 60, 80, 110, 150])

    corpus_unido = "\n".join(textos)

    print("\nBusqueda de patrones: escalado con el texto", flush=True)
    filas += benchmark_busqueda_texto(corpus_unido, "artificial intelligence",
                                      [1700, 3400, 6800, 13600, len(corpus_unido)])

    print("\nBusqueda de patrones: peor caso", flush=True)
    filas += benchmark_peor_caso(5000, [10, 20, 40, 80, 160, 320])

    escribir_csv(filas, os.path.join(dir_salida, "rendimiento.csv"))

    print("\nBusqueda de patrones: preprocesamiento frente a busqueda", flush=True)
    filas_patrones = benchmark_fases_patrones(corpus_unido, PATRONES)
    escribir_csv(filas_patrones, os.path.join(dir_salida, "rendimiento_patrones.csv"),
                 COLUMNAS_PATRONES)

    with open(os.path.join(dir_salida, "entorno.json"), "w", encoding="utf-8") as fh:
        json.dump({
            "python": platform.python_version(),
            "sistema": platform.platform(),
            "procesador": platform.processor(),
            "repeticiones": REPETICIONES,
        }, fh, indent=2)

    print(f"\n{len(filas) + len(filas_patrones)} mediciones escritas en: {dir_salida}")


if __name__ == "__main__":
    main()

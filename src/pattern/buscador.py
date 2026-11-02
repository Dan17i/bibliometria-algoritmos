"""
Buscador de patrones sobre el corpus
=====================================
Aplica los tres algoritmos de coincidencia exacta a los 20 abstracts y
compara sus resultados y su rendimiento.

Los tres deben devolver EXACTAMENTE las mismas posiciones. Si difieren, uno
está mal: no hay margen de interpretación en una búsqueda exacta.
"""

from __future__ import annotations

import gc
import json
import os
import statistics
import time

from src.pattern import boyer_moore, kmp, rabin_karp

ALGORITMOS = {
    "KMP": kmp,
    "Boyer-Moore": boyer_moore,
    "Rabin-Karp": rabin_karp,
}

COMPLEJIDAD = {
    "KMP": {"preproceso": "O(m)", "busqueda": "O(n)", "peor": "O(n + m)"},
    "Boyer-Moore": {"preproceso": "O(m + s)", "busqueda": "O(n/m) mejor caso",
                    "peor": "O(n * m)"},
    "Rabin-Karp": {"preproceso": "O(m)", "busqueda": "O(n + m) promedio",
                   "peor": "O(n * m)"},
}


def buscar_en_corpus(registros: list[dict], patron: str, algoritmo
                     ) -> list[dict]:
    """
    Busca el patrón en los abstracts y devuelve las coincidencias por artículo.

    La búsqueda distingue mayúsculas de minúsculas porque la coincidencia es
    exacta: el patrón y el texto no se transforman.
    """
    resultados = []
    for registro in registros:
        posiciones = algoritmo.buscar(registro["texto"], patron)
        if posiciones:
            resultados.append({
                "id": registro["id"],
                "titulo": registro["titulo"],
                "ocurrencias": len(posiciones),
                "posiciones": posiciones,
                "contextos": [
                    _contexto(registro["texto"], p, len(patron))
                    for p in posiciones[:3]
                ],
            })
    return resultados


def _contexto(texto: str, pos: int, largo: int, margen: int = 40) -> str:
    """Fragmento alrededor de una coincidencia, para mostrarla en contexto."""
    inicio = max(0, pos - margen)
    fin = min(len(texto), pos + largo + margen)
    fragmento = texto[inicio:fin].replace("\n", " ")
    return ("..." if inicio > 0 else "") + fragmento + ("..." if fin < len(texto) else "")


def medir(funcion, texto: str, patron: str, repeticiones: int = 101) -> float:
    """Mediana del tiempo en milisegundos."""
    funcion(texto, patron)
    tiempos = []
    for _ in range(repeticiones):
        gc.collect()
        gc.disable()
        try:
            inicio = time.perf_counter()
            funcion(texto, patron)
            tiempos.append((time.perf_counter() - inicio) * 1000)
        finally:
            gc.enable()
    return statistics.median(tiempos)


def comparar(registros: list[dict], patron: str) -> dict:
    """
    Ejecuta los tres algoritmos sobre el corpus completo y compara.

    Devuelve las coincidencias, los tiempos y las métricas internas de cada
    uno (comparaciones, saltos, colisiones), que es lo que permite explicar
    por qué uno es más rápido que otro sobre los mismos datos.
    """
    corpus = "\n".join(r["texto"] for r in registros)

    salidas, tiempos, metricas = {}, {}, {}
    for nombre, modulo in ALGORITMOS.items():
        salidas[nombre] = modulo.buscar(corpus, patron)
        tiempos[nombre] = medir(modulo.buscar, corpus, patron)
        metricas[nombre] = modulo.buscar_con_metricas(corpus, patron)

    nombres = list(salidas)
    coinciden = all(salidas[nombres[0]] == salidas[o] for o in nombres[1:])

    return {
        "patron": patron,
        "longitud_corpus": len(corpus),
        "coinciden": coinciden,
        "total_ocurrencias": len(salidas[nombres[0]]),
        "tiempos_ms": tiempos,
        "metricas": metricas,
        "por_articulo": buscar_en_corpus(registros, patron, kmp),
    }


def main() -> None:
    ruta = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "data", "processed", "abstracts.json")
    with open(ruta, encoding="utf-8") as fh:
        registros = json.load(fh)

    patrones = ["artificial intelligence", "learning", "assessment",
                "systematic review", "ChatGPT"]

    print(f"Corpus: {len(registros)} abstracts, "
          f"{sum(len(r['texto']) for r in registros)} caracteres\n")

    todos = []
    for patron in patrones:
        r = comparar(registros, patron)
        todos.append(r)

        print(f"PATRON: '{patron}'")
        print(f"  Los tres coinciden : {'SI' if r['coinciden'] else 'NO'}")
        print(f"  Ocurrencias        : {r['total_ocurrencias']} "
              f"en {len(r['por_articulo'])} articulos")
        print(f"  {'ALGORITMO':<14}{'TIEMPO':>12}{'COMPARACIONES':>16}{'EXTRA':>26}")
        for nombre in ALGORITMOS:
            m = r["metricas"][nombre]
            t = r["tiempos_ms"][nombre]
            if nombre == "KMP":
                extra = f"saltos: {m['saltos']}"
            elif nombre == "Boyer-Moore":
                extra = f"alineaciones: {m['alineaciones']}"
            else:
                extra = f"colisiones: {m['colisiones']}"
            print(f"  {nombre:<14}{t:>9.4f} ms{m['comparaciones']:>16,}{extra:>26}")
        print()

    salida = os.path.join(os.path.dirname(ruta), "resultados_patrones.json")
    json.dump(todos, open(salida, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"Resultados en {salida}")


if __name__ == "__main__":
    main()

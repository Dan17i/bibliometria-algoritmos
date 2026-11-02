"""
Caso de estudio del Requerimiento 3
===================================
Ejecuta los tres algoritmos sobre un patrón y una oración de un abstract, y guarda
cada paso en data/resultados/traza_req3.json. De ahí salen las tablas del documento
técnico y las figuras de `python_viz.trazas`.

    python -m src.pattern.traza
"""

from __future__ import annotations

import json
import os

from src.pattern import boyer_moore, kmp, rabin_karp

ID_ARTICULO = "P10"
PATRON = "learning"
INICIO_ORACION = "Trend Two posits"
FIN_ORACION = "approaches."

# Un primo pequeño para mostrar colisiones: con q = 1 000 003 no aparecen en un
# texto tan corto.
PRIMO_PEQUENO = 101


def extraer_oracion(texto: str) -> str:
    """La oración del caso de estudio, localizada con el propio KMP."""
    inicio = kmp.buscar(texto, INICIO_ORACION)[0]
    fin = kmp.buscar(texto[inicio:], FIN_ORACION)[0] + inicio + len(FIN_ORACION)
    return texto[inicio:fin]


def construir_traza(texto: str, patron: str, primo_pequeno: int = PRIMO_PEQUENO) -> dict:
    """Todos los pasos de los tres algoritmos sobre el mismo texto y patrón."""
    n = len(texto)
    lps, pasos_lps = kmp.construir_lps_con_pasos(patron)
    ocurrencias_kmp, pasos_kmp = kmp.buscar_con_pasos(texto, patron)
    ocurrencias_bm, pasos_bm = boyer_moore.buscar_con_pasos(texto, patron, max_pasos=n)
    ocurrencias_rk, ventanas = rabin_karp.buscar_con_pasos(texto, patron, max_pasos=n)
    _, ventanas_q_pequeno = rabin_karp.buscar_con_pasos(texto, patron, max_pasos=n,
                                                        primo=primo_pequeno)

    factor, hash_patron, hash_primera_ventana = rabin_karp.preprocesar(texto, patron)

    return {
        "patron": patron,
        "texto": texto,
        "n": n,
        "m": len(patron),
        "kmp": {
            "lps": lps,
            "pasos_lps": pasos_lps,
            "pasos": pasos_kmp,
            "ocurrencias": ocurrencias_kmp,
            "comparaciones": kmp.buscar_con_metricas(texto, patron)["comparaciones"],
        },
        "boyer_moore": {
            "caracter_malo": boyer_moore.tabla_caracter_malo(patron),
            "sufijo_bueno": boyer_moore.tabla_sufijo_bueno(patron),
            "pasos": pasos_bm,
            "ocurrencias": ocurrencias_bm,
            "comparaciones": boyer_moore.buscar_con_metricas(texto, patron)["comparaciones"],
        },
        "rabin_karp": {
            "base": rabin_karp.BASE,
            "primo": rabin_karp.PRIMO,
            "factor": factor,
            "hash_patron": hash_patron,
            "hash_primera_ventana": hash_primera_ventana,
            "ventanas": ventanas,
            "ocurrencias": ocurrencias_rk,
            "comparaciones": rabin_karp.buscar_con_metricas(texto, patron)["comparaciones"],
            "primo_pequeno": primo_pequeno,
            "ventanas_primo_pequeno": ventanas_q_pequeno,
        },
    }


def main() -> None:
    raiz = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    with open(os.path.join(raiz, "data", "processed", "abstracts.json"), encoding="utf-8") as fh:
        registros = json.load(fh)

    articulo = next(r for r in registros if r["id"] == ID_ARTICULO)
    traza = construir_traza(extraer_oracion(articulo["texto"]), PATRON)
    traza["articulo"] = ID_ARTICULO

    salida = os.path.join(raiz, "data", "resultados", "traza_req3.json")
    with open(salida, "w", encoding="utf-8") as fh:
        json.dump(traza, fh, ensure_ascii=False, indent=1)

    colisiones = [v["posicion"] for v in traza["rabin_karp"]["ventanas_primo_pequeno"] if v["colision"]]
    print(f"Texto ({traza['n']} caracteres): {traza['texto']}")
    print(f"LPS: {traza['kmp']['lps']}")
    print(f"Caracter malo: {traza['boyer_moore']['caracter_malo']}")
    print(f"Sufijo bueno: {traza['boyer_moore']['sufijo_bueno']}")
    print(f"Ocurrencias: {traza['kmp']['ocurrencias']}")
    print(f"Comparaciones: KMP {traza['kmp']['comparaciones']}, "
          f"Boyer-Moore {traza['boyer_moore']['comparaciones']}, "
          f"Rabin-Karp {traza['rabin_karp']['comparaciones']}")
    print(f"Colisiones con q = {PRIMO_PEQUENO}: {colisiones}")
    print(f"Escrito: {salida}")


if __name__ == "__main__":
    main()

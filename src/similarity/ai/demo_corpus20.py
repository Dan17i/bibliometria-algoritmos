"""
Corrida sobre los 20 abstracts reales (data/processed/abstracts.json).
-----------------------------------------------------------------------
Responde a los pendientes de Req. 1 IA / Req. 2 con el corpus completo:

    .venv\\Scripts\\python.exe -m src.similarity.ai.demo_corpus20

Para cada representación (GloVe promediado sin stopwords, GloVe centrado y
MiniLM) calcula la matriz de distancias coseno, aplica los 3 enlaces, evalúa con
CPCC y silueta, y exporta el JSON de fusiones (formato de export.py).

Salidas (en tests/fixtures/daniel/corpus20/):
    - fusiones_<representacion>.json : fusiones de los 3 enlaces.
    - resumen.json                   : CPCC, mejor k y silueta por representación y enlace,
                                       más rango y tau de Kendall entre representaciones.
"""
import json
from pathlib import Path

from src.clustering.evaluation import best_k_by_silhouette, cophenetic_correlation, cut_tree, silhouette
from src.clustering.export import export_merges_json, load_export
from src.clustering.linkage import LINKAGES, hierarchical_clustering
from src.similarity.ai.demo_flujo import kendall_tau, upper_triangle, value_range
from src.similarity.ai.demo_modelos import glove_embeddings, sentence_embeddings
from src.similarity.ai.stopwords import STOPWORDS
from src.similarity.metrics import center_vectors, cosine_distance, cosine_similarity, pairwise_matrix

RAIZ = Path(__file__).resolve().parents[3]
ABSTRACTS = RAIZ / "data" / "processed" / "abstracts.json"
SALIDA = RAIZ / "tests" / "fixtures" / "daniel" / "corpus20"


def load_corpus(path: Path = ABSTRACTS) -> tuple[list[str], list[str]]:
    """(ids, textos) de los abstracts procesados por Cayón (solo lectura)."""
    registros = json.loads(path.read_text(encoding="utf-8"))
    return [r["id"] for r in registros], [r["texto"] for r in registros]


def representaciones(textos: list[str]) -> dict[str, list[list[float]]]:
    """Un vector por abstract según cada enfoque. Costo dominado por el modelo."""
    glove = glove_embeddings(textos, stopwords=STOPWORDS)["vectors"]
    centrado, _ = center_vectors(glove)
    return {
        "glove": glove,
        "glove_centrado": centrado,
        "minilm": sentence_embeddings(textos),
    }


def evaluar(dist: list[list[float]]) -> dict:
    """3 enlaces con CPCC, mejor k por silueta y silueta del corte elegido. O(n^3)."""
    n = len(dist)
    out = {}
    for linkage in LINKAGES:
        merges = hierarchical_clustering(dist, linkage)
        mejor = best_k_by_silhouette(dist, merges)
        out[linkage] = {
            "cpcc": cophenetic_correlation(dist, merges)["cpcc"],
            "mejor_k": mejor["best_k"],
            "silueta_mejor_k": mejor["best_silhouette"],
            "silueta_por_k": mejor["by_k"],
            "corte_mejor_k": cut_tree(merges, n, mejor["best_k"]),
            "altura_final": merges[-1][2],
        }
    return out


def main() -> None:
    ids, textos = load_corpus()
    SALIDA.mkdir(parents=True, exist_ok=True)
    vectores = representaciones(textos)

    resumen = {"n": len(ids), "ids": ids, "representaciones": {}}
    similitudes = {}
    for nombre, vecs in vectores.items():
        sim = upper_triangle(pairwise_matrix(vecs, cosine_similarity))
        similitudes[nombre] = sim
        dist = pairwise_matrix(vecs, cosine_distance)
        ruta = SALIDA / f"fusiones_{nombre}.json"
        export_merges_json(str(ruta), dist, ids)
        load_export(str(ruta))
        resumen["representaciones"][nombre] = {
            "rango_similitud": value_range(sim),
            "enlaces": evaluar(dist),
        }

    resumen["tau_vs_minilm"] = {
        n: kendall_tau(similitudes[n], similitudes["minilm"]) for n in ("glove", "glove_centrado")
    }
    (SALIDA / "resumen.json").write_text(json.dumps(resumen, indent=2), encoding="utf-8")

    print(f"=== {len(ids)} abstracts; salida en {SALIDA} ===")
    for nombre, r in resumen["representaciones"].items():
        print(f"\n{nombre}: rango de similitud {r['rango_similitud']:.4f}")
        for linkage, e in r["enlaces"].items():
            print(f"  {linkage:9} CPCC {e['cpcc']:.4f} | mejor k {e['mejor_k']} "
                  f"silueta {e['silueta_mejor_k']:.4f} | altura final {e['altura_final']:.4f}")
    print("\nTau de Kendall vs MiniLM:", {k: round(v, 3) for k, v in resumen["tau_vs_minilm"].items()})


if __name__ == "__main__":
    main()

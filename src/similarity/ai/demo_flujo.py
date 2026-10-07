"""
Demo de punta a punta con modelos reales (4 abstracts de docs/articulos/).
--------------------------------------------------------------------------
Genera todo lo que pide el documento para P1-P3 y el flujo completo:

    .venv\\Scripts\\python.exe -m src.similarity.ai.demo_flujo

Salidas (en tests/fixtures/daniel/):
    - experimento_centrado.json : rangos de GloVe con/sin stopwords, con/sin centrar.
    - fusiones_p1_p4.json       : fusiones de los 3 enlaces con MiniLM (formato de export.py).
Imprime en pantalla la demostración paso a paso (markdown) y el resumen del clustering.
"""
import json

from src.clustering.evaluation import (
    best_k_by_silhouette,
    cophenetic_correlation,
    cut_tree,
    silhouette,
)
from src.clustering.export import export_merges_json, load_export
from src.clustering.linkage import LINKAGES, hierarchical_clustering
from src.similarity.ai.caso_estudio import SALIDA
from src.similarity.ai.demo_modelos import (
    glove_embeddings,
    load_abstracts,
    sentence_embeddings,
)
from src.similarity.ai.demostracion import demostracion_markdown
from src.similarity.ai.embeddings import load_gensim_vectors, tokenize
from src.similarity.ai.stopwords import STOPWORDS
from src.similarity.metrics import (
    center_vectors,
    cosine_distance,
    cosine_similarity,
    pairwise_matrix,
)


def upper_triangle(matrix: list[list[float]]) -> list[float]:
    """Valores de los pares i < j, en orden de fila."""
    n = len(matrix)
    return [matrix[i][j] for i in range(n) for j in range(i + 1, n)]


def value_range(values: list[float]) -> float:
    """Rango = máximo - mínimo. Complejidad O(m)."""
    return max(values) - min(values)


def kendall_tau(xs: list[float], ys: list[float]) -> float:
    """
    Tau de Kendall (sin empates): (concordantes - discordantes) / pares.
    Mide si dos métodos ordenan igual los pares. Complejidad O(m^2).
    """
    m = len(xs)
    concordant = 0
    discordant = 0
    for i in range(m):
        for j in range(i + 1, m):
            s = (xs[i] - xs[j]) * (ys[i] - ys[j])
            if s > 0:
                concordant += 1
            elif s < 0:
                discordant += 1
    total = m * (m - 1) / 2
    return (concordant - discordant) / total


def experimento_centrado(textos: list[str]) -> dict:
    """
    Hipótesis: el promedio de GloVe casi no discrimina por una dirección común.
    Se resta el vector medio de los abstracts antes del coseno y se mira si el
    rango de similitudes se abre y si se conserva el orden de MiniLM.
    """
    minilm = pairwise_matrix(sentence_embeddings(textos), cosine_similarity)
    ref = upper_triangle(minilm)
    out = {"minilm": {"rango": value_range(ref), "pares": ref}}
    for nombre, stop in (("con_stopwords", None), ("sin_stopwords", STOPWORDS)):
        vectores = glove_embeddings(textos, stopwords=stop)["vectors"]
        normal = upper_triangle(pairwise_matrix(vectores, cosine_similarity))
        centrados, _ = center_vectors(vectores)
        centrado = upper_triangle(pairwise_matrix(centrados, cosine_similarity))
        out[nombre] = {
            "sin_centrar": {"rango": value_range(normal), "tau_vs_minilm": kendall_tau(normal, ref), "pares": normal},
            "centrado": {"rango": value_range(centrado), "tau_vs_minilm": kendall_tau(centrado, ref), "pares": centrado},
        }
    return out


def flujo_clustering(ids: list[str], textos: list[str]) -> dict:
    """3 enlaces + CPCC + silueta + export JSON con embeddings MiniLM."""
    embeddings = sentence_embeddings(textos)
    dist = pairwise_matrix(embeddings, cosine_distance)
    ruta = SALIDA / "fusiones_p1_p4.json"
    export_merges_json(str(ruta), dist, ids)
    load_export(str(ruta))  # valida lo que se escribió

    n = len(ids)
    resumen = {}
    for linkage in LINKAGES:
        merges = hierarchical_clustering(dist, linkage)
        resumen[linkage] = {
            "merges": merges,
            "cpcc": cophenetic_correlation(dist, merges)["cpcc"],
            "mejor_k": best_k_by_silhouette(dist, merges),
            "k2": cut_tree(merges, n, 2),
            "silueta_k2": silhouette(dist, cut_tree(merges, n, 2))["silhouette"],
        }
    return {"distancias": dist, "resumen": resumen, "ruta": str(ruta)}


def main() -> None:
    ids, textos = load_abstracts()
    i, j = ids.index("P1"), ids.index("P3")

    # 1. demostración paso a paso
    vocab = set()
    for t in textos:
        vocab.update(tokenize(t))
    word_vectors = load_gensim_vectors("glove-wiki-gigaword-100", vocab)
    minilm = sentence_embeddings(textos)
    print(demostracion_markdown("P1", "P3", textos[i], textos[j], word_vectors, STOPWORDS, minilm[i], minilm[j]))

    # 3. experimento de centrado
    exp = experimento_centrado(textos)
    (SALIDA / "experimento_centrado.json").write_text(json.dumps(exp, indent=2), encoding="utf-8")
    print("=== Experimento de centrado (rango y tau de Kendall vs MiniLM) ===")
    print(f"MiniLM: rango {exp['minilm']['rango']:.4f}")
    for nombre in ("con_stopwords", "sin_stopwords"):
        for variante in ("sin_centrar", "centrado"):
            d = exp[nombre][variante]
            print(f"GloVe {nombre:14} {variante:11}: rango {d['rango']:.4f}  tau {d['tau_vs_minilm']:+.2f}")

    # 4. clustering de punta a punta
    flujo = flujo_clustering(ids, textos)
    print(f"\n=== Clustering P1-P4 con MiniLM (exportado en {flujo['ruta']}) ===")
    for linkage, r in flujo["resumen"].items():
        fus = [(a, b, round(h, 4), s) for a, b, h, s in r["merges"]]
        print(f"{linkage:9} CPCC {r['cpcc']:.4f} | k=2 {r['k2']} silueta {r['silueta_k2']:.4f} | "
              f"mejor k {r['mejor_k']['best_k']}\n          fusiones {fus}")


if __name__ == "__main__":
    main()

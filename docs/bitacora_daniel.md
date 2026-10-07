# Bitácora de Desarrollo - Daniel

## Requerimientos 1 (IA) y 2 (Clustering jerárquico)

### Registro de actividades

**[2026-10-07]**
- **Actividad:** `src/similarity/metrics.py`, métricas vectoriales a mano.
- **Detalles:**
  - `dot_product`, `norm`, `cosine_similarity`, `cosine_distance`, `euclidean_distance`, `mean_vector` y `angle_degrees`, solo con listas, bucles y aritmética.
  - `cosine_steps` y `euclidean_steps` devuelven los pasos intermedios (producto punto, normas, diferencias, ángulo) para la demostración del documento.
  - Convención: vector cero → coseno 0.0 y distancia coseno 1.0; dimensiones distintas → `ValueError`.
  - `angle_degrees` obtiene arccos por bisección con serie de Taylor de cos, sin `math`.
  - Todas las funciones documentan fórmula y complejidad temporal y espacial (`O(n)` tiempo; `O(1)` espacio, salvo `euclidean_steps` y `mean_vector`).
- **Pruebas:** `tests/test_metrics.py` (idénticos, ortogonales, opuestos, vector cero, dimensiones distintas, casos resueltos a mano).
- **Resultado:** `pytest tests/test_metrics.py` → 14 passed.
- **Próximo paso:** `src/similarity/ai/` (embeddings, previo acuerdo de `requirements.txt`) y `src/clustering/`.

- **Actividad:** `src/similarity/ai/` (Req. 1 IA) y `pairwise_matrix` en `metrics.py`.
- **Detalles:**
  - `embeddings.py`: enfoque 1 = promedio de vectores de palabras (`average_word_vectors`, reporta tokens usados y fuera de vocabulario); enfoque 2 = embeddings de oración (`encode_documents`). Los modelos entran por inyección (dict de vectores / función `encode`), así los tests no descargan nada. Cargadores perezosos `load_word_vectors` (gensim, solo lectura) y `load_sentence_encoder` (sentence-transformers, solo `.encode`), siempre con `.tolist()`.
  - `compare.py`: `compare_embeddings` calcula coseno, ángulo y euclidiana por par con los pasos intermedios.
  - `metrics.pairwise_matrix(vectors, metric)`: matriz de distancias para el clustering, `O(m^2 n)`.
  - `requirements.txt`: se agregaron `pytest`, `gensim`, `sentence-transformers` (aún no instaladas en `.venv`).
- **Pruebas:** `tests/test_ai_embeddings.py`; suite completa: 22 passed.
- **Pendiente:** elegir y descargar modelos reales (Word2Vec/FastText y sentence-transformers) cuando existan los abstracts de Cayón; script del caso de estudio.
- **Próximo paso:** `src/clustering/` (matriz de distancias → 3 enlaces → lista de fusiones `(a, b, distancia, tamaño)` → métrica interna).

- **Actividad:** `src/clustering/` (Req. 2): enlaces y evaluación interna.
- **Detalles:**
  - `linkage.py`: `hierarchical_clustering(dist, linkage, return_steps)` con Single, Complete y Average (UPGMA), actualizando distancias con Lance-Williams. Devuelve `[(a, b, distancia, tamaño)]` (ids 0..n-1 para hojas y n+k para la k-ésima fusión) y, opcionalmente, el historial paso a paso. Desempates deterministas por ids. Tiempo `O(n^3)`, espacio `O(n^2)`.
  - `evaluation.py`: matriz cofenética, correlación cofenética (CPCC) con Pearson a mano, `cut_tree` (corte en k clusters), silueta y `best_k_by_silhouette`. Cada función con fórmula y complejidad.
  - El dendrograma lo dibuja Cayón a partir de la lista de fusiones.
- **Pruebas:** `tests/test_clustering.py` con 4 puntos en una recta resueltos a mano (single/complete/average, CPCC ≈ 0.8985, silueta 5/6, 4/5, 5/9, 9/13). Suite completa: 40 passed.
- **Pendiente:** justificación matemática de CPCC vs silueta para el documento; aplicar sobre los 20 abstracts cuando Cayón entregue `data/processed/`; Ward no se implementó (se eligieron Single/Complete/Average).

- **Actividad:** flujo sintético completo y guardia de transparencia.
- **Detalles:**
  - Revisión: `compare.py` solo importa `cosine_steps` y `euclidean_steps` de `metrics.py`; en `src/` no hay llamadas a `util.cos_sim`, `.similarity()`, `n_similarity` ni `wmdistance` (las menciones están en comentarios).
  - `tests/fixtures/daniel/`: 6 abstracts sintéticos (visión, lenguaje, audio) y vectores de palabras de 3 dimensiones inventados.
  - `tests/test_flujo_sintetico.py`: abstracts → promedio de vectores → matriz de distancias coseno → 3 enlaces → CPCC y silueta. Los 3 enlaces recuperan los 3 temas (CPCC ≈ 0.973, silueta 0.991 en k = 3).
  - `tests/test_transparencia.py`: revisa el AST de mis módulos y falla si importan sklearn/scipy/numpy/difflib o llaman a `cos_sim`, `similarity`, `n_similarity`, `wmdistance`.
- **Pruebas:** suite completa: 58 passed.
- **Aviso al equipo:** `sentence-transformers` instala PyTorch (varios GB); avisar a Samuel y Cayón antes de `pip install -r requirements.txt`.

- **Actividad:** guardia verificada, abstracts "puente" y `requirements-ai.txt`.
- **Detalles:**
  - `test_transparencia.py`: la guardia ahora es la función `violaciones(codigo)`; 10 casos de código falso (import sklearn/scipy/numpy, `x.similarity(y)`, `cos_sim`, `n_similarity`, `wmdistance`) confirman que se detectan, y un caso limpio confirma que no hay falsos positivos (nombres solo en docstrings/comentarios).
  - `requirements-ai.txt` (gensim, sentence-transformers) separado de `requirements.txt` (solo pytest) para que Samuel y Cayón no instalen PyTorch.
  - Fixture ampliado a 8 abstracts: B1 mezcla visión y lenguaje, B2 mezcla lenguaje y audio.
- **Resultados con puentes (distancia coseno):**

  | Enlace | CPCC | Altura de las 2 últimas fusiones | Corte k=3 | Silueta k=3 |
  |---|---|---|---|---|
  | single | 0.671 | 0.062 / 0.067 | {V1,V2,B1} {L1,L2,B2} {A1,A2} | 0.746 |
  | complete | 0.749 | 0.270 / 0.278 | igual | 0.746 |
  | average | 0.751 | 0.183 / 0.198 | igual | 0.746 |

  - Los tres cortes en k=3 coinciden; la diferencia está en las alturas y en el CPCC. Single une los temas a alturas ~4 veces menores que complete (el puente basta para conectarlos): efecto de encadenamiento, pérdida de contraste en el dendrograma y menor correlación cofenética. Average es el que mejor conserva las distancias originales.
  - Para la justificación: la silueta, calculada sobre un corte, no distingue los enlaces cuando coinciden en la partición; el CPCC sí, porque mide todo el árbol.
- **Pruebas:** suite completa: 72 passed.

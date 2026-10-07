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

---

## Borrador: justificación matemática de la evaluación interna (Req. 2)

> Borrador para el documento de diseño. Los números salen del ejemplo sintético (`tests/fixtures/daniel/`, 8 abstracts con 2 "puente", distancia coseno sobre promedio de vectores de palabras). Hay que repetir el cálculo con los 20 abstracts reales.

### 1. Correlación cofenética (CPCC)

**Distancia cofenética.** Dado un dendrograma, la distancia cofenética $C(x,y)$ entre dos elementos es la altura $h$ de la fusión que los reúne por primera vez en un mismo cluster. En el código, para cada fusión $(a, b, h, \cdot)$ todo par $(x \in a,\ y \in b)$ recibe $C(x,y)=h$ (`cophenetic_matrix`).

**Definición.** Sea $d(x,y)$ la distancia original y sean $x<y$ los $m = n(n-1)/2$ pares de elementos:

$$
\mathrm{CPCC}=\frac{\sum_{x<y}\bigl(d(x,y)-\bar d\bigr)\bigl(C(x,y)-\bar C\bigr)}
{\sqrt{\sum_{x<y}\bigl(d(x,y)-\bar d\bigr)^2\;\sum_{x<y}\bigl(C(x,y)-\bar C\bigr)^2}},
\qquad \bar d=\tfrac1m\sum_{x<y} d(x,y),\ \ \bar C=\tfrac1m\sum_{x<y} C(x,y)
$$

Es el coeficiente de correlación de Pearson entre las dos listas de $m$ distancias. Toma valores en $[-1,1]$; en la práctica queda en $(0,1]$ y **más cerca de 1 significa que el dendrograma conserva mejor las distancias originales**. Si una de las series es constante la varianza es 0 y el código devuelve 0.0 (`pearson`).

**Qué mide.** Evalúa **el árbol completo**: usa los $m$ pares y todas las alturas de fusión, sin elegir un número de clusters. Es la medida natural cuando el entregable es el dendrograma.

**Ejemplo a mano** (4 puntos en la recta 0, 1, 5, 7; single linkage). Distancias originales $[1,5,7,4,6,2]$ y cofenéticas $[1,4,4,4,4,2]$, con $\bar d = 25/6$ y $\bar C = 19/6$:
$\sum (d-\bar d)(C-\bar C)=83/6$, $\sum (d-\bar d)^2=161/6$, $\sum (C-\bar C)^2=53/6$, luego $\mathrm{CPCC}=\dfrac{83/6}{\sqrt{(161/6)(53/6)}}\approx 0.8985$ (verificado en `tests/test_clustering.py`).

### 2. Coeficiente de silueta

**Definición.** Dada una partición en $k$ clusters (un corte del dendrograma, `cut_tree`) y un elemento $i$ del cluster $C_i$:

$$
a(i)=\frac{1}{|C_i|-1}\sum_{j\in C_i,\ j\ne i} d(i,j),\qquad
b(i)=\min_{C\ne C_i}\ \frac1{|C|}\sum_{j\in C} d(i,j),\qquad
s(i)=\frac{b(i)-a(i)}{\max\{a(i),\,b(i)\}}
$$

con $s(i)=0$ si $|C_i|=1$ (o si $\max\{a,b\}=0$). La silueta global es $S=\frac1n\sum_i s(i)\in[-1,1]$. $a(i)$ es la cohesión (distancia media a su propio cluster) y $b(i)$ la separación (distancia media al cluster vecino más cercano): $s(i)\to1$ si $i$ está bien asignado, $\approx 0$ si está en la frontera y $<0$ si probablemente está en el cluster equivocado.

**Qué mide.** Evalúa **una partición concreta** (un corte en $k$ clusters), no el árbol. Depende de $k$: `best_k_by_silhouette` recorre $k=2,\dots,n-1$ y elige el mejor.

**Ejemplo a mano** (mismos 4 puntos, etiquetas $[0,0,1,1]$): $s=\left(\tfrac56,\ \tfrac45,\ \tfrac59,\ \tfrac9{13}\right)$ y $S\approx0.7203$ (verificado en los tests).

### 3. Complejidad (con $n$ abstracts, $m=n(n-1)/2$ pares)

| Cálculo | Tiempo | Espacio | Justificación |
|---|---|---|---|
| Matriz de distancias (`pairwise_matrix`, dim. $p$) | $O(n^2 p)$ | $O(n^2)$ | $m$ evaluaciones de una métrica $O(p)$ |
| Clustering jerárquico (`hierarchical_clustering`) | $O(n^3)$ | $O(n^2)$ | $n-1$ iteraciones; buscar el mínimo recorre $O(k^2)$ pares y actualizar cuesta $O(k)$; $\sum_{k} k^2=O(n^3)$ |
| Matriz cofenética | $O(n^2)$ | $O(n^2)$ | cada par se asigna una sola vez, en la fusión que lo reúne |
| CPCC (cofenética + Pearson) | $O(n^2)$ | $O(n^2)$ | Pearson hace 4 recorridos sobre $m$ valores; el costo total dado el árbol es $O(n^2)$ |
| Silueta de un corte | $O(n^2)$ | $O(n)$ | para cada $i$ se recorren sus $n-1$ distancias |
| Silueta para todos los cortes (`best_k_by_silhouette`) | $O(n^3)$ | $O(n)$ | $n-2$ cortes, cada uno con `cut_tree` y silueta $O(n^2)$ |

Con $n=20$ todo es inmediato; el clustering ($O(n^3)$) domina el costo en ambos casos y la evaluación no lo cambia asintóticamente.

### 4. Por qué el CPCC es la métrica principal para comparar Single, Complete y Average

Resultados del ejemplo sintético con abstracts puente:

| Enlace | CPCC | Altura de las 2 últimas fusiones | Silueta en $k=3$ |
|---|---|---|---|
| Single | 0.671 | 0.062 / 0.067 | 0.746 |
| Complete | 0.749 | 0.270 / 0.278 | 0.746 |
| Average | 0.751 | 0.183 / 0.198 | 0.746 |

1. **El entregable es un dendrograma por criterio.** La pregunta es qué enlace produce el *árbol* más fiel, y el CPCC es la única de las dos que evalúa el árbol entero. La silueta evalúa un corte.
2. **La silueta no discrimina aquí.** Los tres enlaces dan el mismo corte en $k=3$ ($\{V_1,V_2,B_1\}$, $\{L_1,L_2,B_2\}$, $\{A_1,A_2\}$) y, por tanto, exactamente la misma silueta (0.746). Dos criterios distintos quedan empatados porque la silueta solo ve la partición, no las alturas a las que se construyó.
3. **El CPCC sí los separa** ($0.671<0.749<0.751$) y lo hace por una causa interpretable: el **encadenamiento de Single**. Single define $D(A,B)=\min d(x,y)$, así que un abstract puente basta para unir dos temas; por eso reúne los temas a alturas $\approx4$ veces menores que Complete (0.067 vs 0.278). Muchos pares quedan con una distancia cofenética muy inferior a la real, lo que reduce la correlación y aplana el dendrograma (poco contraste entre las últimas fusiones).
4. **No depende de $k$.** No hay que fijar de antemano el número de clusters, que además no es conocido en un corpus real de 20 abstracts.
5. **Es comparable entre enlaces** porque usa la misma matriz $d$ y la misma escala para todos.

**Limitaciones (a declarar en el documento).**
- El CPCC mide **fidelidad a las distancias**, no coherencia temática. Un árbol puede reproducir bien las distancias de una representación pobre de los abstracts.
- Está sesgado a favor de Average: en UPGMA la altura de cada fusión es la media de las distancias originales entre los dos clusters, es decir, el árbol "promedia" por construcción la matriz $d$. La diferencia Average–Complete (0.751 vs 0.749) es pequeña y no es concluyente por sí sola.
- La silueta sigue siendo útil como **complemento**: una vez elegido el enlace, sirve para escoger el corte ($k$) y detectar elementos mal asignados (en el ejemplo los puentes tienen $s(B_1)=0.209$ y $s(B_2)=0.326$, frente a $0.87$–$0.98$ de los abstracts puros).
- El ejemplo es sintético y de 8 elementos: la conclusión se debe contrastar con los 20 abstracts reales y, si es posible, con una clasificación temática de referencia.

**Decisión propuesta:** comparar los tres enlaces con el CPCC (métrica principal), usar la silueta de `best_k_by_silhouette` para elegir el corte y reportar ambas, junto con una lectura de las alturas del dendrograma.

---

- **Actividad:** exportación de fusiones a JSON para los dendrogramas de Cayón.
- **Detalles:**
  - `src/clustering/export.py`: `build_export`, `export_merges_json(path, dist, labels)` y `load_export` (valida el archivo). Usa solo el módulo `json`.
  - Formato (documentado en el docstring del módulo): `version`, `n`, `labels`, `ids` y, por enlace (`single`, `complete`, `average`), `merges` = lista de `[a, b, distancia, tamaño]` y `cpcc`. Ids: hojas `0..n-1`, la fusión `k` crea el cluster `n+k`; alturas no decrecientes.
  - Se serializa antes de abrir el archivo y se rechaza NaN/Infinity (`allow_nan=False`), para no dejar JSON a medias o inválido.
- **Pruebas:** `tests/test_clustering_export.py`.
- **Pendiente:** acordar con Cayón la ruta de salida del JSON (no se escribe nada en `data/processed/`).

- **Actividad:** modelos preentrenados reales sobre los abstracts de `docs/articulos/`.
- **Detalles:**
  - Se instaló `requirements-ai.txt` en `.venv` (torch 2.14.1 CPU, transformers, gensim 4.4.0, sentence-transformers 6.1.0). Arrastra `numpy`, `scipy` y `scikit-learn` como dependencias transitivas: están instaladas pero **no se importan** (lo vigila `test_transparencia.py`).
  - Fixture `tests/fixtures/daniel/abstracts_pdfs.json`: abstracts de los 4 PDFs copiados a mano (P1–P4). Son de IA en educación, no de IA generativa.
  - `embeddings.py`: `load_gensim_vectors` (solo convierte a listas las palabras necesarias) y `load_sentence_encoder` con `all-MiniLM-L6-v2` por defecto. `average_word_vectors` ahora prueba las partes de las palabras compuestas ("problem-solving") si el token completo no está en el vocabulario.
  - `src/similarity/ai/demo_modelos.py` (`python -m src.similarity.ai.demo_modelos`) imprime ambas matrices; el coseno sale de `metrics.py`.
  - Pruebas con descarga en `tests/test_ai_modelos.py`: se saltan salvo que `RUN_MODEL_TESTS=1`.
- **Resultados (similitud coseno):**

  | | GloVe-100 promedio | | | | MiniLM-L6-v2 | | | |
  |---|---|---|---|---|---|---|---|---|
  | | P1 | P2 | P3 | P4 | P1 | P2 | P3 | P4 |
  | P1 | 1.000 | 0.988 | 0.994 | 0.988 | 1.000 | 0.590 | 0.735 | 0.431 |
  | P2 | | 1.000 | 0.989 | 0.987 | | 1.000 | 0.592 | 0.431 |
  | P3 | | | 1.000 | 0.990 | | | 1.000 | 0.483 |
  | P4 | | | | 1.000 | | | | 1.000 |

  - Cobertura de GloVe: 0 a 5 tokens fuera de vocabulario por abstract (`aied`, `decremental`).
  - **Hallazgo para el análisis crítico:** el promedio de vectores GloVe sin quitar stopwords da similitudes casi idénticas (0.987–0.994) porque las palabras funcionales dominan el promedio; no discrimina. MiniLM sí separa (0.43–0.74) y el par más cercano es P1–P3 (alfabetización y concepciones erróneas de IA), el más alejado P4 (resolución de problemas complejos).
- **Pruebas:** por defecto 84 passed, 2 skipped; con `RUN_MODEL_TESTS=1`, los 2 de modelos pasan (77 s).
- **Pendiente:** probar el promedio de GloVe quitando stopwords (permitido por el enunciado) o ponderando; comparar con los 4 clásicos de Samuel sobre el mismo par.

- **Actividad:** enfoque 1 sin stopwords y caso de estudio P1–P3.
- **Detalles:**
  - `src/similarity/ai/stopwords.py`: lista de stopwords en inglés escrita a mano (sin NLTK ni spaCy) y `remove_stopwords`. `average_word_vectors(..., stopwords=...)` las elimina antes de promediar y las reporta en `removed_stopwords`.
  - `src/similarity/ai/caso_estudio.py`: corre sobre un par los clásicos de Samuel importados sin modificar (`levenshtein_distance`, `jaccard_similarity`, `cosine_similarity`) y los enfoques de IA; Needleman-Wunsch y coseno TF-IDF quedan marcados como pendientes de Samuel. Guarda `tests/fixtures/daniel/matrices_p1_p4.json` y `tests/fixtures/daniel/caso_estudio_p1_p3.json`.
  - Elección del caso: P1–P3 es el par más parecido según MiniLM (0.735).
- **Efecto de las stopwords en el enfoque 1 (similitud coseno):**

  | Par | GloVe con stopwords | GloVe sin stopwords | MiniLM |
  |---|---|---|---|
  | P1–P2 | 0.9884 | 0.9760 | 0.5898 |
  | P1–P3 | 0.9938 | 0.9810 | 0.7353 |
  | P1–P4 | 0.9882 | 0.9592 | 0.4307 |
  | P2–P3 | 0.9885 | 0.9657 | 0.5916 |
  | P2–P4 | 0.9874 | 0.9664 | 0.4309 |
  | P3–P4 | 0.9897 | 0.9657 | 0.4825 |
  | Rango (máx − mín) | 0.0064 | 0.0218 | 0.3044 |

  Quitar stopwords baja todos los valores y triplica el rango (0.006 → 0.022), pero el promedio de GloVe sigue casi sin discriminar frente a MiniLM (0.30). Queda como hipótesis, sin probar, que los vectores de GloVe comparten una dirección dominante que el promedio conserva; una prueba posible es restar el vector medio del corpus (no implementada). El orden relativo se mantiene: P1–P3 sigue siendo el par más cercano con los tres métodos.
- **Caso de estudio P1 vs P3:**

  | Algoritmo | Tipo | Resultado | Similitud |
  |---|---|---|---|
  | Levenshtein (caracteres) | clásico, PD | distancia 1100 de 1544 | 0.2876 (normalizada) |
  | Jaccard (palabras) | clásico, conjuntos | \|∩\| = 35, \|∪\| = 221 | 0.1584 |
  | Coseno TF | clásico, espacio vectorial | 221 dimensiones | 0.7360 |
  | Needleman-Wunsch | clásico, PD | pendiente (Samuel) | — |
  | Coseno TF-IDF | clásico, espacio vectorial | pendiente (Samuel) | — |
  | GloVe-100 promedio, con stopwords | IA | ángulo 6.4°, euclidiana 0.465 | 0.9938 |
  | GloVe-100 promedio, sin stopwords | IA | ángulo 11.2°, euclidiana 0.606 | 0.9810 |
  | all-MiniLM-L6-v2 | IA | ángulo 42.7°, euclidiana 0.728 | 0.7353 |

  Lectura preliminar (para el análisis crítico): Jaccard y Levenshtein dan valores bajos porque miden coincidencia literal (palabras o caracteres) y los dos abstracts tratan lo mismo con vocabulario distinto; el coseno TF (0.736) y MiniLM (0.735) coinciden casi por azar, pues miden cosas distintas (frecuencia de palabras exactas frente a significado). GloVe promedio satura cerca de 1. La normalización de Levenshtein (1 − d/máx) es nuestra, no de Samuel. Levenshtein sobre caracteres de textos de ~1500 caracteres es O(m·n) ≈ 1.6 M celdas y mide poco semántica.
- **Observación para Samuel (solo lectura):** `src/similarity/classic/jaccard.py:30-31` y `src/similarity/classic/cosine.py:22-23` eliminan toda la puntuación con `replace`, así que "problem-solving" se vuelve "problemsolving" y "AI's" se vuelve "ais"; además no quitan stopwords.
- **Pruebas:** `tests/test_ai_caso_estudio.py`; por defecto 94 passed, 2 skipped.

# Análisis de algoritmos en el contexto de la bibliometría

Proyecto de **Análisis de Algoritmos** · Ingeniería de Sistemas y Computación · Universidad del Quindío · 2026-2.

Implementa, **desde cero y sin librerías que resuelvan el algoritmo**, técnicas de análisis textual sobre los *abstracts* de un corpus de 20 artículos científicos sobre inteligencia artificial en educación, y compara los enfoques clásicos con los basados en IA:

- **Similitud textual** entre dos o más abstracts: Levenshtein, Needleman-Wunsch, Jaccard, coseno con TF-IDF, promedio de vectores GloVe y embeddings de `all-MiniLM-L6-v2`.
- **Agrupamiento jerárquico** de los 20 abstracts: Single, Complete y Average, con su dendrograma y evaluación interna (CPCC y silueta).
- **Búsqueda de patrones**: KMP, Boyer-Moore y Rabin-Karp, con búsqueda exacta y análisis de su complejidad contrastado con mediciones.

**Aplicación publicada:** https://bibliometria-algoritmos.onrender.com (la primera carga después de un rato sin uso puede tardar unos 20 segundos).

Cada resultado de la aplicación tiene un botón «Ver demostración», que ejecuta ese algoritmo en el momento y muestra qué hace, sus fórmulas con los valores reales y cada paso del cálculo.

## Tabla de contenido

1. [Regla de implementación explícita](#regla-de-implementación-explícita)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Instalación](#instalación)
4. [Uso](#uso)
5. [Pruebas](#pruebas)
6. [Resultados](#resultados)
7. [Documentación](#documentación)
8. [Equipo](#equipo)
9. [Uso de IA generativa](#uso-de-ia-generativa)

## Regla de implementación explícita

No se usan funciones de alto nivel que implementen los algoritmos. Todo el cálculo se escribe con estructuras básicas de Python (listas, diccionarios, bucles y aritmética), de modo que el funcionamiento interno sea transparente y analizable, y las funciones devuelven los **pasos intermedios** (matriz de programación dinámica, pesos, historial de fusiones, comparaciones) para poder demostrarlos paso a paso.

| No se usa para los algoritmos | Sí se permite |
|---|---|
| `scikit-learn`, `scipy` (incluido `scipy.cluster.hierarchy`) y `numpy` | Extraer texto del PDF (`PyMuPDF`) |
| `difflib`, `rapidfuzz`, `textdistance`, `jellyfish` y similares | Listas de *stopwords* |
| `model.similarity`, `n_similarity`, `wmdistance`, `util.cos_sim` | Modelos preentrenados, **solo para obtener los vectores** |
| En el núcleo de los algoritmos: `min`, `max`, `sum`, `sorted`, `set`, `lower`, `split`, `replace`, `find`, `index`, `count`, `intersection`, `union`, `math` | `matplotlib` para graficar |

Sí se usan `len`, `range`, `ord`, `chr` y `enumerate`, que son parte de la estructura básica del lenguaje. Los modelos entregan vectores; el promedio, el producto punto, la norma, el coseno, la distancia euclidiana, la matriz de distancias, la fusión de clusters y las métricas de evaluación se calculan a mano. La regla se verifica automáticamente en [`tests/test_transparencia.py`](tests/test_transparencia.py), que recorre el árbol sintáctico del código y falla si se incumple.

## Estructura del repositorio

```
bibliometria-algoritmos/
├── app/
│   ├── servidor.py               # Rutas de la aplicación (Flask sobre waitress)
│   ├── explicaciones*.py         # Arman cada demostración: idea, fórmulas, pasos y conclusiones
│   ├── lanzar.py                 # Lanzador local
│   └── static/                   # Página, estilos, scripts y KaTeX
├── data/
│   ├── processed/                # Abstracts extraídos, vectores y los 190 pares calculados
│   ├── resultados/               # Fusiones, caso de estudio y trazas (JSON)
│   ├── benchmark/                # Tiempos medidos (CSV) y datos del entorno
│   └── figuras/                  # Dendrogramas, gráficas y diagramas (PNG)
├── docs/
│   ├── Documento_Tecnico.docx    # Diseño, arquitectura y explicación de cada requerimiento
│   ├── Justificacion_uso_IA.docx # Declaración del uso de IA generativa
│   ├── bitacora_samuel.md        # Registro del trabajo en los algoritmos clásicos
│   ├── bitacora_daniel.md        # Registro del trabajo en IA y clustering
│   ├── enunciado/                # Descripción del proyecto y plan de trabajo (PDF)
│   └── articulos/                # Los 20 PDF del corpus
├── python_viz/                   # Dendrogramas, gráficas, trazas y diagramas (matplotlib)
├── src/
│   ├── preprocessing/            # Extractor de los PDF y transformación del abstract
│   ├── similarity/
│   │   ├── classic/              # Levenshtein, Needleman-Wunsch, Jaccard, coseno y tokenizador
│   │   ├── ai/                   # Embeddings, comparación, caso de estudio y corridas
│   │   ├── metrics.py            # Producto punto, norma, coseno, euclidiana y ángulo a mano
│   │   └── traza_clasicos.py     # Pasos de la demostración de los clásicos
│   ├── clustering/               # Enlaces jerárquicos, evaluación interna y exportación
│   ├── pattern/                  # KMP, Boyer-Moore, Rabin-Karp y caso de estudio
│   └── benchmark/                # Medición de tiempos
├── tests/                        # Pruebas con pytest
├── Iniciar aplicacion.bat        # Abre la aplicación con doble clic en Windows
├── render.yaml                   # Configuración del despliegue
├── requirements.txt              # Desarrollo: pytest, pymupdf, matplotlib, flask, waitress
├── requirements-ai.txt           # Modelos: gensim y sentence-transformers (pesadas)
└── requirements-web.txt          # Servidor publicado: flask y waitress
```

La sección 2.8 del [documento técnico](docs/Documento_Tecnico.docx) dice en qué carpeta, archivo y funciones está cada algoritmo.

## Instalación

Las pruebas y mediciones se hicieron con **Python 3.11**; el servidor publicado usa Python 3.12.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

Con eso basta para correr la aplicación y las pruebas, porque los resultados ya vienen guardados en `data/`. Las librerías de IA van aparte porque **`sentence-transformers` instala PyTorch (varios GB)**, y solo hacen falta para regenerar los vectores:

```bash
pip install -r requirements-ai.txt
```

Instalar `requirements-ai.txt` arrastra `numpy`, `scipy` y `scikit-learn` como dependencias de esas librerías. Quedan instalados, pero el código de los algoritmos no los importa. La primera vez que se usan los modelos se descargan automáticamente (GloVe ≈ 130 MB y MiniLM ≈ 90 MB).

## Uso

Todos los ejemplos se ejecutan desde la raíz del repositorio.

**Abrir la aplicación**

En Windows, doble clic en `Iniciar aplicacion.bat`. En cualquier sistema:

```bash
python -m app.lanzar
```

El lanzador arranca el servidor y abre el navegador. Si ya hay una instancia abierta, la reutiliza: para cargar una versión nueva del código hay que cerrar primero la anterior con Ctrl+C.

**Regenerar todos los resultados**, desde los PDF hasta las figuras (requiere `requirements-ai.txt`):

```bash
python -m src.preprocessing.extractor         # PDF -> data/processed/abstracts.csv y .json
python -m src.similarity.ai.precalculo        # vectores.json, glove_palabras.json y similitudes_pd.json
python -m src.similarity.ai.demo_corpus20     # fusiones por representación y resumen.json
python -m src.similarity.ai.demo_clasicos20   # comparación de los 190 pares
python -m src.similarity.ai.caso_estudio      # caso de estudio P6-P9 y su traza
python -m src.pattern.traza                   # caso de estudio de la búsqueda de patrones
python -m python_viz.dendrograma              # dendrogramas -> data/figuras/
python -m python_viz.trazas                   # iteraciones y desplazamientos de la búsqueda
python -m src.benchmark.runner                # tiempos -> data/benchmark/
python -m python_viz.graficas                 # gráficas de rendimiento
python -m python_viz.arquitectura             # diagrama general
python -m python_viz.diagramas                # paquetes, módulos y secuencias
```

**Usar los módulos directamente**

```python
from src.similarity.metrics import cosine_steps, euclidean_distance

pasos = cosine_steps([1, 1], [1, 0])
print(pasos["dot"], pasos["norm_a"], pasos["cosine"], pasos["angle_degrees"])  # 1 1.414... 0.7071... 45.0
print(euclidean_distance([0, 0], [3, 4]))                                     # 5.0
```

```python
from src.similarity.metrics import cosine_distance, pairwise_matrix
from src.clustering.linkage import hierarchical_clustering
from src.clustering.evaluation import cophenetic_correlation, cut_tree, silhouette

dist = pairwise_matrix(embeddings, cosine_distance)          # matriz de distancias n x n
merges = hierarchical_clustering(dist, "average")            # [(a, b, distancia, tamaño), ...]
print(cophenetic_correlation(dist, merges)["cpcc"])
print(silhouette(dist, cut_tree(merges, len(dist), 3))["silhouette"])
```

El formato del JSON de fusiones (ids de hojas, alturas y orden) está en el docstring de [`src/clustering/export.py`](src/clustering/export.py).

## Pruebas

```bash
python -m pytest tests -q
```

Son **604 pruebas**: 602 corren siempre y 2 se omiten por defecto porque descargan los modelos reales. Para incluirlas:

```bash
set RUN_MODEL_TESTS=1           # PowerShell: $env:RUN_MODEL_TESTS = "1"
python -m pytest tests -q
```

Incluyen casos pequeños resueltos a mano, un flujo completo con datos sintéticos, la extracción de los 20 PDF, la guardia de transparencia, la aplicación web y la comparación de los tres algoritmos de búsqueda contra una búsqueda ingenua. 232 de ellas verifican que los diagramas de secuencia del documento técnico coincidan con el código.

## Resultados

El análisis completo está en el [documento técnico](docs/Documento_Tecnico.docx).

**Caso de estudio P6–P9** (el par más parecido según MiniLM entre los 190 del corpus):

| Algoritmo | Similitud |
|---|---|
| Levenshtein (normalizada) | 0.2454 |
| Needleman-Wunsch (normalizada) | 0.3146 |
| Jaccard | 0.1552 |
| Coseno TF (referencia) | 0.6808 |
| Coseno TF-IDF | 0.1600 |
| GloVe-100 promediado | 0.9792 |
| all-MiniLM-L6-v2 | 0.8500 |

Las medidas literales dan valores bajos porque miden coincidencia de caracteres o palabras, no de tema. TF-IDF baja mucho frente al coseno TF porque las palabras presentes en todos los abstracts dejan de contar. GloVe promediado casi no discrimina: todos los pares del corpus quedan entre 0.91 y 0.98. MiniLM va de 0.42 a 0.85.

**Agrupamiento de los 20 abstracts** (CPCC; distancia = 1 − coseno calculado a mano):

| Representación | Single | Complete | Average |
|---|---|---|---|
| all-MiniLM-L6-v2 | 0.7973 | 0.7685 | **0.8371** |
| GloVe promediado | 0.6939 | 0.6154 | **0.7473** |
| GloVe promediado y centrado | 0.3976 | 0.5644 | **0.6242** |
| Jaccard | 0.6473 | 0.5450 | **0.7113** |

**Average** es el enlace más coherente en todas las representaciones, así que la elección no depende del modelo. La silueta es baja en todos los casos (con MiniLM, 0.188, 0.204 y 0.220 en k = 2) porque el corpus es de un solo tema; por eso el CPCC, que evalúa el árbol completo, es la métrica principal.

**Búsqueda de patrones.** Con «artificial intelligence» sobre los 20 abstracts, Boyer-Moore tarda 1.15 ms, KMP 4.49 ms y Rabin-Karp 13.87 ms (mediana de 101 repeticiones). El preprocesamiento es como máximo el 2.73 % del tiempo total. Los exponentes medidos sobre prefijos crecientes son 0.95 (KMP), 0.96 (Boyer-Moore) y 1.03 (Rabin-Karp).

**Limitaciones.** El CPCC mide fidelidad a las distancias, no coherencia temática, y favorece por construcción a Average. El tau de Kendall mide acuerdo entre métodos, no cuál acierta. El corpus es de 20 abstracts de un mismo tema.

## Documentación

- [`docs/Documento_Tecnico.docx`](docs/Documento_Tecnico.docx): arquitectura, diagramas, demostración de cada algoritmo, análisis crítico, complejidad, pruebas y despliegue.
- [`docs/Justificacion_uso_IA.docx`](docs/Justificacion_uso_IA.docx): qué herramientas de IA generativa se usaron, para qué y cómo se validó.
- [`docs/bitacora_samuel.md`](docs/bitacora_samuel.md) y [`docs/bitacora_daniel.md`](docs/bitacora_daniel.md): registro del trabajo, con los valores de cada momento.
- [`docs/enunciado/`](docs/enunciado/): descripción del proyecto y plan de trabajo.
- Cada función algorítmica documenta su fórmula y su complejidad temporal y espacial en el *docstring*.

## Equipo

| Integrante | Responsabilidades |
|---|---|
| **Samuel Castaño** | Requerimiento 1, algoritmos clásicos y su análisis matemático; auditoría final, demostraciones en la aplicación y actualización de la documentación |
| **Daniel Jurado Celemin** | Requerimiento 1, enfoques con IA y análisis crítico; Requerimiento 2, enlaces jerárquicos y evaluación interna |
| **Juan Esteban Cayón Marmolejo** | Requerimiento 3, búsqueda de patrones; preprocesamiento de los PDF, visualización, mediciones, despliegue y documento de diseño |

## Uso de IA generativa

En el desarrollo se usaron **Claude Code** y **Claude** (Anthropic) como asistentes de programación y redacción. Las herramientas fueron un apoyo y no reemplazan el diseño algorítmico ni el análisis formal, que son del equipo. El detalle de qué herramienta usó cada integrante, para qué y cómo se validó está en [`docs/Justificacion_uso_IA.docx`](docs/Justificacion_uso_IA.docx).

Los modelos `all-MiniLM-L6-v2` y GloVe son otra cosa: forman parte del requerimiento 1 y solo se usan para obtener vectores.

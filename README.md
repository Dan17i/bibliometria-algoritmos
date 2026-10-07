# Análisis de algoritmos en el contexto de la bibliometría

Proyecto de **Análisis de Algoritmos** · Ingeniería de Sistemas y Computación · Universidad del Quindío.

Implementa, **desde cero y sin librerías que resuelvan el algoritmo**, técnicas de análisis textual sobre los *abstracts* de un corpus de 20 artículos científicos (dominio: IA generativa), y compara los enfoques clásicos con los basados en IA:

- **Similitud textual** entre abstracts: Levenshtein, Needleman-Wunsch, Jaccard, coseno con TF-IDF y similitud sobre *embeddings* de modelos preentrenados.
- **Clustering jerárquico** de los 20 abstracts: tres criterios de enlace (Single, Complete, Average), evaluación interna y datos para dibujar los dendrogramas.

> **Estado:** proyecto en desarrollo. La aplicación desplegada y el procesamiento de los 20 PDFs aún no están; ver [Estado del proyecto](#estado-del-proyecto).

## Tabla de contenido

1. [Estado del proyecto](#estado-del-proyecto)
2. [Regla de implementación explícita](#regla-de-implementación-explícita)
3. [Estructura del repositorio](#estructura-del-repositorio)
4. [Instalación](#instalación)
5. [Uso](#uso)
6. [Pruebas](#pruebas)
7. [Resultados preliminares](#resultados-preliminares)
8. [Documentación](#documentación)
9. [Equipo](#equipo)
10. [Uso de IA generativa](#uso-de-ia-generativa)

## Estado del proyecto

| Componente | Estado |
|---|---|
| Similitud: Levenshtein, Jaccard, coseno de frecuencias (TF) | Implementados (Samuel) |
| Similitud: Needleman-Wunsch | Pendiente (el archivo existe pero está vacío) |
| Similitud: coseno con TF-IDF | Pendiente (el coseno actual usa solo TF) |
| Similitud con IA: promedio de vectores GloVe y `all-MiniLM-L6-v2`, con coseno y euclidiana a mano | Implementada y con pruebas |
| Clustering jerárquico: Single, Complete, Average | Implementado y con pruebas |
| Evaluación interna: correlación cofenética (CPCC) y silueta | Implementada y con pruebas |
| Exportación de las fusiones a JSON para los dendrogramas | Implementada y con pruebas |
| Extracción de abstracts de los 20 PDFs | Pendiente |
| Dendrogramas, gráficas de rendimiento y benchmarks (`python_viz/`, `src/benchmark/`) | Pendiente |
| Aplicación desplegada (`app/`) y documento de diseño final | Pendiente |

Los resultados que se muestran más abajo se obtuvieron con cuatro artículos de ejemplo (`docs/articulos/`) y con datos sintéticos; **todavía no se han corrido sobre los 20 artículos del corpus**.

## Regla de implementación explícita

El enunciado prohíbe usar funciones de alto nivel que implementen los algoritmos pedidos. Todo el cálculo se escribe con estructuras básicas de Python (listas, diccionarios, bucles y aritmética), de modo que el funcionamiento interno sea transparente y analizable, y las funciones devuelven los **pasos intermedios** (matriz de programación dinámica, pesos, historial de fusiones) para poder demostrarlos paso a paso.

| No se usa para los algoritmos | Sí se permite |
|---|---|
| `scikit-learn`, `scipy` (incluido `scipy.cluster.hierarchy`) y `numpy` | Extraer texto del PDF (`pdfplumber` / `PyMuPDF`) |
| `difflib`, `rapidfuzz`, `textdistance`, `jellyfish` y similares | Listas de *stopwords* |
| `model.similarity`, `n_similarity`, `wmdistance`, `util.cos_sim` | Modelos preentrenados, **solo para obtener los vectores** |
| | `matplotlib` para graficar y `pandas` para leer/escribir CSV |

Los modelos de embeddings entregan vectores; el promedio, el producto punto, la norma, el coseno, la distancia euclidiana, la matriz de distancias, la fusión de clusters y las métricas de evaluación se calculan a mano. Esta regla se vigila automáticamente en [`tests/test_transparencia.py`](tests/test_transparencia.py), que revisa el código del módulo de IA, las métricas y el clustering y falla si importa `sklearn`, `scipy`, `numpy` o `difflib`, o si llama a las similitudes de los modelos.

## Estructura del repositorio

```
bibliometria-algoritmos/
├── app/                      # Aplicación desplegable (pendiente)
├── data/
│   ├── raw/                  # PDFs del docente (ignorados por git)
│   └── processed/            # Abstracts, títulos y autores extraídos
├── docs/
│   ├── enunciado/            # Enunciado y plan de trabajo (PDF)
│   ├── articulos/            # 4 artículos de ejemplo para pruebas
│   ├── bitacora_samuel.md    # Bitácora de los algoritmos clásicos
│   └── bitacora_daniel.md    # Bitácora de IA y clustering, con la justificación matemática
├── python_viz/               # Visualización y benchmarks (pendiente)
├── src/
│   ├── preprocessing/        # Extracción y limpieza de texto de los PDFs
│   ├── similarity/
│   │   ├── classic/          # Levenshtein, Needleman-Wunsch, Jaccard y coseno
│   │   ├── ai/               # Embeddings, comparación y caso de estudio
│   │   └── metrics.py        # Producto punto, norma, coseno y euclidiana a mano
│   ├── clustering/           # Enlaces jerárquicos, evaluación interna y exportación
│   └── benchmark/            # Medición de tiempos y exportación de CSV
├── tests/                    # Pruebas con pytest y datos de prueba
├── requirements.txt          # Dependencias base
└── requirements-ai.txt       # Librerías de IA (pesadas, opcionales)
```

## Instalación

Requiere **Python 3.13** (el desarrollo y las pruebas se hicieron con esa versión).

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux / macOS

pip install -r requirements.txt
```

`requirements.txt` solo trae lo necesario para el código base y las pruebas. Las librerías de IA van aparte porque **`sentence-transformers` instala PyTorch (varios GB)**:

```bash
pip install -r requirements-ai.txt     # opcional: gensim + sentence-transformers
```

Instalar `requirements-ai.txt` arrastra `numpy`, `scipy` y `scikit-learn` como dependencias de esas librerías. Quedan instalados, pero el código de los algoritmos no los importa.

La primera vez que se usan los modelos se descargan automáticamente (GloVe ≈ 130 MB y MiniLM ≈ 90 MB).

## Uso

Todos los ejemplos se ejecutan desde la raíz del repositorio.

**Similitud entre vectores**

```python
from src.similarity.metrics import cosine_steps, euclidean_distance

pasos = cosine_steps([1, 1], [1, 0])
print(pasos["dot"], pasos["norm_a"], pasos["cosine"], pasos["angle_degrees"])  # 1 1.414... 0.7071... 45.0
print(euclidean_distance([0, 0], [3, 4]))                                     # 5.0
```

**Clustering jerárquico, evaluación y exportación**

```python
from src.similarity.metrics import cosine_distance, pairwise_matrix
from src.clustering.linkage import hierarchical_clustering
from src.clustering.evaluation import cophenetic_correlation, cut_tree, silhouette
from src.clustering.export import export_merges_json

dist = pairwise_matrix(embeddings, cosine_distance)          # matriz de distancias n x n
merges = hierarchical_clustering(dist, "average")            # [(a, b, distancia, tamaño), ...]
print(cophenetic_correlation(dist, merges)["cpcc"])
print(silhouette(dist, cut_tree(merges, len(dist), 3))["silhouette"])

export_merges_json("fusiones.json", dist, etiquetas)         # los tres enlaces, listos para dibujar
```

El formato del JSON (ids de hojas, alturas y orden de las fusiones) está documentado en el docstring de [`src/clustering/export.py`](src/clustering/export.py).

**Demo con modelos preentrenados** (requiere `requirements-ai.txt`)

```bash
python -m src.similarity.ai.demo_modelos      # matrices de similitud de los 4 artículos de ejemplo
python -m src.similarity.ai.caso_estudio      # tabla comparativa clásicos vs IA para un par de abstracts
```

## Pruebas

```bash
python -m pytest tests -q
```

Hay pruebas con casos pequeños resueltos a mano (por ejemplo, un clustering de 4 puntos en una recta) y un flujo completo con datos sintéticos. Las pruebas que descargan modelos reales **no corren por defecto**:

```bash
set RUN_MODEL_TESTS=1           # PowerShell: $env:RUN_MODEL_TESTS = "1"
python -m pytest tests/test_ai_modelos.py -v
```

## Resultados preliminares

Con cuatro artículos de ejemplo de IA en educación (`docs/articulos/`), el par más parecido según `all-MiniLM-L6-v2` es P1–P3. Sobre ese par:

| Algoritmo | Similitud |
|---|---|
| Levenshtein (normalizada) | 0.2876 |
| Jaccard | 0.1584 |
| Coseno TF | 0.7360 |
| Needleman-Wunsch / coseno TF-IDF | pendientes |
| GloVe-100 promedio (con / sin stopwords) | 0.9938 / 0.9810 |
| all-MiniLM-L6-v2 | 0.7353 |

El promedio de vectores GloVe casi no discrimina entre abstracts (todos los pares quedan entre 0.96 y 0.99), mientras que MiniLM separa claramente (0.43 a 0.74). En el clustering, con datos sintéticos que incluyen abstracts «puente» entre temas, Single linkage encadena los grupos y reduce el CPCC a 0.67, frente a 0.75 de Complete y Average. Estos resultados son preliminares: se repetirán con los 20 abstracts. El análisis completo está en [`docs/bitacora_daniel.md`](docs/bitacora_daniel.md).

## Documentación

- [`docs/enunciado/`](docs/enunciado/): enunciado del proyecto y plan de trabajo.
- [`docs/bitacora_samuel.md`](docs/bitacora_samuel.md): algoritmos clásicos, con su fundamento matemático.
- [`docs/bitacora_daniel.md`](docs/bitacora_daniel.md): métricas, embeddings, clustering, justificación de la evaluación interna (CPCC frente a silueta) y caso de estudio.
- Cada función algorítmica documenta su fórmula y su complejidad temporal y espacial en el *docstring*.
- El documento de diseño final y la declaración de uso de IA generativa se entregarán aparte (pendiente).

## Equipo

| Integrante | Responsabilidades |
|---|---|
| **Samuel** | Requerimiento 1, algoritmos clásicos y su análisis matemático paso a paso |
| **Daniel** | Requerimiento 1, enfoques con IA y análisis crítico; Requerimiento 2, enlaces jerárquicos y evaluación interna |
| **Cayón** | Visualización y benchmarks, preprocesamiento de los PDFs, despliegue y documento de diseño |

## Uso de IA generativa

El enunciado permite la IA generativa solo como apoyo y exige declararla. Durante el desarrollo se usó **Claude Code** (Anthropic) como asistente de programación en el módulo de IA, las métricas, el clustering y sus pruebas, y para redactar parte de la documentación. La IA no reemplaza el diseño algorítmico ni el análisis formal, que son del equipo. La declaración formal completa irá en el documento de diseño final.

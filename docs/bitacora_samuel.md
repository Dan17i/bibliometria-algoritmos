# Bitácora de Desarrollo - Samuel

## Requerimiento 1: Análisis de Similitud Textual (Algoritmos Clásicos)

**Objetivos:**
1. Implementación explícita desde cero de 4 algoritmos clásicos de similitud.
2. Redacción del análisis algorítmico y matemático (demostración paso a paso con 2 *abstracts*).

---

### Registro de Actividades

**[2026-09-29]**
- **Actividad:** Análisis inicial del proyecto y configuración del entorno.
- **Detalles:** 
  - Se revisó el `Plan de Trabajo` para entender la distribución (Samuel, Cayón, Daniel).
  - Se analizó la estructura de carpetas creada por Daniel (`src/similarity/classic/` será nuestro directorio principal de trabajo).
- **Próximo Paso:** 
  - Definir e implementar el primer algoritmo clásico (ej. Distancia de Levenshtein) asegurando que no se usen librerías de alto nivel para su lógicas internas.

- **Actividad:** Implementación de Distancia de Levenshtein.
- **Detalles:** 
  - Se creó el archivo `levenshtein.py` en `src/similarity/classic/`.
  - Se implementó desde cero usando Programación Dinámica ($O(m \times n)$).
  - La función retorna no solo la distancia, sino también la matriz generada. Esto es un diseño estratégico para facilitar la posterior "demostración matemática paso a paso".
  - Se verificó que las carpetas `data/raw/` y `data/processed/` están vacías. No necesitamos los datos finales de los artículos para construir y probar la lógica de los algoritmos.

**[2026-09-29]**
- **Actividad:** Implementación explícita del Algoritmo Needleman-Wunsch.
- **Detalles Teóricos y Matemáticos:**
  - Se creó el script `needleman_wunsch.py` en `src/similarity/classic/`.
  - **Fundamento Matemático:** A diferencia de Levenshtein (que se enfoca en *minimizar el costo* de penalizaciones puras), Needleman-Wunsch busca *maximizar un puntaje de alineamiento global* ($F$).
  - La matriz $F$ se inicializa con los gaps base ($F_{i,0} = i \times d$ y $F_{0,j} = j \times d$).
  - La relación de recurrencia empleada es $F(i,j) = \max(F(i-1,j-1) + S(x_i,y_j),\; F(i-1,j) + d,\; F(i,j-1) + d)$.
  - **Transparencia Algorítmica:** No se utilizó ningún paquete pre-entrenado ni abstracciones como `BioPython` o librerías de NLP. El cálculo del puntaje, los arreglos y la validación de las métricas (`Match=+1`, `Mismatch=-1`, `Gap=-1`) se codificaron secuencialmente. 
  - **Importancia para el Documento Final:** Al igual que en Levenshtein, se retorna la `score matrix` generada. Esto significa que para la entrega del "análisis algorítmico y matemático paso a paso con 2 abstracts", solo tendremos que imprimir el estado de esta matriz y explicar cómo el algoritmo tomó las decisiones en cada celda.

- **Actividad:** Implementación explícita del Algoritmo de Similitud de Jaccard.
- **Detalles Teóricos y Matemáticos (Documentación Ampliada):**
  - Se creó el archivo `jaccard.py`. A diferencia de los algoritmos de secuencia anteriores (Prog. Dinámica), Jaccard aborda el problema bajo el paradigma de "Bolsa de Palabras" (Bag of Words) binario.
  - **Preprocesamiento transparente:** Se construyó una función auxiliar `preprocess_text_to_set` que simula la tokenización. Elimina la puntuación, transforma a minúsculas y separa por espacios explícitamente, transformando la lista de palabras en una estructura de datos `Set` de Python. Esto garantiza que operemos matemáticamente con conjuntos sin recurrir a paquetes preconstruidos como NLTK o Spacy.
  - **Fórmula Matemática:** La similitud se calcula mediante $J(A,B) = \frac{|A \cap B|}{|A \cup B|}$. Es decir, la cardinalidad de la intersección dividida por la cardinalidad de la unión.
  - **Análisis de Complejidad:**
    - Tiempo: $O(|A| + |B|)$ para convertir a sets y calcular uniones e intersecciones (donde $A$ y $B$ son el número de palabras en cada texto). Esto es computacionalmente más barato que $O(n \times m)$ de Programación Dinámica.
    - Espacio: $O(|A| + |B|)$ necesario para almacenar los conjuntos en memoria.
  - **Justificación de su elección para el proyecto:** Este modelo asume que el orden de las palabras en un *abstract* científico no importa tanto como su ocurrencia. Si dos *abstracts* comparten gran parte de sus términos clave, Jaccard lo reflejará con un valor cercano a 1.0. 
  - Para cumplir la restricción del documento de "demostración matemática paso a paso", la función retorna no solo la similitud (ej: 0.5), sino también los sets generados (la Intersección explícita y la Unión explícita). Así, en tu reporte podrás hacer un cuadro mostrando las palabras exactas que entraron en la intersección.

- **Actividad:** Implementación explícita de Similitud del Coseno (Cosine Similarity).
- **Detalles Teóricos y Matemáticos (Documentación Ampliada):**
  - Se creó el archivo `cosine.py`. Este algoritmo es fundamental porque marca el punto de transición hacia el espacio vectorial de palabras, que es exactamente la misma base que usará Daniel (embeddings de IA), lo que nos permitirá hacer la comparación directa en el Requerimiento 1.
  - **Conversión de Texto a Vectores:** 
    1. Se unificaron las palabras de ambos textos para crear el "espacio de características" (vocabulario global).
    2. Se representó cada texto como un arreglo donde cada posición corresponde a la frecuencia de una palabra del vocabulario (Term Frequency - TF).
  - **Cálculo Transparente (Álgebra Lineal desde cero):** 
    - **Producto Punto:** Se programó un bucle que multiplica las posiciones homologas de los vectores $\sum_{i=1}^{n} A_i \times B_i$.
    - **Magnitudes (Norma L2):** Se calculó usando sumatorias y potenciación a la 0.5 ($\sqrt{\sum A_i^2}$) en lugar de la librería `math` o `numpy`.
    - **Fórmula de la métrica:** $\cos(\theta) = \frac{A \cdot B}{||A|| \times ||B||}$.
  - **Análisis de Complejidad:**
    - Tiempo: $O(N)$ donde $N$ es la suma total de tokens en ambos textos, dominado por la construcción del vocabulario y los vectores.
    - Espacio: $O(|V|)$ donde $V$ es el tamaño del vocabulario unificado, necesario para almacenar los vectores densos.
  - Al retornar explícitamente el vocabulario unificado, el Vector A y el Vector B, la demostración paso a paso del caso de estudio (los 2 *abstracts*) será trivial: solo se copiará la tabla de frecuencias de los vectores arrojados por la consola y se mostrará la multiplicación manual.

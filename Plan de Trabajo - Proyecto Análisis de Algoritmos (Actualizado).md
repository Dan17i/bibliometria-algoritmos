# **Plan de Trabajo y Distribución de Tareas**

## **Proyecto: Análisis de algoritmos en el contexto de la bibliometría**

**Equipo:** Samuel, Cayón, Daniel  
**Referencia base:** Repositorio *Analisis-Algoritmos-Proy-Final-Financiero* (Arquitectura de algoritmos backend \+ Graficación visual independiente)

### ---

**1\. Requerimiento 1: Análisis de Similitud Textual**

> * **Samuel (Algoritmos Clásicos):**  
  * Implementación explícita desde cero de 4 algoritmos clásicos (ej. Programación Dinámica como Levenshtein/Needleman-Wunsch, Similitud del Coseno, Jaccard).  
  * Redacción del análisis algorítmico y matemático: demostración paso a paso (ej. llenado de la matriz) para un caso de estudio concreto con 2 *abstracts*.  
> * **Daniel (Inteligencia Artificial):**  
  * Implementación de 2 enfoques de similitud apoyados en IA mediante la generación de embeddings (ej. Word2Vec, FastText).  
  * Cálculo métrico de similitud (euclidiana o coseno) sobre los vectores generados.  
  * Análisis crítico comparando los resultados empíricos de los modelos de IA frente a los clásicos desarrollados por Samuel en el caso de estudio.

### **2\. Requerimiento 2: Agrupamiento Jerárquico (Clustering)**

> * **Daniel:**  
  * Implementación desde cero de 3 criterios de agrupamiento jerárquico (ej. Single Linkage, Complete Linkage, Average/Ward).  
  * Cálculo y justificación de métricas matemáticas de evaluación interna para determinar la coherencia de los agrupamientos.

### **3\. Módulo de Graficación y Benchmarking (Basado en Repositorio de Referencia)**

> * **Cayón:**  
  * Desarrollo del módulo de visualización de datos (replicando la arquitectura del directorio python\_viz y la exportación de CSVs de rendimiento del repositorio de referencia).  
  * Generación automatizada de las gráficas comparativas de rendimiento (benchmarks) de todos los algoritmos.  
  * Graficación visual (renderizado) de los 3 dendrogramas exigidos en el Requerimiento 2\.

### **4\. Requerimiento 6 y Transversales: Datos, Despliegue y Documentación**

> * **Cayón:**  
  * Preprocesamiento riguroso de los 20 artículos PDF proporcionados por el docente (extracción precisa del *abstract*, título y autores).  
  * Despliegue final de la aplicación asegurando la integración entre los algoritmos y la visualización.  
  * Estructuración del **Documento de Diseño Final**, unificando las explicaciones técnicas y redactando la declaración formal y explícita del uso de herramientas de IA Generativa.

### **5\. Restricciones Técnicas Absolutas**

> * **Transparencia Algorítmica:** Está terminantemente prohibido usar funciones de alto nivel que resuelvan los algoritmos directamente. Todo el comportamiento interno debe ser transparente y analizable.  
> * **Uso de IA:** No reemplaza bajo ninguna circunstancia el diseño algorítmico formal exigido.
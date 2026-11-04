def levenshtein_distance(str1: str, str2: str) -> tuple[int, list[list[int]]]:
    """
    Calcula la distancia de Levenshtein entre dos cadenas utilizando Programación Dinámica.
    
    La distancia de Levenshtein mide el número mínimo de operaciones requeridas para
    transformar una cadena en otra. Las operaciones permitidas son:
    - Inserción
    - Eliminación
    - Sustitución

    Fórmula Matemática:
        D(i, 0) = i,  D(0, j) = j
        D(i, j) = min(
            D(i-1, j) + 1,      # Eliminación
            D(i, j-1) + 1,      # Inserción
            D(i-1, j-1) + c     # Sustitución: c = 0 si str1[i-1] == str2[j-1], c = 1 si no
        )

    Complejidad Temporal: O(m * n), donde m y n son las longitudes de las cadenas.
    Complejidad Espacial: O(m * n) para almacenar la matriz de Programación Dinámica.
    
    Args:
        str1 (str): Primera cadena de texto.
        str2 (str): Segunda cadena de texto.
        
    Returns:
        tuple[int, list[list[int]]]: 
            - El primer elemento es la distancia mínima (int).
            - El segundo elemento es la matriz completa de programación dinámica (list[list[int]]).
              (Se retorna la matriz para facilitar el análisis matemático y demostración paso a paso).
    """
    m = len(str1)
    n = len(str2)
    
    # Inicialización de la matriz (m+1 filas por n+1 columnas) con ceros
    matriz = [[0 for _ in range(n + 1)] for _ in range(m + 1)]
    
    # Llenado de la primera fila y primera columna (casos base)
    for i in range(m + 1):
        matriz[i][0] = i
    for j in range(n + 1):
        matriz[0][j] = j
        
    # Llenado del resto de la matriz usando Programación Dinámica
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                costo_sustitucion = 0
            else:
                costo_sustitucion = 1
                
            eliminacion = matriz[i - 1][j] + 1
            insercion = matriz[i][j - 1] + 1
            sustitucion = matriz[i - 1][j - 1] + costo_sustitucion

            # La celda toma el mínimo de las tres operaciones
            menor = eliminacion
            if insercion < menor:
                menor = insercion
            if sustitucion < menor:
                menor = sustitucion
            matriz[i][j] = menor

    # La distancia final se encuentra en la esquina inferior derecha
    distancia_final = matriz[m][n]
    
    return distancia_final, matriz


if __name__ == "__main__":
    # Prueba rápida de escritorio para validar transparencia algorítmica
    palabra1 = "gato"
    palabra2 = "pato"
    
    distancia, matriz_resultante = levenshtein_distance(palabra1, palabra2)
    
    print(f"Distancia de Levenshtein entre '{palabra1}' y '{palabra2}': {distancia}")
    print("Matriz resultante:")
    for fila in matriz_resultante:
        print(fila)

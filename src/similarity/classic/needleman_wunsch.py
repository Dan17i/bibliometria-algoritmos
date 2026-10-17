"""
Módulo de Alineamiento Global Needleman-Wunsch
-----------------------------------------------
Implementación explícita del algoritmo clásico de Needleman-Wunsch para el cálculo
del alineamiento global de secuencias textuales mediante Programación Dinámica.
"""


def needleman_wunsch(str1: str, str2: str, match_score: int = 1, mismatch_penalty: int = -1, gap_penalty: int = -1) -> tuple[int, list[list[int]]]:
    """
    Calcula el puntaje de similitud (alineamiento global) entre dos secuencias de texto
    utilizando el algoritmo clásico de Needleman-Wunsch.

    A diferencia de la distancia de Levenshtein (que minimiza un costo de edición),
    Needleman-Wunsch busca maximizar un puntaje de similitud mediante la asignación de
    pesos a las coincidencias (matches), discrepancias (mismatches) y huecos (gaps).

    Complejidad Temporal: O(n * m) donde n y m son las longitudes de las cadenas.
    Complejidad Espacial: O(n * m) para almacenar la matriz de Programación Dinámica.

    Args:
        str1 (str): Primera cadena a alinear.
        str2 (str): Segunda cadena a alinear.
        match_score (int): Puntaje sumado cuando dos caracteres coinciden (por defecto 1).
        mismatch_penalty (int): Penalización restada cuando dos caracteres no coinciden (por defecto -1).
        gap_penalty (int): Penalización restada al insertar o eliminar un caracter (por defecto -1).

    Returns:
        tuple[int, list[list[int]]]:
            - El puntaje máximo de alineamiento global (int).
            - La matriz resultante de Programación Dinámica (list[list[int]]), la cual 
              sirve para realizar la demostración matemática en el documento final.
    """
    m = len(str1)
    n = len(str2)

    # 1. Inicialización explícita de la matriz de puntajes (score matrix)
    # Tamaño (m + 1) x (n + 1) inicializada en 0.
    matriz = [[0 for _ in range(n + 1)] for _ in range(m + 1)]

    # 2. Casos base: Llenado de primera fila y primera columna con penalizaciones acumuladas de gap.
    for i in range(m + 1):
        matriz[i][0] = i * gap_penalty
    for j in range(n + 1):
        matriz[0][j] = j * gap_penalty

    # 3. Llenado dinámico de la matriz usando la relación de recurrencia:
    # F(i,j) = max(
    #    F(i-1, j-1) + S(x_i, y_j),  # Alineamiento (Match / Mismatch)
    #    F(i-1, j) + d,              # Gap en la segunda cadena
    #    F(i, j-1) + d               # Gap en la primera cadena
    # )
    for i in range(1, m + 1):
        char1 = str1[i - 1]
        fila_anterior = matriz[i - 1]
        fila_actual = matriz[i]
        for j in range(1, n + 1):
            char2 = str2[j - 1]
            score_actual = match_score if char1 == char2 else mismatch_penalty

            match_mismatch = fila_anterior[j - 1] + score_actual
            gap_arriba = fila_anterior[j] + gap_penalty
            gap_izq = fila_actual[j - 1] + gap_penalty

            fila_actual[j] = max(match_mismatch, gap_arriba, gap_izq)

    # El puntaje óptimo de alineamiento se encuentra en la esquina inferior derecha
    puntaje_final = matriz[m][n]

    return puntaje_final, matriz


def normalized_needleman_wunsch(str1: str, str2: str, match_score: int = 1, mismatch_penalty: int = -1, gap_penalty: int = -1) -> float:
    """
    Retorna la similitud normalizada en el rango [0, 1] basada en Needleman-Wunsch.
    
    Con L = max(m, n), el puntaje queda acotado entre L * gap_penalty (todo
    falla) y L * match_score (cadenas idénticas). La normalización mapea
    linealmente ese rango a [0, 1]:
        sim = (score - L * gap) / (L * match - L * gap)
    La cota inferior vale con los pesos por defecto (mismatch >= gap); con otros
    pesos el resultado se recorta a [0, 1].
    """
    score, _ = needleman_wunsch(str1, str2, match_score, mismatch_penalty, gap_penalty)
    max_score = max(len(str1), len(str2)) * match_score
    if max_score == 0:
        return 1.0
    # Mapeo lineal relativo a un score mínimo teórico
    # min_score = max(len(str1), len(str2)) * min(mismatch_penalty, gap_penalty)
    min_score = max(len(str1), len(str2)) * gap_penalty
    norm = (score - min_score) / (max_score - min_score)
    return max(0.0, min(1.0, norm))


if __name__ == "__main__":
    seq1 = "ATCG"
    seq2 = "ACG"
    puntaje, matriz_resultante = needleman_wunsch(seq1, seq2)
    print(f"Puntaje de alineamiento Needleman-Wunsch entre '{seq1}' y '{seq2}': {puntaje}")
    print(f"Similitud normalizada: {normalized_needleman_wunsch(seq1, seq2):.4f}")


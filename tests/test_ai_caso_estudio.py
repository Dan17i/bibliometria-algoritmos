"""Pruebas sin descargas: stopwords, tabla comparativa y clásicos de Samuel importados."""
import math

import pytest

from src.similarity.ai.caso_estudio import (
    ai_scores,
    classic_scores,
    format_table,
    operaciones_levenshtein,
    operaciones_needleman,
    traza_clasicos,
)
from src.similarity.ai.embeddings import average_word_vectors
from src.similarity.ai.stopwords import STOPWORDS, remove_stopwords

VECTORES = {"gato": [1.0, 0.0], "perro": [0.0, 1.0], "the": [5.0, 5.0], "of": [5.0, 5.0]}


def test_lista_de_stopwords():
    assert {"the", "of", "and", "is", "to"} <= STOPWORDS
    # palabras de contenido y negaciones no se eliminan
    assert not ({"ai", "learning", "review", "not", "no"} & STOPWORDS)
    assert all(w == w.lower() for w in STOPWORDS)


def test_remove_stopwords_conserva_orden_y_repeticiones():
    assert remove_stopwords(["the", "ai", "of", "ai", "review"]) == ["ai", "ai", "review"]
    assert remove_stopwords([]) == []


def test_promedio_sin_stopwords():
    # con stopwords: gato, the, of -> ((1+5+5)/3, (0+5+5)/3); sin ellas: solo gato
    con = average_word_vectors("gato the of", VECTORES)
    sin = average_word_vectors("gato the of", VECTORES, stopwords=STOPWORDS)
    assert con["vector"] == pytest.approx([11 / 3, 10 / 3])
    assert sin["vector"] == [1.0, 0.0]
    assert sin["removed_stopwords"] == ["the", "of"]
    assert sin["tokens"] == ["gato"]
    assert con["removed_stopwords"] == []


def test_stopwords_en_partes_de_palabras_compuestas():
    # "ai's" -> partes "ai" y "s"; "s" es stopword y no debe entrar al promedio
    vec = {"ai": [1.0, 0.0], "s": [9.0, 9.0]}
    sin = average_word_vectors("AI's", vec, stopwords=STOPWORDS)
    assert sin["vector"] == [1.0, 0.0]
    assert average_word_vectors("AI's", vec)["vector"] == [5.0, 4.5]


def test_solo_stopwords_lanza_error():
    with pytest.raises(ValueError):
        average_word_vectors("the of", VECTORES, stopwords=STOPWORDS)


def test_clasicos_de_samuel_sobre_un_par_pequeno():
    filas = classic_scores("the cat sat", "the cat ran")
    por_nombre = {f["algoritmo"]: f for f in filas}

    lev = por_nombre["Levenshtein (caracteres)"]
    # "sat" -> "ran": s->r y t->n (la 'a' coincide) = 2 sustituciones sobre 11 caracteres
    assert lev["valor"] == "distancia 2 de 11"
    assert lev["similitud"] == pytest.approx(1 - 2 / 11)

    jac = por_nombre["Jaccard (palabras)"]
    # {the, cat, sat} y {the, cat, ran}: 2 / 4
    assert jac["similitud"] == 0.5
    assert jac["valor"] == "|∩| = 2, |∪| = 4"

    # coseno TF: vocabulario {cat, ran, sat, the}; (1,0,1,1)·(1,1,0,1)=2 ; 2/(3)
    assert por_nombre["Coseno TF (frecuencias)"]["similitud"] == pytest.approx(2 / 3)


def test_needleman_wunsch_y_tfidf_sobre_un_par_pequeno():
    corpus = ["the cat sat", "the cat ran", "a dog barks loudly"]
    filas = classic_scores("the cat sat", "the cat ran", corpus)
    por_nombre = {f["algoritmo"]: f for f in filas}

    nw = por_nombre["Needleman-Wunsch"]
    # 11 caracteres, solo "sat" vs "ran" difiere en 2: 9 aciertos y 2 fallos = 9 - 2 = 7
    assert nw["valor"] == "puntaje 7 (L = 11)"
    assert nw["similitud"] == pytest.approx((7 + 11) / 22)

    # idf: the, cat -> ln(3/2); sat, ran -> ln(3). Pesos A=(cat,ran,sat,the)=(a,0,b,a), B=(a,b,0,a)
    a, b = math.log(1.5), math.log(3)
    esperado = (2 * a * a) / (2 * a * a + b * b)
    assert por_nombre["Coseno TF-IDF"]["similitud"] == pytest.approx(esperado)
    # el IDF baja el coseno respecto al TF (2/3) porque pondera menos lo común
    assert por_nombre["Coseno TF-IDF"]["similitud"] < por_nombre["Coseno TF (frecuencias)"]["similitud"]
    assert all(f["similitud"] is not None for f in filas)
    assert "—" not in format_table(filas)


def test_operaciones_coinciden_con_el_resultado():
    from src.similarity.classic.levenshtein import levenshtein_distance
    from src.similarity.classic.needleman_wunsch import needleman_wunsch

    distancia, matriz = levenshtein_distance("kitten", "sitting")
    lev = operaciones_levenshtein("kitten", "sitting", matriz)
    assert lev["sustituciones"] + lev["inserciones"] + lev["eliminaciones"] == distancia == 3

    puntaje, matriz = needleman_wunsch("ATCG", "ACG")
    nw = operaciones_needleman("ATCG", "ACG", matriz)
    assert nw["coincidencias"] - nw["discrepancias"] - nw["huecos"] == puntaje == 2


def test_traza_clasicos_es_coherente():
    corpus = ["the cat sat on the mat", "the dog ran far away", "a bird flew over it"]
    t = traza_clasicos(corpus[0], corpus[1], corpus, lado=4)
    assert t["celdas"] == 23 * 21
    assert t["levenshtein"]["esquina_final"][-1][-1] == t["levenshtein"]["distancia"]
    assert t["needleman_wunsch"]["esquina_final"][-1][-1] == t["needleman_wunsch"]["puntaje"]
    assert t["jaccard"]["tam_interseccion"] == len(t["jaccard"]["interseccion"]) == 1
    # "the" está en 2 de 3 documentos y aparece 2 veces en el primero y 1 en el segundo
    primero = t["coseno_tfidf"]["terminos_que_mas_aportan"][0]
    assert primero["termino"] == "the" and primero["tf_a"] == 2 and primero["tf_b"] == 1


def test_ai_scores_usa_metrics():
    par = ([1.0, 0.0], [0.0, 1.0])
    iguales = ([1.0, 2.0], [2.0, 4.0])
    filas = ai_scores(par, iguales, par)
    assert [f["nota"] for f in filas] == ["con stopwords", "sin stopwords", "384 dimensiones"]
    assert filas[0]["similitud"] == 0.0
    assert filas[1]["similitud"] == pytest.approx(1.0)
    assert "90.0°" in filas[0]["valor"] and "euclidiana 1.414" in filas[0]["valor"]

"""Pruebas sin descargas: stopwords, tabla comparativa y clásicos de Samuel importados."""
import pytest

from src.similarity.ai.caso_estudio import PENDIENTE, ai_scores, classic_scores, format_table
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


def test_pendientes_de_samuel_marcados_en_la_tabla():
    filas = classic_scores("a b", "a c")
    por_nombre = {f["algoritmo"]: f for f in filas}
    for nombre in ("Needleman-Wunsch", "Coseno TF-IDF"):
        assert por_nombre[nombre]["valor"] == PENDIENTE
        assert por_nombre[nombre]["similitud"] is None
    tabla = format_table(filas)
    assert tabla.count(PENDIENTE) == 2
    assert "—" in tabla


def test_ai_scores_usa_metrics():
    par = ([1.0, 0.0], [0.0, 1.0])
    iguales = ([1.0, 2.0], [2.0, 4.0])
    filas = ai_scores(par, iguales, par)
    assert [f["nota"] for f in filas] == ["con stopwords", "sin stopwords", "384 dimensiones"]
    assert filas[0]["similitud"] == 0.0
    assert filas[1]["similitud"] == pytest.approx(1.0)
    assert "90.0°" in filas[0]["valor"] and "euclidiana 1.414" in filas[0]["valor"]

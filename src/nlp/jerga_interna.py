import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nlp.stop_words import STOP_WORDS


TOKEN_PATTERN = re.compile(r"(?u)\b\w{3,}\b")


def _tokenizar(textos):
    return [" ".join(TOKEN_PATTERN.findall(t.lower())) for t in textos]


def detectar_jerga_interna(
    df_mensajes: pd.DataFrame,
    col_texto: str = "texto",
    col_usuario: str = "id_usuario",
    ngram_range=(1, 3),
    max_features=500,
    umbral_exclusividad: float = 0.15,
) -> dict:
    """
    Identifica terminos de jerga interna del grupo mediante TF-IDF.

    Calcula TF-IDF sobre todos los mensajes; los terminos con mayor
    puntuacion son candidatos a jerga del grupo (vocabulario compartido
    que no aparece en un corpus externo de referencia).

    Retorna:
        - terminos_jerga: lista de terminos candidatos
        - tfidf_matrix: matriz TF-IDF completa
        - vocabulary: dict {termino: indice}
        - df_terminos: DataFrame con termino y peso medio
    """
    textos_limpios = _tokenizar(df_mensajes[col_texto].fillna("").tolist())

    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=2,
        max_df=0.85,
        stop_words=STOP_WORDS,
    )
    tfidf_matrix = vectorizer.fit_transform(textos_limpios)
    vocabulary = vectorizer.vocabulary_

    # Peso medio de cada termino en el corpus
    pesos_medios = np.array(tfidf_matrix.mean(axis=0)).flatten()
    terminos = [None] * len(vocabulary)
    for term, idx in vocabulary.items():
        terminos[idx] = term

    df_terminos = pd.DataFrame({"termino": terminos, "peso_tfidf": pesos_medios})
    df_terminos = df_terminos.sort_values("peso_tfidf", ascending=False)

    # Seleccionar candidatos a jerga: terminos con peso alto
    # (exclusividad: aparecen mucho en este grupo vs. lo esperable)
    # Usamos el peso TF-IDF como proxy de especificidad
    umbral_peso = df_terminos["peso_tfidf"].quantile(1 - umbral_exclusividad)
    terminos_jerga = df_terminos[df_terminos["peso_tfidf"] >= umbral_peso]

    return {
        "terminos_jerga": terminos_jerga["termino"].tolist(),
        "tfidf_matrix": tfidf_matrix,
        "vocabulary": vocabulary,
        "df_terminos": df_terminos,
        "vectorizer": vectorizer,
    }


def obtener_terminos_grupo(
    df_mensajes: pd.DataFrame,
    col_texto: str = "texto",
    min_frecuencia: int = 3,
) -> dict:
    """
    Extrae terminos frecuentes del grupo como vocabulario compartido.
    Analisis complementario: n-gramas de frecuencia.
    """
    textos = df_mensajes[col_texto].fillna("").tolist()
    tokens = []
    for t in textos:
        tokens.extend(TOKEN_PATTERN.findall(t.lower()))

    frecuencias = Counter(tokens)
    terminos_frecuentes = {
        word: count
        for word, count in frecuencias.most_common(100)
        if count >= min_frecuencia
    }

    # Bigramas
    bigramas = Counter()
    for t in textos:
        pts = TOKEN_PATTERN.findall(t.lower())
        for i in range(len(pts) - 1):
            bigramas[f"{pts[i]} {pts[i+1]}"] += 1

    bigramas_frecuentes = {
        bg: count
        for bg, count in bigramas.most_common(50)
        if count >= min_frecuencia
    }

    return {
        "unigramas": terminos_frecuentes,
        "bigramas": bigramas_frecuentes,
        "total_palabras": sum(frecuencias.values()),
        "vocabulario_unico": len(frecuencias),
    }

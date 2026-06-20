import re

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nlp.stop_words import STOP_WORDS


TOKEN_PATTERN = re.compile(r"(?u)\b\w{3,}\b")


def _tokenizar(textos):
    return [" ".join(TOKEN_PATTERN.findall(t.lower())) for t in textos]


def modelar_topicos(
    df_mensajes: pd.DataFrame,
    col_texto: str = "texto",
    n_topicos: int = 5,
    max_features: int = 500,
    min_df: int = 2,
    max_df: float = 0.85,
) -> dict:
    """
    Modela topicos latentes en los mensajes usando LDA.

    Retorna:
        - model: modelo LDA entrenado
        - vectorizer: CountVectorizer
        - df_topicos: palabras principales por topico
        - matriz_topicos: distribucion de topicos por mensaje
        - topico_dominante: topico principal de cada mensaje
    """
    textos_limpios = _tokenizar(df_mensajes[col_texto].fillna("").tolist())

    vectorizer = CountVectorizer(
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=STOP_WORDS,
    )
    X = vectorizer.fit_transform(textos_limpios)
    vocabulary = vectorizer.get_feature_names_out()

    model = LatentDirichletAllocation(
        n_components=n_topicos,
        random_state=42,
        max_iter=100,
        learning_method="batch",
    )
    topicos_dist = model.fit_transform(X)
    topico_dominante = topicos_dist.argmax(axis=1)

    # Extraer palabras principales por topico
    n_words = 10
    palabras_topicos = []
    for topic_idx, topic in enumerate(model.components_):
        top_words_idx = topic.argsort()[:-n_words - 1:-1]
        top_words = [vocabulary[i] for i in top_words_idx]
        top_weights = [round(topic[i], 4) for i in top_words_idx]
        palabras_topicos.append({
            "topico": topic_idx,
            "palabras": top_words,
            "pesos": top_weights,
        })

    df_topicos = pd.DataFrame(palabras_topicos)

    return {
        "model": model,
        "vectorizer": vectorizer,
        "df_topicos": df_topicos,
        "matriz_topicos": topicos_dist,
        "topico_dominante": topico_dominante,
        "vocabulary": vocabulary,
    }


def extraer_topicos_espiral(
    df_mensajes: pd.DataFrame,
    df_perfiles: pd.DataFrame,
    col_texto: str = "texto",
    col_usuario: str = "id_usuario",
    col_perfil: str = "perfil",
    n_topicos: int = 5,
) -> dict:
    """
    Analiza distribucion de topicos por perfil para detectar
    espiral del silencio: topicos que los perfiles perifericos
    evitan discutir.

    Hipotesis: si un topico es dominante en el nucleo pero
    esta ausente en perifericos, puede indicar auto-censura
    (espiral del silencio).

    Retorna:
        - distribucion_topicos_por_perfil: matriz perfil x topico
        - topicos_evitados: topicos con menor presencia en perifericos
        - df_topicos_mensajes: DataFrame con topico por mensaje
    """
    resultado = modelar_topicos(df_mensajes, col_texto, n_topicos)
    topicos_dist = resultado["matriz_topicos"]
    topico_dominante = resultado["topico_dominante"]
    df_topicos = resultado["df_topicos"]

    perfil_map = df_perfiles.set_index(col_usuario)[col_perfil].to_dict()

    # Asignar topico a cada mensaje
    rows = []
    for i, (_, msg) in enumerate(df_mensajes.iterrows()):
        usuario = msg[col_usuario]
        rows.append({
            col_usuario: usuario,
            "perfil": perfil_map.get(usuario, "desconocido"),
            "topico_dominante": int(topico_dominante[i]),
            "confianza_topico": round(float(topicos_dist[i].max()), 4),
            **{f"peso_topico_{t}": round(float(topicos_dist[i][t]), 4)
               for t in range(n_topicos)},
        })

    df_topicos_mensajes = pd.DataFrame(rows)

    # Distribucion de topicos por perfil
    distribucion = (
        df_topicos_mensajes
        .groupby(["perfil", "topico_dominante"])
        .size()
        .unstack(fill_value=0)
    )

    # Normalizar por fila (perfil)
    distribucion_pct = distribucion.div(distribucion.sum(axis=1), axis=0)

    # Detectar topicos evitados: diferencia nucleo vs periferico
    topicos_evitados = []
    for t in range(n_topicos):
        if t in distribucion_pct.columns:
            en_nucleo = distribucion_pct.loc["nucleo", t] if "nucleo" in distribucion_pct.index else 0
            en_periferico = distribucion_pct.loc["periferico", t] if "periferico" in distribucion_pct.index else 0
            diferencia = en_nucleo - en_periferico
            if diferencia > 0.1:
                palabras = df_topicos[df_topicos["topico"] == t]["palabras"].values[0]
                topicos_evitados.append({
                    "topico": t,
                    "palabras_clave": palabras[:5],
                    "presencia_nucleo": round(en_nucleo, 4),
                    "presencia_periferico": round(en_periferico, 4),
                    "diferencia": round(diferencia, 4),
                    "posible_espiral_silencio": diferencia > 0.2,
                })

    return {
        "distribucion_topicos_por_perfil": distribucion_pct,
        "topicos_evitados": topicos_evitados,
        "df_topicos_mensajes": df_topicos_mensajes,
        "resultado_lda": resultado,
    }

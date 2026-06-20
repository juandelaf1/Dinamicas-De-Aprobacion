import re

import numpy as np
import pandas as pd


# Lexico de sentimiento en espanol e ingles (cubrimiento bilingual)
# Palabras positivas y negativas de uso general
LEXICO_POSITIVO = {
    "good", "great", "awesome", "love", "beautiful", "amazing", "wonderful",
    "fantastic", "excellent", "happy", "glad", "nice", "perfect", "best",
    "fun", "cool", "cute", "adorable", "brilliant", "gorgeous", "sweet",
    "lovely", "wow", "beautiful", "incredible", "magnificent", "superb",
    "bien", "bueno", "genial", "excelente", "hermoso", "maravilloso",
    "fantastico", "feliz", "alegre", "bonito", "perfecto", "mejor",
    "divertido", "encantador", "impresionante", "precioso",
}

LEXICO_NEGATIVO = {
    "bad", "terrible", "awful", "horrible", "hate", "ugly", "worst",
    "sad", "angry", "angry", "disgusting", "stupid", "dumb", "annoying",
    "boring", "ugly", "gross", "cringe", "trash", "garbage", "toxic",
    "malo", "terrible", "horrible", "odio", "feo", "peor", "triste",
    "enojado", "aburrido", "estupido", "asco",
}

INTENSIFICADORES = {
    "very", "so", "really", "extremely", "super", "ultra", "absolutely",
    "completely", "totally", "muy", "tan", "realmente", "extremadamente",
    "super", "absolutamente", "totalmente",
}


def _tokenizar(texto: str) -> list:
    return re.findall(r"(?u)\b\w+\b", texto.lower())


def puntuar_sentimiento(texto: str) -> dict:
    """
    Puntua el sentimiento de un texto usando lexico载gado.
    Retorna {positivo, negativo, neto, intensidad, polaridad}.
    """
    tokens = _tokenizar(texto)
    if not tokens:
        return {"positivo": 0, "negativo": 0, "neto": 0.0, "intensidad": 0.0, "polaridad": "neutro"}

    pos = sum(1 for t in tokens if t in LEXICO_POSITIVO)
    neg = sum(1 for t in tokens if t in LEXICO_NEGATIVO)
    intensidad = sum(1 for t in tokens if t in INTENSIFICADORES)

    neto = (pos - neg) / len(tokens)

    # Polaridad categorica
    if neto > 0.05:
        polaridad = "positivo"
    elif neto < -0.05:
        polaridad = "negativo"
    else:
        polaridad = "neutro"

    return {
        "positivo": pos,
        "negativo": neg,
        "neto": round(neto, 4),
        "intensidad": intensidad,
        "polaridad": polaridad,
    }


def analizar_sentimiento_perfil(
    df_mensajes: pd.DataFrame,
    df_perfiles: pd.DataFrame,
    col_texto: str = "texto",
    col_usuario: str = "id_usuario",
    col_perfil: str = "perfil",
) -> pd.DataFrame:
    """
    Calcula metricas de sentimiento agregadas por perfil.

    Retorna DataFrame con:
        - sentimiento_medio: neto promedio
        - proporcion_positivos: fraccion de mensajes positivos
        - proporcion_negativos: fraccion de mensajes negativos
        - intensidad_media: intensificadores promedio por mensaje
        - volatilidad_emocional: desviacion estandar del sentimiento neto
    """
    resultados = []
    for _, msg in df_mensajes.iterrows():
        score = puntuar_sentimiento(str(msg[col_texto]))
        resultados.append({
            col_usuario: msg[col_usuario],
            "sentimiento_neto": score["neto"],
            "polaridad": score["polaridad"],
            "intensidad": score["intensidad"],
        })

    df_sent = pd.DataFrame(resultados)

    # Agregar perfil
    perfil_map = df_perfiles.set_index(col_usuario)[col_perfil].to_dict()
    df_sent["perfil"] = df_sent[col_usuario].map(perfil_map)

    # Agregacion por perfil
    agg = df_sent.groupby("perfil").agg(
        sentimiento_medio=("sentimiento_neto", "mean"),
        volatilidad_emocional=("sentimiento_neto", "std"),
        intensidad_media=("intensidad", "mean"),
        mensajes_totales=("sentimiento_neto", "count"),
    ).reset_index()

    # Proporciones de polaridad
    for pol in ["positivo", "negativo", "neutro"]:
        prop = (
            df_sent[df_sent["polaridad"] == pol]
            .groupby("perfil")
            .size()
            .div(df_sent.groupby("perfil").size())
            .fillna(0)
        )
        agg[f"proporcion_{pol}s"] = agg["perfil"].map(prop).fillna(0)

    return agg, df_sent

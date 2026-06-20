import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


TOKEN_PATTERN = re.compile(r"(?u)\b\w+\b")


def _tokenizar(texto: str) -> list:
    return [t.lower() for t in TOKEN_PATTERN.findall(str(texto))]


def _extraer_estilo(texto: str) -> dict:
    """
    Extrae rasgos estilisticos de un texto:
    - longitud media de palabra
    - proporcion de mayusculas
    - repeticion de puntuacion (!!!, ???)
    - ratio de emojis aproximado (caracteres no-ASCII)
    - densidad de palabras raras (>7 caracteres)
    """
    tokens = _tokenizar(texto)
    if not tokens:
        return {
            "longitud_palabra_media": 0,
            "prop_mayusculas": 0,
            "exclamaciones_multiple": 0,
            "interrogantes_multiple": 0,
            "prop_no_ascii": 0,
            "palabras_largas": 0,
            "num_tokens": 0,
        }

    num_tokens = len(tokens)
    letras_totales = sum(len(t) for t in tokens)
    mayusculas = sum(1 for t in tokens if t.isupper() and len(t) > 1)
    palabras_largas = sum(1 for t in tokens if len(t) > 7)

    # Puntuacion repetida
    exc = len(re.findall(r"!{2,}", texto))
    ints = len(re.findall(r"\?{2,}", texto))

    # Caracteres no-ASCII (proxy de emojis/unicode)
    no_ascii = sum(1 for c in texto if ord(c) > 127)

    return {
        "longitud_palabra_media": round(letras_totales / num_tokens, 2) if num_tokens > 0 else 0,
        "prop_mayusculas": round(mayusculas / num_tokens, 4) if num_tokens > 0 else 0,
        "exclamaciones_multiple": exc,
        "interrogantes_multiple": ints,
        "prop_no_ascii": round(no_ascii / len(texto), 4) if len(texto) > 0 else 0,
        "palabras_largas": palabras_largas,
        "num_tokens": num_tokens,
    }


def comparar_estilo_imitativo(
    df_mensajes: pd.DataFrame,
    df_perfiles: pd.DataFrame = None,
    col_texto: str = "texto",
    col_usuario: str = "id_usuario",
    col_perfil: str = "perfil",
) -> dict:
    """
    Analiza similitud estilistica entre usuarios.

    Detecta posibles casos de lenguaje imitativo (usuarios cuyo
    estilo linguistico es anormalmente similar al del nucleo).

    Estrategia:
    1. Extraer rasgos estilisticos por usuario
    2. Calcular distancia euclidiana entre perfiles estilisticos
    3. Identificar pares con similitud alta (posible mimetismo)

    Retorna:
        - df_estilo: rasgos estilisticos por usuario
        - pares_similares: pares de usuarios con estilo cercano
        - imitadores_sospechosos: usuarios con estilo similar al nucleo
    """
    # Rasgos estilisticos por mensaje
    rasgos = []
    for _, msg in df_mensajes.iterrows():
        estilo = _extraer_estilo(str(msg[col_texto]))
        estilo[col_usuario] = msg[col_usuario]
        estilo["id_mensaje"] = msg.get("id_mensaje", "")
        rasgos.append(estilo)

    df_estilo = pd.DataFrame(rasgos)

    # Agregar rasgos por usuario (media)
    cols_estilo = [
        "longitud_palabra_media", "prop_mayusculas",
        "exclamaciones_multiple", "interrogantes_multiple",
        "prop_no_ascii", "palabras_largas", "num_tokens",
    ]

    df_usuario_estilo = df_estilo.groupby(col_usuario)[cols_estilo].mean().reset_index()

    # Si hay perfiles, agregarlos
    if df_perfiles is not None and col_perfil in df_perfiles.columns:
        perfil_map = df_perfiles.set_index(col_usuario)[col_perfil].to_dict()
        df_usuario_estilo["perfil"] = df_usuario_estilo[col_usuario].map(perfil_map)

    # Matriz de similitud coseno entre vectores de estilo
    X = df_usuario_estilo[cols_estilo].fillna(0).values
    if len(X) < 2:
        return {
            "df_estilo": df_usuario_estilo,
            "pares_similares": [],
            "imitadores_sospechosos": [],
        }

    sim_matrix = cosine_similarity(X)

    # Encontrar pares con similitud alta (>0.9)
    pares = []
    n = len(sim_matrix)
    for i in range(n):
        for j in range(i + 1, n):
            sim = sim_matrix[i, j]
            if sim > 0.9:
                pares.append({
                    "usuario_a": df_usuario_estilo.iloc[i][col_usuario],
                    "usuario_b": df_usuario_estilo.iloc[j][col_usuario],
                    "similitud_estilo": round(sim, 4),
                    "perfil_a": df_usuario_estilo.iloc[i].get("perfil", "desconocido"),
                    "perfil_b": df_usuario_estilo.iloc[j].get("perfil", "desconocido"),
                })

    pares.sort(key=lambda x: x["similitud_estilo"], reverse=True)

    # Identificar imitadores: perifericos con estilo similar a nucleo
    imitadores = []
    if df_perfiles is not None:
        nucleo_ids = set(
            df_perfiles[df_perfiles[col_perfil] == "nucleo"][col_usuario].tolist()
        )
        for p in pares:
            a_en_nucleo = p["usuario_a"] in nucleo_ids
            b_en_nucleo = p["usuario_b"] in nucleo_ids
            if a_en_nucleo ^ b_en_nucleo:
                imitador = p["usuario_b"] if a_en_nucleo else p["usuario_a"]
                imitadores.append({
                    "usuario_imitador": imitador,
                    "modelo_a_seguir": p["usuario_a"] if a_en_nucleo else p["usuario_b"],
                    "similitud": p["similitud_estilo"],
                })

    return {
        "df_estilo": df_usuario_estilo,
        "pares_similares": pares,
        "imitadores_sospechosos": imitadores,
    }

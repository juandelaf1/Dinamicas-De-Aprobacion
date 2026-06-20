import re

import pandas as pd


PATRONES_BORRADO = [
    re.compile(r"\[deleted\]", re.IGNORECASE),
    re.compile(r"\[removed\]", re.IGNORECASE),
    re.compile(r"\(mensaje eliminado\)", re.IGNORECASE),
    re.compile(r"\(deleted\)", re.IGNORECASE),
    re.compile(r"^\.$"),  # mensaje de solo un punto
    re.compile(r"^\s*$"),  # mensaje vacio
    re.compile(r"\[this post has been deleted\]", re.IGNORECASE),
    re.compile(r"contenido no disponible", re.IGNORECASE),
    re.compile(r"contenido eliminado", re.IGNORECASE),
    re.compile(r"unavailable", re.IGNORECASE),
    re.compile(r"mensaje no disponible", re.IGNORECASE),
]


def detectar_borrados(
    df_mensajes: pd.DataFrame,
    col_texto: str = "texto",
    col_usuario: str = "id_usuario",
) -> pd.DataFrame:
    """
    Detecta mensajes que parecen haber sido borrados o eliminados.

    Busca patrones textuales tipicos de mensajes eliminados en
    plataformas de chat / redes sociales.

    Retorna el DataFrame original con columnas adicionales:
        - es_borrado: bool
        - patron_borrado: patron que coincidio (o None)
        - motivo_borrado: categoria del borrado
    """
    df = df_mensajes.copy()

    def _check_borrado(texto):
        texto_str = str(texto)
        for pat in PATRONES_BORRADO:
            if pat.search(texto_str):
                return True, pat.pattern
        return False, None

    resultados = df[col_texto].apply(_check_borrado)
    df["es_borrado"] = resultados.apply(lambda x: x[0])
    df["patron_borrado"] = resultados.apply(lambda x: x[1])

    def _categorizar(patron, es_borrado):
        if not es_borrado:
            return None
        if "deleted" in str(patron).lower() or "eliminado" in str(patron).lower():
            return "eliminado_por_usuario"
        if "removed" in str(patron).lower() or "no disponible" in str(patron).lower():
            return "removido_por_moderacion"
        if patron == r"^\s*$" or patron == r"^\.$":
            return "vacio_o_placeholder"
        return "otro"

    df["motivo_borrado"] = df.apply(
        lambda r: _categorizar(r["patron_borrado"], r["es_borrado"]),
        axis=1,
    )

    return df


def resumen_borrados(df_borrados: pd.DataFrame, df_perfiles: pd.DataFrame = None) -> dict:
    """Genera resumen de mensajes borrados."""
    total = len(df_borrados)
    n_borrados = df_borrados["es_borrado"].sum()

    res = {
        "total_mensajes": total,
        "mensajes_borrados": int(n_borrados),
        "porcentaje_borrado": round(n_borrados / total * 100, 2) if total > 0 else 0,
        "por_motivo": df_borrados[df_borrados["es_borrado"]]["motivo_borrado"].value_counts().to_dict(),
    }

    if df_perfiles is not None and "id_usuario" in df_borrados.columns:
        merged = df_borrados[df_borrados["es_borrado"]].merge(
            df_perfiles[["id_usuario", "perfil"]], on="id_usuario", how="left"
        )
        res["por_perfil"] = merged["perfil"].value_counts().to_dict()

    return res

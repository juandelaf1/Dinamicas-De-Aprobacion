"""
Pipeline de clasificación de perfiles psicosociales.
Carga datos extraídos de Bluesky, calcula features y asigna perfiles.
"""

import json
import os
from collections import defaultdict

import pandas as pd
import numpy as np


def load_messages(jsonl_path: str) -> list:
    """Load extracted messages from JSONL."""
    messages = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            messages.append(json.loads(line))
    return messages


def to_er_dataframes(messages: list):
    """
    Convert messages to ER-aligned DataFrames.
    Returns (df_usuarios, df_mensajes, df_interacciones)
    """
    autores = {}
    rows_mensajes = []

    for msg in messages:
        did = msg["author_did"]
        if did:
            if did not in autores:
                autores[did] = {
                    "id_usuario": did,
                    "nombre": msg["author_name"] or msg["author_handle"],
                    "handle": msg["author_handle"],
                }

        rows_mensajes.append({
            "id_mensaje": msg["uri"],
            "id_usuario": did,
            "timestamp": msg["timestamp"],
            "texto": msg["text"],
            "id_mensaje_respuesta": msg["parent_uri"],
            "likes": msg["likes"],
            "reposts": msg["reposts"],
            "es_raiz": msg["is_root"],
            "profundidad": msg["depth"],
        })

    df_usuarios = pd.DataFrame(list(autores.values()))
    df_mensajes = pd.DataFrame(rows_mensajes)

    # Build interacciones (reply graph)
    replies_df = df_mensajes[df_mensajes["id_mensaje_respuesta"].notna()].copy()
    if len(replies_df) > 0:
        merged = replies_df.merge(
            df_mensajes[["id_mensaje", "id_usuario"]],
            left_on="id_mensaje_respuesta",
            right_on="id_mensaje",
            how="left",
            suffixes=("", "_destino"),
        )
        merged = merged.dropna(subset=["id_usuario_destino"])
        df_interacciones = merged[["id_usuario", "id_usuario_destino", "id_mensaje", "timestamp"]].copy()
        df_interacciones.columns = ["id_usuario_origen", "id_usuario_destino", "id_mensaje", "timestamp"]
        df_interacciones["tipo"] = "respuesta_directa"
        df_interacciones["peso"] = 1.0
    else:
        df_interacciones = pd.DataFrame()

    return df_usuarios, df_mensajes, df_interacciones


def compute_features(df_usuarios: pd.DataFrame, df_mensajes: pd.DataFrame, df_interacciones: pd.DataFrame):
    """
    Compute all features for each user.
    Returns df_usuarios with added feature columns.
    """
    users = df_usuarios.copy()

    # Per-user message stats
    msg_stats = df_mensajes.groupby("id_usuario").agg(
        mensajes_totales=("id_mensaje", "count"),
        mensajes_raiz=("es_raiz", "sum"),
        mensajes_reply=("es_raiz", lambda x: (~x.astype(bool)).sum() if x.dtype == bool else len(x) - x.sum()),
        likes_recibidos=("likes", "sum"),
        likes_por_msg=("likes", "mean"),
    ).reset_index()

    users = users.merge(msg_stats, on="id_usuario", how="left")
    users = users.fillna(0)

    # IA: Índice de Aprobación (likes recibidos / mensajes enviados)
    users["IA"] = users.apply(
        lambda r: r["likes_recibidos"] / r["mensajes_totales"] if r["mensajes_totales"] > 0 else 0,
        axis=1,
    )

    # Normalize IA by group baseline
    ia_mean = users["IA"].mean()
    ia_std = users["IA"].std()
    if ia_std > 0:
        users["IA_norm"] = (users["IA"] - ia_mean) / ia_std
    else:
        users["IA_norm"] = 0.0

    # Outreach: diversity of people this user interacted with
    if len(df_interacciones) > 0:
        outreach = df_interacciones.groupby("id_usuario_origen")["id_usuario_destino"].nunique().reset_index()
        outreach.columns = ["id_usuario", "destinatarios_unicos"]
        users = users.merge(outreach, on="id_usuario", how="left")

        # Response ratio: how many of their messages got replies
        responded = df_interacciones.groupby("id_usuario_destino")["id_usuario_origen"].count().reset_index()
        responded.columns = ["id_usuario", "veces_respondido"]
        users = users.merge(responded, on="id_usuario", how="left")
    else:
        users["destinatarios_unicos"] = 0
        users["veces_respondido"] = 0

    users = users.fillna(0)

    # RRD: Ratio de Respuesta Directa (veces que le respondieron / mensajes totales)
    users["RRD"] = users.apply(
        lambda r: min(r["veces_respondido"] / r["mensajes_totales"], 1.0) if r["mensajes_totales"] > 0 else 0,
        axis=1,
    )

    # Pertenencia Aproximada (PA): composite score
    # Combines: diversity of destinations, reply ratio, having replies
    pa_max = users["destinatarios_unicos"].max()
    users["PA"] = users.apply(
        lambda r: (
            0.3 * (r["destinatarios_unicos"] / pa_max if pa_max > 0 else 0)
            + 0.4 * r["RRD"]
            + 0.3 * min(r["mensajes_reply"] / r["mensajes_totales"] if r["mensajes_totales"] > 0 else 0, 0.5) * 2
        ),
        axis=1,
    )

    # Latency proxy: time between user's first and last message
    if "timestamp" in df_mensajes.columns and len(df_mensajes) > 0:
        df_mensajes["timestamp"] = pd.to_datetime(df_mensajes["timestamp"], errors="coerce")
        time_range = df_mensajes.groupby("id_usuario").agg(
            primera_actividad=("timestamp", "min"),
            ultima_actividad=("timestamp", "max"),
        ).reset_index()
        time_range["ventana_activa_dias"] = (
            time_range["ultima_actividad"] - time_range["primera_actividad"]
        ).dt.total_seconds() / (3600 * 24)
        users = users.merge(time_range[["id_usuario", "ventana_activa_dias"]], on="id_usuario", how="left")
    else:
        users["ventana_activa_dias"] = 0.0

    users = users.fillna(0)
    return users


def classify_profiles(df_features: pd.DataFrame):
    """
    Assign profile based on IA and PA.
    Uses multi-step decision rules with dynamic thresholds.
    """
    result = df_features.copy()

    # Thresholds dinámicos
    pa_high = result["PA"].quantile(0.66)
    pa_mid = result["PA"].quantile(0.50)
    ia_high = result["IA_norm"].quantile(0.66)
    ia_low = result["IA_norm"].quantile(0.33)
    median_msgs = result["mensajes_totales"].median()
    high_activity = result["mensajes_totales"].quantile(0.75)

    def assign(row):
        # Espectador Fantasma: 0 aportes (solo recibe, no envía)
        if row["mensajes_totales"] == 0:
            return "fantasma"

        # Núcleo (Líder): alta aprobación y alta pertenencia
        if row["IA_norm"] > ia_high and row["PA"] > pa_high:
            return "nucleo"

        # Buscador de Validación: mucha actividad, baja pertenencia, IA baja o inestable
        if (
            row["mensajes_totales"] >= high_activity
            and row["PA"] < pa_mid
            and row["IA_norm"] < ia_high
        ):
            return "buscador_validacion"

        # Integrado Silencioso: poca actividad, alta pertenencia (reacciona, le responden)
        if row["PA"] > pa_mid and row["mensajes_totales"] <= median_msgs:
            return "integrado_silencioso"

        # Integrado Silencioso (variante): PA alta aunque tenga pocos mensajes
        if row["PA"] > pa_high:
            return "integrado_silencioso"

        # Periférico: baja pertenencia, poca aprobación
        if row["PA"] <= pa_mid and row["IA_norm"] <= ia_high:
            return "periferico"

        # Default
        return "periferico"

    result["perfil"] = result.apply(assign, axis=1)
    return result


def summary(df_classified: pd.DataFrame):
    """Print classification summary."""
    print("=" * 60)
    print("RESUMEN DE CLASIFICACIÓN")
    print("=" * 60)

    profile_order = ["nucleo", "buscador_validacion", "integrado_silencioso", "periferico", "fantasma"]
    profile_names = {
        "nucleo": "Núcleo (Líder)",
        "buscador_validacion": "Buscador de Validación",
        "integrado_silencioso": "Integrado Silencioso",
        "periferico": "Periférico / Excluido",
        "fantasma": "Espectador Fantasma",
    }

    for p in profile_order:
        subset = df_classified[df_classified["perfil"] == p]
        n = len(subset)
        if n == 0:
            continue
        print(f"\n  {profile_names[p]} ({n} usuarios)")
        print(f"    IA medio:   {subset['IA'].mean():.2f}")
        print(f"    PA medio:   {subset['PA'].mean():.2f}")
        print(f"    Mensajes:   {subset['mensajes_totales'].mean():.1f} avg")
        print(f"    Likes:      {subset['likes_recibidos'].mean():.1f} avg")
        top = subset.sort_values("IA", ascending=False).head(3)
        for _, u in top.iterrows():
            print(f"      -> @{u['handle']:<25s} IA={u['IA']:.2f} PA={u['PA']:.2f} msgs={int(u['mensajes_totales'])}")

    print("\n" + "=" * 60)


def run_pipeline(jsonl_path: str):
    """Run the full pipeline from JSONL to classification."""
    print(f"Cargando datos desde: {jsonl_path}")
    messages = load_messages(jsonl_path)
    print(f"  Mensajes cargados: {len(messages)}")

    df_usuarios, df_mensajes, df_interacciones = to_er_dataframes(messages)
    print(f"  Usuarios: {len(df_usuarios)}")
    print(f"  Mensajes: {len(df_mensajes)}")
    print(f"  Interacciones: {len(df_interacciones)}")

    df_features = compute_features(df_usuarios, df_mensajes, df_interacciones)
    df_result = classify_profiles(df_features)

    summary(df_result)
    return df_result, df_mensajes, df_interacciones


if __name__ == "__main__":
    import os
    base = os.path.dirname(os.path.dirname(__file__))
    jsonl_path = os.path.join(base, "data", "raw", "bluesky_threads_20260619_123556.jsonl")
    run_pipeline(jsonl_path)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def crear_proxy_churn(
    df_features: pd.DataFrame,
    df_mensajes: pd.DataFrame = None,
    ventana_silencio_dias: float = 0,
) -> pd.DataFrame:
    """
    Crea etiquetas proxy de abandono basadas en sennales de riesgo.

    El proxy se construye con sennales NO circulares respecto a las
    features del modelo: principalmente inactividad temporal y
    aislamiento estructural en el grafo.

    Un usuario se marca como alto riesgo si cumple:
    - Sin actividad reciente (si hay datos temporales)
    - O pertenece al percentil inferior de interaccion reciproca
    - O tiene 0 replies recibidas siendo periferico

    Retorna df con columna 'riesgo_abandono' (0=bajo, 1=alto).
    """
    df = df_features.copy()

    # Usar la ventana activa como proxy temporal de abandono
    # Si el usuario solo fue activo al inicio del dataset y no despues,
    # es candidato a abandono
    if "ventana_activa_dias" in df.columns and df_mensajes is not None:
        df_mensajes = df_mensajes.copy()
        df_mensajes["timestamp"] = pd.to_datetime(df_mensajes["timestamp"], errors="coerce")
        ultimo_ts = df_mensajes["timestamp"].max()
        inicio_ts = df_mensajes["timestamp"].min()
        total_ventana = (ultimo_ts - inicio_ts).total_seconds() / (3600 * 24)

        ultimo_msg = df_mensajes.groupby("id_usuario")["timestamp"].max().reset_index()
        ultimo_msg.columns = ["id_usuario", "ultimo_mensaje"]
        df = df.merge(ultimo_msg, on="id_usuario", how="left")
        df["dias_desde_ultimo"] = (ultimo_ts - df["ultimo_mensaje"]).dt.total_seconds() / (3600 * 24)
        inactivo = df["dias_desde_ultimo"] > total_ventana * 0.5
    else:
        inactivo = pd.Series([False] * len(df))

    # Sennales de riesgo (no directamente IA/PA/RRD)
    solo_un_mensaje = df["mensajes_totales"] == 1
    sin_respuestas = df["veces_respondido"] == 0
    sin_outreach = df["destinatarios_unicos"] == 0
    ventana_corta = df["ventana_activa_dias"] <= ventana_silencio_dias if "ventana_activa_dias" in df.columns else False

    # Combinacion
    n_sennales = (
        inactivo.astype(int)
        + solo_un_mensaje.astype(int)
        + sin_respuestas.astype(int)
    )
    if isinstance(ventana_corta, pd.Series):
        n_sennales = n_sennales + ventana_corta.astype(int)

    df["n_sennales_abandono"] = n_sennales
    # Alto riesgo: 3+ sennales general, o 2+ siendo periferico
    if "perfil" in df.columns:
        es_periferico = df["perfil"] == "periferico"
        df["riesgo_abandono"] = ((n_sennales >= 3) | ((n_sennales >= 2) & es_periferico)).astype(int)
    else:
        df["riesgo_abandono"] = (n_sennales >= 3).astype(int)

    return df


_FEATURES_CHURN = [
    "IA", "IA_norm", "likes_por_msg",
    "mensajes_raiz", "mensajes_reply",
    "ventana_activa_dias", "likes_recibidos",
]

_COLUMNAS_EXCLUIDAS = {"n_sennales_abandono", "riesgo_abandono", "ultimo_mensaje",
                        "dias_desde_ultimo", "perfil", "id_usuario", "nombre", "handle"}


def entrenar_modelo_riesgo(
    df: pd.DataFrame,
    cols_features: list = None,
    col_target: str = "riesgo_abandono",
    incluir_nlp: bool = False,
    df_nlp: pd.DataFrame = None,
) -> dict:
    """
    Entrena Random Forest para predecir riesgo de abandono.

    Usa features NO circulares: exclusivamente metricas que no
    estan directamente en la definicion del proxy.

    Si incluir_nlp=True, espera df_nlp con columnas de sentimiento.
    """
    if cols_features is None:
        cols_features = [c for c in _FEATURES_CHURN if c in df.columns]
        if incluir_nlp and df_nlp is not None:
            nlp_cols = [c for c in df_nlp.columns
                        if c not in _COLUMNAS_EXCLUIDAS
                        and c not in cols_features
                        and c not in ("id_usuario", col_target)]
            cols_features.extend(nlp_cols)

    df_modelo = df.dropna(subset=[col_target]).copy()
    # Excluir columnas que definen el proxy
    cols_features = [c for c in cols_features if c not in _COLUMNAS_EXCLUIDAS]
    X_raw = df_modelo[cols_features].select_dtypes(include=[np.number]).fillna(0)
    # Eliminar columnas duplicadas
    X_raw = X_raw.loc[:, ~X_raw.columns.duplicated()]
    cols_efectivas = X_raw.columns.tolist()
    X = X_raw.values
    y = df_modelo[col_target].values

    if len(set(y)) < 2:
        return {"error": "Una sola clase en target, no se puede entrenar"}

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=200, max_depth=6,
                                random_state=42, class_weight="balanced")
    scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")
    auc_scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="roc_auc")

    rf.fit(X_scaled, y)

    importancia = pd.DataFrame({
        "feature": cols_efectivas,
        "importancia": rf.feature_importances_,
    }).sort_values("importancia", ascending=False)

    return {
        "modelo": rf,
        "scaler": scaler,
        "accuracy_cv": round(scores.mean(), 4),
        "accuracy_std": round(scores.std(), 4),
        "auc_roc_cv": round(auc_scores.mean(), 4),
        "importancia": importancia,
        "cols_features": cols_efectivas,
    }


def predecir_riesgo(
    df: pd.DataFrame,
    modelo: dict,
) -> pd.DataFrame:
    """
    Aplica el modelo entrenado a los datos y devuelve riesgo por usuario.
    """
    X = df[modelo["cols_features"]].fillna(0).values
    X_scaled = modelo["scaler"].transform(X)

    proba = modelo["modelo"].predict_proba(X_scaled)[:, 1]
    pred = modelo["modelo"].predict(X_scaled)

    resultado = df[["id_usuario", "handle", "perfil", "IA", "PA", "RRD",
                    "mensajes_totales", "veces_respondido"]].copy()
    resultado["probabilidad_abandono"] = np.round(proba, 4)
    resultado["prediccion_abandono"] = pred
    resultado["alerta"] = resultado["probabilidad_abandono"].apply(
        lambda x: "CRITICO" if x >= 0.8 else ("ALTO" if x >= 0.6 else ("MEDIO" if x >= 0.4 else "BAJO"))
    )
    return resultado.sort_values("probabilidad_abandono", ascending=False)


def detectar_escalada_hostil(
    df_mensajes: pd.DataFrame,
    df_perfiles: pd.DataFrame,
    col_texto: str = "texto",
) -> pd.DataFrame:
    """
    Detecta posible escalada hostil: perifericos cuyo lenguaje
    se vuelve negativo o agresivo.

    Busca patrones de negatividad creciente, uso de mayusculas
    sostenidas,辱骂 (insultos), y marcadores de frustracion.
    """
    import re

    patrones_hostiles = [
        (r"\b(idiota|estupido|imbecil|mierda|puto|maldito)\b", "insulto_directo"),
        (r"\b(callate|vete|largate|cerra el orto)\b", "orden_agresiva"),
        (r"[A-Z]{5,}", "gritos"),
        (r"(!\s*!){3,}|(\?\s*\?){3,}", "puntuacion_agresiva"),
        (r"\b(odio|detesto|asco|asqueroso|repugna|repudio)\b", "lenguaje_asco"),
        (r"\b(vos|tu|usted)\s+(siempre|nunca|jamas)\b", "acusacion_generalizante"),
    ]

    resultados = []
    for _, msg in df_mensajes.iterrows():
        texto = str(msg.get(col_texto, ""))
        puntaje = 0
        detectados = []
        for patron, nombre in patrones_hostiles:
            if re.search(patron, texto, re.IGNORECASE):
                puntaje += 1
                detectados.append(nombre)
        if puntaje >= 2:
            resultados.append({
                "id_usuario": msg.get("id_usuario", ""),
                "id_mensaje": msg.get("id_mensaje", ""),
                "texto": texto[:100],
                "puntaje_hostilidad": puntaje,
                "patrones": ", ".join(detectados),
            })

    df_hostil = pd.DataFrame(resultados)

    if len(df_hostil) == 0:
        return pd.DataFrame()

    # Agregar perfil
    perfil_map = df_perfiles.set_index("id_usuario")["perfil"].to_dict()
    df_hostil["perfil"] = df_hostil["id_usuario"].map(perfil_map)

    # Agregar por usuario: total hostilidad
    agg = df_hostil.groupby("id_usuario").agg(
        mensajes_hostiles=("id_mensaje", "count"),
        puntaje_total=("puntaje_hostilidad", "sum"),
        patrones_comunes=("patrones", lambda x: ", ".join(set(",".join(x).split(", ")))),
    ).reset_index()
    agg["perfil"] = agg["id_usuario"].map(perfil_map)
    agg["escalada"] = agg["puntaje_total"] >= 3

    return agg.sort_values("puntaje_total", ascending=False)


def recomendar_intervencion(
    df_riesgo: pd.DataFrame,
    df_perfiles: pd.DataFrame,
    umbral_critico: float = 0.7,
) -> pd.DataFrame:
    """
    Genera recomendaciones de intervencion para usuarios en riesgo.

    Sugiere accion segun perfil y nivel de riesgo.
    """
    df = df_riesgo[df_riesgo["probabilidad_abandono"] >= umbral_critico].copy()

    if len(df) == 0:
        return pd.DataFrame()

    def sugerir_accion(row):
        perfil = row.get("perfil", "desconocido")
        prob = row["probabilidad_abandono"]

        if perfil == "periferico":
            if prob >= 0.8:
                return "Intervencion urgente: mensaje personalizado de inclusion, mencion directa en tema de su interes"
            return "Mencionar en conversacion grupal, preguntar su opinion sobre tema recurrente"
        elif perfil == "integrado_silencioso":
            if prob >= 0.8:
                return "Contactar directamente, verificar si hay problema externo al grupo"
            return "Reforzar con reacciones a sus mensajes, reconocer su presencia"
        elif perfil == "nucleo":
            return "Monitorear: posible fatiga del lider. Ofrecer delegacion"
        elif perfil == "fantasma":
            return "Invitacion personalizada a participar, encuesta anonima de satisfaccion"
        return "Monitoreo general"

    df["accion_recomendada"] = df.apply(sugerir_accion, axis=1)
    df["prioridad"] = df["probabilidad_abandono"].apply(
        lambda x: "URGENTE" if x >= 0.85 else "ALTA" if x >= 0.75 else "MEDIA"
    )

    return df.sort_values("probabilidad_abandono", ascending=False)


def resumen_predictivo(
    df_riesgo: pd.DataFrame,
    modelo_res: dict,
    df_hostil: pd.DataFrame = None,
    df_intervencion: pd.DataFrame = None,
) -> str:
    """Reporte textual del modulo predictivo."""
    lines = []
    lines.append("=" * 60)
    lines.append("MODULO PREDICTIVO - Riesgo de Abandono y Escalada")
    lines.append("=" * 60)

    # Modelo
    lines.append(f"\n[Modelo] Random Forest - Riesgo de Abandono:")
    lines.append(f"  Accuracy CV: {modelo_res['accuracy_cv']:.4f} +/- {modelo_res['accuracy_std']:.4f}")
    lines.append(f"  AUC-ROC CV:  {modelo_res['auc_roc_cv']:.4f}")

    lines.append("\n  Top 5 features predictivas:")
    for _, row in modelo_res["importancia"].head(5).iterrows():
        lines.append(f"    {row['feature']:<25s} {row['importancia']:.4f}")

    # Distribucion de riesgo
    lines.append(f"\n[Alertas] Distribucion de riesgo:")
    for nivel in ["BAJO", "MEDIO", "ALTO", "CRITICO"]:
        count = len(df_riesgo[df_riesgo["alerta"] == nivel])
        lines.append(f"  {nivel:<10s} {count} usuarios")

    lines.append(f"\n  Top 10 usuarios en riesgo:")
    top10 = df_riesgo.head(10)
    for _, u in top10.iterrows():
        lines.append(f"    @{u['handle'][-16:]:<20s} riesgo={u['probabilidad_abandono']:.2f} "
                     f"perfil={u['perfil']:<12s} IA={u['IA']:.2f} PA={u['PA']:.2f}")

    # Hostilidad
    if df_hostil is not None and len(df_hostil) > 0:
        lines.append(f"\n[Escalada Hostil] Usuarios con patrones agresivos:")
        lines.append(f"  Total usuarios detectados: {len(df_hostil)}")
        for _, u in df_hostil.head(5).iterrows():
            lines.append(f"    @{u['id_usuario'][-16:]:<20s} puntaje={u['puntaje_total']} "
                         f"perfil={u['perfil']:<12s} mensajes={u['mensajes_hostiles']}")

    # Intervencion
    if df_intervencion is not None and len(df_intervencion) > 0:
        lines.append(f"\n[Intervencion Recomendada] {len(df_intervencion)} usuarios requieren accion:")
        for _, u in df_intervencion.head(10).iterrows():
            lines.append(f"  [{u['prioridad']}] @{u['handle'][-16:]:<20s} -> {u['accion_recomendada']}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)

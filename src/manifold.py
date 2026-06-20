import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


def reducir_dimension(
    df_features: pd.DataFrame,
    cols_features: list = None,
    metodo: str = "tsne",
    n_components: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Reduce dimensionalidad del feature space a 2D.

    Parametros:
        metodo: "tsne" | "pca"
    Retorna df con columnas x, y.
    """
    if cols_features is None:
        cols_features = [c for c in df_features.columns
                         if c not in ("id_usuario", "perfil", "handle", "nombre",
                                      "id_usuario_origen", "id_usuario_destino")]

    # Seleccionar solo columnas numericas
    X = df_features[cols_features].select_dtypes(include=[np.number]).fillna(0).values

    # Estandarizar
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if metodo == "tsne":
        reducer = TSNE(n_components=n_components, random_state=random_state,
                       perplexity=min(30, len(X) - 1), max_iter=1000)
        coords = reducer.fit_transform(X_scaled)
    elif metodo == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state)
        coords = reducer.fit_transform(X_scaled)
    else:
        raise ValueError(f"Metodo no soportado: {metodo}")

    result = df_features[["id_usuario"]].copy()
    result["x"] = coords[:, 0]
    result["y"] = coords[:, 1]
    return result


def validar_clusters(
    df_features: pd.DataFrame,
    cols_features: list = None,
    metodo_clustering: str = "kmeans",
    n_clusters_range: range = range(2, 8),
) -> dict:
    """
    Valida cuantos clusters naturales existen en los datos.

    Para cada k en n_clusters_range, calcula:
    - silhouette_score
    - inercia (KMeans)
    - correlacion con perfil actual (ajusted_rand_score)

    Retorna dict con scores y mejor k.
    """
    if cols_features is None:
        cols_features = [c for c in df_features.columns
                         if c not in ("id_usuario", "perfil", "handle", "nombre")]
    X = df_features[cols_features].select_dtypes(include=[np.number]).fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    resultados = []
    for k in n_clusters_range:
        if metodo_clustering == "kmeans":
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            etiquetas = model.fit_predict(X_scaled)
            sil = silhouette_score(X_scaled, etiquetas)
            resultados.append({
                "k": k,
                "silhouette": round(sil, 4),
                "inercia": round(model.inertia_, 2),
            })
        elif metodo_clustering == "dbscan":
            continue  # DBSCAN no tiene parametro k

    # Mejor k por silhouette
    if resultados:
        mejor_k = max(resultados, key=lambda r: r["silhouette"])
    else:
        mejor_k = None

    # DBSCAN
    if metodo_clustering in ("dbscan", "ambos"):
        for eps in [0.3, 0.5, 0.8, 1.0, 1.5]:
            model = DBSCAN(eps=eps, min_samples=3)
            etiquetas = model.fit_predict(X_scaled)
            n_clusters = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
            n_ruido = sum(etiquetas == -1)
            sil = None
            if n_clusters > 1 and n_clusters < len(X):
                try:
                    sil = round(silhouette_score(X_scaled, etiquetas), 4)
                except Exception:
                    pass
            resultados.append({
                "k": f"DBSCAN eps={eps}",
                "silhouette": sil,
                "n_clusters": n_clusters,
                "n_ruido": n_ruido,
            })

    # Correlacion con perfil actual (si existe)
    if "perfil" in df_features.columns:
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        perfil_encoded = le.fit_transform(df_features["perfil"])
        for r in resultados:
            if isinstance(r["k"], int):
                model = KMeans(n_clusters=r["k"], random_state=42, n_init=10)
                etiquetas = model.fit_predict(X_scaled)
                r["ari_vs_perfil"] = round(adjusted_rand_score(perfil_encoded, etiquetas), 4)
            else:
                r["ari_vs_perfil"] = None

    return {
        "resultados": resultados,
        "mejor_k": mejor_k,
        "X_scaled": X_scaled,
    }


def clasificador_ml(
    df_features: pd.DataFrame,
    col_perfil: str = "perfil",
    cols_features: list = None,
    test_size: float = 0.3,
    random_state: int = 42,
) -> dict:
    """
    Entrena un Random Forest para clasificar perfiles.

    Usa las features calculadas (IA, PA, RRD, etc.) como predictores.
    Retorna: accuracy, feature_importance, modelo, score CV.
    """
    if cols_features is None:
        cols_features = [c for c in df_features.columns
                         if c not in ("id_usuario", "perfil", "handle", "nombre")]

    df = df_features.dropna(subset=[col_perfil]).copy()
    # Excluir perfiles con un solo ejemplar
    perfiles_count = df[col_perfil].value_counts()
    perfiles_validos = perfiles_count[perfiles_count > 1].index
    df = df[df[col_perfil].isin(perfiles_validos)]

    X = df[cols_features].select_dtypes(include=[np.number]).fillna(0).values
    y = df[col_perfil].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Random Forest
    rf = RandomForestClassifier(n_estimators=200, max_depth=8,
                                random_state=random_state, class_weight="balanced")
    scores = cross_val_score(rf, X_scaled, y, cv=5, scoring="accuracy")

    # Entrenar full para importancia
    rf.fit(X_scaled, y)

    importancia = pd.DataFrame({
        "feature": cols_features,
        "importancia": rf.feature_importances_,
    }).sort_values("importancia", ascending=False)

    return {
        "modelo": rf,
        "scaler": scaler,
        "accuracy_cv_mean": round(scores.mean(), 4),
        "accuracy_cv_std": round(scores.std(), 4),
        "importancia": importancia,
        "n_classes": len(set(y)),
        "clases": list(set(y)),
    }


def resumen_validacion(
    df_coords: pd.DataFrame,
    validacion_clusters: dict,
    clasificador: dict = None,
    df_features: pd.DataFrame = None,
) -> str:
    """Genera reporte textual del analisis de manifold + clustering + ML."""
    lines = []
    lines.append("=" * 60)
    lines.append("VALIDACION DE TAXONOMIA - UMAP/t-SNE + Clustering + ML")
    lines.append("=" * 60)

    # Coordenadas
    lines.append(f"\n[t-SNE] Coordenadas generadas para {len(df_coords)} usuarios")

    # Clustering
    vc = validacion_clusters
    lines.append(f"\n[Clustering] Validacion de clusters:")
    for r in vc["resultados"]:
        if isinstance(r["k"], int):
            ari = r.get("ari_vs_perfil", "N/A")
            lines.append(f"  KMeans k={r['k']}: silhouette={r['silhouette']}  inercia={r['inercia']}  ARI vs perfil={ari}")
        else:
            lines.append(f"  {r['k']}: silhouette={r['silhouette']}  clusters={r.get('n_clusters','?')}  ruido={r.get('n_ruido','?')}")

    if vc["mejor_k"]:
        mk = vc["mejor_k"]
        lines.append(f"\n  ** Mejor k (silhouette): {mk['k']} (s={mk['silhouette']}) **")
        if mk["k"] == 5:
            lines.append("  -> Coincide con la taxonomia de 5 perfiles!")
        elif mk["k"] < 5:
            lines.append(f"  -> SuGIERE {mk['k']} perfiles (menos de los 5 teoricos)")
        else:
            lines.append(f"  -> SuGIERE {mk['k']} perfiles (mas subdivision de la esperada)")

    # Clasificador ML
    if clasificador:
        lines.append(f"\n[Random Forest] Clasificador de perfiles:")
        lines.append(f"  Accuracy (CV 5-fold): {clasificador['accuracy_cv_mean']:.4f} +/- {clasificador['accuracy_cv_std']:.4f}")
        lines.append(f"  Clases: {clasificador['n_classes']}")
        lines.append(f"\n  Top 5 features por importancia:")
        for _, row in clasificador["importancia"].head(5).iterrows():
            lines.append(f"    {row['feature']:<25s} {row['importancia']:.4f}")

    # Correlacion con perfil actual
    if df_features is not None and "perfil" in df_features.columns:
        lines.append(f"\n[Resumen] Distribucion de perfiles:")
        for perfil, count in df_features["perfil"].value_counts().items():
            lines.append(f"  {perfil:<25s} {count} usuarios")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)

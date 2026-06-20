import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
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
    n_clusters_range: range = range(2, 8),
    incluir_gmm: bool = True,
    incluir_spectral: bool = True,
    incluir_dbscan: bool = True,
) -> dict:
    """
    Valida cuantos clusters naturales existen en los datos usando
    multiples algoritmos: KMeans, GMM, Spectral, DBSCAN.

    Para cada k en n_clusters_range, calcula:
    - silhouette_score
    - ARI vs perfil actual
    - NMI vs perfil actual

    Retorna dict con scores, mejor k (por silhouette) y etiquetas.
    """
    if cols_features is None:
        cols_features = [c for c in df_features.columns
                         if c not in ("id_usuario", "perfil", "handle", "nombre")]

    X = df_features[cols_features].select_dtypes(include=[np.number]).fillna(0).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Codificar perfiles actuales para comparacion
    if "perfil" in df_features.columns:
        le = LabelEncoder()
        perfil_encoded = le.fit_transform(df_features["perfil"])
        clases_perfil = le.classes_
    else:
        perfil_encoded = None
        clases_perfil = None

    resultados = []

    def _evaluar(etiquetas, metodo, k_label):
        """Evalua un clustering y retorna dict con metricas."""
        n_uniq = len(set(etiquetas)) - (1 if -1 in etiquetas else 0)
        n_ruido = sum(etiquetas == -1) if -1 in etiquetas else 0
        mask_valido = etiquetas != -1
        sil = None
        if n_uniq > 1 and n_uniq < len(etiquetas):
            try:
                if sum(mask_valido) > 1:
                    sil = round(silhouette_score(X_scaled[mask_valido],
                                                  etiquetas[mask_valido]), 4)
                else:
                    sil = round(silhouette_score(X_scaled, etiquetas), 4)
            except Exception:
                pass
        ari = None
        nmi = None
        if perfil_encoded is not None and n_uniq > 1:
            ari = round(adjusted_rand_score(perfil_encoded, etiquetas), 4)
            try:
                nmi = round(normalized_mutual_info_score(perfil_encoded, etiquetas), 4)
            except Exception:
                pass
        r = {
            "metodo": metodo,
            "k_label": k_label,
            "n_clusters": n_uniq,
            "n_ruido": n_ruido,
            "silhouette": sil,
            "ari_vs_perfil": ari,
            "nmi_vs_perfil": nmi,
        }
        if metodo == "kmeans" and isinstance(k_label, int):
            model = KMeans(n_clusters=k_label, random_state=42, n_init=10)
            model.fit(X_scaled)
            r["inercia"] = round(model.inertia_, 2)
        return r

    # 1. KMeans
    for k in n_clusters_range:
        model = KMeans(n_clusters=k, random_state=42, n_init=10)
        etiquetas = model.fit_predict(X_scaled)
        r = _evaluar(etiquetas, "kmeans", k)
        r["inercia"] = round(model.inertia_, 2)
        resultados.append(r)

    # 2. GMM (Gaussian Mixture)
    if incluir_gmm:
        for k in n_clusters_range:
            model = GaussianMixture(n_components=k, random_state=42, max_iter=200)
            etiquetas = model.fit_predict(X_scaled)
            r = _evaluar(etiquetas, "gmm", k)
            r["bic"] = round(model.bic(X_scaled), 2)
            resultados.append(r)

    # 3. Spectral Clustering
    if incluir_spectral:
        max_k = max(n_clusters_range)
        if max_k < len(X_scaled):
            for k in n_clusters_range:
                model = SpectralClustering(n_clusters=k, random_state=42,
                                           affinity="nearest_neighbors",
                                           n_neighbors=min(15, len(X_scaled) - 1))
                etiquetas = model.fit_predict(X_scaled)
                resultados.append(_evaluar(etiquetas, "spectral", k))

    # 4. DBSCAN (barrido de eps)
    if incluir_dbscan:
        for eps in [0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]:
            model = DBSCAN(eps=eps, min_samples=3)
            etiquetas = model.fit_predict(X_scaled)
            r = _evaluar(etiquetas, "dbscan", f"eps={eps}")
            resultados.append(r)

    # Mejor k global (promedio de silhouette entre metodos)
    mejores_por_metodo = {}
    for r in resultados:
        if r["silhouette"] is not None:
            metodo = r["metodo"]
            if metodo not in mejores_por_metodo or r["silhouette"] > mejores_por_metodo[metodo]["silhouette"]:
                mejores_por_metodo[metodo] = r

    # Mejor KMeans
    kmeans_results = [r for r in resultados if r["metodo"] == "kmeans" and r["silhouette"] is not None]
    mejor_kmeans = max(kmeans_results, key=lambda r: r["silhouette"]) if kmeans_results else None

    return {
        "resultados": resultados,
        "mejor_kmeans": mejor_kmeans,
        "mejores_por_metodo": mejores_por_metodo,
        "X_scaled": X_scaled,
        "clases_perfil": clases_perfil,
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

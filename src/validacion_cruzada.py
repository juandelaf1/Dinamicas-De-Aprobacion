"""
Validacion cruzada de la taxonomia con 4 algoritmos de clustering.
Analiza donde coinciden y donde divergen los perfiles teoricos vs clusters naturales.
"""
import warnings, sys, os, json
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from manifold import validar_clusters, reducir_dimension, clasificador_ml

base = os.path.join("data", "raw", "reddit_merged_20260620.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)
n_users = len(df_r)

cols = [c for c in df_r.columns
        if c not in ("id_usuario", "perfil", "handle", "nombre")]
X = df_r[cols].select_dtypes(include=[np.number]).fillna(0).values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

le = LabelEncoder()
perfil_encoded = le.fit_transform(df_r["perfil"])
clases_perfil = le.classes_

# === 1. CLUSTERING MULTI-METODO ===
print("=" * 72)
print("VALIDACION CRUZADA DE TAXONOMIA (4 algoritmos)")
print("=" * 72)

val = validar_clusters(df_r, cols_features=cols,
                       n_clusters_range=range(2, 8),
                       incluir_gmm=True, incluir_spectral=True, incluir_dbscan=True)

# Tabla comparativa por metodo
print("\n--- Mejor configuracion por metodo ---")
for metodo, r in val["mejores_por_metodo"].items():
    k = r.get("k_label", "?")
    n_cl = r["n_clusters"]
    sil = r["silhouette"]
    ari = r.get("ari_vs_perfil", "-")
    nmi = r.get("nmi_vs_perfil", "-")
    ruido = r.get("n_ruido", 0)
    ruido_pct = f"{ruido/n_users*100:.1f}%"
    print(f"  {metodo:<12s} k={str(k):<8s} clusters={n_cl}  "
          f"sil={sil:.4f}  ARI={ari}  NMI={nmi}  ruido={ruido_pct}")

# === 2. MATRIZ DE CONSENSO (k=5) ===
print("\n\n--- Consenso entre metodos para k=5 ---")
consenso = {}
for r in val["resultados"]:
    if r["metodo"] in ("kmeans", "gmm", "spectral"):
        if r["k_label"] == 5:
            met = r["metodo"]
            consenso[met] = {
                "silhouette": r["silhouette"],
                "ari_vs_perfil": r.get("ari_vs_perfil"),
                "nmi_vs_perfil": r.get("nmi_vs_perfil"),
            }
            print(f"  {met:<12s} sil={r['silhouette']:.4f}  ARI={r['ari_vs_perfil']:.4f}  NMI={r['nmi_vs_perfil']:.4f}")

# === 3. DIVERGENCIAS: DONDE DISCREPAN LOS PERFILES ===
print("\n\n--- Divergencias: perfil teorico vs kmeans k=4 ---")
from sklearn.cluster import KMeans

for k in [4, 5, 6]:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    etiquetas_cluster = km.fit_predict(X_scaled)

    # Tabla de confusion: perfil teorico x cluster
    df_comp = df_r[["id_usuario", "perfil"]].copy()
    df_comp["cluster"] = etiquetas_cluster

    tabla = pd.crosstab(df_comp["perfil"], df_comp["cluster"],
                         normalize="columns")

    # Para cada cluster, ver que perfil predomina
    print(f"\n  k={k} — Distribucion de perfiles dentro de cada cluster:")
    for cluster in sorted(tabla.columns):
        print(f"    Cluster {cluster}:")
        for perfil in tabla.index:
            pct = tabla.loc[perfil, cluster] * 100
            if pct > 5:
                print(f"      {perfil:<25s} {pct:.1f}%")

# === 4. USUARIOS MAL CLASIFICADOS (k=5) ===
print("\n\n--- Usuarios con discrepancia perfil vs cluster (k=5) ---")
km5 = KMeans(n_clusters=5, random_state=42, n_init=10)
cluster5 = km5.fit_predict(X_scaled)
df_comp5 = df_r[["id_usuario", "perfil", "IA", "PA", "RRD"]].copy()
df_comp5["cluster"] = cluster5

# Mapa: cluster -> perfil mayoritario
mapa_cluster = {}
for c in range(5):
    perfiles_en_c = df_comp5[df_comp5["cluster"] == c]["perfil"].value_counts()
    if len(perfiles_en_c) > 0:
        mapa_cluster[c] = perfiles_en_c.index[0]

df_comp5["perfil_asignado_cluster"] = df_comp5["cluster"].map(mapa_cluster)
df_comp5["discrepa"] = df_comp5["perfil"] != df_comp5["perfil_asignado_cluster"]

n_discrepan = df_comp5["discrepa"].sum()
print(f"  Usuarios que discrepan: {n_discrepan} de {n_users} ({n_discrepan/n_users*100:.1f}%)")
if n_discrepan > 0:
    print("\n  Perfiles con mas discrepancias:")
    print(df_comp5[df_comp5["discrepa"]]["perfil"].value_counts().to_string())

    # Mapa cluster -> perfil
    print("\n  Mapa cluster -> perfil mayoritario:")
    for c, p in sorted(mapa_cluster.items()):
        print(f"    Cluster {c} -> {p}")

    # Mostrar algunos casos concretos
    print("\n  Casos de discrepancia (primeros 10):")
    for _, u in df_comp5[df_comp5["discrepa"]].head(10).iterrows():
        print(f"    @{u['id_usuario'][-16:]:<20s} perfil={u['perfil']:<20s} "
              f"cluster_asigna={u['perfil_asignado_cluster']:<20s} "
              f"IA={u['IA']:.1f} PA={u['PA']:.3f} RRD={u['RRD']:.2f}")

# === 5. CONCLUSION ===
print("\n\n" + "=" * 72)
print("CONCLUSION DE VALIDACION CRUZADA")
print("=" * 72)

# Mejor k global
metodos_con_k = [r for r in val["resultados"] if r["metodo"] in ("kmeans", "gmm", "spectral") and r["silhouette"] is not None]
mejor_global = max(metodos_con_k, key=lambda r: r["silhouette"])

print(f"\n  Mejor configuracion global: {mejor_global['metodo']} k={mejor_global['k_label']} "
      f"(silhouette={mejor_global['silhouette']:.4f})")
print(f"  ARI maximo vs perfiles: {max(r['ari_vs_perfil'] for r in metodos_con_k if r.get('ari_vs_perfil')):.4f}")
print(f"  NMI maximo vs perfiles: {max(r['nmi_vs_perfil'] for r in metodos_con_k if r.get('nmi_vs_perfil')):.4f}")

print(f"\n  Numero de clusters mas consistente entre metodos:")
from collections import Counter
ks_metodos = [r["k_label"] for r in val["mejores_por_metodo"].values() if isinstance(r.get("k_label"), int)]
print(f"    {Counter(ks_metodos).most_common()}")

print(f"\n  Tasa de discrepancia (k=5): {n_discrepan/n_users*100:.1f}%")
print(f"  -> {'Los perfiles coinciden bien con clusters naturales' if n_discrepan/n_users < 0.3 else 'Hay divergencia significativa entre perfiles teoricos y clusters'}")

# Guardar
out = os.path.join("data", "validacion_cruzada.txt")
with open(out, "w", encoding="utf-8") as f:
    f.write(f"Validacion cruzada de taxonomia\n")
    f.write(f"4 algoritmos: KMeans, GMM, Spectral, DBSCAN\n")
    f.write(f"Usuarios: {n_users}\n\n")
    f.write(f"Mejor config: {mejor_global['metodo']} k={mejor_global['k_label']} (sil={mejor_global['silhouette']:.4f})\n")
    f.write(f"ARI max vs perfiles: {max(r['ari_vs_perfil'] for r in metodos_con_k if r.get('ari_vs_perfil')):.4f}\n")
    f.write(f"NMI max vs perfiles: {max(r['nmi_vs_perfil'] for r in metodos_con_k if r.get('nmi_vs_perfil')):.4f}\n")
    f.write(f"Discrepancias k=5: {n_discrepan} ({n_discrepan/n_users*100:.1f}%)\n")
    for r in val["resultados"]:
        if r["silhouette"] is not None:
            f.write(f"{r['metodo']:>10s} k={str(r['k_label']):<8s} "
                    f"sil={r['silhouette']:.4f}  ari={r.get('ari_vs_perfil','-'):<8}  "
                    f"nmi={r.get('nmi_vs_perfil','-'):<8}\n")

print(f"\nResultados guardados en: {out}")

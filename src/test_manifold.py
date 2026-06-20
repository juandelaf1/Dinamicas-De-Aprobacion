import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from manifold import (
    reducir_dimension, validar_clusters, clasificador_ml,
    resumen_validacion
)

base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "data", "raw", "bluesky_threads_20260619_123556.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)

print("Features disponibles:", list(df_r.columns))
print()

# 1. t-SNE
print("--- t-SNE ---")
coords = reducir_dimension(df_r, metodo="tsne")
print(f"Coordenadas: {len(coords)} usuarios, min_x={coords['x'].min():.2f} max_x={coords['x'].max():.2f}")
print()

# 2. PCA
print("--- PCA ---")
coords_pca = reducir_dimension(df_r, metodo="pca")
print(f"PCA: {len(coords_pca)} usuarios")
expl = coords_pca[["x", "y"]].values
print(f"  Varianza explicada: x={abs(coords_pca['x']).mean():.4f} y={abs(coords_pca['y']).mean():.4f}")
print()

# 3. Validacion de clusters (KMeans)
print("--- Clustering Validation ---")
val = validar_clusters(df_r, n_clusters_range=range(2, 8), metodo_clustering="kmeans")
for r in val["resultados"]:
    if isinstance(r["k"], int):
        print(f"  k={r['k']}: silhouette={r['silhouette']}  inercia={r['inercia']}  ari={r.get('ari_vs_perfil','?')}")
print(f"  Mejor k: {val['mejor_k']}")
print()

# 4. Random Forest classifier
print("--- ML Classifier (Random Forest) ---")
clf = clasificador_ml(df_r)
print(f"  Accuracy CV: {clf['accuracy_cv_mean']:.4f} +/- {clf['accuracy_cv_std']:.4f}")
print(f"  Clases: {clf['n_classes']}")
print("  Feature importance (top 8):")
print(clf["importancia"].head(8).to_string(index=False))
print()

# 5. Visualizacion t-SNE
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

merged = df_r[["id_usuario", "perfil", "handle", "IA", "PA"]].merge(coords, on="id_usuario")
color_map = {"nucleo": "#e74c3c", "buscador_validacion": "#e67e22",
             "integrado_silencioso": "#2ecc71", "periferico": "#3498db", "fantasma": "#95a5a6"}
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for perfil, color in color_map.items():
    subset = merged[merged["perfil"] == perfil]
    if len(subset) > 0:
        ax.scatter(subset["x"], subset["y"], c=color, label=perfil.replace("_", " ").title(),
                   alpha=0.7, edgecolors="white", linewidth=0.3, s=40)
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=c, edgecolor="white", label=p.replace("_", " ").title())
                   for p, c in color_map.items() if len(merged[merged["perfil"] == p]) > 0]
ax.legend(handles=legend_elements, loc="upper right", fontsize=9)
ax.set_title("t-SNE: Espacio de Features Coloreado por Perfil", fontsize=14)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
fig.savefig(os.path.join(out_dir, "tsne_perfiles.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("t-SNE plot guardado en data/tsne_perfiles.png")

# 6. Scatter IA vs PA
fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
for perfil, color in color_map.items():
    subset = merged[merged["perfil"] == perfil]
    if len(subset) > 0:
        ax2.scatter(subset["PA"], subset["IA"], c=color, label=perfil.replace("_", " ").title(),
                    alpha=0.7, edgecolors="white", linewidth=0.3, s=50)
ax2.legend(handles=legend_elements, loc="upper right", fontsize=9)
ax2.set_title("IA vs PA por Perfil", fontsize=14)
ax2.set_xlabel("PA (Pertenencia Aproximada)")
ax2.set_ylabel("IA (Indice de Aprobacion)")
ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
ax2.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
fig2.savefig(os.path.join(out_dir, "ia_vs_pa.png"), dpi=150, bbox_inches="tight")
plt.close(fig2)
print("Scatter IA vs PA guardado en data/ia_vs_pa.png")

# 7. Resumen
print()
print(resumen_validacion(coords, val, clf, df_r))

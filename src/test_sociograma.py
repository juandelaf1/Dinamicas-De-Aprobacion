import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from sociograma import (
    construir_grafo, metricas_centralidad, detectar_comunidades,
    analizar_aislamiento, resumen_sociograma
)
from visualizar_sociograma import (
    visualizar_sociograma, visualizar_centralidad, visualizar_comunidades
)

base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "data", "raw", "bluesky_threads_20260619_123556.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)

print(f"Interacciones disponibles: {len(df_i)}")

# Construir grafo dirigido
G = construir_grafo(df_i, df_perfiles=df_r, df_features=df_f, dirigido=True)
print(f"Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

# Centralidad
df_cent = metricas_centralidad(G)
print(f"Centralidad calculada: {len(df_cent)} nodos")
print("Columnas:", list(df_cent.columns))

# Comunidades
df_com = detectar_comunidades(G)
print(f"Comunidades detectadas: {df_com['comunidad'].nunique()}")
print(df_com["comunidad"].value_counts().sort_index().to_string())

# Aislamiento
aisl = analizar_aislamiento(G, df_r)
print(f"\nAislados: {aisl['n_aislados']} (de {G.number_of_nodes()} nodos)")
print(f"Componentes: {aisl['n_componentes']}")
print(f"Densidad: {aisl['densidad_red']}")
print(f"Clustering: {aisl['coeficiente_clustering']}")
for i, (idx, tam) in enumerate(aisl["componentes"][:3]):
    print(f"  Componente {idx}: {tam} nodos")

# Resumen textual
print()
print(resumen_sociograma(G, df_cent, aisl))

# Visualizaciones
out_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(out_dir, exist_ok=True)

print("\nGenerando visualizaciones...")
vis = visualizar_sociograma(G, os.path.join(out_dir, "sociograma_perfiles.png"))
print(f"  Sociograma: {vis}")

vis2 = visualizar_centralidad(df_cent, os.path.join(out_dir, "centralidad_pagerank.png"))
print(f"  Centralidad: {vis2}")

vis3 = visualizar_comunidades(G, df_com, os.path.join(out_dir, "comunidades.png"))
print(f"  Comunidades: {vis3}")
print("\nVisualizaciones generadas en data/")

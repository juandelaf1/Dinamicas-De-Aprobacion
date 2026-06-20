import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from sociograma import construir_grafo, detectar_comunidades
from subgrupos import (
    detectar_subgrupos_excluyentes, usuarios_aislados_en_subgrupo,
    puentes_entre_comunidades, resumen_subgrupos
)

base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "data", "raw", "bluesky_threads_20260619_123556.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)

print("=== Construyendo grafo ===")
G = construir_grafo(df_i, df_perfiles=df_r, df_features=df_f, dirigido=True)
print(f"Grafo: {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")

print("\n=== Comunidades ===")
df_com = detectar_comunidades(G)
print(f"Comunidades: {df_com['comunidad'].nunique()}")

print("\n=== 3.4 Subgrupos Excluyentes ===")
df_excl = detectar_subgrupos_excluyentes(G, df_com, umbral_exclusividad=3.0)
print(f"Excluyentes: {df_excl['excluyente'].sum()} de {len(df_excl)} comunidades")
print(df_excl[["comunidad", "miembros", "densidad_interna", "ratio_exclusividad", "excluyente"]].head(15).to_string(index=False))

print("\n=== Marginados Internos ===")
df_aisl = usuarios_aislados_en_subgrupo(G, df_com, df_r)
print(f"Usuarios marginados/aislados: {len(df_aisl)}")
if len(df_aisl) > 0:
    print(df_aisl["perfil"].value_counts().to_string())
    print()
    print(df_aisl.head(10).to_string(index=False))

print("\n=== Puentes entre Comunidades ===")
df_puentes = puentes_entre_comunidades(G, df_com)
print(f"Usuarios puente: {len(df_puentes)}")
if len(df_puentes) > 0:
    print(df_puentes.head(10).to_string(index=False))

print("\n" + resumen_subgrupos(df_excl, df_aisl, df_puentes, G))

import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from sociograma import construir_grafo, metricas_centralidad, detectar_comunidades, analizar_aislamiento
from manifold import reducir_dimension, validar_clusters, clasificador_ml
from nlp.analisis_nlp import AnalisisNLP

base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "data", "raw", "bluesky_threads_20260619_123556.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)

# NLP
nlp = AnalisisNLP(df_m, df_r)
nlp.ejecutar_jerga_interna()
nlp.ejecutar_sentimiento()
nlp.ejecutar_borrados()
nlp.ejecutar_estilo()
nlp.ejecutar_topicos(n_topicos=4)

# Sociograma
G = construir_grafo(df_i, df_perfiles=df_r, df_features=df_f, dirigido=True)
df_cent = metricas_centralidad(G)
df_com = detectar_comunidades(G)
aisl = analizar_aislamiento(G, df_r)

# Manifold
coords = reducir_dimension(df_r, metodo="tsne")
val = validar_clusters(df_r, n_clusters_range=range(2, 8))
clf = clasificador_ml(df_r)

print("=" * 60)
print("PRUEBA END-TO-END COMPLETA: TODOS LOS MODULOS OK")
print("=" * 60)
print(f"Pipeline:      {len(df_r)} usuarios clasificados")
print(f"NLP:           {len(nlp.resultados['jerga']['terminos_jerga'])} terminos jerga")
print(f"Sociograma:    {G.number_of_nodes()} nodos, {G.number_of_edges()} aristas")
print(f"Manifold:      {len(coords)} coordenadas t-SNE")
print(f"Clustering:    mejor k={val['mejor_k']['k']} (silhouette={val['mejor_k']['silhouette']})")
print(f"ML Accuracy:   {clf['accuracy_cv_mean']:.4f}")
print("=" * 60)

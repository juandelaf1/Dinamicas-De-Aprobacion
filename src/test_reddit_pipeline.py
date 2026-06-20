import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from sociograma import construir_grafo, metricas_centralidad, detectar_comunidades, analizar_aislamiento
from manifold import reducir_dimension, validar_clusters, clasificador_ml
from nlp.analisis_nlp import AnalisisNLP

base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "data", "raw", "reddit_merged_20260620.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)

print("=" * 60)
print("PIPELINE COMPLETO - Reddit (3 subreddits)")
print("=" * 60)
print(f"\nUsuarios: {len(df_r)}")
print(f"Mensajes: {len(df_m)}")
print(f"Interacciones (reply graph): {len(df_i)}")
print(f"Perfiles:")
print(df_r["perfil"].value_counts().to_string())
print(f"\nIA medio por perfil:")
print(df_r.groupby("perfil")["IA"].mean().round(2).to_string())
print(f"\nPA medio por perfil:")
print(df_r.groupby("perfil")["PA"].mean().round(4).to_string())

# Sociograma
print("\n--- Sociograma ---")
G = construir_grafo(df_i, df_perfiles=df_r, df_features=df_f, dirigido=True)
print(f"Nodos: {G.number_of_nodes()}, Aristas: {G.number_of_edges()}")
aisl = analizar_aislamiento(G, df_r)
print(f"Aislados: {aisl['n_aislados']} ({aisl['n_aislados']/max(G.number_of_nodes(),1)*100:.1f}%)")
print(f"Densidad: {aisl['densidad_red']}")

# NLP
print("\n--- NLP ---")
nlp = AnalisisNLP(df_m, df_r)
nlp.ejecutar_jerga_interna()
nlp.ejecutar_sentimiento()
nlp.ejecutar_borrados()
nlp.ejecutar_topicos(n_topicos=4)
jerga = nlp.resultados["jerga"]
print(f"Terminos jerga: {len(jerga['terminos_jerga'])}")
print(f"Top 10: {jerga['terminos_jerga'][:10]}")
tg = nlp.resultados["terminos_grupo"]
print(f"Vocabulario unico: {tg['vocabulario_unico']}")

sa = nlp.resultados.get("sentimiento_agregado")
if sa is not None:
    print(f"\nSentimiento por perfil:")
    print(sa[["perfil", "sentimiento_medio", "volatilidad_emocional"]].to_string(index=False))

rb = nlp.resultados.get("resumen_borrados", {})
if rb:
    print(f"\nBorrados: {rb['mensajes_borrados']} ({rb['porcentaje_borrado']}%)")

tp = nlp.resultados.get("topicos", {})
if tp.get("topicos_evitados"):
    print(f"\nTopicos evitados: {len(tp['topicos_evitados'])}")
    for te in tp["topicos_evitados"]:
        print(f"  Topico {te['topico']}: {te['palabras_clave'][:3]} nucleo={te['presencia_nucleo']:.1%} perif={te['presencia_periferico']:.1%}")

# Manifold
print("\n--- Validacion ---")
coords = reducir_dimension(df_r, metodo="tsne")
val = validar_clusters(df_r, n_clusters_range=range(2, 8))
clf = clasificador_ml(df_r)
print(f"Mejor k: {val['mejor_k']}")
if clf:
    print(f"RF Accuracy CV: {clf['accuracy_cv_mean']:.4f}")
    print(f"Top features:")
    print(clf["importancia"].head(5).to_string(index=False))

print("\n" + "=" * 60)
print("COMPLETADO")
print("=" * 60)

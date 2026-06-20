import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from nlp.analisis_nlp import AnalisisNLP

base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "data", "raw", "bluesky_threads_20260619_123556.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)

print(f"Usuarios: {len(df_r)}, Mensajes: {len(df_m)}")

nlp = AnalisisNLP(df_m, df_r)

print("--- 2.1 Jerga interna ---")
j = nlp.ejecutar_jerga_interna()
print(f"Terminos jerga: {len(j['terminos_jerga'])}")
print(f"Top 10: {j['terminos_jerga'][:10]}")

tg = nlp.resultados["terminos_grupo"]
print(f"Vocabulario unico: {tg['vocabulario_unico']}")
top_uni = list(tg["unigramas"].keys())[:5]
print(f"Top unigramas: {top_uni}")

print()
print("--- 2.2 Sentimiento por perfil ---")
sa = nlp.ejecutar_sentimiento()
cols = ["perfil", "sentimiento_medio", "volatilidad_emocional",
        "proporcion_positivos", "proporcion_negativos"]
print(sa[cols].to_string(index=False))

print()
print("--- 2.3 Mensajes borrados ---")
rb = nlp.ejecutar_borrados()
print(f"Borrados: {rb['mensajes_borrados']}/{rb['total_mensajes']} ({rb['porcentaje_borrado']}%)")
print(f"Por motivo: {rb['por_motivo']}")

print()
print("--- 2.4 Estilo imitativo ---")
est = nlp.ejecutar_estilo()
print(f"Pares con estilo similar (>0.9): {len(est['pares_similares'])}")
print(f"Imitadores sospechosos: {len(est['imitadores_sospechosos'])}")
if est["pares_similares"]:
    for p in est["pares_similares"][:3]:
        print(f"  @{p['usuario_a'][-12:]} <-> @{p['usuario_b'][-12:]} sim={p['similitud_estilo']:.3f}")

print()
print("--- 2.5 Topic modeling y Espiral del silencio ---")
tp = nlp.ejecutar_topicos(n_topicos=4)
print(f"Topicos evitados: {len(tp['topicos_evitados'])}")
for te in tp["topicos_evitados"]:
    flag = " **ESPIRAL**" if te["posible_espiral_silencio"] else ""
    print(f"  Topico {te['topico']}: {te['palabras_clave'][:3]} nucleo={te['presencia_nucleo']:.1%} perif={te['presencia_periferico']:.1%}{flag}")
print()
print("Distribucion topicos por perfil:")
print(tp["distribucion_topicos_por_perfil"].to_string())

print()
print(nlp.reporte_texto())

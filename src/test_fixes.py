import warnings, sys, os
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from nlp.topic_modeling import extraer_topicos_espiral
from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from predictivo import detectar_escalada_hostil

base = os.path.join("data", "raw", "reddit_merged_20260620.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)

# Test topic modeling with new stop words
tp = extraer_topicos_espiral(df_m, df_r, n_topicos=4)
print("=== TOPICOS (con stop words ES+EN) ===")
for t in tp["resultado_lda"]["df_topicos"].itertuples():
    print(f"  Topico {t.topico}: {t.palabras[:8]}")
print()
print(f"Topicos evitados: {len(tp['topicos_evitados'])}")
for te in tp["topicos_evitados"]:
    kw = te["palabras_clave"][:5]
    dif = te["diferencia"]
    print(f"  T{te['topico']}: {kw}  dif={dif:.2f}")

# Test hostility detection
df_hostil = detectar_escalada_hostil(df_m, df_r)
print(f"\n=== HOSTILIDAD (umbral >=2 patrones) ===")
print(f"Usuarios detectados: {len(df_hostil)} de {len(df_r)} ({len(df_hostil)/len(df_r)*100:.1f}%)")
if len(df_hostil) > 0:
    print(f"Top perfil hostil: {df_hostil['perfil'].value_counts().index[0]}")
    print(f"\nTop 5:")
    for _, u in df_hostil.head(5).iterrows():
        print(f"  @{u['id_usuario'][-16:]:<20s} puntaje={u['puntaje_total']} msgs={u['mensajes_hostiles']} perfil={u['perfil']}")

import sys, os, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles

messages = load_messages("C:\\Users\\JUAN\\Desktop\\Proyectos\\aprobacion-social-chats\\data\\raw\\bluesky_threads_20260619_123556.jsonl")
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_c = classify_profiles(df_f)

# Check the most active user
target = "did:plc:mo5ygs7v7bkqypewszyndc3w"
user = df_c[df_c["id_usuario"] == target]
print(f"Usuario más activo:")
for col in ["handle", "mensajes_totales", "IA", "PA", "RRD", "likes_recibidos", "perfil"]:
    print(f"  {col}: {user[col].values[0]}")

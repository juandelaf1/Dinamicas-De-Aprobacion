import sys, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from predictivo import (
    crear_proxy_churn, entrenar_modelo_riesgo, predecir_riesgo,
    detectar_escalada_hostil, recomendar_intervencion, resumen_predictivo
)
from nlp.analisis_nlp import AnalisisNLP

base = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                    "data", "raw", "bluesky_threads_20260619_123556.jsonl")
messages = load_messages(base)
df_u, df_m, df_i = to_er_dataframes(messages)
df_f = compute_features(df_u, df_m, df_i)
df_r = classify_profiles(df_f)

# NLP features para el modelo
print("Calculando features NLP...")
nlp = AnalisisNLP(df_m, df_r)
nlp.ejecutar_sentimiento()
df_sent = nlp.resultados["sentimiento_detalle"]
# Agregar sentimiento medio por usuario
sent_agg = df_sent.groupby("id_usuario").agg(
    sentimiento_neto_medio=("sentimiento_neto", "mean"),
    intensidad_media=("intensidad", "mean"),
).reset_index()

print("=== 3.1 Churn Proxy ===")
df_churn = crear_proxy_churn(df_r, df_mensajes=df_m)
# Merge con sentimiento
df_churn = df_churn.merge(sent_agg, on="id_usuario", how="left").fillna(0)
print(f"Etiquetas creadas: {df_churn['riesgo_abandono'].sum()} alto riesgo / {len(df_churn)} totales")
print(df_churn["riesgo_abandono"].value_counts().to_string())

print("\n=== 3.1 Modelo Random Forest (sin features circulares) ===")
modelo = entrenar_modelo_riesgo(df_churn, incluir_nlp=True, df_nlp=df_churn)
print(f"Accuracy CV: {modelo['accuracy_cv']:.4f} +/- {modelo['accuracy_std']:.4f}")
print(f"AUC-ROC CV:  {modelo['auc_roc_cv']:.4f}")
print("Feature importance:")
print(modelo["importancia"].head(10).to_string(index=False))

print("\n=== Prediccion de Riesgo ===")
df_riesgo = predecir_riesgo(df_churn, modelo)
print(f"Alertas: {df_riesgo['alerta'].value_counts().to_string()}")
print("\nTop 10 en riesgo:")
for _, u in df_riesgo.head(10).iterrows():
    print(f"  @{u['handle'][-20:]:<22s} riesgo={u['probabilidad_abandono']:.2f} "
          f"alerta={u['alerta']:<8s} perfil={u['perfil']:<10s} IA={u['IA']:.2f} PA={u['PA']:.2f}")

print("\n=== 3.2 Escalada Hostil ===")
df_hostil = detectar_escalada_hostil(df_m, df_r)
if len(df_hostil) > 0:
    print(f"Usuarios con patrones hostiles: {len(df_hostil)}")
    for _, u in df_hostil.head(5).iterrows():
        print(f"  @{u['id_usuario'][-20:]:<22s} puntaje={u['puntaje_total']} "
              f"perfil={u['perfil']:<10s} msgs={u['mensajes_hostiles']}")
else:
    print("No se detectaron patrones hostiles significativos")

print("\n=== 3.3 Recomendacion de Intervencion ===")
df_interv = recomendar_intervencion(df_riesgo, df_r)
if len(df_interv) > 0:
    print(f"{len(df_interv)} usuarios requieren intervencion:")
    for _, u in df_interv.head(10).iterrows():
        print(f"  [{u['prioridad']}] @{u['handle'][-20:]:<22s} -> {u['accion_recomendada']}")
else:
    print("Ningun usuario supera el umbral critico de intervencion")

print("\n" + resumen_predictivo(df_riesgo, modelo, df_hostil, df_interv))

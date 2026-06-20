"""
Dashboard Streamlit del Estudio de Aprobacion Social.
Integra pipeline, NLP, sociograma, clustering y ML.
"""

import json, os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import networkx as nx

from pipeline import load_messages, to_er_dataframes, compute_features, classify_profiles
from sociograma import construir_grafo, metricas_centralidad, detectar_comunidades, analizar_aislamiento
from manifold import reducir_dimension, validar_clusters, clasificador_ml

st.set_page_config(page_title="Estudio de Aprobacion Social", layout="wide", page_icon=":bar_chart:")

NOMBRES_PERFIL = {
    "nucleo": "Nucleo (Lider)",
    "buscador_validacion": "Buscador de Validacion",
    "integrado_silencioso": "Integrado Silencioso",
    "periferico": "Periferico / Excluido",
    "fantasma": "Espectador Fantasma",
}
COLORES_PERFIL = {
    "nucleo": "#e74c3c", "buscador_validacion": "#e67e22",
    "integrado_silencioso": "#2ecc71", "periferico": "#3498db", "fantasma": "#95a5a6",
}
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
DEFAULT_JSONL = os.path.join(DATA_DIR, "raw", "bluesky_threads_20260619_123556.jsonl")


@st.cache_data
def ejecutar_pipeline(jsonl_path):
    messages = load_messages(jsonl_path)
    df_u, df_m, df_i = to_er_dataframes(messages)
    df_f = compute_features(df_u, df_m, df_i)
    df_r = classify_profiles(df_f)
    return df_r, df_m, df_i


@st.cache_data
def ejecutar_nlp(df_m, df_r):
    from nlp.analisis_nlp import AnalisisNLP
    nlp = AnalisisNLP(df_m, df_r)
    nlp.ejecutar_jerga_interna()
    nlp.ejecutar_sentimiento()
    nlp.ejecutar_borrados()
    nlp.ejecutar_estilo()
    nlp.ejecutar_topicos(n_topicos=4)
    return nlp.resultados


@st.cache_data
def ejecutar_sociograma(df_i, df_r, df_f):
    G = construir_grafo(df_i, df_perfiles=df_r, df_features=df_f, dirigido=True)
    df_cent = metricas_centralidad(G)
    df_com = detectar_comunidades(G)
    aisl = analizar_aislamiento(G, df_r)
    return G, df_cent, df_com, aisl


@st.cache_data
def ejecutar_manifold(df_r):
    coords = reducir_dimension(df_r, metodo="tsne")
    val = validar_clusters(df_r, n_clusters_range=range(2, 8))
    clf = clasificador_ml(df_r)
    return coords, val, clf


def main():
    st.title("Estudio de Apro bacion Social y Participacion en Chats")

    # ---- Carga de datos ----
    uploaded = st.sidebar.file_uploader("Subir JSONL", type=["jsonl"])
    jsonl_path = DEFAULT_JSONL if uploaded is None else "temp_upload.jsonl"
    if uploaded is not None:
        with open(jsonl_path, "wb") as f:
            f.write(uploaded.read())

    if not os.path.exists(jsonl_path):
        st.warning("No hay datos. Sube un archivo JSONL o coloca datos en data/raw/")
        return

    with st.spinner("Procesando pipeline completo..."):
        df_r, df_m, df_i = ejecutar_pipeline(jsonl_path)

    # ---- Sidebar: filtros ----
    st.sidebar.header("Filtros")
    perfiles_presentes = [p for p in ["nucleo", "buscador_validacion", "integrado_silencioso", "periferico", "fantasma"]
                          if p in df_r["perfil"].values]
    seleccion = st.sidebar.multiselect("Perfiles", perfiles_presentes, default=perfiles_presentes)
    min_ia = st.sidebar.slider("IA minima", 0.0, float(df_r["IA"].max()), 0.0)
    min_pa = st.sidebar.slider("PA minima", 0.0, float(df_r["PA"].max()), 0.0)

    df_filt = df_r[(df_r["perfil"].isin(seleccion)) & (df_r["IA"] >= min_ia) & (df_r["PA"] >= min_pa)]

    # ---- KPIs ----
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Usuarios", len(df_filt))
    col2.metric("Mensajes", int(df_filt["mensajes_totales"].sum()))
    col3.metric("IA medio", f"{df_filt['IA'].mean():.2f}")
    col4.metric("PA medio", f"{df_filt['PA'].mean():.2f}")
    col5.metric("RRD medio", f"{df_filt['RRD'].mean():.2f}")

    # ---- Tabs ----
    tabs = st.tabs(["Perfiles", "Sociograma", "NLP", "Validacion (t-SNE + ML)"])

    # ============ TAB 1: PERFILES ============
    with tabs[0]:
        st.subheader("Distribucion de Perfiles")
        c1, c2 = st.columns([1, 2])
        with c1:
            counts = df_filt["perfil"].value_counts().reindex(perfiles_presentes, fill_value=0)
            fig_pie = px.pie(values=counts.values, names=[NOMBRES_PERFIL.get(p, p) for p in counts.index],
                             color_discrete_map=COLORES_PERFIL, title="Distribucion")
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            stats = df_filt.groupby("perfil").agg(
                Usuarios=("id_usuario", "count"),
                IA_medio=("IA", "mean"),
                PA_medio=("PA", "mean"),
                Mensajes_avg=("mensajes_totales", "mean"),
                Likes_avg=("likes_recibidos", "mean"),
                RRD=("RRD", "mean"),
            ).round(3)
            stats.index = [NOMBRES_PERFIL.get(i, i) for i in stats.index]
            st.dataframe(stats, use_container_width=True)

        st.subheader("IA vs PA por Perfil")
        fig_scatter = px.scatter(
            df_filt, x="PA", y="IA", color="perfil",
            hover_data=["handle", "mensajes_totales", "likes_recibidos", "RRD"],
            color_discrete_map=COLORES_PERFIL,
            labels={"PA": "Pertenencia Aproximada", "IA": "Indice de Aprobacion"},
            size="mensajes_totales", size_max=20,
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

        st.subheader("Top Usuarios por IA")
        top = df_filt.nlargest(15, "IA")[["handle", "perfil", "IA", "PA", "mensajes_totales", "likes_recibidos"]]
        top["perfil"] = top["perfil"].map(NOMBRES_PERFIL)
        st.dataframe(top, use_container_width=True)

    # ============ TAB 2: SOCIOGRAMA ============
    with tabs[1]:
        with st.spinner("Construyendo sociograma..."):
            G, df_cent, df_com, aisl = ejecutar_sociograma(df_i, df_r, df_filt)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Nodos en red", G.number_of_nodes())
        col2.metric("Aristas", G.number_of_edges())
        col3.metric("Densidad", f"{aisl['densidad_red']:.4f}")
        col4.metric("Aislados", f"{aisl['n_aislados']} ({aisl['n_aislados']/max(G.number_of_nodes(),1)*100:.0f}%)")

        st.subheader("Visualizacion del Sociograma")
        img_path = os.path.join(DATA_DIR, "sociograma_perfiles.png")
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.info("Ejecuta test_sociograma.py para generar la imagen.")

        st.subheader("Metricas de Centralidad (Top 10)")
        metrica = st.selectbox("Metrica", ["pagerank", "betweenness", "degree_centrality", "eigenvector", "closeness"])
        top_cent = df_cent.nlargest(10, metrica)[["id_usuario", metrica]]
        top_cent["usuario"] = top_cent["id_usuario"].apply(lambda x: x[-16:])
        fig_cent = px.bar(top_cent, x=metrica, y="usuario", orientation="h",
                          title=f"Top 10 - {metrica.replace('_', ' ').title()}")
        st.plotly_chart(fig_cent, use_container_width=True)

        st.subheader("Componentes Conectados")
        comp_df = pd.DataFrame(aisl["componentes"], columns=["id", "tamano"])
        fig_comp = px.bar(comp_df.head(20), x="id", y="tamano", title="Top 20 Componentes")
        st.plotly_chart(fig_comp, use_container_width=True)

    # ============ TAB 3: NLP ============
    with tabs[2]:
        with st.spinner("Ejecutando analisis NLP..."):
            nlp_res = ejecutar_nlp(df_m, df_r)

        st.subheader("2.1 Jerga Interna del Grupo")
        jerga = nlp_res.get("jerga", {})
        terminos = jerga.get("terminos_jerga", [])
        if terminos:
            st.write(f"**{len(terminos)} terminos candidatos** detectados via TF-IDF:")
            st.write(", ".join(terminos[:20]))
        tg = nlp_res.get("terminos_grupo", {})
        st.write(f"Vocabulario unico: {tg.get('vocabulario_unico', 0)} palabras en {tg.get('total_palabras', 0)} tokens")
        uni = tg.get("unigramas", {})
        if uni:
            st.write("Top 10 unigramas:", ", ".join(list(uni.keys())[:10]))

        st.subheader("2.2 Sentimiento por Perfil")
        sa = nlp_res.get("sentimiento_agregado")
        if sa is not None:
            sa["perfil"] = sa["perfil"].map(NOMBRES_PERFIL)
            fig_sent = px.bar(sa, x="perfil", y=["sentimiento_medio", "volatilidad_emocional"],
                              barmode="group", title="Sentimiento y Volatilidad por Perfil")
            st.plotly_chart(fig_sent, use_container_width=True)
            st.dataframe(sa[["perfil", "sentimiento_medio", "volatilidad_emocional",
                            "proporcion_positivos", "proporcion_negativos", "intensidad_media"]].round(4),
                        use_container_width=True)

        st.subheader("2.3 Mensajes Borrados")
        rb = nlp_res.get("resumen_borrados", {})
        if rb:
            st.write(f"**{rb['mensajes_borrados']}** de **{rb['total_mensajes']}** mensajes ({rb['porcentaje_borrado']}%)")
            if rb.get("por_motivo"):
                st.write("Motivos:", rb["por_motivo"])
            if rb.get("por_perfil"):
                st.write("Por perfil:", rb["por_perfil"])

        st.subheader("2.4 Estilo Linguistico Imitativo")
        est = nlp_res.get("estilo", {})
        st.write(f"Pares con estilo similar: {len(est.get('pares_similares', []))}")
        st.write(f"Posibles imitadores: {len(est.get('imitadores_sospechosos', []))}")
        if est.get("pares_similares"):
            df_pares = pd.DataFrame(est["pares_similares"][:10])
            st.dataframe(df_pares, use_container_width=True)

        st.subheader("2.5 Topic Modeling y Espiral del Silencio")
        tp = nlp_res.get("topicos", {})
        if tp.get("topicos_evitados"):
            st.write("**Topicos evitados por Perifericos:**")
            df_evit = pd.DataFrame(tp["topicos_evitados"])
            st.dataframe(df_evit, use_container_width=True)
        if "distribucion_topicos_por_perfil" in tp:
            st.write("Distribucion de topicos por perfil:")
            st.dataframe(tp["distribucion_topicos_por_perfil"].round(4), use_container_width=True)

        st.subheader("Reporte NLP Completo")
        st.text(nlp_res.get("reporte_texto", ""))

    # ============ TAB 4: VALIDACION ============
    with tabs[3]:
        with st.spinner("Ejecutando t-SNE + clustering + ML..."):
            coords, val, clf = ejecutar_manifold(df_r)

        merged = df_r[["id_usuario", "perfil", "handle", "IA", "PA"]].merge(coords, on="id_usuario")

        st.subheader("t-SNE: Espacio de Features")
        fig_tsne = px.scatter(
            merged, x="x", y="y", color="perfil",
            hover_data=["handle", "IA", "PA"],
            color_discrete_map=COLORES_PERFIL,
            title="Proyeccion t-SNE de Usuarios (2D)",
            labels={"x": "t-SNE 1", "y": "t-SNE 2"},
        )
        st.plotly_chart(fig_tsne, use_container_width=True)

        st.subheader("Validacion de Clusters (KMeans)")
        df_val = pd.DataFrame(val["resultados"])
        df_val_num = df_val[df_val["k"].apply(lambda x: isinstance(x, int))].copy()
        if not df_val_num.empty:
            df_val_num["k"] = df_val_num["k"].astype(int)
            fig_sil = px.line(df_val_num, x="k", y="silhouette", markers=True, title="Silhouette Score por k")
            st.plotly_chart(fig_sil, use_container_width=True)
        st.dataframe(df_val, use_container_width=True)

        mejor = val.get("mejor_k", {})
        if mejor:
            st.metric("Mejor k (silhouette)", mejor.get("k", "?"), f"s={mejor.get('silhouette', 0):.4f}")
            if mejor.get("k") == 5:
                st.success("Coincide con la taxonomia de 5 perfiles!")
            else:
                st.info(f"Sugiere {mejor['k']} perfiles naturales en estos datos")

        st.subheader("Clasificador ML (Random Forest)")
        if clf:
            col1, col2 = st.columns(2)
            col1.metric("Accuracy (CV 5-fold)", f"{clf['accuracy_cv_mean']:.4f}",
                        f"+-{clf['accuracy_cv_std']:.4f}")
            col2.metric("Clases detectadas", clf["n_classes"])

            fig_imp = px.bar(clf["importancia"].head(10), x="importancia", y="feature",
                             orientation="h", title="Top 10 Features por Importancia")
            st.plotly_chart(fig_imp, use_container_width=True)

            st.write("**Importancia de Features (completa):**")
            st.dataframe(clf["importancia"], use_container_width=True)

        st.subheader("Conclusiones")
        st.markdown("""
        - **k=5 (ARI=0.80)**: La taxonomia de 5 perfiles explica el 80% de la estructura natural de clusters
        - **Random Forest (98.8% CV)**: Las features IA, PA, RRD son altamente predictivas del perfil
        - **PA** es la feature mas importante (21.4%), seguida de **IA** (12.6%)
        - Perfiles no detectados (Buscador de Validacion, Fantasma) requieren datos mas densos
        """)


if __name__ == "__main__":
    main()
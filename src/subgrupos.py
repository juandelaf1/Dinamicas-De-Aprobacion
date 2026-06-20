import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations


def detectar_subgrupos_excluyentes(
    G: nx.Graph,
    df_comunidades: pd.DataFrame = None,
    umbral_exclusividad: float = 3.0,
) -> pd.DataFrame:
    """
    Detecta subgrupos (comunidades) con comportamiento excluyente.

    Un subgrupo es excluyente si:
    - Alta densidad interna (se hablan mucho entre si)
    - Baja densidad externa (ignoran a los demas)
    - Ratio exclusividad = densidad_interna / densidad_externa > umbral

    Retorna DataFrame con metricas por comunidad.
    """
    undirected = G if not isinstance(G, nx.DiGraph) else G.to_undirected()

    if df_comunidades is not None:
        comunidades = df_comunidades.groupby("comunidad")["id_usuario"].apply(list).to_dict()
    else:
        from networkx.algorithms.community import louvain_communities
        try:
            comms = louvain_communities(undirected, weight="weight")
        except Exception:
            from networkx.algorithms.community import greedy_modularity_communities
            comms = greedy_modularity_communities(undirected, weight="weight")
        comunidades = {i: list(c) for i, c in enumerate(comms)}

    total_nodos = undirected.number_of_nodes()

    rows = []
    for cid, miembros in comunidades.items():
        tam = len(miembros)
        if tam < 2:
            continue

        subgraph = undirected.subgraph(miembros)
        aristas_internas = subgraph.number_of_edges()
        posibles_internas = tam * (tam - 1) / 2
        densidad_interna = aristas_internas / posibles_internas if posibles_internas > 0 else 0

        # Aristas hacia afuera
        aristas_externas = 0
        for n in miembros:
            for vecino in undirected.neighbors(n):
                if vecino not in miembros:
                    aristas_externas += 1
        # Normalizar
        nodos_externos = total_nodos - tam
        posibles_externas = tam * nodos_externos
        densidad_externa = aristas_externas / posibles_externas if posibles_externas > 0 else 0

        # Ratio de exclusividad
        ratio = densidad_interna / densidad_externa if densidad_externa > 0 else float("inf")

        # Centralidad media del subgrupo
        try:
            pr = nx.pagerank(undirected, weight="weight")
            centralidad_media = np.mean([pr.get(n, 0) for n in miembros])
        except Exception:
            centralidad_media = 0

        rows.append({
            "comunidad": cid,
            "miembros": tam,
            "aristas_internas": aristas_internas,
            "aristas_externas": aristas_externas,
            "densidad_interna": round(densidad_interna, 4),
            "densidad_externa": round(densidad_externa, 6),
            "ratio_exclusividad": round(ratio, 2) if ratio != float("inf") else None,
            "centralidad_media": round(centralidad_media, 6),
            "excluyente": ratio > umbral_exclusividad if ratio != float("inf") else True,
        })

    return pd.DataFrame(rows).sort_values("ratio_exclusividad", ascending=False, na_position="last")


def usuarios_aislados_en_subgrupo(
    G: nx.Graph,
    df_comunidades: pd.DataFrame,
    df_perfiles: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Identifica usuarios que estan aislados DENTRO de su propia comunidad.
    Reciben poca o ninguna atencion de miembros de su mismo subgrupo.

    Esto detecta "marginados internos": pertenecen al grupo pero
    son ignorados incluso por su propia comunidad.
    """
    undirected = G if not isinstance(G, nx.DiGraph) else G.to_undirected()

    comunidad_map = {}
    for _, row in df_comunidades.iterrows():
        comunidad_map[row["id_usuario"]] = row["comunidad"]

    rows = []
    for nodo in undirected.nodes():
        cid = comunidad_map.get(nodo)
        if cid is None:
            continue

        # Vecinos en la misma comunidad
        vecinos = list(undirected.neighbors(nodo))
        vecinos_misma_comunidad = [v for v in vecinos if comunidad_map.get(v) == cid]

        # Grado total vs grado intra-comunidad
        grado_total = len(vecinos)
        grado_intra = len(vecinos_misma_comunidad)

        # Si tiene grado total > 0 pero todo es hacia afuera
        if grado_total > 0 and grado_intra == 0:
            rows.append({
                "id_usuario": nodo,
                "comunidad": cid,
                "grado_total": grado_total,
                "grado_intra_comunidad": grado_intra,
                "tipo_aislamiento": "ignorado_por_su_comunidad",
            })
        elif grado_total == 0:
            rows.append({
                "id_usuario": nodo,
                "comunidad": cid,
                "grado_total": 0,
                "grado_intra_comunidad": 0,
                "tipo_aislamiento": "totalmente_aislado",
            })

    df_aislados = pd.DataFrame(rows)

    if df_perfiles is not None and len(df_aislados) > 0:
        perfil_map = df_perfiles.set_index("id_usuario")["perfil"].to_dict()
        df_aislados["perfil"] = df_aislados["id_usuario"].map(perfil_map)

    return df_aislados


def puentes_entre_comunidades(
    G: nx.Graph,
    df_comunidades: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identifica usuarios que actuan como puentes entre comunidades.
    Tienen aristas hacia 2+ comunidades diferentes.

    Los puentes son cruciales para la cohesion del grupo:
    - Mantienen la integracion
    - Su ausencia fragmenta la red
    """
    undirected = G if not isinstance(G, nx.DiGraph) else G.to_undirected()

    comunidad_map = {}
    for _, row in df_comunidades.iterrows():
        comunidad_map[row["id_usuario"]] = row["comunidad"]

    rows = []
    for nodo in undirected.nodes():
        cid_origen = comunidad_map.get(nodo)
        if cid_origen is None:
            continue

        vecinos = list(undirected.neighbors(nodo))
        comunidades_vecinas = set()
        for v in vecinos:
            cid = comunidad_map.get(v)
            if cid is not None and cid != cid_origen:
                comunidades_vecinas.add(cid)

        if len(comunidades_vecinas) > 0:
            rows.append({
                "id_usuario": nodo,
                "comunidad_origen": cid_origen,
                "comunidades_conectadas": len(comunidades_vecinas),
                "ids_comunidades": sorted(comunidades_vecinas),
                "vecinos_totales": len(vecinos),
                "vecinos_externos": len(comunidades_vecinas),
            })

    if not rows:
        return pd.DataFrame(columns=["id_usuario", "comunidad_origen", "comunidades_conectadas",
                                     "vecinos_totales", "vecinos_externos", "betweenness"])

    df_puentes = pd.DataFrame(rows).sort_values("comunidades_conectadas", ascending=False)

    # Betweenness como medida de importancia del puente
    try:
        betweenness = nx.betweenness_centrality(undirected, weight="weight", k=min(500, len(undirected)))
        df_puentes["betweenness"] = df_puentes["id_usuario"].map(betweenness).fillna(0)
    except Exception:
        df_puentes["betweenness"] = 0

    return df_puentes


def resumen_subgrupos(
    df_excluyentes: pd.DataFrame,
    df_aislados: pd.DataFrame,
    df_puentes: pd.DataFrame,
    G: nx.Graph = None,
) -> str:
    """Reporte textual del analisis de subgrupos."""
    lines = []
    lines.append("=" * 60)
    lines.append("SUBGRUPOS EXCLUYENTES - Analisis de Dinamicas de Grupo")
    lines.append("=" * 60)

    # Subgrupos excluyentes
    excl = df_excluyentes[df_excluyentes["excluyente"] == True]
    lines.append(f"\n[Subgrupos Excluyentes] {len(excl)} comunidades detectadas:")
    for _, row in excl.head(10).iterrows():
        lines.append(f"  Comunidad {row['comunidad']}: {row['miembros']} miembros, "
                     f"densidad_interna={row['densidad_interna']:.3f}, "
                     f"ratio_excl={row['ratio_exclusividad']}")

    # No excluyentes
    no_excl = df_excluyentes[df_excluyentes["excluyente"] == False]
    lines.append(f"\n[Comunidades Integradas] {len(no_excl)} con baja exclusividad:")
    for _, row in no_excl.head(5).iterrows():
        lines.append(f"  Comunidad {row['comunidad']}: {row['miembros']} miembros, "
                     f"ratio={row['ratio_exclusividad']}")

    # Aislados internos
    if len(df_aislados) > 0:
        lines.append(f"\n[Marginados Internos] {len(df_aislados)} usuarios:")
        por_tipo = df_aislados["tipo_aislamiento"].value_counts().to_dict()
        lines.append(f"  Tipos: {por_tipo}")
        if df_perfiles_available(df_aislados):
            por_perfil = df_aislados["perfil"].value_counts().to_dict()
            lines.append(f"  Por perfil: {por_perfil}")

    # Puentes
    if len(df_puentes) > 0:
        lines.append(f"\n[Puentes entre Comunidades] {len(df_puentes)} usuarios conectan grupos:")
        top_puentes = df_puentes.head(5)
        for _, row in top_puentes.iterrows():
            lines.append(f"  @{row['id_usuario'][-16:]:<20s} conecta {row['comunidades_conectadas']} comunidades, "
                         f"betweenness={row.get('betweenness', 0):.4f}")

    # Resumen general
    if G is not None:
        undirected = G if not isinstance(G, nx.DiGraph) else G.to_undirected()
        n_components = nx.number_connected_components(undirected)
        lines.append(f"\n[Resumen de Red]")
        lines.append(f"  Componentes conectados: {n_components}")
        lines.append(f"  Comunidades detectadas: {len(df_excluyentes)}")
        lines.append(f"  Subgrupos excluyentes: {len(excl)}")
        lines.append(f"  Puentes entre grupos: {len(df_puentes)}")
        lines.append(f"  Usuarios marginados: {len(df_aislados)}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def df_perfiles_available(df):
    return "perfil" in df.columns and len(df) > 0

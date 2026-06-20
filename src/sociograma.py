import pandas as pd
import numpy as np
import networkx as nx
from itertools import combinations


def construir_grafo(
    df_interacciones: pd.DataFrame,
    df_perfiles: pd.DataFrame = None,
    df_features: pd.DataFrame = None,
    dirigido: bool = True,
) -> nx.Graph:
    """
    Construye el grafo de interacciones a partir de las respuestas.

    Nodos = usuarios. Aristas = reply de un usuario a otro.
    Peso = cantidad de replies entre el par.

    Si dirigido=True, grafo dirigido (quien responde a quien).
    Si dirigido=False, grafo no dirigido (interaccion reciproca).
    """
    if dirigido:
        G = nx.DiGraph()
    else:
        G = nx.Graph()

    # Agregar todos los usuarios como nodos (incluso sin aristas)
    if df_perfiles is not None:
        for _, row in df_perfiles.iterrows():
            G.add_node(row["id_usuario"])
    elif df_features is not None:
        for _, row in df_features.iterrows():
            G.add_node(row["id_usuario"])

    if len(df_interacciones) > 0:
        # Agregar aristas con peso = conteo de interacciones
        interacciones_agg = (
            df_interacciones.groupby(["id_usuario_origen", "id_usuario_destino"])
            .size()
            .reset_index(name="peso")
        )

        for _, row in interacciones_agg.iterrows():
            G.add_edge(
                row["id_usuario_origen"],
                row["id_usuario_destino"],
                weight=row["peso"],
            )

    # Agregar atributos de perfil
    if df_perfiles is not None:
        perfil_map = df_perfiles.set_index("id_usuario").to_dict("index")
        for node in G.nodes():
            attrs = perfil_map.get(node, {})
            if attrs:
                G.nodes[node].update(attrs)

    # Agregar features de usuario
    if df_features is not None:
        feat_map = df_features.set_index("id_usuario").to_dict("index")
        for node in G.nodes():
            attrs = feat_map.get(node, {})
            if attrs:
                G.nodes[node].update(attrs)

    return G


def metricas_centralidad(G: nx.Graph) -> pd.DataFrame:
    """
    Calcula metricas de centralidad para todos los nodos del grafo.

    Retorna DataFrame con:
        - degree_centrality
        - betweenness_centrality
        - closeness_centrality (solo para grafos conectados)
        - eigenvector_centrality
        - pagerank
        - (en grafos dirigidos) in_degree, out_degree
    """
    metricas = {}

    # Degree (in/out si dirigido)
    if isinstance(G, nx.DiGraph):
        in_deg = dict(G.in_degree(weight="weight"))
        out_deg = dict(G.out_degree(weight="weight"))
        total_deg = {n: in_deg.get(n, 0) + out_deg.get(n, 0) for n in G.nodes()}
        metricas["in_degree"] = in_deg
        metricas["out_degree"] = out_deg
        metricas["total_degree"] = total_deg
    else:
        metricas["degree"] = dict(G.degree(weight="weight"))

    # Centralidad de grado normalizada
    try:
        metricas["degree_centrality"] = nx.degree_centrality(G)
    except Exception:
        pass

    # Betweenness
    try:
        # Usar muestra si el grafo es grande (>500 nodos)
        k = min(500, len(G))
        metricas["betweenness"] = nx.betweenness_centrality(
            G, weight="weight", k=k if k < len(G) else None
        )
    except Exception:
        pass

    # Closeness (solo componente gigante si hay varios componentes)
    try:
        # Si hay multiples componentes, calcular solo en el principal
        if nx.is_connected(G if not isinstance(G, nx.DiGraph) else G.to_undirected()):
            metricas["closeness"] = nx.closeness_centrality(G)
        else:
            # Componente gigante
            undirected = G if not isinstance(G, nx.DiGraph) else G.to_undirected()
            largest = max(nx.connected_components(undirected), key=len)
            subgraph = G.subgraph(largest)
            closeness = nx.closeness_centrality(subgraph)
            metricas["closeness"] = {n: closeness.get(n, 0) for n in G.nodes()}
    except Exception:
        pass

    # Eigenvector (solo componente gigante, grafo no dirigido)
    try:
        undirected = G if not isinstance(G, nx.DiGraph) else G.to_undirected()
        largest = max(nx.connected_components(undirected), key=len)
        subgraph = undirected.subgraph(largest)
        eig = nx.eigenvector_centrality_numpy(subgraph, weight="weight")
        metricas["eigenvector"] = {n: eig.get(n, 0) for n in G.nodes()}
    except Exception:
        pass

    # PageRank
    try:
        pr = nx.pagerank(G, weight="weight")
        metricas["pagerank"] = pr
    except Exception:
        pass

    # Convertir a DataFrame
    df = pd.DataFrame(metricas).fillna(0)
    df.index.name = "id_usuario"
    df = df.reset_index()
    return df


def detectar_comunidades(G: nx.Graph, resolucion: float = 1.0) -> pd.DataFrame:
    """
    Detecta comunidades en el grafo usando Louvain (greedy modularity).

    Retorna DataFrame con {id_usuario, comunidad}.
    """
    # Convertir a no dirigido si es necesario
    if isinstance(G, nx.DiGraph):
        H = G.to_undirected()
    else:
        H = G.copy()

    if len(H) < 2:
        return pd.DataFrame({"id_usuario": list(H.nodes()), "comunidad": [0] * len(H)})

    try:
        from networkx.algorithms.community import louvain_communities

        communities = louvain_communities(H, weight="weight", resolution=resolucion)
        comunidad_map = {}
        for idx, miembros in enumerate(communities):
            for m in miembros:
                comunidad_map[m] = idx

        return pd.DataFrame(
            {"id_usuario": list(comunidad_map.keys()),
             "comunidad": list(comunidad_map.values())}
        )
    except ImportError:
        # Fallback: greedy modularity
        from networkx.algorithms.community import greedy_modularity_communities

        communities = greedy_modularity_communities(H, weight="weight")
        comunidad_map = {}
        for idx, miembros in enumerate(communities):
            for m in miembros:
                comunidad_map[m] = idx

        return pd.DataFrame(
            {"id_usuario": list(comunidad_map.keys()),
             "comunidad": list(comunidad_map.values())}
        )


def analizar_aislamiento(
    G: nx.Graph,
    df_perfiles: pd.DataFrame = None,
    col_perfil: str = "perfil",
) -> dict:
    """
    Analiza el aislamiento de nodos en el grafo.

    Retorna:
        - nodos_aislados: lista de nodos sin aristas
        - componentes: lista de componentes conectados
        - perifericos_aislados: nodos perifericos que estan aislados
        - densidad_red: densidad del grafo (completa / observada)
        - coeficiente_clustering: transitivity global
    """
    undirected = G if not isinstance(G, nx.DiGraph) else G.to_undirected()

    # Nodos aislados (grado 0)
    aislados = [n for n in undirected.nodes() if undirected.degree(n) == 0]

    # Componentes conectados
    componentes = list(nx.connected_components(undirected))
    componentes.sort(key=len, reverse=True)

    # Tamaño de componentes
    size_componentes = [(i, len(c)) for i, c in enumerate(componentes)]

    # Perifericos aislados
    perifericos_aislados = []
    if df_perfiles is not None:
        perfil_map = df_perfiles.set_index("id_usuario")[col_perfil].to_dict()
        for n in aislados:
            perfil = perfil_map.get(n, "desconocido")
            if perfil == "periferico":
                perifericos_aislados.append({
                    "id_usuario": n,
                    "perfil": perfil,
                })

    # Densidad
    densidad = nx.density(undirected)

    # Coeficiente de clustering global
    try:
        clustering = nx.transitivity(undirected)
    except Exception:
        clustering = 0.0

    return {
        "nodos_aislados": aislados,
        "n_aislados": len(aislados),
        "componentes": size_componentes,
        "n_componentes": len(componentes),
        "perifericos_aislados": perifericos_aislados,
        "densidad_red": round(densidad, 6),
        "coeficiente_clustering": round(clustering, 6),
    }


def resumen_sociograma(G: nx.Graph, df_centralidad: pd.DataFrame, aislamiento: dict) -> str:
    """Genera reporte textual del analisis de sociograma."""
    lines = []
    lines.append("=" * 60)
    lines.append("SOCI OGRAMA - Analisis de Red de Interacciones")
    lines.append("=" * 60)
    lines.append(f"\n  Nodos: {G.number_of_nodes()}")
    lines.append(f"  Aristas: {G.number_of_edges()}")
    lines.append(f"  Densidad: {aislamiento['densidad_red']}")
    lines.append(f"  Clustering: {aislamiento['coeficiente_clustering']}")

    if isinstance(G, nx.DiGraph):
        lines.append(f"  Componentes: {aislamiento['n_componentes']}")

    lines.append(f"\n  Componentes conectados: {aislamiento['n_componentes']}")
    for i, (idx, tam) in enumerate(aislamiento["componentes"][:5]):
        label = " (gigante)" if i == 0 else ""
        lines.append(f"    Componente {idx}: {tam} nodos{label}")

    lines.append(f"\n  Nodos aislados: {aislamiento['n_aislados']} ({aislamiento['n_aislados']/G.number_of_nodes()*100:.1f}%)")
    if aislamiento["perifericos_aislados"]:
        lines.append(f"  Perifericos aislados: {len(aislamiento['perifericos_aislados'])}")

    if df_centralidad is not None and len(df_centralidad) > 0:
        lines.append("\n  Top 5 PageRank:")
        top_pr = df_centralidad.nlargest(5, "pagerank")
        for _, row in top_pr.iterrows():
            lines.append(f"    @{row['id_usuario'][-16:]:<20s} PR={row['pagerank']:.4f}")

        lines.append("\n  Top 5 Betweenness:")
        top_btw = df_centralidad.nlargest(5, "betweenness")
        for _, row in top_btw.iterrows():
            lines.append(f"    @{row['id_usuario'][-16:]:<20s} BTW={row['betweenness']:.4f}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)

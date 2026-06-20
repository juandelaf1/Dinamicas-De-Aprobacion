import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import os


COLORES_PERFIL = {
    "nucleo": "#e74c3c",
    "buscador_validacion": "#e67e22",
    "integrado_silencioso": "#2ecc71",
    "periferico": "#3498db",
    "fantasma": "#95a5a6",
    "desconocido": "#bdc3c7",
}


def visualizar_sociograma(
    G: nx.Graph,
    output_path: str,
    colorear_por: str = "perfil",
    resaltar_aislados: bool = True,
    figsize=(16, 12),
    dpi: int = 150,
    mostrar_etiquetas: bool = False,
) -> str:
    """
    Genera visualizacion del sociograma con:
    - Layout ForceAtlas2 (spring_layout)
    - Nodos coloreados por perfil
    - Tamaño de nodo proporcional a PageRank
    - Aristas con transparencia segun peso
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if G.number_of_nodes() == 0:
        ax.text(0.5, 0.5, "Grafo vacio", ha="center", va="center", fontsize=14)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # Layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    # Colores de nodos
    if colorear_por == "perfil":
        colores = []
        for node in G.nodes():
            perfil = G.nodes[node].get("perfil", "desconocido")
            colores.append(COLORES_PERFIL.get(perfil, COLORES_PERFIL["desconocido"]))
    else:
        colores = ["#3498db"] * G.number_of_nodes()

    # Tamaño de nodos (proporcional a PageRank o degree)
    try:
        pr = nx.pagerank(G, weight="weight")
        tamanos = [300 + pr.get(n, 0) * 5000 for n in G.nodes()]
    except Exception:
        tamanos = [300] * G.number_of_nodes()

    # Aristas
    aristas = G.edges(data=True)
    if aristas:
        pesos = [d.get("weight", 1) for _, _, d in aristas]
        max_peso = max(pesos) if pesos else 1
        widths = [0.5 + (w / max_peso) * 3 for w in pesos]
        alphas = [0.1 + (w / max_peso) * 0.5 for w in pesos]

        nx.draw_networkx_edges(
            G, pos, ax=ax,
            width=widths,
            alpha=alphas,
            edge_color="#7f8c8d",
            arrows=isinstance(G, nx.DiGraph),
            arrowstyle="->",
            arrowsize=10,
        )

    # Nodos
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colores,
        node_size=tamanos,
        edgecolors="white",
        linewidths=0.5,
        alpha=0.85,
    )

    # Etiquetas (opcional)
    if mostrar_etiquetas:
        labels = {}
        for node in G.nodes():
            handle = G.nodes[node].get("handle", "")
            labels[node] = f"@{handle}" if handle else node[-12:]
        nx.draw_networkx_labels(
            G, pos, ax=ax,
            labels=labels,
            font_size=5,
            alpha=0.7,
        )

    # Resaltar aislados con borde punteado
    if resaltar_aislados:
        undirected = G if not isinstance(G, nx.DiGraph) else G.to_undirected()
        aislados = [n for n in undirected.nodes() if undirected.degree(n) == 0]
        aislados_con_pos = [n for n in aislados if n in pos]
        if aislados_con_pos:
            nx.draw_networkx_nodes(
                G, pos, ax=ax,
                nodelist=aislados_con_pos,
                node_color="none",
                edgecolors="#e74c3c",
                linewidths=2,
                node_size=[tamanos[list(G.nodes()).index(n)] for n in aislados_con_pos],
            )

    # Leyenda
    from matplotlib.patches import Patch
    legend_elements = []
    for perfil, color in COLORES_PERFIL.items():
        # Verificar si este perfil existe en el grafo
        perfiles_en_grafo = set()
        for node in G.nodes():
            perfiles_en_grafo.add(G.nodes[node].get("perfil", "desconocido"))
        if perfil in perfiles_en_grafo or perfil == "desconocido":
            legend_elements.append(
                Patch(facecolor=color, edgecolor="white", label=perfil.replace("_", " ").title())
            )
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8)

    ax.set_title("Sociograma de Interacciones", fontsize=14, fontweight="bold")
    ax.axis("off")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def visualizar_centralidad(
    df_centralidad: pd.DataFrame,
    output_path: str,
    top_n: int = 15,
    metrica: str = "pagerank",
    figsize=(10, 6),
    dpi: int = 150,
) -> str:
    """
    Grafico de barras de la metrica de centralidad top N.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if metrica not in df_centralidad.columns:
        ax.text(0.5, 0.5, f"Columna '{metrica}' no encontrada", ha="center", va="center")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    top = df_centralidad.nlargest(top_n, metrica)
    handles = [h[-12:] for h in top["id_usuario"].tolist()]
    valores = top[metrica].tolist()

    bars = ax.barh(range(len(handles)), valores, color="#3498db", edgecolor="white")
    ax.set_yticks(range(len(handles)))
    ax.set_yticklabels(handles, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metrica.replace("_", " ").title())
    ax.set_title(f"Top {top_n} - {metrica.replace('_', ' ').title()}", fontsize=12)

    for bar, val in zip(bars, valores):
        ax.text(val + max(valores) * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path


def visualizar_comunidades(
    G: nx.Graph,
    df_comunidades: pd.DataFrame,
    output_path: str,
    figsize=(14, 10),
    dpi: int = 150,
) -> str:
    """
    Visualiza el grafo coloreado por comunidad detectada.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if G.number_of_nodes() == 0 or df_comunidades.empty:
        ax.text(0.5, 0.5, "Sin datos", ha="center", va="center")
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        return output_path

    comunidad_map = df_comunidades.set_index("id_usuario")["comunidad"].to_dict()
    n_com = df_comunidades["comunidad"].nunique()

    # Colores por comunidad
    cmap = plt.cm.get_cmap("tab20", n_com)
    colores = []
    for node in G.nodes():
        cid = comunidad_map.get(node, -1)
        if cid >= 0:
            colores.append(cmap(cid % 20))
        else:
            colores.append("#bdc3c7")

    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)

    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.15, edge_color="#7f8c8d")

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=colores,
        node_size=200,
        edgecolors="white",
        linewidths=0.3,
        alpha=0.8,
    )

    ax.set_title(f"Comunidades Detectadas ({n_com} grupos)", fontsize=14)
    ax.axis("off")

    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    return output_path

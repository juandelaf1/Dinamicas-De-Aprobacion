"""
Reddit/Pushshift extractor.
Extrae comentarios de subreddits via Pushshift API (sin auth).
Los subreddits funcionan como grupos de interes con:
- reply chains (conversaciones)
- upvotes (aprobacion social)
- usuarios recurrentes
"""

import json
import time
import os
from datetime import datetime, timezone

import requests

PUSHSHIFT_BASE = "https://api.pullpush.io/reddit"
USER_AGENT = "aprobacion-social/0.2.0"
RATE_LIMIT_DELAY = 0.5
MAX_RETRIES = 3


def _get_json(url: str, params: dict = None) -> dict:
    """GET request con retry y rate limiting."""
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=30)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                wait = 2 ** attempt * 2
                print(f"  Rate limited, esperando {wait}s...")
                time.sleep(wait)
            else:
                print(f"  HTTP {resp.status_code}")
                return {}
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(1)
    return {}


def buscar_comentarios(
    subreddit: str,
    tamanio: int = 500,
    antes: int = None,
    despues: int = None,
    enlace: str = None,
) -> list:
    """
    Busca comentarios en un subreddit via Pushshift.

    Parametros:
        subreddit: nombre del sub (sin r/)
        tamanio: max comentarios por llamada (max 1000)
        antes/despues: timestamp UNIX para paginacion
        enlace: link_id (t3_xxx) para filtrar por post especifico

    Retorna lista de comentarios con:
        - author, body, score, subreddit
        - parent_id (para reply chain)
        - link_id (post padre)
        - created_utc (timestamp)
    """
    params = {
        "subreddit": subreddit,
        "size": min(tamanio, 1000),
        "sort": "desc",
        "sort_type": "created_utc",
    }
    if antes:
        params["before"] = antes
    if despues:
        params["after"] = despues
    if enlace:
        params["link_id"] = enlace

    url = f"{PUSHSHIFT_BASE}/search/comment/"
    data = _get_json(url, params)
    return data.get("data", [])


def buscar_posts(
    subreddit: str,
    tamanio: int = 100,
    antes: int = None,
    despues: int = None,
) -> list:
    """Busca posts en un subreddit."""
    params = {
        "subreddit": subreddit,
        "size": min(tamanio, 500),
        "sort": "desc",
        "sort_type": "created_utc",
    }
    if antes:
        params["before"] = antes
    if despues:
        params["after"] = despues

    url = f"{PUSHSHIFT_BASE}/search/submission/"
    data = _get_json(url, params)
    return data.get("data", [])


def comentarios_a_mensajes(comentarios: list) -> list:
    """
    Convierte comentarios de Reddit al formato estandar de mensajes.
    Los parent_id pueden ser t1_ (reply a comentario) o t3_ (reply a post).
    """
    mensajes = []
    for c in comentarios:
        author = c.get("author", "[deleted]")
        parent = c.get("parent_id", "")
        link = c.get("link_id", "")
        cid = c.get("id", "")

        # uri unico para este comentario
        uri = f"reddit_{c.get('subreddit','')}_{cid}"

        # parent_uri: apunta al id del padre (post o comentario)
        parent_uri = None
        parent_tipo = None
        if parent.startswith("t1_"):
            # Reply a otro comentario
            parent_uri = f"reddit_{c.get('subreddit','')}_{parent[3:]}"
            parent_tipo = "comentario"
        elif parent.startswith("t3_"):
            # Reply al post directamente
            parent_uri = f"reddit_{c.get('subreddit','')}_{link[3:]}"
            parent_tipo = "post"

        mensajes.append({
            "uri": uri,
            "parent_uri": parent_uri,
            "parent_tipo": parent_tipo,
            "author_did": f"reddit_{author}",
            "author_handle": author,
            "author_name": author,
            "text": c.get("body", ""),
            "timestamp": datetime.fromtimestamp(c.get("created_utc", 0), tz=timezone.utc).isoformat(),
            "likes": c.get("score", 0),
            "replies_count": 0,
            "reposts": 0,
            "depth": 1 if parent_tipo == "post" else 2,
            "is_root": False,
            "subreddit": c.get("subreddit", ""),
            "post_id": link,
        })
    return mensajes


def posts_a_mensajes(posts: list) -> list:
    """Convierte posts de Reddit a formato estandar (como mensajes raiz)."""
    mensajes = []
    for p in posts:
        author = p.get("author", "[deleted]")
        pid = p.get("id", "")
        mensajes.append({
            "uri": f"reddit_{p.get('subreddit','')}_{pid}",
            "parent_uri": None,
            "parent_tipo": None,
            "author_did": f"reddit_{author}",
            "author_handle": author,
            "author_name": author,
            "text": p.get("title", "") + "\n\n" + (p.get("selftext", "") or ""),
            "timestamp": datetime.fromtimestamp(p.get("created_utc", 0), tz=timezone.utc).isoformat(),
            "likes": p.get("score", 0),
            "replies_count": p.get("num_comments", 0),
            "reposts": 0,
            "depth": 0,
            "is_root": True,
            "subreddit": p.get("subreddit", ""),
            "post_id": pid,
        })
    return mensajes


def extraer_subreddit(
    subreddit: str,
    output_dir: str,
    n_comentarios: int = 2000,
    incluir_posts: bool = True,
    n_posts: int = 200,
) -> str:
    """
    Extrae comentarios y posts de un subreddit.

    Los comentarios tienen reply chains naturales que forman
    conversaciones, y scores (upvotes) que sirven como IA.

    Retorna ruta del archivo JSONL generado.
    """
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nExtrayendo r/{subreddit}...")

    # Estrategia: extraer comentarios en paralelo por ventanas de tiempo
    # Pushshift tiene datos limitados post-2023, asi que usamos ventanas
    # de 6 meses hacia atras para cubrir mas rango temporal.
    VENTANAS = [
        (None, None),                                        # mas recientes
    ]

    todos_comentarios = []
    for inicio, fin in VENTANAS:
        comentarios = buscar_comentarios(subreddit, tamanio=1000, despues=inicio, antes=fin)
        if comentarios:
            todos_comentarios.extend(comentarios)
        print(f"  Ventana {inicio}-{fin}: {len(comentarios)} comentarios")
        time.sleep(RATE_LIMIT_DELAY)

    # Paginar mas atras si no llegamos
    if len(todos_comentarios) >= 900:
        ultimo_ts = todos_comentarios[-1].get("created_utc")
        for intento in range(3):
            comentarios = buscar_comentarios(subreddit, tamanio=1000, antes=ultimo_ts)
            if not comentarios:
                break
            todos_comentarios.extend(comentarios)
            ultimo_ts = comentarios[-1].get("created_utc")
            print(f"  Paginacion: {len(todos_comentarios)} comentarios")
            time.sleep(RATE_LIMIT_DELAY)

    print(f"  Total comentarios: {len(todos_comentarios)}")

    # Extraer posts
    todos_posts = []
    if incluir_posts:
        ultimo_ts_p = None
        for batch in range(0, n_posts, 100):
            size = min(100, n_posts - len(todos_posts))
            posts = buscar_posts(subreddit, tamanio=size, antes=ultimo_ts_p)
            if not posts:
                break
            todos_posts.extend(posts)
            ultimo_ts_p = posts[-1].get("created_utc")
            print(f"  Posts: {len(todos_posts)}/{n_posts}")
            time.sleep(RATE_LIMIT_DELAY)

    # Convertir a formato estandar
    mensajes = posts_a_mensajes(todos_posts) + comentarios_a_mensajes(todos_comentarios)
    print(f"  Total mensajes: {len(mensajes)}")

    # Identificar reply chains entre comentarios
    # Pushshift no devuelve replies anidados directamente,
    # pero el parent_id permite reconstruir la cadena
    chain_map = {}
    for m in mensajes:
        if m["parent_uri"] and m["parent_uri"].startswith("https://reddit.com"):
            parent_id = m["parent_uri"].split("/")[-1]
            chain_map[m["uri"]] = parent_id

    # Guardar
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"reddit_{subreddit}_{timestamp}.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        for msg in mensajes:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    # Resumen
    autores = set(m["author_did"] for m in mensajes)
    print(f"\n{'=' * 50}")
    print(f"Extraccion completada: r/{subreddit}")
    print(f"  Posts: {len(todos_posts)}")
    print(f"  Comentarios: {len(todos_comentarios)}")
    print(f"  Mensajes totales: {len(mensajes)}")
    print(f"  Autores unicos: {len(autores)}")
    print(f"  Archivo: {output_file}")
    print(f"{'=' * 50}")

    return output_file, mensajes


if __name__ == "__main__":
    import sys
    sub = sys.argv[1] if len(sys.argv) > 1 else "argentina"
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    extraer_subreddit(sub, out)

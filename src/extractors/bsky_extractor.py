"""
Bluesky thread extractor.
Fetches full conversation threads via the public Bluesky API.
Uses requests directly (Bluesky returns JSON, not HTML).
"""

import json
import time
import os
from datetime import datetime

import requests
import pandas as pd

BSKY_API = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread"
USER_AGENT = "aprobacion-social-chats/0.1.0"
RATE_LIMIT_DELAY = 0.35


def parse_actor_id(actor_id: str) -> str:
    return actor_id.split("_", 1)[1]


def parse_post_id(post_id: str) -> str:
    return post_id.split("_", 1)[1]


def build_uri(did: str, rkey: str) -> str:
    return f"at://{did}/app.bsky.feed.post/{rkey}"


def extract_replies(reply_list, parent_uri: str, depth: int = 0):
    results = []
    for reply in reply_list:
        post = reply.get("post", {})
        record = post.get("record", {})
        author = post.get("author", {})

        entry = {
            "uri": post.get("uri", ""),
            "parent_uri": parent_uri,
            "author_did": author.get("did", ""),
            "author_handle": author.get("handle", ""),
            "author_name": author.get("displayName", ""),
            "text": record.get("text", ""),
            "timestamp": record.get("createdAt", ""),
            "likes": post.get("likeCount", 0),
            "replies_count": post.get("replyCount", 0),
            "reposts": post.get("repostCount", 0),
            "depth": depth,
            "is_root": False,
        }
        results.append(entry)

        nested = reply.get("replies", [])
        if nested:
            results.extend(extract_replies(nested, entry["uri"], depth + 1))

    return results


def fetch_thread(did: str, rkey: str):
    uri = build_uri(did, rkey)
    url = f"{BSKY_API}?uri={uri}"

    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=15)

    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"

    data = resp.json()
    thread = data.get("thread", {})
    post = thread.get("post", {})
    record = post.get("record", {})
    author = post.get("author", {})

    root = {
        "uri": post.get("uri", ""),
        "parent_uri": None,
        "author_did": author.get("did", ""),
        "author_handle": author.get("handle", ""),
        "author_name": author.get("displayName", ""),
        "text": record.get("text", ""),
        "timestamp": record.get("createdAt", ""),
        "likes": post.get("likeCount", 0),
        "replies_count": post.get("replyCount", 0),
        "reposts": post.get("repostCount", 0),
        "depth": 0,
        "is_root": True,
    }

    all_messages = [root]
    replies = thread.get("replies", [])
    all_messages.extend(extract_replies(replies, root["uri"], depth=1))

    return all_messages, None


def extract_all(source_parquet: str, output_dir: str, max_posts: int = None):
    os.makedirs(output_dir, exist_ok=True)

    posts = pd.read_parquet(source_parquet)
    print(f"Cargados {len(posts)} posts desde {source_parquet}")

    # Prioritize posts with comments
    posts = posts.sort_values("comments", ascending=False)

    if max_posts:
        posts = posts.head(max_posts)

    all_messages = []
    stats = {"success": 0, "failed": 0, "total_messages": 0, "total_replies": 0}

    for idx, (_, row) in enumerate(posts.iterrows()):
        did = parse_actor_id(row["actor_id"])
        rkey = parse_post_id(row["post_id"])
        print(f"[{idx+1}/{len(posts)}] {did[-24:]} ... rkey={rkey}")

        messages, error = fetch_thread(did, rkey)

        if messages:
            all_messages.extend(messages)
            stats["success"] += 1
            replies = [m for m in messages if not m["is_root"]]
            stats["total_messages"] += len(messages)
            stats["total_replies"] += len(replies)
            print(f"  -> {len(messages)} msgs ({len(replies)} replies, {messages[0]['likes']} likes)")
        else:
            stats["failed"] += 1
            print(f"  -> Error: {error}")

        time.sleep(RATE_LIMIT_DELAY)

    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(output_dir, f"bluesky_threads_{timestamp}.jsonl")

    with open(output_file, "w", encoding="utf-8") as f:
        for msg in all_messages:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    # Summary
    unique_authors = set(m["author_did"] for m in all_messages)
    print(f"\n{'='*50}")
    print(f"Extracción completada")
    print(f"Posts procesados: {stats['success']} OK / {stats['failed']} FAIL")
    print(f"Mensajes totales: {stats['total_messages']} ({stats['total_replies']} replies)")
    print(f"Autores únicos: {len(unique_authors)}")
    print(f"Output: {output_file}")
    print(f"{'='*50}")

    return output_file, all_messages


def to_dataframes(messages: list):
    """
    Convert extracted messages to ER-aligned DataFrames.
    Returns (df_usuarios, df_mensajes, df_interacciones)
    """
    rows_mensajes = []
    autores_set = {}

    for msg in messages:
        author_did = msg["author_did"]
        if author_did and author_did not in autores_set:
            autores_set[author_did] = {
                "id_usuario": author_did,
                "nombre": msg["author_name"] or msg["author_handle"],
                "handle": msg["author_handle"],
            }

        rows_mensajes.append({
            "id_mensaje": msg["uri"],
            "id_usuario": author_did,
            "timestamp": msg["timestamp"],
            "texto": msg["text"],
            "id_mensaje_respuesta": msg["parent_uri"],
            "likes": msg["likes"],
            "reposts": msg["reposts"],
            "es_raiz": msg["is_root"],
            "profundidad": msg["depth"],
        })

    df_usuarios = pd.DataFrame(list(autores_set.values()))
    df_mensajes = pd.DataFrame(rows_mensajes)

    # Build interacciones (reply graph)
    replies_df = df_mensajes[df_mensajes["id_mensaje_respuesta"].notna()].copy()
    if len(replies_df) > 0:
        # Merge to get origen/destino
        merged = replies_df.merge(
            df_mensajes[["id_mensaje", "id_usuario"]],
            left_on="id_mensaje_respuesta",
            right_on="id_mensaje",
            suffixes=("_origen", "_destino"),
            how="left"
        )
        df_interacciones = merged.rename(columns={
            "id_usuario_origen": "id_usuario_origen",
            "id_usuario_destino": "id_usuario_destino",
        })[["id_usuario_origen", "id_usuario_destino", "id_mensaje"]].copy()
        df_interacciones["tipo"] = "respuesta_directa"
        df_interacciones["timestamp"] = replies_df["timestamp"].values
        df_interacciones["peso"] = 1.0
    else:
        df_interacciones = pd.DataFrame()

    return df_usuarios, df_mensajes, df_interacciones


if __name__ == "__main__":
    source = "C:\\Users\\JUAN\\Desktop\\Proyectos\\attention_observatory\\data\\bronze\\bluesky_posts_20260611_005914.parquet"
    output = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    extract_all(source, output, max_posts=None)

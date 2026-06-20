"""
Fusiona todos los datasets individuales de Reddit en un solo archivo JSONL.
"""
import json, os, glob
from collections import Counter

raw_dir = os.path.join("data", "raw")
files = sorted(glob.glob(os.path.join(raw_dir, "reddit_*.jsonl")))
# Excluir merged previo
files = [f for f in files if "merged" not in f]

print(f"Archivos a fusionar: {len(files)}")
todos = []
autores = set()
stats_por_sub = Counter()

for path in files:
    sub = os.path.basename(path).split("_")[1] if "reddit_" in os.path.basename(path) else "unknown"
    n = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            # Normalizar campos
            if "author_did" not in msg:
                msg["author_did"] = f"reddit_{msg.get('author', msg.get('author_handle', 'unknown'))}"
            if "id_mensaje" not in msg:
                msg["id_mensaje"] = msg.get("uri", f"msg_{n}")
            if "es_root" not in msg:
                msg["es_root"] = msg.get("is_root", False)
            if "id_usuario" not in msg:
                msg["id_usuario"] = msg["author_did"]
            # Asegurar likes como entero
            try:
                msg["likes"] = int(msg.get("likes", 0))
            except (ValueError, TypeError):
                msg["likes"] = 0
            todos.append(msg)
            autores.add(msg["author_did"])
            n += 1
    stats_por_sub[sub] = n

# Deduplicar por uri (quedarse con la ultima version)
vistos = {}
for msg in todos:
    uri = msg.get("uri", "")
    vistos[uri] = msg
todos = list(vistos.values())

total = len(todos)
print(f"\nTotal mensajes: {total}")
print(f"Autores unicos: {len(autores)}")
print(f"Por subreddit:")
for sub, n in sorted(stats_por_sub.items()):
    print(f"  r/{sub}: {n}")

# Guardar fusionado
ts = "20260621"
out = os.path.join(raw_dir, f"reddit_merged_{ts}.jsonl")
with open(out, "w", encoding="utf-8") as f:
    for msg in todos:
        f.write(json.dumps(msg, ensure_ascii=False) + "\n")

print(f"\nGuardado: {out}")
print(f"Tamanio: {os.path.getsize(out) / 1024:.0f} KB")

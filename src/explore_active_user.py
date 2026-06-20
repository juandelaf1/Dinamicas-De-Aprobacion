import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json

with open("C:\\Users\\JUAN\\Desktop\\Proyectos\\aprobacion-social-chats\\data\\raw\\bluesky_threads_20260619_123556.jsonl", "r", encoding="utf-8") as f:
    messages = [json.loads(l) for l in f]

# Check the most active user
target_did = None
from collections import Counter
author_counts = Counter(m["author_did"] for m in messages)
most_active = author_counts.most_common(1)[0][0]

print(f"Usuario más activo: {most_active}")
print(f"Mensajes: {author_counts[most_active]}")
print()

user_msgs = [m for m in messages if m["author_did"] == most_active]
for m in user_msgs:
    tag = "ROOT" if m["is_root"] else "REPLY"
    print(f"  [{tag}] likes={m['likes']:3d} | parent={str(m['parent_uri'])[:40]:40s} | {m['text'][:80]}")

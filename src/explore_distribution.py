import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
from collections import Counter

with open("C:\\Users\\JUAN\\Desktop\\Proyectos\\aprobacion-social-chats\\data\\raw\\bluesky_threads_20260619_123556.jsonl", "r", encoding="utf-8") as f:
    messages = [json.loads(l) for l in f]

author_counts = Counter(m["author_did"] for m in messages)
print("Distribución de mensajes por autor:")
for count, freq in sorted(Counter(author_counts.values()).items()):
    bar = "#" * freq
    print(f"  {count:2d} mensajes: {freq:3d} usuarios {bar}")

print(f"\nTotal usuarios: {len(author_counts)}")
print(f"Mensajes totales: {len(messages)}")
print(f"Media: {len(messages)/len(author_counts):.1f} msgs/usuario")
print(f"Mediana: {sorted(author_counts.values())[len(author_counts)//2]}")

# Also check likes distribution
likes_per_user = {}
for m in messages:
    did = m["author_did"]
    likes_per_user.setdefault(did, 0)
    likes_per_user[did] += m["likes"]

print("\nTop 10 usuarios por likes recibidos:")
for did, likes in sorted(likes_per_user.items(), key=lambda x: -x[1])[:10]:
    cnt = author_counts[did]
    handle = next(m["author_handle"] for m in messages if m["author_did"] == did)
    print(f"  @{handle:<30s} likes={likes:3d} msgs={cnt}")

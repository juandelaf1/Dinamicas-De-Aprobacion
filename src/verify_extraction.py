import json

with open("C:\\Users\\JUAN\\Desktop\\Proyectos\\aprobacion-social-chats\\src\\data\\raw\\bluesky_threads_20260619_123303.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

print(f"Total messages: {len(lines)}")
roots = sum(1 for l in lines if json.loads(l)["is_root"])
replies = sum(1 for l in lines if not json.loads(l)["is_root"])
authors = set(json.loads(l)["author_did"] for l in lines)
likes_total = sum(json.loads(l)["likes"] for l in lines)
print(f"Root posts: {roots}")
print(f"Replies: {replies}")
print(f"Unique authors: {len(authors)}")
print(f"Total likes: {likes_total}")
print()

for l in lines[:5]:
    m = json.loads(l)
    tag = "ROOT" if m["is_root"] else "REPLY"
    handle = m["author_handle"][:20]
    print(f"  [{tag}] @{handle:20s} | likes: {m['likes']:3d} | {m['text'][:60]}")

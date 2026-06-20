import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from raspal import Fetcher
import json

f = Fetcher()
url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri=at://did:plc:enar5exlzyhomuhj4fgosm6g/app.bsky.feed.post/3mnxsfs5vpc2z"
result = f.fetch(url, engine="auto", cache_ttl=3600)

# Check various content attributes
print(f"html type: {type(result.html)}")
print(f"html is None: {result.html is None}")
print(f"html len: {len(result.html) if result.html else 0}")
print()

# Try model_dump
d = result.model_dump()
content_keys = [k for k in d.keys() if d.get(k) is not None]
print(f"Non-null keys in model_dump: {content_keys}")
print()

# The json method returns a string
json_str = result.json()
print(f"json() type: {type(json_str)}")
print(f"json() len: {len(json_str)}")
if len(json_str) > 50:
    print(f"json()[:200]: {json_str[:200]}")
    parsed = json.loads(json_str)
    if isinstance(parsed, dict):
        print(f"Parsed keys: {list(parsed.keys())}")
        thread = parsed.get("thread", {})
        post = thread.get("post", {})
        print(f"Likes: {post.get('likeCount')}, Replies: {post.get('replyCount')}")

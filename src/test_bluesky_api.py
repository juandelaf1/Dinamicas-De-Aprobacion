import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests
import json

# Test Bluesky public API - fetch a post thread
did = "did:plc:m2yuqynl2cttvi4k5453yegh"
rkey = "3mnxslkjnkvh2"
uri = f"at://{did}/app.bsky.feed.post/{rkey}"
url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri={uri}"

print(f"Fetching: {url}")
resp = requests.get(url, headers={"User-Agent": "raspal-test/0.1.0"})
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    thread = data.get("thread", {})
    post = thread.get("post", {})
    record = post.get("record", {})
    author = post.get("author", {})
    print(f"\nAuthor: {author.get('displayName')} (@{author.get('handle')})")
    print(f"Text: {str(record.get('text', ''))[:150]}")
    print(f"Likes: {post.get('likeCount', 'N/A')}")
    print(f"Replies: {post.get('replyCount', 'N/A')}")
    print(f"Reposts: {post.get('repostCount', 'N/A')}")

    replies = thread.get("replies", [])
    print(f"\nTotal replies in thread: {len(replies)}")
    for i, reply in enumerate(replies[:5]):
        r_post = reply.get("post", {})
        r_author = r_post.get("author", {})
        r_record = r_post.get("record", {})
        print(f"\n  Reply {i+1}:")
        print(f"    Author: {r_author.get('displayName')} (@{r_author.get('handle')})")
        print(f"    Text: {str(r_record.get('text', ''))[:120]}")
        print(f"    Likes: {r_post.get('likeCount', 0)}")

        # Check for nested replies
        r_replies = reply.get("replies", [])
        if r_replies:
            print(f"    Nested replies: {len(r_replies)}")
else:
    print(f"Error: {resp.text[:500]}")

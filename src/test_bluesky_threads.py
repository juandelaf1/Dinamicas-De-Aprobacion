import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd
import requests
import time
import json

posts = pd.read_parquet("C:\\Users\\JUAN\\Desktop\\Proyectos\\attention_observatory\\data\\bronze\\bluesky_posts_20260611_005914.parquet")

# Find posts with replies
with_comments = posts[posts['comments'] > 0].sort_values('comments', ascending=False)
print(f"Posts with comments > 0: {len(with_comments)}")
print(f"Max comments: {with_comments['comments'].max()}")
print()

# For the top 10, try to fetch their thread
def post_id_to_uri(post_id):
    parts = post_id.split("_", 1)
    rkey = parts[1]
    # we need the DID - get it from the posts dataframe
    return rkey

def actor_to_did(actor_id):
    parts = actor_id.split("_", 1)
    return parts[1]

count = 0
for _, row in with_comments.head(10).iterrows():
    did = actor_to_did(row['actor_id'])
    rkey = post_id_to_uri(row['post_id'])
    uri = f"at://{did}/app.bsky.feed.post/{rkey}"
    url = f"https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri={uri}"

    print(f"\n--- Post {row['post_id']} (comments: {row['comments']}, likes: {row['likes']}) ---")
    print(f"  Text: {str(row['content_text'])[:100]}")

    resp = requests.get(url, headers={"User-Agent": "raspal/0.1.0"})
    if resp.status_code == 200:
        data = resp.json()
        thread = data.get("thread", {})
        post = thread.get("post", {})
        replies = thread.get("replies", [])
        print(f"  API says replies: {len(replies)}")
        for i, reply in enumerate(replies[:3]):
            r_post = reply.get("post", {})
            r_author = r_post.get("author", {})
            r_record = r_post.get("record", {})
            print(f"    [{i+1}] @{r_author.get('handle', '?')}: {str(r_record.get('text', ''))[:80]}")
            print(f"         likes: {r_post.get('likeCount', 0)} | time: {str(r_record.get('createdAt', '?'))[:19]}")
    else:
        print(f"  Error: {resp.status_code}")

    count += 1
    if count >= 5:
        break
    time.sleep(0.5)

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import requests, json

# Search Bluesky for posts with replies
search_url = "https://public.api.bsky.app/xrpc/app.bsky.feed.searchPosts"
params = {
    "q": "artificial intelligence",
    "sort": "latest",
    "limit": 25
}

print(f"Searching Bluesky for: '{params['q']}'")
resp = requests.get(search_url, params=params, headers={"User-Agent": "raspal-test/0.1.0"})
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    data = resp.json()
    posts = data.get("posts", [])
    print(f"Total posts found: {len(posts)}")
    print()

    # Find posts with replies
    with_replies = [p for p in posts if p.get("replyCount", 0) > 0]
    print(f"Posts with replies: {len(with_replies)}")

    for p in with_replies[:10]:
        author = p.get("author", {})
        record = p.get("record", {})
        print(f"\n  --- Post ---")
        print(f"  Author:  {author.get('displayName', '?')} (@{author.get('handle', '?')})")
        print(f"  Text:    {str(record.get('text', ''))[:100]}")
        print(f"  Likes:   {p.get('likeCount', 0)} | Replies: {p.get('replyCount', 0)} | Reposts: {p.get('repostCount', 0)}")
        print(f"  URI:     {p.get('uri', '?')}")
        print(f"  Time:    {record.get('createdAt', '?')}")

    posts_with_likes = [p for p in posts if p.get("likeCount", 0) > 0]
    print(f"\nPosts with likes > 0: {len(posts_with_likes)}")
    print(f"Posts with replies > 0: {len(with_replies)}")
else:
    print(f"Error: {resp.text[:500]}")

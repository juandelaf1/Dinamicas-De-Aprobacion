"""Test available real data sources."""
import requests
import sys
import json

headers = {"User-Agent": "aprobacion-social/0.1.0"}

print("=== Reddit (old.reddit.com JSON) ===")
try:
    url = "https://old.reddit.com/r/argentina.json?limit=3"
    resp = requests.get(url, headers=headers, timeout=15)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        posts = data["data"]["children"]
        print(f"  Posts: {len(posts)}")
        for c in posts[:3]:
            print(f"    r/{c['data']['subreddit']}: {c['data']['title'][:60]}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Pushshift (Reddit archive) ===")
try:
    url = "https://api.pullpush.io/reddit/search/comment/?subreddit=argentina&size=3&sort=desc"
    resp = requests.get(url, headers=headers, timeout=15)
    print(f"  Status: {resp.status_code}")
    if resp.status_code == 200:
        data = resp.json()
        comments = data.get("data", [])
        print(f"  Comments: {len(comments)}")
        for c in comments[:3]:
            print(f"    u/{c['author']}: {c['body'][:80]}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== Hacker News API ===")
try:
    url = "https://hacker-news.firebaseio.com/v0/topstories.json"
    resp = requests.get(url, headers=headers, timeout=15)
    top_ids = resp.json()[:3]
    for id_ in top_ids:
        item = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{id_}.json",
                           headers=headers, timeout=15).json()
        print(f"  HN: {item.get('title', '')[:60]} ({item.get('score', 0)} pts)")
        # Get comments
        kids = item.get("kids", [])[:3]
        for k in kids:
            comment = requests.get(f"https://hacker-news.firebaseio.com/v0/item/{k}.json",
                                  headers=headers, timeout=15).json()
            print(f"    reply by {comment.get('by', '?')}: {comment.get('text', '')[:80]}")
except Exception as e:
    print(f"  Error: {e}")

print("\n=== DummyJSON (placeholder) ===")
try:
    url = "https://dummyjson.com/comments?limit=3"
    resp = requests.get(url, headers=headers, timeout=15)
    print(f"  Status: {resp.status_code}")
except Exception as e:
    print(f"  Error: {e}")

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from raspal import Fetcher

f = Fetcher()
url = "https://public.api.bsky.app/xrpc/app.bsky.feed.getPostThread?uri=at://did:plc:enar5exlzyhomuhj4fgosm6g/app.bsky.feed.post/3mnxsfs5vpc2z"
result = f.fetch(url, engine="auto", cache_ttl=3600)

print(f"Type: {type(result)}")
print(f"Dir: {[a for a in dir(result) if not a.startswith('_')]}")
print(f"Status: {getattr(result, 'status', 'N/A')}")
print(f"Text length: {len(getattr(result, 'text', '') or '')}")
print(f"HTML length: {len(getattr(result, 'html', '') or '')}")
print(f"Headers: {getattr(result, 'headers', 'N/A')}")
print()

# Check result.text 
text = getattr(result, 'text', None)
print(f"text is None: {text is None}")
if text:
    print(f"text[:200]: {text[:200]}")
else:
    # Try result.html
    html = getattr(result, 'html', None)
    print(f"html is None: {html is None}")
    if html:
        print(f"html[:300]: {html[:300]}")
    # Also check json method
    try:
        j = result.json()
        print(f"\nJSON keys: {list(j.keys()) if isinstance(j, dict) else type(j)}")
    except Exception as e:
        print(f"json() error: {e}")

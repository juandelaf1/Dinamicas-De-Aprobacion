import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

posts = pd.read_parquet("C:\\Users\\JUAN\\Desktop\\Proyectos\\attention_observatory\\data\\bronze\\bluesky_posts_20260611_005914.parquet")

print("Sample post_ids:")
for _, row in posts.head(10).iterrows():
    print(f"  {row['post_id']}")

print("\nFirst actor_id:")
print(f"  {posts['actor_id'].iloc[0]}")

print("\nFirst content_text (first 200 chars):")
print(f"  {posts['content_text'].iloc[0][:200]}")

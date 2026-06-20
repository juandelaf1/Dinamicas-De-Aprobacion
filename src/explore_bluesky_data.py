import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import pandas as pd

# Bronze posts
posts = pd.read_parquet("C:\\Users\\JUAN\\Desktop\\Proyectos\\attention_observatory\\data\\bronze\\bluesky_posts_20260611_005914.parquet")
print("=== BRONZE POSTS ===")
print(f"Shape: {posts.shape}")
print(f"Columns: {list(posts.columns)}")
print()
print("Sample posts:")
for _, row in posts.head(5).iterrows():
    print(f"  actor: {str(row['actor_id'])[:35]:35s} | likes: {row['likes']:3d} | text: {str(row['content_text'])[:80]}")

print(f"\nUnique actors in posts: {posts['actor_id'].nunique()}")
print(f"Total posts: {len(posts)}")
print(f"Posts with likes > 0: {(posts['likes'] > 0).sum()}")
print(f"Posts with comments > 0: {(posts['comments'] > 0).sum()}")

# Bronze actors
actors = pd.read_parquet("C:\\Users\\JUAN\\Desktop\\Proyectos\\attention_observatory\\data\\bronze\\bluesky_actors_20260611_005914.parquet")
print("\n=== BRONZE ACTORS ===")
print(f"Shape: {actors.shape}")
print(f"Columns: {list(actors.columns)}")
print(f"Unique actor_ids: {actors['actor_id'].nunique()}")
print(f"Total actors with followers > 0: {(actors['followers'] > 0).sum()}")
print(f"Total actors with posts_count > 0: {(actors['posts_count'] > 0).sum()}")

# Check overlap: which actors in posts also appear in actors table
post_actors = set(posts['actor_id'].unique())
actor_table = set(actors['actor_id'].unique())
print(f"\nOverlap post_actors ∩ actor_table: {len(post_actors & actor_table)}")
print(f"Post actors NOT in actor table: {len(post_actors - actor_table)}")

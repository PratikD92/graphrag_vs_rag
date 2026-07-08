import pandas as pd

df = pd.read_parquet("output/communities.parquet")

print(
    df[["community", "level", "parent", "children"]].sort_values(["level", "community"])
)

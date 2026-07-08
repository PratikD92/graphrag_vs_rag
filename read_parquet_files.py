import pandas as pd
import os

parquet_dir = "output"
parquet_files = os.listdir(parquet_dir)

# print(parquet_files)

# Read only .parquet files
parquet_files = [f for f in parquet_files if f.endswith(".parquet")]

# Read parquet files
for file in parquet_files:
    df = pd.read_parquet(f"{parquet_dir}/{file}")
    print(f"--- {file} ---")
    print(f"Columns: {list(df.columns)}")

    # match = df[df["id"] == "8fd98c7c-d361-4632-ad6b-9e756235b6b9"]
    # if not match.empty:
    #     print(match["description"].iloc[0])
    # print(df.dtypes)
    # print(df.head())
    # if file == "relationships.parquet":
    #     print(df["source"].iloc[0])
    #     print(df["target"].iloc[0])
    #     print(df["description"].iloc[0])
    #     print(df["combined_degree"].iloc[0])
    print("\n")
    # break

import pandas as pd
from rich import print
import sys

files = [
    "output/community_reports.parquet",
    "output/relationships.parquet",
    "output/entities.parquet",
    "output/communities.parquet",
    "output/text_units.parquet",
]
# [Data: Entities (29); Relationships (17)]
entities_file = files[2]
relationships_file = files[1]

for file in files:
    print(f"File: {file}")
    df = pd.read_parquet(file)
    print(df.iloc[0].T)
    print(df.iloc[1].T)
    print("\n")

sys.exit()

entities = pd.read_parquet(entities_file)

seventeen = entities.loc[entities["human_readable_id"] == 30]
# print(seventeen.iloc[0][["title", "description", "text_unit_ids"]])
print(seventeen.iloc[0])
# print(seventeen.iloc[0]["description"])
print(seventeen.iloc[0]["text_unit_ids"])


relationships = pd.read_parquet(relationships_file)
# print(relationships.columns)
fourteen = relationships.loc[relationships["human_readable_id"] == 38]
print(fourteen.iloc[0])
print(fourteen.iloc[0]["text_unit_ids"])


text_unit_file = files[4]
text_units = pd.read_parquet(text_unit_file)
print(text_units.columns)
for id in fourteen.iloc[0]["text_unit_ids"]:
    print(text_units.loc[text_units["id"] == id].iloc[0]["text"])
    print("\n")

print("========================================\n")
print("========================================\n")

for id in seventeen.iloc[0]["text_unit_ids"]:
    print(text_units.loc[text_units["id"] == id].iloc[0]["text"])
    print("========================================\n")

# print(text_units.loc[text_units["id"] == fourteen.iloc[0]["text_unit_ids"][0]])


print(
    entities.loc[entities["human_readable_id"] == 30].iloc[0][["title", "description"]]
)

print(
    relationships.loc[relationships["human_readable_id"] == 38].iloc[0][
        ["source", "target", "description"]
    ]
)

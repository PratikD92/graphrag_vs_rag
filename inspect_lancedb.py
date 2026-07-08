import lancedb

db = lancedb.connect("output/lancedb")

print(db.table_names())

for table_name in db.table_names():
    table = db.open_table(table_name)
    print(f"\n{table_name}:")
    # print(table.head(5))
    df = table.to_pandas()
    print(df["id"].iloc[0])

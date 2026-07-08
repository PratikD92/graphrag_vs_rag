import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

entities = pd.read_parquet("output/entities.parquet")

relationships = pd.read_parquet("output/relationships.parquet")

G = nx.Graph()

for _, row in relationships.iterrows():
    G.add_edge(row["source"], row["target"])

plt.figure(figsize=(15, 10))
nx.draw(G, with_labels=True, node_size=500, font_size=8)

plt.show()

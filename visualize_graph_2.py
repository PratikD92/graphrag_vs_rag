import pandas as pd
from pyvis.network import Network

relationships = pd.read_parquet("output/relationships.parquet")

net = Network(
    height="900px", width="100%", bgcolor="#ffffff", font_color="black", notebook=False
)

for _, row in relationships.iterrows():
    source = str(row["source"])
    target = str(row["target"])

    net.add_node(source, label=source, size=20)
    net.add_node(target, label=target, size=20)

    net.add_edge(source, target)

# Increase node spacing
# net.barnes_hut(
#     gravity=-30000,
#     central_gravity=0.3,
#     spring_length=0.2,  # <-- increase this
#     spring_strength=0.01,
#     damping=0.09,
# )

# Physics engine allows dragging and automatic layout
net.show_buttons(filter_=["physics"])

net.save_graph("graphrag_graph3.html")

print("Open graphrag_graph.html in browser")

import chokepoints
from metrics import ttl_to_graph
import networkx as nx

ttl_path = "/home/sausage69/OneDrive/GreI/4º/2Semestre/tfg/tfg/recursos/ontologies/r2.ttl"


g = ttl_to_graph(ttl_path)

g0 = chokepoints.get_chokepoint_subgraph(g, 0)
g1 = chokepoints.get_chokepoint_subgraph(g, 1)
g2 = chokepoints.get_chokepoint_subgraph(g, 2)


# Print nx.DiGraph subgraphs to file
with open("g0.txt", "w") as f:
    for u, v, p in g0.edges(data=True):
        f.write(f"{u} {p["predicate"]} {v}\n")


with open("g1.txt", "w") as f:
    for u, v, p in g1.edges(data=True):
        f.write(f"{u} {p["predicate"]} {v}\n")

with open("g2.txt", "w") as f:
    for u, v, p in g2.edges(data=True):
        f.write(f"{u} {p["predicate"]} {v}\n")

with open("g.txt", "w") as f:
    for u, v, p in g.edges(data=True):
        f.write(f"{u} {p["predicate"]} {v}\n")
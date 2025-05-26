from metrics import ttl_to_graph

g = ttl_to_graph("/home/sausage69/OneDrive/GreI/4º/2Semestre/tfg/tfg/recursos/ontologies/r4.ttl")

with open("grafo_tripletas.txt", "w") as f:
    for u, v, data in g.edges(data=True):
        predicado = data.get("predicate", "relacion")
        f.write(f"{u}\t{predicado}\t{v}\n")
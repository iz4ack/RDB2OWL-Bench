from rdflib.namespace import RDF, RDFS, OWL
import networkx as nx

BASIC = {
    0: {
        "predicates": {
            RDFS.range, RDFS.domain, RDFS.subClassOf
        },
        "rdf_type_objects": {
            "owl:Class",
            "owl:DatatypeProperty",
            "owl:DatatypeProperty"
        }
    }
}

def get_basic_subgraph(G):
    predicates = BASIC[0].get("predicates", set())
    rdf_type_objs = BASIC[0].get("rdf_type_objects", set())

    nodes = G.nodes(data=True)

    edges_cp = []
    for u, v, d in G.edges(data=True):
        pred = d.get("predicate")
        
        # If node has atribute blankNode = 1 skip it
        if nodes[u].get("blankNode") or nodes[v].get("blankNode"):
            continue

        # Caso 1: predicado explícito permitido
        if pred in predicates:
            edges_cp.append((u, v, d))
        
        # Caso 2: rdf:type con objeto en rdf_type_objects
        elif pred == RDF.type and v in rdf_type_objs:
            edges_cp.append((u, v, d))

    # Construir subgrafo
    G_cp = nx.DiGraph()
    for u, v, d in edges_cp:
        G_cp.add_node(u, **G.nodes[u])
        G_cp.add_node(v, **G.nodes[v])
        G_cp.add_edge(u, v, **d)

    return G_cp

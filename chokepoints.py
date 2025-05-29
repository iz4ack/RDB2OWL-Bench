from rdflib.namespace import RDF, RDFS, OWL
import networkx as nx

BASE = {
    0: {
        "predicates": {
            RDFS.range, RDFS.domain
        },
        "rdf_type_objects": {
            OWL.Class,
            OWL.ObjectProperty,
            OWL.DatatypeProperty
        }
    },
    1: {
        "predicates": {
            RDFS.subClassOf,
            OWL.withRestrictions,
            OWL.onProperty,
            OWL.onDatatype,
            OWL.allValuesFrom,
            OWL.someValuesFrom,
            OWL.hasValue,
            OWL.cardinality,
            OWL.minCardinality,
            OWL.maxCardinality
        },
        "rdf_type_objects": {
            OWL.Restriction
        }
    }
}


def build_chokepoints():
    cumulative = {}
    pred_union = set()
    rdf_types_union = set()

    for level in sorted(BASE.keys()):
        pred_union |= BASE[level]["predicates"]
        rdf_types_union |= BASE[level]["rdf_type_objects"]
        cumulative[level] = {
            "predicates": set(pred_union),
            "rdf_type_objects": set(rdf_types_union)
        }

    return cumulative


CHOKEPOINTS = build_chokepoints()

def get_chokepoint_subgraph(G, level):
    if level not in CHOKEPOINTS:
        raise ValueError(f"Nivel {level} no está definido.")

    predicates = CHOKEPOINTS[level].get("predicates", set())
    rdf_type_objs = {str(obj) for obj in CHOKEPOINTS[level].get("rdf_type_objects", set())}

    edges_cp = []
    for u, v, d in G.edges(data=True):
        pred = d.get("predicate")
        
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

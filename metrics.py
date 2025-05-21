import rdflib
from rdflib.collection import Collection
from graph_tool import Graph
from graph_tool.topology import isomorphism
from graph_tool.clustering import motifs
import networkx as nx
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import numpy as np
from sentence_transformers import SentenceTransformer
from collections import Counter


torch = __import__('torch')
from torch_geometric.nn import SGConv
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx


def shorten(uri, graph):
    try:
        # If it's a known prefix like "rdf", "rdfs", "owl", "xsd", ":", etc.
        return graph.namespace_manager.qname(uri)
    except:
        # If there's no prefix, try extracting the localname manually
        uri = str(uri)
        if "#" in uri:
            return ":" + uri.split("#")[-1]
        elif "/" in uri:
            return uri.rsplit("/", 1)[-1]
        else:
            return uri

def ttl_to_graph(ttl_file_path):
    # Load the ontology
    g = rdflib.Graph()

    # Register known prefixes
    g.namespace_manager.bind(":", "http://example.org/ontology#")
    g.namespace_manager.bind("owl", "http://www.w3.org/2002/07/owl#")
    g.namespace_manager.bind("rdf", "http://www.w3.org/1999/02/22-rdf-syntax-ns#")
    g.namespace_manager.bind("xsd", "http://www.w3.org/2001/XMLSchema#")
    g.namespace_manager.bind("rdfs", "http://www.w3.org/2000/01/rdf-schema#")

    try:
        g.parse(ttl_file_path, format="ttl")
    except Exception as e:
        print(f"Error parsing {ttl_file_path}: {e}")
        return None

    def title(t, g):
        t = shorten(t, g)
        # Remove the prefix if it exists
        if ":" in t:
            prefix = t.split(":")[0]
            if prefix not in [":", "rdf", "rdfs", "owl", "xsd"]:
                t = t.split(":")[-1]  # Quita el prefijo si no está permitido

        return t

    def describe_blank_node(graph, bnode, seen=None):
        """Generates a human-readable (and deterministic) representation of a blank node."""
        if seen is None:
            seen = set()

        if bnode in seen:
            return "_:recursion"
        seen.add(bnode)

        if (bnode, rdflib.RDF.first, None) in graph:
            try:
                items = list(Collection(graph, bnode))
                return "(" + ", ".join(format_node(i, graph, seen=seen) for i in items) + ")"
            except:
                return "_:badList"

        props = sorted(graph.predicate_objects(bnode), key=lambda x: shorten(x[0], graph))
        parts = []
        for p, o in props:
            p_str = shorten(p, graph) if isinstance(p, rdflib.URIRef) else str(p)
            o_str = format_node(o, graph, seen=seen)
            parts.append(f"{p_str}={o_str}")
        return "[" + "; ".join(parts) + "]"

    def format_node(n, g, seen=None):
        return str(n) if not isinstance(n, rdflib.BNode) else describe_blank_node(g, n, seen=seen)
    
    # Create a directed graph
    G = nx.DiGraph()
    for subj, pred, obj in g:
        # Ensure nodes exist with 'title' attribute
        if pred in [rdflib.RDFS.comment, rdflib.RDFS.label]:
                continue
        subj_label = format_node(subj, g)
        obj_label = format_node(obj, g)
        if subj not in G:
            G.add_node(subj_label, title=title(subj, g) if isinstance(subj, rdflib.URIRef) else subj_label)
        if obj not in G:
            G.add_node(obj_label, title=title(obj, g) if isinstance(obj, rdflib.URIRef) else obj_label)
        # Special case: treat owl:hasKey as a semantic relation
        if pred == rdflib.OWL.hasKey and isinstance(obj, rdflib.BNode):
            try:
                keys = list(Collection(g, obj))
                for key in keys:
                    key_label = format_node(key, g)
                    G.add_edge(subj_label, key_label, predicate="hasKey")
            except:
                pass
        else:
            # Add edge (subj -> obj) with predicate as optional attribute
            G.add_edge(subj_label, obj_label, predicate=pred)

    return G

def embed_nodes(
    nodes_sys: list, 
    nodes_gold: list, 
    model_name: str = 'all-MiniLM-L6-v2'
    ) -> tuple:
    texts      = [str(u) for u in nodes_sys] + [str(v) for v in nodes_gold]
    model   = SentenceTransformer(model_name)
    emb_all = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    emb_sys = emb_all[:len(nodes_sys)]
    emb_gold = emb_all[len(nodes_sys):]
    return emb_sys, emb_gold

def literal_f1(G_pred: nx.Graph, G_true: nx.Graph):
    if len(G_pred) == 0 or len(G_true) == 0:
        return 0, 0, 0

    edges_G = {(G_pred.nodes[u]['title'], G_pred.nodes[v]['title']) for u, v in G_pred.edges}
    edges_G_true = {(G_true.nodes[u]['title'], G_true.nodes[v]['title']) for u, v in G_true.edges}

    precision = len(edges_G & edges_G_true) / len(edges_G)
    recall = len(edges_G & edges_G_true) / len(edges_G_true)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return precision, recall, f1

def fuzzy_f1(
    G_sys: nx.DiGraph,
    G_gold: nx.DiGraph,
    threshold: float = 0.9,
    model_name: str = 'all-MiniLM-L6-v2'
) -> tuple:
    """
    Computes the Fuzzy F1 metric between two directed graphs.

    Parameters
    ----------
    G_sys : nx.DiGraph
        Graph generated by the system (E').
    G_gold : nx.DiGraph
        Reference (gold standard) graph (E).
    threshold : float, optional
        Semantic similarity threshold (t). Default is 0.9.
    model_name : str, optional
        Name of the SentenceTransformer model. Default is 'all-MiniLM-L6-v2'.

    Returns
    -------
    - precision: Fuzzy precision
    - recall:    Fuzzy recall
    - f1:        Fuzzy F1 = 2·P·R / (P+R) (0 if P+R == 0)
    """
    # Extract nodes and calculate embeddings
    nodes_sys   = list(G_sys.nodes())
    nodes_gold  = list(G_gold.nodes())
    node_sys_title = {n: G_sys.nodes[n]['title'] for n in nodes_sys}
    node_gold_title = {n: G_gold.nodes[n]['title'] for n in nodes_gold}
    emb_sys, emb_gold = embed_nodes(node_sys_title, node_gold_title, model_name)

    # Calculate similarity matrix
    #    sim[i,j] = cosine(emb_sys[i], emb_gold[j])
    sim = cosine_similarity(emb_sys, emb_gold)

    # For each row of sim (nodes of sys), store the nodes of gold that exceed the threshold
    sim_sys_to_gold = {
        nodes_sys[i]: { nodes_gold[j] for j in np.where(sim[i] > threshold)[0] }
        for i in range(len(nodes_sys))
    }
    # For each column of sim (nodes of gold), store the nodes of sys that exceed the threshold
    sim_gold_to_sys = {
        nodes_gold[j]: { nodes_sys[i] for i in np.where(sim[:,j] > threshold)[0] }
        for j in range(len(nodes_gold))
    }

    # Fuzzy precision
    # |{(u′, v′) ∈ E′ | ∃(u, v) ∈ E. NodeSim(u, u′) > t ∧ NodeSim(v, v′) > t}|
    E_sys  = list(G_sys.edges())
    match_sys = 0
    for u_sys, v_sys in E_sys:
        # ¿Does (u_sys, v_sys) match with any (u, v) in G_gold?
        gold_u_cand = sim_sys_to_gold.get(u_sys, set())
        gold_v_cand = sim_sys_to_gold.get(v_sys, set())
        found = any((u, v) in G_gold.edges() for u in gold_u_cand for v in gold_v_cand)
        if found:
            match_sys += 1

    # Fuzzy recall
    # |{(u, v) ∈ E | ∃(u′, v′) ∈ E′. NodeSim(u, u′) > t ∧ NodeSim(v, v′) > t}|
    E_gold = list(G_gold.edges())
    match_gold = 0
    for u, v in E_gold:
        sys_u_cand = sim_gold_to_sys.get(u, set())
        sys_v_cand = sim_gold_to_sys.get(v, set())
        found = any((u_p, v_p) in G_sys.edges() for u_p in sys_u_cand for v_p in sys_v_cand)
        if found:
            match_gold += 1

    # Metrics
    P = match_sys  / len(E_sys)  if E_sys  else 0.0
    R = match_gold / len(E_gold) if E_gold else 0.0
    F1 = 2 * P * R / (P + R)     if (P + R) > 0 else 0.0

    return P, R, F1


def continuous_f1(
    G_sys: nx.DiGraph,
    G_gold: nx.DiGraph,
    model_name: str = 'all-MiniLM-L6-v2'
) -> tuple:
    """
    Computes the Continuous F1 metric between two directed graphs.

    Continuous precision = scont / |E_sys|
    Continuous recall    = scont / |E_gold|
    Continuous F1        = 2·P·R / (P+R)

    where scont is the sum of scores from the optimal edge matching,
    and the score for matching (u_sys→v_sys) with (u_gold→v_gold) is:
        min(NodeSim(u_sys, u_gold), NodeSim(v_sys, v_gold))

    Parameters
    ----------
    G_sys : nx.DiGraph
        Graph generated by the system (E').
    G_gold : nx.DiGraph
        Reference graph (E).
    model_name : str, optional
        SentenceTransformer model name (default is 'all-MiniLM-L6-v2').

    Returns
    -------
    - precision: continuous precision
    - recall:    continuous recall
    - f1:        continuous F1
    """
    # 1) Extract nodes and calculate embeddings
    nodes_sys   = list(G_sys.nodes())
    nodes_gold  = list(G_gold.nodes())
    emb_sys, emb_gold = embed_nodes(nodes_sys, nodes_gold, model_name)

    # 2) Similarity matrix
    # sim_nodes[i,j] = NodeSim(nodes_sys[i], nodes_gold[j])
    sim_nodes = cosine_similarity(emb_sys, emb_gold)

    # 3) Index edges of G_sys and G_gold
    E_sys   = list(G_sys.edges())
    E_gold  = list(G_gold.edges())
    m, n    = len(E_sys), len(E_gold)
    if m == 0 or n == 0:
        return 0, 0, 0

    # Map nodes to indices
    idx_sys  = { u:i for i, u in enumerate(nodes_sys) }
    idx_gold = { v:j for j, v in enumerate(nodes_gold) }

    # Build score matrix (m × n)
    # score[i,j] = min(sim(u_sys_i, u_gold_j), sim(v_sys_i, v_gold_j))
    S = np.zeros((m, n), dtype=float)
    for i, (u_sys, v_sys) in enumerate(E_sys):
        for j, (u_gold, v_gold) in enumerate(E_gold):
            s_u = sim_nodes[idx_sys[u_sys], idx_gold[u_gold]]
            s_v = sim_nodes[idx_sys[v_sys], idx_gold[v_gold]]
            S[i, j] = min(s_u, s_v)

    # Optimal matching using the Hungarian algorithm
    cost = -S
    row_ind, col_ind = linear_sum_assignment(cost)
    scont = S[row_ind, col_ind].sum()

    # 6) Metrics
    P = scont / m
    R = scont / n
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    return P, R, F1


def graph_f1(
    G_sys: nx.DiGraph,
    G_gold: nx.DiGraph,
    model_name: str = 'all-MiniLM-L6-v2',
    K: int = 2
) -> tuple:
    """
    Computes Graph F1 using message passing over embeddings (SGConv).

    Parameters
    ----------
    G_sys : nx.DiGraph
        System-generated graph.
    G_gold : nx.DiGraph
        Reference graph.
    model_name : str
        SentenceTransformer model to use.
    K : int
        Number of propagation steps in SGConv.

    Returns
    -------
    - precision: graph-based precision
    - recall:    graph-based recall
    - f1:        graph-based F1 score
    """
    # 1) Extract nodes and calculate embeddings
    nodes_sys   = list(G_sys.nodes())
    nodes_gold  = list(G_gold.nodes())
    emb_sys, emb_gold = embed_nodes(nodes_sys, nodes_gold, model_name)

    # Asociate each node with its embedding
    for i, n in enumerate(nodes_sys):
        G_sys.nodes[n]['x'] = torch.from_numpy(emb_sys[i]).float()
    for i, n in enumerate(nodes_gold):
        G_gold.nodes[n]['x'] = torch.from_numpy(emb_gold[i]).float()

    # Convert to PyTorch Geometric Data
    data_sys = from_networkx(G_sys, group_node_attrs=['x'])
    xs_sys   = np.vstack(data_sys.x)             
    data_sys.x = torch.from_numpy(xs_sys).float()

    data_gold = from_networkx(G_gold, group_node_attrs=['x'])
    xs_gold   = np.vstack(data_gold.x)
    data_gold.x = torch.from_numpy(xs_gold).float()

    # SGConv, Simple Graph Convolution, with added self-loops and identity weights
    d    = emb_sys.shape[1]
    conv = SGConv(d, d, K=K, cached=False, add_self_loops=True, bias=False)
    with torch.no_grad():
        conv.lin.weight.copy_(torch.eye(d))
    conv.lin.weight.requires_grad_(False)

    # 5) Convolution
    H_sys  = conv(data_sys.x,  data_sys.edge_index).cpu().numpy()
    H_gold = conv(data_gold.x, data_gold.edge_index).cpu().numpy()

    # 6) Similarity matrix and optimal matching
    sim            = cosine_similarity(H_sys, H_gold)
    row_ind, col_ind = linear_sum_assignment(-sim)
    sgraph         = sim[row_ind, col_ind].sum()

    n_sys, n_gold = len(nodes_sys), len(nodes_gold)
    P = sgraph / n_sys  if n_sys  > 0 else 0.0
    R = sgraph / n_gold if n_gold > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    return P, R, F1


def motif_distance(G1_nx: nx.DiGraph,
                   G2_nx: nx.DiGraph,
                   k: int = 3,
                   p: float = 1.0) -> float:
    """
    Computes the motif distance between G1_nx and G2_nx as the Total Variation Distance (TVD)
    between the distributions of their k-size directed subgraphs, grouping motifs by exact isomorphism.

    Parameters
    ----------
    G1_nx : nx.DiGraph
        First graph.
    G2_nx : nx.DiGraph
        Second graph.
    k : int
        Motif size (number of nodes).
    p : float
        Sampling probability for motif enumeration.

    Returns
    -------
    float
        Motif distance (TVD).
    """
    # Convert NetworkX graphs to graph-tool
    def nx_to_gt(G_nx: nx.DiGraph) -> Graph:
        Gt = Graph(directed=True)
        n = G_nx.number_of_nodes()
        Gt.add_vertex(n)

        idx = {v: i for i, v in enumerate(G_nx.nodes())}

        for u, v in G_nx.edges():
            Gt.add_edge(idx[u], idx[v])
        return Gt

    # Convert G1_nx and G2_nx to graph-tool graphs
    G1 = nx_to_gt(G1_nx)
    G2 = nx_to_gt(G2_nx)

    # Count motifs in both graphs
    motifs1, counts1 = motifs(G1, k, p=p)
    motifs2, counts2 = motifs(G2, k, p=p)

    # Agregate motifs by isomorphism
    all_motifs = list(motifs1)  
    for m2 in motifs2:
        # If m2 is not isomorphic to any motif in all_motifs, add it
        if not any(isomorphism(m2, m1) for m1 in all_motifs):
            all_motifs.append(m2)

    # Initialize counts for all motifs
    all_counts1 = np.zeros(len(all_motifs), dtype=float)
    all_counts2 = np.zeros(len(all_motifs), dtype=float)

    # Fill all_counts1
    for m, c in zip(motifs1, counts1):
        for j, gm in enumerate(all_motifs):
            if isomorphism(m, gm):
                all_counts1[j] = c
                break

    # Fill all_counts2
    for m, c in zip(motifs2, counts2):
        for j, gm in enumerate(all_motifs):
            if isomorphism(m, gm):
                all_counts2[j] = c
                break

    # Probability distribution normalization
    total1 = all_counts1.sum() or 1.0
    total2 = all_counts2.sum() or 1.0
    p_dist = all_counts1 / total1
    q_dist = all_counts2 / total2

    # Motif distance (TVD)
    tvd = 0.5 * np.abs(p_dist - q_dist).sum()
    return tvd


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py generatedOntology.ttl goldStandardOntology.ttl")
        sys.exit(1)

    ttl_file1 = sys.argv[1]
    ttl_file2 = sys.argv[2]

    G1 = ttl_to_graph(ttl_file1)
    G2 = ttl_to_graph(ttl_file2)

    # Metrics
    precisionLiteral, recallLiteral, f1Literal = literal_f1(G1, G2)
    pFuzzy, rFuzzy, f1Fuzzy = fuzzy_f1(G1, G2)
    pCont, rCont, f1Cont = continuous_f1(G1, G2)
    pGraph, rGraph, f1Graph = graph_f1(G1, G2)
    mDistance = motif_distance(G1, G2)

    print(f"Literal       : P={precisionLiteral:.4f} R={recallLiteral:.4f} F1={f1Literal:.4f}")
    print(f"Fuzzy F1      : P={pFuzzy:.4f} R={rFuzzy:.4f} F1={f1Fuzzy:.4f}")
    print(f"Continuous F1 : P={pCont:.4f} R={rCont:.4f} F1={f1Cont:.4f}")
    print(f"Graph F1      : P={pGraph:.4f} R={rGraph:.4f} F1={f1Graph:.4f}")
    print(f"Motif distance: {mDistance:.4f}")
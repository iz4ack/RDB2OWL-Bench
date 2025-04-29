import rdflib
import networkx as nx
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import numpy as np
from sentence_transformers import SentenceTransformer



from graph_metrics import (
    graph_precision_recall_f1,
    fuzzy_and_continuous_precision_recall_f1,
    literal_prec_recall_f1,
    motif_distance
)

def ttl_to_graph(ttl_file_path):
    # Cargar la ontología
    g = rdflib.Graph()
    g.parse(ttl_file_path, format="ttl")

    # Crear un grafo dirigido
    G = nx.DiGraph()

    for subj, pred, obj in g:
        # Asegurar que los nodos existan con atributo 'title'
        if pred != rdflib.RDFS.comment and pred != rdflib.RDFS.label: 
            if subj not in G:
                G.add_node(subj, title=str(subj))
            if obj not in G:
                G.add_node(obj, title=str(obj))

            # Agregar la arista (subj -> obj) con predicado como atributo opcional
            G.add_edge(subj, obj, predicate=str(pred))

    return G

def draw_graph(G, title="Graph", filename=None):
    mpl.rcParams['text.usetex'] = False  # <--- desactiva LaTeX

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    labels = {node: data.get('title', str(node)) for node, data in G.nodes(data=True)}
    edge_labels = {(u, v): d.get('predicate', '') for u, v, d in G.edges(data=True)}
    
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=500, node_color='skyblue', font_size=8, arrows=True)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    plt.title(title)
    if filename:
        plt.savefig(filename, format='png')
    plt.close()

def parse_title(t):
    # Parsear el titulo para quedarse con el nombre
    # http://example.org/universidad#Evaluacion , quedarse con Evaluacion
    if "#" in t:
        return t.split("#")[-1]
    elif "/" in t:
        return t.split("/")[-1]
    else:
        return t

def title(G, n):
    t = G.nodes[n]["title"]
    return parse_title(t)


def fuzzy_f1(
    G_sys: nx.DiGraph,
    G_gold: nx.DiGraph,
    threshold: float = 0.436,
    model_name: str = 'all-MiniLM-L6-v2'
) -> dict:
    """
    Calcula la métrica Fuzzy F1 entre dos grafos dirigidos.
    
    Parámetros
    ----------
    G_sys : nx.DiGraph
        Grafo generado por el sistema (E').
    G_gold : nx.DiGraph
        Grafo de referencia (E).
    threshold : float, opcional
        Umbral de similitud semántica (t). Por defecto 0.436.
    model_name : str, opcional
        Nombre del modelo de SentenceTransformer. Por defecto 'all-MiniLM-L6-v2'.
    
    Devuelve
    -------
          - 'precision': Fuzzy precision
          - 'recall':    Fuzzy recall
          - 'f1':        Fuzzy F1 = 2·P·R/(P+R) (0 si P+R == 0)
    """
    # 1. Extraer nodos y preparar textos
    nodes_sys   = list(G_sys.nodes())
    nodes_gold  = list(G_gold.nodes())
    texts_sys   = [parse_title(str(u)) for u in nodes_sys]
    texts_gold  = [parse_title(str(u)) for u in nodes_gold]

    # 2. Cargar modelo y calcular embeddings (numpy arrays)
    model = SentenceTransformer(model_name)
    all_texts = texts_sys + texts_gold
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    emb_sys  = embeddings[: len(nodes_sys)]
    emb_gold = embeddings[len(nodes_sys): ]

    # 3. Calcular matriz de similitud coseno
    #    sim[i,j] = cosine(emb_sys[i], emb_gold[j])
    #norm_sys  = np.linalg.norm(emb_sys,  axis=1, keepdims=True)
    #norm_gold = np.linalg.norm(emb_gold, axis=1, keepdims=True)
    #sim = (emb_sys @ emb_gold.T) / (norm_sys * norm_gold.T + 1e-8)
    sim = cosine_similarity(emb_sys, emb_gold)

    # 4. Para cada nodo, precomputar con quién supera el umbral
    # Para cada fila de sim(nodos de sys), guardar los nodos de gold que superan el umbral
    sim_sys_to_gold = {
        nodes_sys[i]: { nodes_gold[j] for j in np.where(sim[i] > threshold)[0] }
        for i in range(len(nodes_sys))
    }
    # Para cada columna de sim(nodos de gold), guardar los nodos de sys que superan el umbral
    sim_gold_to_sys = {
        nodes_gold[j]: { nodes_sys[i] for i in np.where(sim[:,j] > threshold)[0] }
        for j in range(len(nodes_gold))
    }

    # 5. Contar falsas positivas (precisión Fuzzy)
    # |{(u′, v′) ∈ E′ | ∃(u, v) ∈ E. NodeSim(u, u′) > t ∧ NodeSim(v, v′) > t}|
    E_sys  = list(G_sys.edges())
    match_sys = 0
    for u_sys, v_sys in E_sys:
        # ¿existe algún (u,v) en G_gold tal que u∼u_sys y v∼v_sys?
        gold_u_cand = sim_sys_to_gold.get(u_sys, set())
        gold_v_cand = sim_sys_to_gold.get(v_sys, set())
        found = any((u, v) in G_gold.edges() for u in gold_u_cand for v in gold_v_cand)
        if found:
            match_sys += 1

    # 6. Contar falsas negativas (recall Fuzzy)
    # |{(u, v) ∈ E | ∃(u′, v′) ∈ E′. NodeSim(u, u′) > t ∧ NodeSim(v, v′) > t}|
    E_gold = list(G_gold.edges())
    match_gold = 0
    for u, v in E_gold:
        sys_u_cand = sim_gold_to_sys.get(u, set())
        sys_v_cand = sim_gold_to_sys.get(v, set())
        found = any((u_p, v_p) in G_sys.edges() for u_p in sys_u_cand for v_p in sys_v_cand)
        if found:
            match_gold += 1

    # 7. Calcular métricas
    P = match_sys  / len(E_sys)  if E_sys  else 0.0
    R = match_gold / len(E_gold) if E_gold else 0.0
    F1 = 2 * P * R / (P + R)     if (P + R) > 0 else 0.0

    return P, R, F1


def continuous_f1(
    G_sys: nx.DiGraph,
    G_gold: nx.DiGraph,
    model_name: str = 'all-MiniLM-L6-v2'
) -> dict:
    """
    Calcula la métrica Continuous F1 entre dos grafos dirigidos.

    Continuous precision = scont / |E_sys|
    Continuous recall    = scont / |E_gold|
    Continuous F1        = 2·P·R/(P+R)

    donde scont es la suma de scores del matching óptimo de aristas,
    y el score de emparejar (u_sys→v_sys) con (u_gold→v_gold) es
        min(NodeSim(u_sys, u_gold), NodeSim(v_sys, v_gold))

    Parámetros
    ----------
    G_sys : nx.DiGraph
        Grafo generado por el sistema (E′).
    G_gold : nx.DiGraph
        Grafo de referencia (E).
    model_name : str, opcional
        Nombre del modelo SentenceTransformer (por defecto 'all-MiniLM-L6-v2').

    Devuelve
    -------
      - 'scont'    : puntuación total del matching
      - 'precision': continuous precision
      - 'recall'   : continuous recall
      - 'f1'       : continuous F1
    """
    # 1) Extraer nodos y calcular embeddings
    nodes_sys  = list(G_sys.nodes())
    nodes_gold = list(G_gold.nodes())
    texts      = [str(u) for u in nodes_sys] + [str(v) for v in nodes_gold]

    model     = SentenceTransformer(model_name)
    emb_all   = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    emb_sys   = emb_all[:len(nodes_sys)]
    emb_gold  = emb_all[len(nodes_sys):]

    # 2) Matriz de similitud nodo-sistema vs nodo-oro
    sim_nodes = cosine_similarity(emb_sys, emb_gold)
    # sim_nodes[i,j] = NodeSim(nodes_sys[i], nodes_gold[j])

    # 3) Indexar aristas
    E_sys   = list(G_sys.edges())
    E_gold  = list(G_gold.edges())
    m, n    = len(E_sys), len(E_gold)
    if m == 0 or n == 0:
        return 0, 0, 0

    # Mapeos de nodo → índice para acceder a sim_nodes
    idx_sys  = { u:i for i, u in enumerate(nodes_sys) }
    idx_gold = { v:j for j, v in enumerate(nodes_gold) }

    # 4) Construir matriz de score (m × n)
    #    score[i,j] = min(sim(u_sys_i, u_gold_j), sim(v_sys_i, v_gold_j))
    S = np.zeros((m, n), dtype=float)
    for i, (u_sys, v_sys) in enumerate(E_sys):
        for j, (u_gold, v_gold) in enumerate(E_gold):
            s_u = sim_nodes[idx_sys[u_sys], idx_gold[u_gold]]
            s_v = sim_nodes[idx_sys[v_sys], idx_gold[v_gold]]
            S[i, j] = min(s_u, s_v)

    # 5) Resolver matching óptimo (Hungarian) para maximizar sum(S)
    #    convertimos a coste negando S
    cost = -S
    row_ind, col_ind = linear_sum_assignment(cost)
    scont = S[row_ind, col_ind].sum()

    # 6) Calcular precision, recall y F1 continuos
    P = scont / m
    R = scont / n
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

    return P, R, F1



if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python script.py ontologia1.ttl ontologia2.ttl")
        sys.exit(1)

    ttl_file1 = sys.argv[1]
    ttl_file2 = sys.argv[2]

    G1 = ttl_to_graph(ttl_file1)
    G2 = ttl_to_graph(ttl_file2)

    # Graficar los grafos
    #draw_graph(G1, title="Ontology 1 Graph", filename="ontology1_graph.png")
    #draw_graph(G2, title="Ontology 2 Graph", filename="ontology2_graph.png")

    #Metricas 
    precisionLiteral, recallLiteral, f1Literal = literal_prec_recall_f1(G1, G2)
    pFuzzy, rFuzzy, f1Fuzzy = fuzzy_f1(G1, G2)
    pCont, rCont, f1Cont = continuous_f1(G1, G2)

    print(f"Literal     : P={precisionLiteral:.4f} R={recallLiteral:.4f} F1={f1Literal:.4f}")
    print(f"Fuzzy F1      : P={pFuzzy:.4f} R={rFuzzy:.4f} F1={f1Fuzzy:.4f}")
    print(f"Continuous F1 : P={pCont:.4f} R={rCont:.4f} F1={f1Cont:.4f}")


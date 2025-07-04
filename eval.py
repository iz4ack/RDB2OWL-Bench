import os
import sys
import argparse
import csv
from metrics import (   
    literal_f1, fuzzy_f1, continuous_f1, graph_f1, motif_distance, ttl_to_graph, resource_f1
)
from chokepoints import get_basic_subgraph
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates generated ontologies against gold standards using multiple metrics."
    )
    parser.add_argument("--gen-dir", required=True, help="Directory with generated ontologies (.ttl)")
    parser.add_argument("--gold-dir", help="Directory with gold standard ontologies (.ttl)", default="recursos/ontologies")
    parser.add_argument("--output-csv", "-o", default="evaluation_results.csv", help="Output CSV file name with metrics, it will be saved in the gen_dir")
    args = parser.parse_args()
    
    output_path = os.path.join(args.gen_dir, args.output_csv)

    with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = [
            "file",
            "literal_P", "literal_R", "literal_F1",
            "fuzzy_P", "fuzzy_R", "fuzzy_F1",
            "cont_P", "cont_R", "cont_F1",
            "graph_P", "graph_R", "graph_F1",
            "motif_dist", 
            "resource_P", "resource_R", "resource_F1"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        ontologies = []
        for o in os.listdir(args.gen_dir):
            if o.endswith((".ttl", ".xml")):
                ontologies.append(o)
        
        for fname in tqdm(sorted(ontologies), desc="Evaluating ontologies"):
            if fname.endswith(".ttl"): fmt = "ttl"
            elif fname.endswith(".xml"): fmt = "xml"
            else: continue  # Skip non-TTL/XML files

            gold_name = os.path.splitext(fname)[0]
            gold_name += ".ttl"

            gen_path = os.path.join(args.gen_dir, fname)
            gold_path = os.path.join(args.gold_dir, gold_name)

            if not os.path.exists(gold_path):
                print(f"Warning: gold standard not found for {fname}, skipping.", file=sys.stderr)
                continue

            G_sys = ttl_to_graph(gen_path, fmt=fmt)
            G_gold = ttl_to_graph(gold_path, fmt="ttl")

            if G_sys is None or G_gold is None:
                print(f"Warning: failed to parse {fname}, skipping.", file=sys.stderr)
                writer.writerow({
                    "file": fname,
                    "literal_P": 0, "literal_R": 0, "literal_F1": 0,
                    "fuzzy_P": 0, "fuzzy_R": 0, "fuzzy_F1": 0,
                    "cont_P": 0, "cont_R": 0, "cont_F1": 0,
                    "graph_P": 0, "graph_R": 0, "graph_F1": 0,
                    "motif_dist": 1,
                    "resource_P": 0, "resource_R": 0, "resource_F1": 0
                })
                continue
            
            # Run metrics for each chokepoint

            G_sys0 = get_basic_subgraph(G_sys)
            G_gold0 = get_basic_subgraph(G_gold)

            for G_sys_cp, G_gold_cp, name in [(G_sys0, G_gold0, "basic"), (G_sys, G_gold, "full")]:
                # Compute metrics
                litP, litR, litF = literal_f1(G_sys_cp, G_gold_cp)
                fzP, fzR, fzF = fuzzy_f1(G_sys_cp, G_gold_cp)
                cP, cR, cF = continuous_f1(G_sys_cp, G_gold_cp)
                gP, gR, gF = graph_f1(G_sys_cp, G_gold_cp)
                m_dist = motif_distance(G_sys_cp, G_gold_cp)
                resourceP, resourceR, resourceF = resource_f1(G_sys_cp, G_gold_cp)

                writer.writerow({
                    "file": f"{fname} ({name})",
                    "literal_P": f"{litP:.4f}", "literal_R": f"{litR:.4f}", "literal_F1": f"{litF:.4f}",
                    "fuzzy_P": f"{fzP:.4f}", "fuzzy_R": f"{fzR:.4f}", "fuzzy_F1": f"{fzF:.4f}",
                    "cont_P": f"{cP:.4f}", "cont_R": f"{cR:.4f}", "cont_F1": f"{cF:.4f}",
                    "graph_P": f"{gP:.4f}", "graph_R": f"{gR:.4f}", "graph_F1": f"{gF:.4f}",
                    "motif_dist": f"{m_dist:.4f}",
                    "resource_P": f"{resourceP:.4f}", "resource_R": f"{resourceR:.4f}", "resource_F1": f"{resourceF:.4f}"
                })
                #print(f"\nEvaluated {fname} (chokepoint {i})")
    print(f"\nEvaluation completed. Results in {args.output_csv}")

if __name__ == "__main__":
    main()

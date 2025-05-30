import os
import sys
import argparse
import csv
from metrics import (   
    literal_f1, fuzzy_f1, continuous_f1, graph_f1, motif_distance, ttl_to_graph
)
from chokepoints import get_chokepoint_subgraph
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser(
        description="Evaluates generated ontologies against gold standards using multiple metrics."
    )
    parser.add_argument("--gen-dir", required=True, help="Directory with generated ontologies (.ttl)")
    parser.add_argument("--gold-dir", help="Directory with gold standard ontologies (.ttl)", default="recursos/ontologies")
    parser.add_argument("--output-csv", "-o", default="evaluation_results.csv", help="Output CSV file name with metrics, it will be saved in the gen_dir")
    parser.add_argument("--chokepoint", "-c", type=int, default=2, help="Max chokepoint level to evaluate (0-2)")
    args = parser.parse_args()
    
    output_path = os.path.join(args.gen_dir, args.output_csv)

    with open(output_path, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = [
            "file",
            "literal_P", "literal_R", "literal_F1",
            "fuzzy_P", "fuzzy_R", "fuzzy_F1",
            "cont_P", "cont_R", "cont_F1",
            "graph_P", "graph_R", "graph_F1",
            "motif_dist"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for fname in sorted(os.listdir(args.gen_dir)):
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
                    "literal_P": "N/A", "literal_R": "N/A", "literal_F1": "N/A",
                    "fuzzy_P": "N/A", "fuzzy_R": "N/A", "fuzzy_F1": "N/A",
                    "cont_P": "N/A", "cont_R": "N/A", "cont_F1": "N/A",
                    "graph_P": "N/A", "graph_R": "N/A", "graph_F1": "N/A",
                    "motif_dist": "N/A"
                })
                continue
            
            # Run metrics for each chokepoint
            for i in tqdm(range(args.chokepoint + 1), desc=f"Evaluating {fname}"):
                G_sys_cp = get_chokepoint_subgraph(G_sys, i) if i != 2 else G_sys
                G_gold_cp = get_chokepoint_subgraph(G_gold, i) if i != 2 else G_gold

                if G_sys_cp is None or G_gold_cp is None:
                    print(f"Warning: failed to apply chokepoint {i} for {fname}, skipping.", file=sys.stderr)
                    continue
                # Compute metrics
                litP, litR, litF = literal_f1(G_sys_cp, G_gold_cp)
                fzP, fzR, fzF = fuzzy_f1(G_sys_cp, G_gold_cp)
                cP, cR, cF = continuous_f1(G_sys_cp, G_gold_cp)
                gP, gR, gF = graph_f1(G_sys_cp, G_gold_cp)
                m_dist = motif_distance(G_sys_cp, G_gold_cp)

                writer.writerow({
                    "file": f"{fname} (chokepoint {i})",
                    "literal_P": f"{litP:.4f}", "literal_R": f"{litR:.4f}", "literal_F1": f"{litF:.4f}",
                    "fuzzy_P": f"{fzP:.4f}", "fuzzy_R": f"{fzR:.4f}", "fuzzy_F1": f"{fzF:.4f}",
                    "cont_P": f"{cP:.4f}", "cont_R": f"{cR:.4f}", "cont_F1": f"{cF:.4f}",
                    "graph_P": f"{gP:.4f}", "graph_R": f"{gR:.4f}", "graph_F1": f"{gF:.4f}",
                    "motif_dist": f"{m_dist:.4f}"
                })
                #print(f"\nEvaluated {fname} (chokepoint {i})")
    print(f"\nEvaluation completed. Results in {args.output_csv}")

if __name__ == "__main__":
    main()

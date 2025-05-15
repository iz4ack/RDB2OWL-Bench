import os
import sys
import argparse
import csv
from metrics import (   
    literal_f1, fuzzy_f1, continuous_f1, graph_f1, motif_distance, ttl_to_graph
)


def main():
    parser = argparse.ArgumentParser(
        description="Evalúa ontologías generadas contra gold standards usando múltiples métricas."
    )
    parser.add_argument("--gen-dir", required=True, help="Directorio con ontologías generadas (.ttl)")
    parser.add_argument("--gold-dir", help="Directorio con ontologías gold estándar (.ttl)", default="recursos/ontologies")
    parser.add_argument("--output-csv", default="evaluation_results.csv", help="Fichero CSV de salida con métricas")
    args = parser.parse_args()

    with open(args.output_csv, "w", newline='', encoding="utf-8") as csvfile:
        fieldnames = [
            "file",
            "literal_P","literal_R","literal_F1",
            "fuzzy_P","fuzzy_R","fuzzy_F1",
            "cont_P","cont_R","cont_F1",
            "graph_P","graph_R","graph_F1",
            "motif_dist"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for fname in sorted(os.listdir(args.gen_dir)):
            if not fname.endswith(".ttl"): continue
            gen_path = os.path.join(args.gen_dir, fname)
            gold_path = os.path.join(args.gold_dir, fname)
            if not os.path.exists(gold_path):
                print(f"Aviso: no encontrado gold para {fname}, se omite.", file=sys.stderr)
                continue

            G_sys = ttl_to_graph(gen_path)
            G_gold = ttl_to_graph(gold_path)

            # Calcula métricas
            litP, litR, litF = literal_f1(G_sys, G_gold)
            fzP, fzR, fzF = fuzzy_f1(G_sys, G_gold)
            cP, cR, cF = continuous_f1(G_sys, G_gold)
            gP, gR, gF = graph_f1(G_sys, G_gold)
            m_dist = motif_distance(G_sys, G_gold)

            writer.writerow({
                "file": fname,
                "literal_P": f"{litP:.4f}", "literal_R": f"{litR:.4f}", "literal_F1": f"{litF:.4f}",
                "fuzzy_P": f"{fzP:.4f}", "fuzzy_R": f"{fzR:.4f}", "fuzzy_F1": f"{fzF:.4f}",
                "cont_P": f"{cP:.4f}", "cont_R": f"{cR:.4f}", "cont_F1": f"{cF:.4f}",
                "graph_P": f"{gP:.4f}", "graph_R": f"{gR:.4f}", "graph_F1": f"{gF:.4f}",
                "motif_dist": f"{m_dist:.4f}"
            })
            print(f"Evaluado {fname}")
    print(f"Evaluación completada. Resultados en {args.output_csv}")

if __name__ == "__main__":
    main()

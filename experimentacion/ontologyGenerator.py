#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import time
import csv
from huggingface_hub import InferenceClient
from tqdm import tqdm
import re

def parse_args():
    parser = argparse.ArgumentParser(
        description="Automatically generates OWL ontologies (Turtle) from SQL schemas using an LLM model from Hugging Face."
    )
    parser.add_argument(
        "--input-dir", "-i",
        help="Folder with .sql files (each one is an RDB schema).",
        default="/home/sausage69/OneDrive/GreI/4º/2Semestre/tfg/tfg/recursos/rdbSchemas"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Folder where the generated .ttl files will be written.",
        default="/home/sausage69/OneDrive/GreI/4º/2Semestre/tfg/tfg/experimentacion/results"
    )
    parser.add_argument(
        "--log-file", "-l", default="generation_log.csv",
        help="Path to the CSV where generation times and status will be logged."
    )
    parser.add_argument(
        "--model", "-m", default="meta-llama/Llama-3.3-70B-Instruct",
        help="Model name on Hugging Face Hub."
    )
    parser.add_argument(
        "--provider", "-p", default="together",
        help="Provider for InferenceClient (e.g., 'together', 'huggingface').",
    )
    parser.add_argument(
        "--api-key", "-k", default=None,
        help="API key for Hugging Face (if not using the HF_API_KEY environment variable)."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Maximum number of generation tokens."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Generation temperature."
    )
    return parser.parse_args()

def load_sql_schema(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def build_prompt(sql_schema: str) -> str:
    return (
        "Please convert the following SQL schema into an OWL ontology using Turtle syntax. "
        "Define classes for each table, object and data properties according to the columns, "
        "include appropriate domains and ranges, and use standard prefixes (rdf, rdfs, owl, xsd). "
        "Return only the Turtle content, with no additional explanations.\n\n"
        "SQL Schema:\n"
        "```sql\n"
        f"{sql_schema}\n"
        "```"
    )

def _sanitize(response):
    match = re.search(r"```turtle(.*?)```", response, re.DOTALL)
    return match.group(1).strip() if match else response

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize HF client
    client = InferenceClient(
        provider=args.provider,
        api_key=args.api_key or os.getenv("HF_API_KEY")
    )

    # Prepare log CSV
    log_path = os.path.join(args.output_dir, args.model)
    log_path = os.path.join(log_path, args.log_file)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "schema_file", "ttl_file", "duration_s", "status", "error_msg"
        ])
        writer.writeheader()

        # Iterate over each .sql file in input-dir
        for fname in tqdm(sorted(os.listdir(args.input_dir))):
            if not fname.lower().endswith(".sql"):
                continue

            schema_path = os.path.join(args.input_dir, fname)
            base = os.path.splitext(fname)[0]
            ttl_path = os.path.join(args.output_dir, args.model)
            ttl_path = os.path.join(ttl_path, base + ".ttl")

            sql_schema = load_sql_schema(schema_path)
            prompt = build_prompt(sql_schema)

            start = time.time()
            try:
                # Build messages
                messages = [
                    {"role": "user", "content": prompt}
                ]
                # API call
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                turtle = _sanitize(completion.choices[0].message["content"])

                # Save TTL
                with open(ttl_path, "w", encoding="utf-8") as out:
                    out.write(turtle.strip() + "\n")

                duration = time.time() - start
                writer.writerow({
                    "schema_file": fname,
                    "ttl_file": os.path.basename(ttl_path),
                    "duration_s": f"{duration:.2f}",
                    "status": "OK",
                    "error_msg": ""
                })

            except Exception as e:
                duration = time.time() - start
                writer.writerow({
                    "schema_file": fname,
                    "ttl_file": "",
                    "duration_s": f"{duration:.2f}",
                    "status": "ERROR",
                    "error_msg": str(e)
                })
                # continue with the next file
                continue

    print(f"Process completed. Ontologies saved in: {args.output_dir}")
    print(f"Execution log: {log_path}")

if __name__ == "__main__":
    main()

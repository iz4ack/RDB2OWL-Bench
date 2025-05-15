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
        description="Genera automáticamente ontologías OWL (Turtle) a partir de esquemas SQL usando un modelo LLM de Hugging Face."
    )
    parser.add_argument(
        "--input-dir", "-i",
        help="Carpeta con archivos .sql (cada uno un esquema RDB).",
        default="/home/sausage69/OneDrive/GreI/4º/2Semestre/tfg/tfg/recursos/rdbSchemas"
    )
    parser.add_argument(
        "--output-dir", "-o",
        help="Carpeta donde se escribirán los archivos .ttl generados.",
        default="/home/sausage69/OneDrive/GreI/4º/2Semestre/tfg/tfg/experimentacion/generatedOntologies"
    )
    parser.add_argument(
        "--log-file", "-l", default="generation_log.csv",
        help="Ruta al CSV donde se guardarán tiempos y estado de generación."
    )
    parser.add_argument(
        "--model", "-m", default="meta-llama/Llama-3.3-70B-Instruct",
        help="Nombre del modelo en Hugging Face Hub."
    )
    parser.add_argument(
        "--provider", "-p", default="together",
        help="Proveedor para InferenceClient (ej. 'together', 'huggingface').",
    )
    parser.add_argument(
        "--api-key", "-k", default=None,
        help="API key para Hugging Face (si no se usa HF_API_KEY env var)."
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8192,
        help="Máximo tokens de generación."
    )
    parser.add_argument(
        "--temperature", type=float, default=0.3,
        help="Temperatura de generación."
    )
    return parser.parse_args()

def load_sql_schema(path):
    with open(path, encoding="utf-8") as f:
        return f.read()

def build_prompt(sql_schema: str) -> str:
    return (
        "Por favor, convierte el siguiente esquema SQL en una ontología OWL usando sintaxis Turtle. "
        "Define clases para cada tabla, propiedades de objeto y de datos según las columnas, "
        "incluye dominios y rangos apropiados, y usa prefijos estándar (rdf, rdfs, owl, xsd). "
        "Sólo devuelve el contenido Turtle, sin explicaciones adicionales.\n\n"
        "Esquema SQL:\n"
        "```sql\n"
        f"{sql_schema}\n"
        "```"
    )

def _sanitize(respuesta):
	match = re.search(r"```turtle(.*?)```", respuesta, re.DOTALL)
	return match.group(1).strip() if match else respuesta

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Inicializa cliente HF
    client = InferenceClient(
        provider=args.provider,
        api_key=args.api_key or os.getenv("HF_API_KEY")
    )

    # Prepara log CSV
    log_path = os.path.join(args.output_dir, args.log_file)
    with open(log_path, mode="w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=[
            "schema_file", "ttl_file", "duration_s", "status", "error_msg"
        ])
        writer.writeheader()

        # Itera cada .sql en input-dir
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
                # Construye mensajes
                messages = [
                    {"role": "user", "content": prompt}
                ]
                # Llamada a la API
                completion = client.chat.completions.create(
                    model=args.model,
                    messages=messages,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                turtle = _sanitize(completion.choices[0].message["content"])

                # Guarda TTL
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
                # continúa con el siguiente
                continue

    print(f"Proceso completado. Ontologías en: {args.output_dir}")
    print(f"Registro de ejecución: {log_path}")

if __name__ == "__main__":
    main()

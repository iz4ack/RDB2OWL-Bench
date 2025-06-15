# Ontology Generation Benchmark from Relational Databases

This repository provides a **benchmarking framework for the automatic generation of ontologies** from relational databases. It includes:

- A collection of aligned **SQL schemas and their corresponding ontologies** (in Turtle format).
- **Evaluation metrics** for comparing generated ontologies to reference ontologies.
- A **small-scale experimentation** with **LLMs** (e.g., ChatGPT, LLaMA, DeepSeek, etc.) to assess their performance on this task.

## Project Structure

The repository is organized as follows:

- The `recursos/` directory contains the core benchmark resources: relational database schemas (`.sql`) and their associated reference ontologies (`.ttl`).
- The `experimentacion/` folder includes scripts, prompts, model outputs, evaluation results, and visualizations from the experiments conducted using various LLMs.
- The `metrics.py`, and `chokepoints.py` files implement the ontology comparison metrics used in the evaluation.
- The `eval.py` script runs the complete evaluation pipeline.
- The `pixi.toml` and `pixi.lock` files define the environment using Pixi, a reproducible package manager.

## Evaluation Metrics

The benchmark includes several ontology comparison metrics, such as:

- Literal F1 score
- Fuzzy F1 score
- Continous F1 score
- Graph F1 score
- Motif distribution distance

These metrics aim to capture both structural and semantic correspondence between ontologies.

## LLM-Based Ontology Generation

In the `experimentacion/` directory, various LLMs are evaluated for their ability to generate ontologies from SQL schemas. For each model, the directory contains:

- The generated ontologies (`.ttl` files)
- Evaluation and generation logs (`.csv`)
- Performance plots (`.png`)

## Visualizations

Plots comparing the performance of different models and metrics are available in the subdirectories under `experimentacion/graficos/`.

## Requirements

To install dependencies and run the evaluation:

```bash
pixi install
```

## Scripts Usage
# === eval.py ===
Evaluates a folder of generated ontologies against gold standards using multiple metrics

python eval.py --gen-dir <generated_ontologies_folder> \
               [--gold-dir <gold_standard_folder>] \
               [--output-csv <output_csv_name>]

Example:
python eval.py --gen-dir experimentacion/results/my-model/default


# === ontologyGenerator.py ===
Generates OWL ontologies (Turtle) from SQL schemas using an LLM via Hugging Face

python experimentacion/ontologyGenerator.py \
  [--input-dir <sql_input_dir>] \
  [--output-dir <ttl_output_dir>] \
  [--log-file <log_csv>] \
  [--model <hf_model_name>] \
  [--prompt-name <prompt_name>] \
  [--provider <provider>] \
  [--api-key <your_api_key>] \
  [--max-tokens <int>] \
  [--temperature <float>]

Example:
python experimentacion/ontologyGenerator.py \
  --input-dir recursos/rdbSchemas \
  --output-dir experimentacion/results/my-model/default \
  --model meta-llama/Llama-3.3-70B-Instruct \
  --provider together \
  --log-file generation_log.csv


# === metrics.py ===
Evaluates a single pair of ontologies: one generated and one gold standard

python metrics.py <generatedOntology.ttl> <goldStandardOntology.ttl>

Example:
python metrics.py r3_generated.ttl r3_gold.ttl


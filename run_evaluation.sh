echo "Running ontology evaluation with various models..."

echo "Evaluating DeepSeek V3 model..."
python eval.py --gen-dir experimentacion/results/deepseek-ai/DeepSeek-V3/default

echo "Evaluating gemma-3-27b-it model..."
python eval.py --gen-dir experimentacion/results/google/gemma-3-27b-it/default

echo "Evaluating Llama-4-Maverick-17B-128E-Instruct..."
python eval.py --gen-dir experimentacion/results/meta-llama/Llama-4-Maverick-17B-128E-Instruct/default

echo "Evaluating Llama-3.3-70B-Instruct..."
python eval.py --gen-dir experimentacion/results/meta-llama/Llama-3.3-70B-Instruct/default

echo "Evaluating chatgpt-4o..."
python eval.py --gen-dir experimentacion/results/openai/chatgpt-4o/default

echo "Evaluating chatgpt-o4-mini-high..."
python eval.py --gen-dir experimentacion/results/openai/chatgpt-o4-mini-high/default


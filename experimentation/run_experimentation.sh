echo "Running ontology generation with various models..."

echo "Running DeepSeek V3 model..."
python ontologyGenerator.py -m deepseek-ai/DeepSeek-V3 

echo "Running gemma-3-27b-it model..."
python ontologyGenerator.py -m google/gemma-3-27b-it -p nebius

echo "Running Llama 4 Maverick..."
python ontologyGenerator.py -m meta-llama/Llama-4-Maverick-17B-128E-Instruct -p fireworks-ai

#echo "Running Mixtral 8x22B model..."
#python ontologyGenerator.py -m mistralai/Mixtral-8x22B-Instruct-v0.1

echo "Running Llama 3.3 model..."
python ontologyGenerator.py -m meta-llama/Llama-3.3-70B-Instruct



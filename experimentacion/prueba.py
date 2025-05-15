import os

output_dir = "/home/sausage69/OneDrive/GreI/4º/2Semestre/tfg/tfg/experimentacion/generatedOntologies"
model = "meta-llama/Llama-3.3-70B-Instruct"
base = "schemaR1"

turtle = "hola que tal"

ttl_path = os.path.join(output_dir, model)
ttl_path = os.path.join(ttl_path, base + ".ttl")

os.makedirs(os.path.dirname(ttl_path), exist_ok=True)

with open(ttl_path, "w+", encoding="utf-8") as out:
    out.write(turtle.strip() + "\n")

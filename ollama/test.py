import ollama   

# Obtener la lista de modelos instalados
def get_installed_models():
    response: ollama.ListResponse = ollama.list()
    return [model.model for model in response.models]


# Mostrar modelos disponibles
available_models = get_installed_models()

print("Modelos disponibles:", ", ".join(available_models))
modelo = input("Selecciona un modelo: ")

# Verificar si el modelo seleccionado está instalado
while modelo not in available_models:
    print("Modelo no encontrado. Inténtalo de nuevo.")
    modelo = input("Selecciona un modelo: ")

message = input("> ")

while message.lower() != 'exit':
    response = ollama.chat(model=modelo, messages=[
        {'role': 'user', 'content': message}
    ], stream=True)

    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)
    
    message = input("\n> ")

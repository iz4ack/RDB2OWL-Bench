import ollama   

message = input("> ")

while message != 'exit':
    response = ollama.chat(model='llama3.2:1b', messages=[
    {
        'role': 'user',
        'content': message,
    }], 
    stream=True)

    for chunk in response:
        print(chunk['message']['content'], end='', flush=True)
    
    message = input("\n> ")
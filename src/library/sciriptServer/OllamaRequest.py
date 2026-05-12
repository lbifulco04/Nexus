import requests

def chiedi_a_granite(prompt):
    """
    Invia un prompt a Ollama e restituisce la risposta come stringa.
    """
    url = "http://127.0.0.1:11434/api/generate"
    payload = {
        "model": "granite-code:3b",  # Assicurati che il nome sia esatto
        "prompt": prompt,
        "stream": False              # Aspetta la risposta completa
    }
    
    try:
        risposta = requests.post(url, json=payload)
        risposta.raise_for_status()  # Verifica che la richiesta sia andata a buon fine
        return risposta.json()["response"]
    except requests.exceptions.RequestException as e:
        return f"Errore di connessione: {e}"

# --- ESEMPIO DI UTILIZZO ---
if __name__ == "__main__":
    mio_prompt = input("Inserisci un prompt da inviare a granite-code:3b")
    print("In attesa di Granite...")
    
    risultato = chiedi_a_granite(mio_prompt)
    print("\n--- RISPOSTA ---")
    print(risultato)
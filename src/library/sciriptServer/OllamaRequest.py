import json
import requests
import os
import time
from tqdm import tqdm  # Libreria per le barre di avanzamento

# --- CONFIGURAZIONE ---
# Specifichiamo il modello Granite scaricato localmente su Ollama
MODELLO = "granite-code:3b" 
# Indirizzo API standard di Ollama per la generazione di testo
URL_OLLAMA = "http://127.0.0.1:11434/api/generate"

def interroga_granite(testo_input):
    """
    Invia il prompt evoluto a Granite per l'ottimizzazione tecnica.
    Il prompt funge da 'istruttore digitale' per correggere la guida umana.
    """
    
    # Questo testo spiega all'IA la fisica del simulatore e come reagire
    istruzioni_tecniche = (
    "Sei un Ingegnere che deve pilotare un auto su TORCS. Obiettivo: velocità e stabilità.\n"
    "1. RETTILINEI: sensore>150m, accel=1.0, steer=0.\n"
    "2. CURVE: fuori-dentro-fuori. Parzializza accel.\n"
    "3. ERBA: trackPos>0.9, accel=0.3, steer verso 0.\n"
    "4. SBANDATA: angle>0.15, CONTROSTERZO, accel=0.\n"
    "RISPONDI SOLO JSON: {'steer': float, 'accel': float, 'brake': float}"
)


    # Preparazione del pacchetto dati (payload) da inviare a Ollama
    payload = {
        "model": MODELLO,
        "prompt": f"{istruzioni_tecniche}\n\nInput: {testo_input}",
        "stream": False,  # False per ricevere la risposta completa in un colpo solo
        "format": "json"  # Forza l'IA a rispondere esclusivamente in formato JSON
    }

    try:
        # Esegue la richiesta HTTP POST al server locale di Ollama
        risposta = requests.post(URL_OLLAMA, json=payload, timeout=60)
        risposta.raise_for_status() # Genera un errore se la richiesta fallisce
        return risposta.json()['response'] # Restituisce il testo generato dall'IA
    except Exception as e:
        # Se c'è un errore (es. Ollama spento), lo stampa senza fermare il programma
        tqdm.write(f"❌ Errore Ollama: {e}")
        return None

def elabora_tutti_i_settori():
    """
    Cerca tutti i file pronti nella cartella e li processa uno dopo l'altro
    senza interruzioni manuali.
    """
    # Crea una lista di tutti i file che finiscono con '_READY_FOR_OLLAMA.jsonl'
    # Cerca tutti i file .jsonl, ma esclude quelli già processati (_FINALE_WATSON) per evitare loop
    file_da_elaborare = [f for f in os.listdir('.') if f.endswith('.jsonl') and "_FINALE_WATSON" not in f]
    
    if not file_da_elaborare:
        print("❓ Nessun file '_READY_FOR_OLLAMA.jsonl' trovato.")
        return

    print(f"🚀 Inizio ottimizzazione batch con Granite...")
    inizio_totale = time.time() # Tempo di inizio per calcolare la durata totale

    # Barra di avanzamento principale: indica quanti file sono stati completati
    pbar_file = tqdm(file_da_elaborare, desc="Progresso Totale", unit="file")

    for nome_file in pbar_file:
        # Crea il nome del file finale sostituendo il suffisso
        file_output = nome_file.replace("_READY_FOR_OLLAMA.jsonl", "_FINALE_WATSON_3B.jsonl")
        
        # Conta quante righe ci sono nel file per inizializzare la barra interna
        with open(nome_file, 'r') as f:
            totale_righe = sum(1 for _ in f)

        # Barra di avanzamento interna: mostra il progresso riga per riga del file attuale
        pbar_righe = tqdm(total=totale_righe, desc=f" ↳ {nome_file}", leave=False, unit="riga")

        # Apre il file di input in lettura e quello di output in scrittura
        with open(nome_file, 'r') as f_in, open(file_output, 'w') as f_out:
            for riga in f_in:
                if not riga.strip(): 
                    pbar_righe.update(1)
                    continue
                
                # Decodifica la riga JSON per estrarre il prompt di telemetria
                dato = json.loads(riga)
                testo_telemetria = dato['input']

                # Chiama la funzione per ottenere i comandi 'puliti' dall'IA
                risposta_ai = interroga_granite(testo_telemetria)

                if risposta_ai:
                    # Costruisce la struttura finale richiesta da IBM Watsonx.ai
                    voce_dataset = {
                        "input": testo_telemetria,
                        "output": risposta_ai.strip()
                    }
                    # Scrive la nuova riga nel file finale
                    f_out.write(json.dumps(voce_dataset) + "\n")
                
                time.sleep(0.5)
                
                # Aggiorna la barra delle righe
                pbar_righe.update(1)
        
        # Chiude la barra del file attuale e stampa un messaggio di successo
        pbar_righe.close()
        tqdm.write(f"✅ Completato: {file_output}")

    # Calcolo del tempo totale impiegato per l'intera operazione
    fine_totale = time.time()
    durata_minuti = (fine_totale - inizio_totale) / 60
    print(f"\n✨ OPERAZIONE TERMINATA in {durata_minuti:.2f} minuti")

# Punto di ingresso del programma
if __name__ == "__main__":
    elabora_tutti_i_settori()
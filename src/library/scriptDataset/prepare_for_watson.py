import json
import os

def prepara_dataset_per_ottimizzazione():
    nome_file_input = input("📄 Inserisci il nome del file da preparare (es: data_curva1.json): ").strip()
    if not os.path.exists(nome_file_input):
        print("❌ File non trovato!")
        return

    nome_file_output = nome_file_input.replace(".json", "_READY_FOR_OLLAMA.jsonl")
    conteggio = 0

    print(f"🔄 Preparazione di {nome_file_input}...")

    with open(nome_file_input, 'r') as f_in, open(nome_file_output, 'w') as f_out:
        for riga in f_in:
            try:
                data = json.loads(riga)
                sensori = data['sensors']
                azioni = data['actions']

                # --- FILTRO ---
                # Scartiamo se l'auto è fuori pista o ferma/incastrata
                if -1.0 in sensori['track'] or sensori['speedX'] < 5.0:
                    continue

                # 1. Costruzione dell'Input (Il contesto per Granite e poi per Watson)
                # Arrotondiamo i valori per rendere il testo più leggibile al modello
                input_testo = (
                    f"Telemetria: velocità {sensori['speedX']:.2f} km/h, "
                    f"RPM {sensori['rpm']:.0f}, "
                    f"marcia {azioni.get('gear', 1)}, "
                    f"angolo {sensori['angle']:.4f}, "
                    f"posizione {sensori['trackPos']:.4f}. "
                    f"Sensori pista: {[round(s, 2) for s in sensori['track']]}. "
                    f"Genera i comandi ottimali di sterzo, accelerazione e freno."
                )

                # 2. Struttura provvisoria
                # Inseriamo i comandi umani come base, che Granite andrà a correggere
                jsonl_entry = {
                    "input": input_testo,
                    "output": "" # Questo verrà riempito da Granite
                }

                f_out.write(json.dumps(jsonl_entry) + '\n')
                conteggio += 1

            except Exception as e:
                continue

    print(f"✅ Successo! Creato file '{nome_file_output}' con {conteggio} righe pulite.")
    print(f"👉 Ora puoi passare questo file allo script di Ollama per riempire gli 'output'.")

if __name__ == "__main__":
    prepara_dataset_per_ottimizzazione()
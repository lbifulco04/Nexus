import json

INPUT_FILE = 'expert_data_cleaned.json'
OUTPUT_FILE = 'granite_race_train.jsonl'

def create_fine_tuning_dataset():
    count = 0
    print(f"Avvio conversione di {INPUT_FILE}...")

    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                sensors = data['sensors']
                actions = data['actions']

                # 1. Costruzione del Prompt (Input per il modello)
                # Includiamo tutti i dati rilevanti per una guida esperta
                input_data = (
                    f"Telemetria: velocità {sensors['speedX']} km/h, "
                    f"RPM {sensors['rpm']}, "
                    f"marcia {sensors.get('gear', 1)}, "
                    f"angolo {sensors['angle']}, "
                    f"posizione {sensors['trackPos']}. "
                    f"Sensori pista: {sensors['track']}. "
                    f"Genera i comandi ottimali di sterzo, accelerazione e freno."
                )

                # 2. Costruzione della Risposta (Output atteso)
                # Formattiamo l'output come un JSON pulito per facilitare il parsing in gara
                output_data = json.dumps({
                    "steer": actions['steer'],
                    "accel": actions['accel'],
                    "brake": actions['brake']
                })

                # 3. Creazione dell'oggetto per watsonx.ai
                jsonl_entry = {
                    "input": input_data,
                    "output": output_data
                }

                # Scrittura su file
                f_out.write(json.dumps(jsonl_entry) + '\n')
                count += 1

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Errore nella riga {count}: {e}")
                continue

    print(f"Successo! Creato file '{OUTPUT_FILE}' con {count} campioni.")

if __name__ == "__main__":
    create_fine_tuning_dataset()
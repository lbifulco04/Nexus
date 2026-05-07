import json

# Configurazione file
INPUT_FILE = 'expert_data_full.json'
OUTPUT_FILE = 'expert_data_cleaned.json'

# Parametri di filtraggio
MIN_SPEED = 5.0    # Soglia di velocità (km/h o m/s a seconda del simulatore)
STEP = 5           # Downsampling: tiene 1 riga ogni 5 (riduce il peso dell'80%)
ACCEL_THRESHOLD = 0.2  # Soglia per considerare una "partenza assistita"

def clean_telemetry():
    count_in = 0
    count_saved = 0
    count_static_discarded = 0
    
    print(f"Inizio elaborazione di {INPUT_FILE}...")

    with open(INPUT_FILE, 'r') as f_in, open(OUTPUT_FILE, 'w') as f_out:
        for i, line in enumerate(f_in):
            count_in += 1
            
            # 1. Downsampling (Campionamento ridotto)
            if i % STEP != 0:
                continue
            
            try:
                data = json.loads(line)
                sensors = data.get('sensors', {})
                actions = data.get('actions', {})
                
                speed = sensors.get('speedX', 0)
                accel = actions.get('accel', 0)

                # 2. Logica Filtro Staticità vs Partenza Assistita
                if speed < MIN_SPEED:
                    # Se l'auto è quasi ferma, salviamo solo se il pilota sta accelerando
                    if accel < ACCEL_THRESHOLD:
                        count_static_discarded += 1
                        continue 
                
                # 3. Ottimizzazione spazio (arrotondamento decimali)
                # Arrotondiamo i sensori principali e le azioni
                sensors['speedX'] = round(speed, 2)
                sensors['angle'] = round(sensors.get('angle', 0), 4)
                sensors['trackPos'] = round(sensors.get('trackPos', 0), 4)
                
                # Arrotondiamo i 19 sensori di pista
                if 'track' in sensors:
                    sensors['track'] = [round(v, 2) for v in sensors['track']]
                
                actions['steer'] = round(actions.get('steer', 0), 4)
                actions['accel'] = round(accel, 2)
                actions['brake'] = round(actions.get('brake', 0), 2)

                # Scrittura riga pulita
                f_out.write(json.dumps(data) + '\n')
                count_saved += 1
                
            except json.JSONDecodeError:
                continue

    print("-" * 30)
    print(f"ELABORAZIONE COMPLETATA")
    print(f"Righe lette totali: {count_in}")
    print(f"Righe scartate (auto ferma): {count_static_discarded}")
    print(f"Righe salvate nel nuovo file: {count_saved}")
    print(f"File creato: {OUTPUT_FILE}")
    print("-" * 30)

if __name__ == "__main__":
    clean_telemetry()
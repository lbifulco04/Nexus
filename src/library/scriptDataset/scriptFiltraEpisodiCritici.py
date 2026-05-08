#Questo script filtra ed unisce in unico file i recuperi tra cui fuoripista,
#testacoda o sbandate

import json

def process_recovery(file_list, output_file):
    processed_count = 0
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for file_name in file_list:
            with open(file_name, 'r') as infile:
                for line in infile:
                    try:
                        data = json.loads(line)
                        sensors = data['sensors']
                        actions = data['actions']
                        
                        # Estrarre parametri chiave
                        pos = sensors['trackPos']
                        angle = sensors['angle']
                        steer = actions['steer']
                        
                        # --- FILTRO QUALITA' RECUPERO ---
                        # Teniamo solo se c'è un errore significativo (> 0.05)
                        # e l'azione non è neutra (steer != 0) o sta correggendo l'angolo.
                        is_error = abs(pos) > 0.05 or abs(angle) > 0.05
                        is_correcting = (pos > 0 and steer < 0) or (pos < 0 and steer > 0) or (abs(angle) > 0.1 and abs(steer) > 0.1)
                        
                        if is_error and (is_correcting or actions['accel'] < 0.8):
                            # Formattazione stringa input coerente con i dati precedenti
                            input_str = (f"Telemetria: velocità {sensors['speedX']:.2f} km/h, "
                                         f"RPM {int(sensors['rpm'])}, marcia {actions['gear']}, "
                                         f"angolo {angle:.4f}, posizione {pos:.4f}. "
                                         f"Sensori pista: {sensors['track']}. "
                                         f"Genera i comandi ottimali di sterzo, accelerazione e freno.")
                            
                            # Formattazione stringa output
                            output_obj = {
                                "steer": round(actions['steer'], 4),
                                "accel": round(actions['accel'], 4),
                                "brake": round(actions['brake'], 4)
                            }
                            
                            json_line = {"input": input_str, "output": json.dumps(output_obj)}
                            outfile.write(json.dumps(json_line) + '\n')
                            processed_count += 1
                    except:
                        continue
    print(f"Conversione completata. Righe di recupero valide salvate: {processed_count}")

# Esecuzione
process_recovery(['data_recupero_sbandata.json', 'data_recupero_erba.json', 'data_recupero_testacoda.json'], 'dataset_recupero_final.jsonl')
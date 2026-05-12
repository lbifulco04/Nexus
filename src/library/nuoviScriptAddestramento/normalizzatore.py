import json
import numpy as np

def normalizza_dataset(percorso_file):
    ingressi_sensori = [] 
    uscite_azioni = []    
    
    with open(percorso_file, 'r') as f:
        for i, riga in enumerate(f):
            riga = riga.strip()
            if not riga:
                continue
            
            try:
                dati = json.loads(riga)
                
                # Estraiamo i blocchi principali
                s = dati["sensors"]
                a = dati["actions"]
                
                # --- 1. PREPARAZIONE INGRESSI (SENSORI) CON NORMALIZZAZIONE DEI DATI ---
                # 19 sensori track (0-200m -> scalati 0-1)
                pista = np.array(s["track"]) / 200.0
                
                # Velocità 
                velocita = np.array([s["speedX"] / 300.0])
                
                # Angolo rispetto alla pista (range -1, 1)
                angolo = np.array([s["angle"] / np.pi])
                
                # Posizione sulla carreggiata (0 centro, 1 bordi)
                posizione_pista = np.array([s["trackPos"]])
                
                # Giri motore (normalizzati su 15.000 RPM)
                giri_motore = np.array([s["rpm"] / 15000.0])

                # Creiamo il vettore di stato (23 elementi totali)
                vettore_stato = np.concatenate([pista, velocita, angolo, posizione_pista, giri_motore])
                
                # --- 2. PREPARAZIONE USCITE (AZIONI) ---
                # Lo sterzo è già in range [-1, 1], accel e brake [0, 1]
                vettore_azione = [a["steer"], a["accel"], a["brake"]]
                
                ingressi_sensori.append(vettore_stato)
                uscite_azioni.append(vettore_azione)
                
            except KeyError as e:
                print(f"⚠️ Errore alla riga {i}: Chiave mancante {e}")
                continue
            except Exception as e:
                print(f"❌ Errore imprevisto alla riga {i}: {e}")
                continue
            
    return np.array(ingressi_sensori), np.array(uscite_azioni)
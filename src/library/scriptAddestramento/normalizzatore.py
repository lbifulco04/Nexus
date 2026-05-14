import json
import numpy as np

def normalizza_dataset(percorso_file):
    ingressi_sensori = [] 
    uscite_azioni = []    # conterrà [steer, accel, brake, gear]
    
    with open(percorso_file, 'r') as f:
        for i, riga in enumerate(f):
            riga = riga.strip()
            if not riga:
                continue
            
            try:
                dati = json.loads(riga)
                s = dati["sensors"]
                a = dati["actions"]
                
                # Sensori
                pista = np.array(s["track"]) / 200.0
                velocita = np.array([s["speedX"] / 300.0])
                angolo = np.array([s["angle"] / np.pi])
                posizione_pista = np.array([s["trackPos"]])
                giri_motore = np.array([s["rpm"] / 15000.0])
                vettore_stato = np.concatenate([pista, velocita, angolo, posizione_pista, giri_motore])
                
                # Azioni (ora includiamo anche la marcia)
                # Convertiamo la marcia in un intero compreso tra -1 e 6
                gear = int(a["gear"])
                # Per sicurezza limitiamo il range (TORCS usa -1..6)
                gear = max(-1, min(gear, 6))
                vettore_azione = [a["steer"], a["accel"], a["brake"], gear]
                
                ingressi_sensori.append(vettore_stato)
                uscite_azioni.append(vettore_azione)
                
            except KeyError as e:
                print(f"⚠️ Errore alla riga {i}: Chiave mancante {e}")
                continue
            except Exception as e:
                print(f"❌ Errore imprevisto alla riga {i}: {e}")
                continue
            
    return np.array(ingressi_sensori), np.array(uscite_azioni)

import json
import os

# Definiamo i settori della pista basati sulla mappatura effettuata
# Ogni settore ha una tupla con (punto_inizio, punto_fine)
SETTORI_PISTA = {
    "curva1": (360, 590),
    "curva2": (685, 890),
    "curva3": (975, 1165),
    "curva4": (1400, 1660),
    "curva5": (1830, 2040),
    "curva6": (2360, 2555),
    "curva7": (2600, 2856),
    "curva8": (2874, 3080),
    "curva9": (3194, 3388)
}

def esegui_filtraggio():
    # Richiediamo il file generato durante la guida
    file_ingresso = input("📄 Inserisci il nome del file da filtrare (es: expert_data_full.json): ").strip()
    
    # Controllo di sicurezza: verifichiamo che il file esista davvero sul disco
    if not os.path.exists(file_ingresso):
        print(f"❌ Errore: Il file {file_ingresso} non è stato trovato.")
        return

    # Creiamo un dizionario per gestire i file aperti di ogni curva
    # Ogni curva avrà il suo file "data_curvaX.json" pronto per la scrittura
    file_output_curve = {nome: open(f"data_{nome}.json", "w") for nome in SETTORI_PISTA.keys()}
    
    # Creiamo un file a parte per raccogliere tutti i dati dei rettilinei
    file_rettilinei = open("data_rettilinei.json", "w")

    # Inizializziamo dei contatori per sapere quanti dati salviamo in ogni file
    contatore_settori = {nome: 0 for nome in SETTORI_PISTA.keys()}
    contatore_rettilinei = 0

    print(f"✂️  Inizio analisi e smistamento di {file_ingresso}...")

    # Apriamo il file originale in modalità lettura
    with open(file_ingresso, "r") as f_leggi:
        for riga in f_leggi:
            # Saltiamo eventuali righe vuote nel file
            if not riga.strip(): 
                continue
            
            try:
                # Trasformiamo la stringa di testo in un oggetto Python (Dizionario)
                dato_telemetria = json.loads(riga)

                if -1.0 in dato_telemetria["sensors"]["track"]:
                    continue
                
                # Estraiamo la distanza percorsa dal sensore specifico
                metri_percorsi = dato_telemetria["sensors"]["distFromStart"]
                
                trovato_in_curva = False
                
                # Controlliamo se la posizione attuale rientra in una delle nostre curve
                for nome_curva, (inizio, fine) in SETTORI_PISTA.items():
                    if inizio <= metri_percorsi <= fine:
                        # Se siamo nel range, scriviamo la riga nel file della curva corrispondente
                        file_output_curve[nome_curva].write(json.dumps(dato_telemetria) + "\n")
                        contatore_settori[nome_curva] += 1
                        trovato_in_curva = True
                        break # Trovata la curva, passiamo alla riga successiva
                
                # Se la distanza non rientra in nessuna curva, è un rettilineo
                if not trovato_in_curva:
                    file_rettilinei.write(json.dumps(dato_telemetria) + "\n")
                    contatore_rettilinei += 1

            except Exception as errore:
                print(f"⚠️  Errore durante l'elaborazione di una riga: {errore}")

    # Molto importante: chiudiamo tutti i file aperti per salvare correttamente i dati
    for f_curva in file_output_curve.values(): 
        f_curva.close()
    file_rettilinei.close()

    # Stampiamo un riepilogo 
    print("\n✅ Operazione completata con successo!")
    print(f"--------------------------------------")
    for nome, tot in contatore_settori.items():
        if tot > 0:
            print(f"🏁 {nome.upper()}: {tot} campioni salvati")
    print(f"🛣️  RETTILINEI: {contatore_rettilinei} campioni salvati")
    print(f"--------------------------------------")

# Avvio dello script
if __name__ == "__main__":
    esegui_filtraggio()
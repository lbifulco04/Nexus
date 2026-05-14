import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import joblib
import snakeoil3_jm2 as snakeoil3

# --- OTTIMIZZAZIONE MAC M1/M2/M3 ---
# Forza l'inferenza su CPU per evitare conflitti Metal/OpenGL che bloccano TORCS
tf.config.set_visible_devices([], 'GPU')

NOME_MODELLO = "best_model.h5"

# Verifica file necessari
for file_name in [NOME_MODELLO, 'scaler.pkl', 'pca.pkl']:
    if not os.path.exists(file_name):
        print(f"❌ Errore: Il file {file_name} non esiste!")
        exit(1)

print(f"🧠 Caricamento del cervello Watson: {NOME_MODELLO}...")
modello = keras.models.load_model(NOME_MODELLO, compile=False)

print("⚙️ Caricamento pipeline di pre-processing (Scaler + PCA)...")
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# --- OTTIMIZZAZIONE GRAFO ---
# Evitiamo che Keras ricompili le funzioni a ogni singolo tick, abbassando la latenza
@tf.function(reduce_retracing=True)
def inferenza_veloce(x):
    return modello(x, training=False)

def prepara_input_pca(sensors):
    """
    Estrae i dati, applica lo StandardScaler e poi la PCA.
    L'ordine DEVE essere identico a quello usato in fase di fit (24 feature).
    """
    track = sensors.get('track', [0.0]*19)
    speedX = sensors.get('speedX', 0.0)
    angle = sensors.get('angle', 0.0)
    trackPos = sensors.get('trackPos', 0.0)
    rpm = sensors.get('rpm', 0.0)
    
    # La PCA è stata addestrata su 24 feature, quindi distFromStart è obbligatorio
    distFromStart = sensors.get('distFromStart', 0.0)
    
    # Creiamo l'array grezzo 1x24
    features_raw = np.array(track + [speedX, angle, trackPos, rpm, distFromStart]).reshape(1, -1)
    
    # Applichiamo la trasformazione
    features_scaled = scaler.transform(features_raw)
    features_pca = pca.transform(features_scaled)
    
    # Convertiamo in tensore float32 per Keras
    return tf.constant(features_pca, dtype=tf.float32)

def main():
    # Connessione rigida a src_server 1 (porta 3001)
    client = snakeoil3.Client(p=3001, vision=False)
    
    print("\n🏎️  PILOTA AUTOMATICO WATSON (con PCA e Fix Tensori) ATTIVO")
    print("In attesa di TORCS (src_server 1)...")

    try:
        while True:
            client.get_servers_input()
            
            # Se il server non ha risposto o i dati sono vuoti, salta il ciclo
            if not client.S.d or 'track' not in client.S.d:
                continue
            
            # 1. Pipeline di trasformazione
            input_ia = prepara_input_pca(client.S.d)

            # 2. Esecuzione rapida
            previsione = inferenza_veloce(input_ia)
            
            # 3. Estrazione tensori SICURA 
            # Flatten converte qualsiasi output in un array monodimensionale piatto [accel, brake, steer]
            valori_predetti = previsione.numpy().flatten()
            
            
            # Basandoci sul nostro script precedente: y = [accel, brake, steer]
            acceleratore = float(valori_predetti[0])
            freno = float(valori_predetti[1])
            sterzo = float(valori_predetti[2])

            # 4. Limitiamo i valori per sicurezza (clipping fisico per le API di TORCS)
            sterzo = max(min(sterzo, 1.0), -1.0)
            acceleratore = max(min(acceleratore, 1.0), 0.0)
            freno = max(min(freno, 1.0), 0.0)

            # Prevenzione conflitto sui pedali (se preme il freno, togliamo gas)
            if freno > 0.05:
                acceleratore = 0.0

            # 5. Logica cambio calibrata per la monoposto di src_server1 (motore ad alti regimi)
            rpm = client.S.d.get('rpm', 0)
            marcia = client.S.d.get('gear', 1)
            
            if rpm > 14500 and marcia < 6: 
                marcia += 1
            elif rpm < 7500 and marcia > 1: 
                marcia -= 1

            # 6. INVIO DATI
            client.R.d.update({
                'steer': sterzo,
                'accel': acceleratore,
                'brake': freno,
                'gear': marcia
            })
            client.respond_to_server()

            # Stampiamo solo ogni 10 tick per non saturare il terminale e creare lag
            ticks = client.S.d.get('ticks', 0)
            if ticks % 10 == 0:
                print(f"\r🤖 Watson | S: {sterzo:5.2f} | A: {acceleratore:4.2f} | F: {freno:4.2f} | M: {marcia} | Ticks: {ticks}", end="")

    except KeyboardInterrupt:
        print("\n\n🛑 Guida interrotta dall'utente.")
        
    finally:
        # Assicuriamoci di chiudere correttamente la connessione UDP e rilasciare i comandi
        try:
            client.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 1.0, 'gear': 0})
            client.respond_to_server()
            client.shutdown()
            print("Socket chiuso e veicolo fermato in sicurezza.")
        except Exception:
            pass

if __name__ == "__main__":
    main()
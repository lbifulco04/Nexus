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

print(f"🧠 Caricamento dell'Agente : {NOME_MODELLO}...")
modello = keras.models.load_model(NOME_MODELLO, compile=False)

print("⚙️ Caricamento pipeline di pre-processing (Scaler + PCA)...")
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')

# --- OTTIMIZZAZIONE GRAFO ---
@tf.function(reduce_retracing=True)
def inferenza_veloce(x):
    return modello(x, training=False)

def prepara_input_pca(sensors):
    """
    Estrae i dati, applica lo StandardScaler e poi la PCA.
    """
    track = sensors.get('track', [0.0]*19)
    speedX = sensors.get('speedX', 0.0)
    angle = sensors.get('angle', 0.0)
    trackPos = sensors.get('trackPos', 0.0)
    rpm = sensors.get('rpm', 0.0)
    distFromStart = sensors.get('distFromStart', 0.0)
    
    features_raw = np.array(track + [speedX, angle, trackPos, rpm, distFromStart]).reshape(1, -1)
    
    features_scaled = scaler.transform(features_raw)
    features_pca = pca.transform(features_scaled)
    
    return tf.constant(features_pca, dtype=tf.float32)

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    
    print("\n🏎️  PILOTA AUTOMATICO (con PCA e Scaler) ATTIVO")
    print("In attesa di TORCS (src_server 1)...")

    try:
        while True:
            client.get_servers_input()
            
            if not client.S.d or 'track' not in client.S.d:
                continue
            
            # 1. Pipeline di trasformazione
            input_ia = prepara_input_pca(client.S.d)

            # 2. Esecuzione rapida
            previsione = inferenza_veloce(input_ia)
            
            # 3. Estrazione tensori
            valori_predetti = previsione.numpy().flatten()
            
            acceleratore = float(valori_predetti[0])
            freno = float(valori_predetti[1])
            sterzo = float(valori_predetti[2])

            # 4. Limitiamo i valori per sicurezza
            sterzo = max(min(sterzo, 1.0), -1.0)
            acceleratore = max(min(acceleratore, 1.0), 0.0)
            freno = max(min(freno, 1.0), 0.0)

            # Prevenzione conflitto sui pedali
            if freno > 0.05:
                acceleratore = 0.0
                
            # Calcoliamo la velocità una sola volta per usarla nelle patch
            velocita_kmh = client.S.d.get('speedX', 0.0) * 3.6
            distanza = client.S.d.get('distFromStart', 0.0)

            # --- 5. PATCH FRENATA PER ULTIMA CURVA
            if 3180 <= distanza <= 3210:
                if velocita_kmh > 105:  
                    freno = 0.5          # Frenata 50 %
                    acceleratore = 0.0
            
            # --- 6. PATCH ANTI-STALLO 
            # Se la velocità crolla troppo  Forziamo il gas.
            if velocita_kmh < 50.0:
                acceleratore = 0.6       
                freno = 0.0              
                sterzo = sterzo * 0.5    # Raddrizzamento per il testacoda testacoda

            # --- 7. LOGICA CAMBIO ---
            rpm = client.S.d.get('rpm', 0)
            marcia = client.S.d.get('gear', 1)
            
            if velocita_kmh < 50.0:      
                marcia = 1
            elif rpm > 14500 and marcia < 6: 
                marcia += 1
            elif rpm < 7500 and marcia > 1: 
                marcia -= 1

            # 8. INVIO DATI
            client.R.d.update({
                'steer': sterzo,
                'accel': acceleratore,
                'brake': freno,
                'gear': marcia
            })
            client.respond_to_server()

            
            # Stampa log
            ticks = client.S.d.get('ticks', 0)
            if ticks % 10 == 0:
                # Aggiungiamo un indicatore visivo nel log quando la patch è attiva
                patch_attiva = "⚠️ FRENATA FORZATA" if (3100 <= distanza <= 3250 and client.S.d.get('speedX', 0)*3.6 > 90) else ""
                print(f"\r🤖 Watson | S: {sterzo:5.2f} | A: {acceleratore:4.2f} | F: {freno:4.2f} | M: {marcia} | Dist: {distanza:.0f}m {patch_attiva}  ", end="")

    except KeyboardInterrupt:
        print("\n\n🛑 Guida interrotta dall'utente.")
        
    finally:
        try:
            client.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 1.0, 'gear': 0})
            client.respond_to_server()
            client.shutdown()
            print("\nSocket chiuso e veicolo fermato in sicurezza.")
        except Exception:
            pass

if __name__ == "__main__":
    main()
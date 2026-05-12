import numpy as np
from tensorflow import keras
import snakeoil3_jm2 as snakeoil3 
import os
import tensorflow as tf

# --- OTTIMIZZAZIONE MAC M1/M2/M3 ---
# Forza l'inferenza su CPU per evitare conflitti Metal/OpenGL che bloccano TORCS
tf.config.set_visible_devices([], 'GPU')

NOME_MODELLO = "modello_keras_v1.h5"

if not os.path.exists(NOME_MODELLO):
    print(f"❌ Errore: Il file {NOME_MODELLO} non esiste!")
    exit()

print(f"🧠 Caricamento del cervello Watson: {NOME_MODELLO}...")
modello = keras.models.load_model(NOME_MODELLO, compile=False)

def normalizza_input_realtime(sensors):
    # Usiamo float32 per coerenza con la rete neurale
    track = np.array(sensors['track'], dtype=np.float32) / 200.0
    speedX = np.array([sensors['speedX'] / 300.0], dtype=np.float32)
    angle = np.array([sensors['angle'] / np.pi], dtype=np.float32)
    trackPos = np.array([sensors['trackPos']], dtype=np.float32)
    rpm = np.array([sensors['rpm'] / 15000.0], dtype=np.float32)
    
    stato = np.concatenate([track, speedX, angle, trackPos, rpm])
    return np.expand_dims(stato, axis=0)

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    
    print("\n🏎️  PILOTA AUTOMATICO WATSON ATTIVO")
    print("In attesa di TORCS...")

    try:
        while True:
            client.get_servers_input()

            
    
            # Se il server non ha risposto o i dati sono vuoti, salta il ciclo
            if not client.S.d or 'track' not in client.S.d:
                continue
            

            input_ia = normalizza_input_realtime(client.S.d)

            # Esecuzione rapida
            previsione = modello(input_ia, training=False)
            
            # --- FIX CRASH: Conversione esplicita in float standard di Python ---
            sterzo = float(previsione[0][0][0])
            acceleratore = float(previsione[1][0][0])
            freno = float(previsione[1][0][1])

            # Limitiamo i valori per sicurezza (clipping)
            sterzo = max(min(sterzo, 1.0), -1.0)
            acceleratore = max(min(acceleratore, 1.0), 0.0)
            freno = max(min(freno, 1.0), 0.0)

            # Logica cambio
            rpm = client.S.d.get('rpm', 0)
            marcia = client.S.d.get('gear', 1)
            if rpm > 9000 and marcia < 6: marcia += 1
            elif rpm < 6500 and marcia > 1: marcia -= 1

            # INVIO DATI
            client.R.d.update({
                'steer': sterzo,
                'accel': acceleratore,
                'brake': freno,
                'gear': marcia
            })
            client.respond_to_server()

            # Stampiamo solo ogni tanto per non saturare il terminale (rallenta il loop)
            ticks = client.S.d.get('ticks', 0)
            if ticks % 10 == 0:
                print(f"\r🤖 Watson | S: {sterzo:5.2f} | A: {acceleratore:4.2f} | F: {freno:4.2f} | Ticks: {ticks}", end="")

    except KeyboardInterrupt:
        print("\n\n🛑 Guida interrotta.")

if __name__ == "__main__":
    main()
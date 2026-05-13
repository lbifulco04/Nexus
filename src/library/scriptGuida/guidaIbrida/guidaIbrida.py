import pygame
import snakeoil3_jm2 as snakeoil3
import json
import sys
import os
import time
import numpy as np
import tensorflow as tf
from tensorflow import keras

# --- OTTIMIZZAZIONE MAC M1/M2/M3 (RISOLUZIONE LAG) ---
tf.config.set_visible_devices([], 'GPU')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

class HybridDriver:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("❌ DualSense non trovato!"); sys.exit()
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        self.AXIS_STEER = 0
        self.AXIS_ACCEL = 5
        self.AXIS_BRAKE = 4
        self.BTN_UP = 0   
        self.BTN_DOWN = 1 

        self.gear = 1 

    def get_human_inputs(self):
        """Legge input e restituisce (controllo_umano, steer, accel, brake, gear)"""
        pygame.event.pump()
        
        raw_steer = self.joy.get_axis(self.AXIS_STEER)
        raw_accel = self.joy.get_axis(self.AXIS_ACCEL)
        raw_brake = self.joy.get_axis(self.AXIS_BRAKE)

        steer = float(-(raw_steer**3) * 0.7)
        accel = float((raw_accel + 1.0) / 2.0)
        brake = float((raw_brake + 1.0) / 2.0)

        cambio_marcia_attivo = False
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == self.BTN_UP:
                    if self.gear < 6: 
                        self.gear += 1
                        cambio_marcia_attivo = True
                elif event.button == self.BTN_DOWN:
                    if self.gear > -1: 
                        self.gear -= 1
                        cambio_marcia_attivo = True

        controlloUmano = abs(raw_steer) > 0.15 or accel > 0.1 or brake > 0.1 or cambio_marcia_attivo
        return controlloUmano, steer, accel, brake, self.gear

def normalizza_input_realtime(sensors):
    track = np.array(sensors['track'], dtype=np.float32) / 200.0
    speedX = np.array([sensors['speedX'] / 300.0], dtype=np.float32)
    angle = np.array([sensors['angle'] / np.pi], dtype=np.float32)
    trackPos = np.array([sensors['trackPos']], dtype=np.float32)
    rpm = np.array([sensors['rpm'] / 15000.0], dtype=np.float32)
    
    stato = np.concatenate([track, speedX, angle, trackPos, rpm])
    return np.expand_dims(stato, axis=0)

def main():
    print("\n" + "="*50)
    print(" 🏎️  SISTEMA DAgger: IA + DUALSENSE OVERRIDE (OTTIMIZZATO)")
    print("="*50)

    NOME_MODELLO = input("Inserisci nome del modello da utilizzare: ") 
    if not os.path.exists(NOME_MODELLO):
        print(f"❌ Errore: Il modello {NOME_MODELLO} non esiste!")
        sys.exit()
    print(f"🧠 Caricamento cervello Watson: {NOME_MODELLO}...")
    modello = keras.models.load_model(NOME_MODELLO, compile=False)

    file_name = input("\n📝 Nome file per salvare i recuperi: ").strip()
    if not file_name: file_name = "dagger_recuperi.json"
    elif not file_name.endswith(".json"): file_name += ".json"

    client = snakeoil3.Client(p=3001, vision=False)
    driver = HybridDriver()
    
    count = 0
    buffer_dati = []  # per scrittura differita
    print(f"\n✅ Sistema Pronto. Salvataggio su: {file_name}")
    print("Premi CTRL+C nel terminale per terminare.\n")
    
    ultimo_sterzo_inviato = 0.0  
    MAX_VELOCITA_VOLANTE = 0.12   # per l'IA (più fluido)

    try:
        with open(file_name, "a") as f:
            while True:
                t0 = time.perf_counter()
                client.get_servers_input()
                if not client.S.d or 'track' not in client.S.d: 
                    continue

                # A. LETTURA INPUT UMANO
                controlloUmano, steer_h, accel_h, brake_h, gear_h = driver.get_human_inputs()

                # B. DECISIONE (IA o Umano)
                if controlloUmano:
                    steer_final = steer_h
                    accel_final = accel_h
                    brake_final = brake_h
                    gear_final  = gear_h
                    stato_guida = "👤 UMANO "

                    distanza = client.S.d.get('distFromStart', 0.0)
                    if client.S.d.get('speedX', 0) > 1.0 and gear_final >= 1:
                        row = {
                            "sensors": {
                                "track": client.S.d.get('track'),
                                "speedX": client.S.d.get('speedX'),
                                "angle": client.S.d.get('angle'),
                                "trackPos": client.S.d.get('trackPos'),
                                "rpm": client.S.d.get('rpm'), 
                                "distFromStart": distanza
                            },
                            "actions": {
                                "steer": steer_final, "accel": accel_final, 
                                "brake": brake_final, "gear": gear_final
                            }
                        }
                        buffer_dati.append(json.dumps(row))
                        count += 1
                        # Scrittura ogni 50 campioni per evitare lag
                        if len(buffer_dati) >= 50:
                            f.write("\n".join(buffer_dati) + "\n")
                            f.flush()
                            buffer_dati.clear()
                else:
                    # IA al comando
                    input_ia = normalizza_input_realtime(client.S.d)
                    previsione = modello(input_ia, training=False)
                    
                    steer_final = float(previsione[0][0][0])
                    accel_final = float(previsione[1][0][0])
                    brake_final = float(previsione[1][0][1])

                    rpm_attuale = client.S.d.get('rpm', 0)
                    gear_ai = client.S.d.get('gear', 1)
                    if rpm_attuale > 9000 and gear_ai < 6: gear_ai += 1
                    elif rpm_attuale < 2500 and gear_ai > 1: gear_ai -= 1

                    gear_final = gear_ai
                    driver.gear = gear_ai 
                    stato_guida = "🤖 Agente"

                    # Smoothing solo per l'IA (nessun ritardo per l'umano)
                    if steer_final > ultimo_sterzo_inviato + MAX_VELOCITA_VOLANTE:
                        steer_final = ultimo_sterzo_inviato + MAX_VELOCITA_VOLANTE
                    elif steer_final < ultimo_sterzo_inviato - MAX_VELOCITA_VOLANTE:
                        steer_final = ultimo_sterzo_inviato - MAX_VELOCITA_VOLANTE

                # Aggiorna memoria sterzo (sia umano che IA)
                ultimo_sterzo_inviato = steer_final

                # Clipping di sicurezza
                steer_final = max(min(steer_final, 1.0), -1.0)
                accel_final = max(min(accel_final, 1.0), 0.0)
                brake_final = max(min(brake_final, 1.0), 0.0)

                # D. INVIO A TORCS
                client.R.d.update({
                    'steer': steer_final, 'accel': accel_final, 
                    'brake': brake_final, 'gear': gear_final
                })
                client.respond_to_server()

                # Stampa diagnostica ogni 5 tick
                ticks = client.S.d.get('ticks', 0)
                distanza = client.S.d.get('distFromStart', 0.0)
                dt = (time.perf_counter() - t0) * 1000.0
                if ticks % 5 == 0:
                    print(f"\r[{stato_guida}] Pos: {distanza:6.1f}m | "
                          f"S: {steer_final:5.2f} A: {accel_final:4.2f} F: {brake_final:4.2f} | "
                          f"dati: {count} | loop: {dt:.1f}ms", end="")

    except KeyboardInterrupt:
        # Salva eventuali dati rimasti nel buffer
        if buffer_dati:
            with open(file_name, "a") as f:
                f.write("\n".join(buffer_dati) + "\n")
        print(f"\n\n✅ Sessione terminata. Modello testato e recuperi salvati.")
        print(f"📂 File: {file_name}")
        print(f"📊 Campioni di correzione umani raccolti: {count}")

if __name__ == "__main__":
    main()
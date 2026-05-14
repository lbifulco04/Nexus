import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
import snakeoil3_jm2 as snakeoil3

# --- OTTIMIZZAZIONE MAC M1/M2/M3 ---
tf.config.set_visible_devices([], 'GPU')

# ──────────────────────────────
# BUG 1: formato coerente con lo script di training
# Se hai salvato con .keras usa .keras; se usi ancora .h5 cambia qui.
NOME_MODELLO = "modello_v7.keras"

if not os.path.exists(NOME_MODELLO):
    # Fallback automatico al vecchio .h5 per compatibilità
    fallback = NOME_MODELLO.replace(".keras", ".h5")
    if os.path.exists(fallback):
        print(f"⚠️  '{NOME_MODELLO}' non trovato. Carico il fallback '{fallback}'.")
        NOME_MODELLO = fallback
    else:
        print(f"❌ Errore: né '{NOME_MODELLO}' né il fallback .h5 esistono!")
        exit(1)

print(f"🧠 Caricamento modello: {NOME_MODELLO}...")
modello = keras.models.load_model(NOME_MODELLO, compile=False)

if len(modello.outputs) != 3:
    print("⚠️  Il modello non ha 3 uscite (sterzo, pedali, marcia)!")
    exit(1)

print(f"✅ Modello caricato. Input: {modello.input_shape} | Output: {[o.shape for o in modello.outputs]}")

# BUG 7: @tf.function evita la ricompilazione del graph a ogni tick
@tf.function(reduce_retracing=True)
def inferenza(x):
    return modello(x, training=False)

# ──────────────────────────────
def normalizza_input_realtime(sensors):
    """
    Converte i sensori grezzi nel vettore normalizzato (23 valori).
    BUG 3: clip di trackPos e rpm per gestire valori fuori pista o motore spento.
    """
    track     = np.clip(np.array(sensors['track'], dtype=np.float32), 0.0, 200.0) / 200.0
    speedX    = np.array([sensors['speedX'] / 300.0],   dtype=np.float32)
    angle     = np.array([sensors['angle']  / np.pi],   dtype=np.float32)
    trackPos  = np.clip(np.array([sensors['trackPos']], dtype=np.float32), -2.0, 2.0)
    rpm       = np.clip(np.array([sensors['rpm']],      dtype=np.float32), 0.0, 15000.0) / 15000.0
    stato = np.concatenate([track, speedX, angle, trackPos, rpm])
    return stato.reshape(1, -1)          # shape (1, 23)

# ──────────────────────────────
# Costanti di guida
TICK_WARMUP        = 10     # primi tick: forza marcia 1, accel leggera
MAX_TICK_VUOTI     = 50     # BUG 2: soglia di timeout server
NEUTRAL_BRAKE_THR  = 0.02   # freno minimo sotto cui non frenare

def main():
    client = snakeoil3.Client(p=3001, vision=False)
    print("\n🏎️  PILOTA AUTOMATICO (modello con 3 uscite) ATTIVO")
    print("   In attesa di TORCS... (Ctrl+C per fermare)\n")

    tick_globale  = 0
    tick_vuoti    = 0

    try:
        while True:
            # BUG 6: eccezioni di rete catturate separatamente
            try:
                client.get_servers_input()
            except Exception as e:
                print(f"\n⚠️  Errore ricezione dati: {e}")
                tick_vuoti += 1
                if tick_vuoti >= MAX_TICK_VUOTI:
                    print("❌ Troppi tick senza risposta. Uscita.")
                    break
                continue

            # Dati mancanti o struttura vuota
            if not client.S.d or 'track' not in client.S.d:
                tick_vuoti += 1
                if tick_vuoti >= MAX_TICK_VUOTI:
                    print("❌ Timeout: server non risponde.")
                    break
                continue

            tick_vuoti = 0   # reset contatore su dati validi

            # ── Normalizzazione input ──
            input_ia = normalizza_input_realtime(client.S.d)
            input_tf = tf.constant(input_ia, dtype=tf.float32)

            # ── Inferenza (compilata con @tf.function) ──
            previsione = inferenza(input_tf)

            # BUG 4: estrazione robusta indipendente dalla shape esatta del tensore
            sterzo     = float(previsione[0].numpy().flatten()[0])
            pedali_arr = previsione[1].numpy().flatten()
            accel      = float(pedali_arr[0])
            brake      = float(pedali_arr[1])

            gear_probs = previsione[2].numpy().flatten()   # shape (N_classi,)
            gear_class = int(np.argmax(gear_probs))
            gear_pred  = gear_class - 1                    # rimappatura a [-1..6]

            # ── Clipping di sicurezza ──
            sterzo    = float(np.clip(sterzo, -1.0,  1.0))
            accel     = float(np.clip(accel,   0.0,  1.0))
            brake     = float(np.clip(brake,   0.0,  1.0))
            gear_pred = int(np.clip(gear_pred, -1, 6))

            # BUG 5: override marcia nei tick di warmup
            # La rete è incerta al tick 0 (input mai visto in esecuzione reale)
            if tick_globale < TICK_WARMUP:
                gear_pred = 1
                accel     = 0.3
                brake     = 0.0

            # Elimina micro-frenate spurie (rumore del modello)
            if brake < NEUTRAL_BRAKE_THR:
                brake = 0.0

            # ── Invio comandi ──
            try:
                client.R.d.update({
                    'steer': sterzo,
                    'accel': accel,
                    'brake': brake,
                    'gear':  gear_pred
                })
                client.respond_to_server()
            except Exception as e:
                print(f"\n⚠️  Errore invio comandi: {e}")
                continue

            # ── Log periodico ──
            tick_globale += 1
            ticks = client.S.d.get('ticks', tick_globale)
            if ticks % 10 == 0:
                speed_kmh = client.S.d.get('speedX', 0) * 3.6
                conf_gear = float(gear_probs[gear_class]) * 100
                print(
                    f"\r🤖 | S:{sterzo:+.2f} A:{accel:.2f} B:{brake:.2f} "
                    f"G:{gear_pred:2d}({conf_gear:.0f}%) "
                    f"V:{speed_kmh:5.1f}km/h T:{ticks}",
                    end=""
                )

    except KeyboardInterrupt:
        print("\n\n🛑 Guida interrotta dall'utente.")
    finally:
        # Cleanup: rilascia i freni e porta in neutro
        try:
            client.R.d.update({'steer': 0.0, 'accel': 0.0, 'brake': 0.0, 'gear': 0})
            client.respond_to_server()
        except Exception:
            pass
        print("   Comandi azzerati. Arrivederci!")

if __name__ == "__main__":
    main()
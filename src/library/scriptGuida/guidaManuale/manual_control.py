#SCRIPT UTILIZZATO PER GIOCARE TRAMITE TASTIERA E RACCOGLIERE I DATI
from pynput.keyboard import Key, Listener
import snakeoil3_jm2 as snakeoil3
import time
import json
import csv
import os

class ArcadeController:
    def __init__(self):
        self.keys = set()
        self.state = {
            'steer': 0.0,
            'accel': 0.0,
            'brake': 0.0,
            'gear': 1
        }
        # Parametri di smoothing (configurabili)
        self.accel_speed = 0.15
        self.brake_speed = 0.3
        self.steer_speed = 0.25

        self.listener = Listener(on_press=self.press, on_release=self.release)
        self.listener.start()

    def press(self, key):
        self.keys.add(key)
        try:
            if hasattr(key, "char"):
                if key.char == 'w':
                    self.state['gear'] += 1
                elif key.char == 's':
                    self.state['gear'] -= 1
        except AttributeError:
            pass

    def release(self, key):
        self.keys.discard(key)

    def update(self, sensors):
        speed = sensors.get('speedX', 0)
        angle = sensors.get('angle', 0)

        # Accel/Brake con interpolazione lineare (Lerp)
        target_accel = 1.0 if Key.up in self.keys else 0.0
        self.state['accel'] += (target_accel - self.state['accel']) * self.accel_speed

        target_brake = 1.0 if Key.down in self.keys else 0.0
        self.state['brake'] += (target_brake - self.state['brake']) * self.brake_speed

        # Steering dinamico basato sulla velocità
        steer_input = 0.0
        if Key.left in self.keys: steer_input += 0.7
        if Key.right in self.keys: steer_input -= 0.7

        # Sensibilità dello sterzo inversamente proporzionale alla velocità
        # Più vai veloce, meno lo sterzo è brusco (previene testacoda)
        max_steer_sens = max(0.2, 1.0 - (speed / 180.0))
        
        # Compensazione angolo (stabilità)
        stability_factor = angle * 0.5
        steer_target = (steer_input * max_steer_sens) - stability_factor

        self.state['steer'] += (steer_target - self.state['steer']) * self.steer_speed

        # Clamp e rifiniture
        self.state['steer'] = max(-1.0, min(1.0, self.state['steer']))
        self.state['accel'] = max(0.0, min(1.0, self.state['accel']))
        self.state['brake'] = max(0.0, min(1.0, self.state['brake']))
        self.state['gear'] = max(-1, min(6, self.state['gear']))

        if abs(self.state['steer']) < 0.01: self.state['steer'] = 0.0

def main():
    # Inizializzazione Client
    client = snakeoil3.Client(p=3001, vision=False)
    controller = ArcadeController()
    
    # Assicurati che il server sia pronto
    client.get_servers_input()

    print("\n--- Arcade Control Ready ---")
    print("Comandi: Frecce (Guida), W/S (Marce), Ctrl+C (Esci e Salva)")

    # Gestione file con 'with' per sicurezza dati
    csv_filename = "manual_log.csv"
    json_filename = "manual_log.json"
    
    log_json = []
    
    try:
        with open(csv_filename, "w", newline='') as f_csv:
            writer = csv.writer(f_csv)
            writer.writerow(["time", "steer", "accel", "brake", "gear", "speedX", "trackPos", "angle", "rpm", "damage"])

            t0 = time.time()
            step = 0

            while True:
                # 1. Ricezione dati dal server
                client.get_servers_input()
                S = client.S.d

                # 2. Update logica di controllo
                controller.update(S)
                a = controller.state

                # 3. Risposta al server
                client.R.d.update({
                    'steer': a['steer'],
                    'accel': a['accel'],
                    'brake': a['brake'],
                    'gear': a['gear'],
                    'clutch': 0.0,
                    'meta': 0
                })
                client.respond_to_server()

                # 4. Logging efficiente
                curr_t = time.time() - t0
                row = [curr_t, a['steer'], a['accel'], a['brake'], a['gear'], 
                       S.get('speedX',0), S.get('trackPos',0), S.get('angle',0), 
                       S.get('rpm',0), S.get('damage',0)]
                
                writer.writerow(row)
                
                # Feedback a video ogni 10 step per non appesantire il terminale
                if step % 10 == 0:
                    print(f"\rVelocità: {S.get('speedX',0):.1f} km/h | Marcia: {a['gear']} | Sterzo: {a['steer']:.2f}", end="")

                # Accumulo per JSON
                log_json.append({
                    "step": step,
                    "time": curr_t,
                    "action": a.copy(),
                    "state": {k: S.get(k, 0) for k in ['speedX', 'trackPos', 'angle', 'rpm', 'damage']}
                })

                step += 1
                # Salva JSON ogni 500 step per performance
                if step % 500 == 0:
                    with open(json_filename, "w") as f_json:
                        json.dump(log_json, f_json, indent=2)

                time.sleep(0.02)

    except KeyboardInterrupt:
        print("\n\nSalvataggio finale in corso...")
        with open(json_filename, "w") as f_json:
            json.dump(log_json, f_json, indent=2)
        print("Dati salvati. Uscita.")

if __name__ == "__main__":
    main()
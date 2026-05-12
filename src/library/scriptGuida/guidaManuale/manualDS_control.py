#SCRIPT UTILIZZATO PER GIOCARE TRAMITE DUALSENSE SU UN MACM1 E RACCOGLIERE DATI
import pygame
import snakeoil3_jm2 as snakeoil3
import json
import sys

class ExpertDriver:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("❌ DualSense non trovato!"); sys.exit()
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        # Mapping DualSense su Mac
        self.AXIS_STEER = 0
        self.AXIS_ACCEL = 5
        self.AXIS_BRAKE = 4
        self.BTN_UP = 0   # Croce (X)
        self.BTN_DOWN = 1 # Cerchio (O)

        self.gear = 1 

    def get_controls(self):
        pygame.event.pump()
        
        steer = float(-self.joy.get_axis(self.AXIS_STEER) * 0.7)
        accel = float((self.joy.get_axis(self.AXIS_ACCEL) + 1.0) / 2.0)
        brake = float((self.joy.get_axis(self.AXIS_BRAKE) + 1.0) / 2.0)

        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if event.button == self.BTN_UP:
                    if self.gear < 6: self.gear += 1
                elif event.button == self.BTN_DOWN:
                    if self.gear > -1: self.gear -= 1

        return steer, accel, brake, self.gear

def main():
    # --- RICHIESTA NOME FILE ---
    print("\n--- CONFIGURAZIONE REGISTRAZIONE ---")
    file_name = input("📝 Inserisci il nome del file (es: curva1 o rettilineo): ").strip()
    
    # Se il nome è vuoto, mettiamo un default
    if not file_name:
        file_name = "expert_data_session"
    
    # Assicuriamoci che finisca con .json
    if not file_name.endswith(".json"):
        file_name += ".json"
    # ---------------------------

    client = snakeoil3.Client(p=3001, vision=False)
    driver = ExpertDriver()
    
    count = 0

    print(f"\n🏎️  RECORDING ON - Salvataggio immediato su: {file_name}")
    print("Premi CTRL+C per terminare la sessione.\n")
    
    try:
        with open(file_name, "a") as f:
            while True:
                client.get_servers_input()
                if not client.S.d: continue

                steer, accel, brake, gear = driver.get_controls()

                # Recuperiamo la distanza dall'inizio della pista
                distanza = client.S.d.get('distFromStart', 0.0)

                client.R.d.update({
                    'steer': steer, 'accel': accel, 
                    'brake': brake, 'gear': gear
                })
                client.respond_to_server()

                
                marcia_attuale = gear

                if client.S.d.get('speedX', 0) > 1.0 and marcia_attuale>=1:
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
                            "steer": steer, "accel": accel, 
                            "brake": brake, "gear": gear
                        }
                    }
                    f.write(json.dumps(row) + "\n")
                    count += 1

                # STAMPA POSIZIONE IN TEMPO REALE
                print(f"\r📍 POSIZIONE: {distanza:7.1f} m | Campioni: {count} | Marcia: {gear} ", end="")

    except KeyboardInterrupt:
        print(f"\n\n✅ Sessione terminata correttamente.")
        print(f"📂 File salvato: {file_name}")
        print(f"📊 Campioni totali: {count}")

if __name__ == "__main__":
    main()
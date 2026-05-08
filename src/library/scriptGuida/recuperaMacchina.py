#Questo script serve per iniziare una comunicazione UDP alla porta 3001 (default torcs) e registrae solo eventi critici
import pygame
import library.scriptGuida.snakeoil3_jm2 as snakeoil3
import json
import sys
import os

class ExpertRecoveryDriver:
    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        if pygame.joystick.get_count() == 0:
            print("❌ DualSense non trovato!"); sys.exit()
        self.joy = pygame.joystick.Joystick(0)
        self.joy.init()

        # Mapping DualSense su Mac (M1 compatible)
        self.AXIS_STEER = 0
        self.AXIS_ACCEL = 5
        self.AXIS_BRAKE = 4
        self.BTN_UP = 0   # Croce (X)
        self.BTN_DOWN = 1 # Cerchio (O)

        self.gear = 1 

    def get_controls(self):
        pygame.event.pump()
        
        # Sterzo ridotto a 0.7 per maggiore precisione nelle correzioni
        steer = float(-self.joy.get_axis(self.AXIS_STEER) * 0.7)
        # Normalizzazione trigger (0 a 1)
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
    print("\n--- 🛠️  MODALITÀ REGISTRAZIONE RECUPERO ---")
    print("Scegli cosa vuoi registrare per istruire Granite:")
    print("1. Rientro dall'erba / Bordo pista")
    print("2. Correzione sbandata (Controsterzo)")
    print("3. Ripartenza dopo testacoda")
    
    scelta = input("\nSeleziona (1-3) o scrivi un nome: ").strip()
    nomi = {"1": "recupero_erba", "2": "recupero_sbandata", "3": "recupero_testacoda"}
    file_tag = nomi.get(scelta, scelta if scelta else "recupero_generico")
    file_name = f"data_{file_tag}.json"

    client = snakeoil3.Client(p=3001, vision=False)
    driver = ExpertRecoveryDriver()
    
    count = 0
    print(f"\n📡 Registrazione ATTIVA su: {file_name}")
    print("Istruzioni: Metti l'auto in difficoltà, poi riprendi il controllo.")
    print("Premi CTRL+C per salvare e uscire.\n")
    
    try:
        with open(file_name, "a") as f:
            while True:
                client.get_servers_input()
                if not client.S.d: continue

                steer, accel, brake, gear = driver.get_controls()
                
                # SENSORI CHIAVE PER IL RECUPERO
                distanza = client.S.d.get('distFromStart', 0.0)
                velocita = client.S.d.get('speedX', 0.0)
                angolo = client.S.d.get('angle', 0.0)
                pos_pista = client.S.d.get('trackPos', 0.0) # 0 = centro, >1 o <-1 = fuori

                client.R.d.update({
                    'steer': steer, 'accel': accel, 
                    'brake': brake, 'gear': gear
                })
                client.respond_to_server()

                # --- LOGICA DI REGISTRAZIONE RECUPERO ---
                # Registriamo solo se:
                # 1. Siamo in marcia avanti (gear >= 1)
                # 2. Ci stiamo muovendo (velocita > 2.0)
                # 3. L'auto è in una condizione di "recupero":
                #    - O siamo molto vicini al bordo/fuori (abs(pos_pista) > 0.6)
                #    - O l'auto è intraversata (abs(angolo) > 0.1)
                #    - O stiamo sterzando attivamente per rientrare
                
                condizione_critica = abs(pos_pista) > 0.5 or abs(angolo) > 0.1
                
                if velocita > 2.0 and gear >= 1 and condizione_critica:
                    row = {
                        "sensors": {
                            "track": client.S.d.get('track'),
                            "speedX": velocita,
                            "angle": angolo,
                            "trackPos": pos_pista,
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
                    status_text = "🔴 REGISTRANDO RECUPERO"
                else:
                    status_text = "⚪ IN ATTESA DI CRITICITÀ"

                print(f"\r{status_text} | Pos: {pos_pista:5.2f} | Angolo: {angolo:5.2f} | Campioni: {count}", end="")

    except KeyboardInterrupt:
        print(f"\n\n✅ Manovre salvate correttamente in {file_name}")

if __name__ == "__main__":
    main()
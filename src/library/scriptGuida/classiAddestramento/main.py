
import os
import sys
import time

from config import (
    FREQUENZA_UPDATE,
    TICK_INTERVAL_S,
    DANNO_RESET_SOGLIA,
)
from agente      import Agente
from connessione import GestoreConnessione
from episodio    import Episodio
from reward      import CalcolatoreReward


def main():
    # ── Cartella di lavoro: stessa directory di main.py
    cartella_out = os.path.dirname(os.path.abspath(__file__))

    # ── Inizializzazione componenti
    try:
        agente = Agente(cartella_out)
    except FileNotFoundError as e:
        print(f"❌  {e}")
        sys.exit(1)

    connessione = GestoreConnessione()
    calcolatore = CalcolatoreReward()

    # ── Connessione iniziale a TORCS 
    try:
        connessione.connetti()
    except ConnectionError as e:
        print(f"❌  {e}")
        sys.exit(1)

    # ── Variabili di stato del loop 
    episodio    = Episodio()
    ep_n        = 0           # numero episodio corrente
    t_prev_tick = time.monotonic() #orologio che va solo in avanti

    print("\n🏁  Watson è pronto. Inizio sessione di addestramento.\n")

    try:
        while True:
            ora        = time.monotonic()
            sleep_time = TICK_INTERVAL_S - (ora - t_prev_tick)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            t_prev_tick = time.monotonic()

            # ── B. LETTURA SENSORI 
            sensori = connessione.leggi_sensori()
            if not sensori:
                continue  # pacchetto incompleto: salta il tick

            danno    = float(sensori.get('damage',  0.0))
            velocita = float(sensori.get('speedX',  0.0))
            stallo   = episodio.e_in_stallo()

            # ── C. RESET (schianto o stallo) 
            if danno > DANNO_RESET_SOGLIA or stallo:
                motivo = "danni gravi" if danno > DANNO_RESET_SOGLIA else "stallo"
                print(f"\n⚠️   Reset per {motivo} "
                      f"(danno={danno:.0f}, v={velocita:.1f})")

                # Riepilogo visivo dell'episodio appena concluso
                episodio.sommario(ep_n)
                ep_n += 1

                # Rimuovi le esperienze tossiche (tick prima dello schianto)
                agente.pulisci_memoria_schianto()

                # Reset connessione TORCS (fix schermata blu UDP)
                connessione.reset_episodio()

                # Ripristina i pesi al best performer (anti-catastrophic forgetting)
                agente.ripristina_pesi_migliori()

                # Salvataggio valutazione periodica
                agente.valuta_e_salva(reward=0.0, ep_n=ep_n)

                # Reset del calcolatore reward (azzera memoria Δdanno)
                calcolatore.reset_episodio()

                # Nuovo episodio
                episodio    = Episodio()
                t_prev_tick = time.monotonic()
                continue

            # ── D. STATO E AZIONE 
            stato = agente.normalizza_input(sensori)

            # L'agente sceglie l'azione e aggiorna epsilon
            sterzo, accel, freno = agente.scegli_azione(stato)

            # ── E. CAMBIO MARCIA E INVIO COMANDI 
            rpm    = float(sensori.get('rpm',  0.0))
            marcia = int(sensori.get('gear',   1))
            marcia = connessione.calcola_marcia(rpm, marcia)

            connessione.invia_comandi(sterzo, accel, freno, marcia)

            # ── F. CALCOLO REWARD 
            # La reward usa i sensori DOPO l'azione (stato risultante)
            reward, breakdown = calcolatore.calcola(sensori)

            # ── G. MEMORIZZAZIONE ESPERIENZA 
            stato_next = agente.normalizza_input(sensori)
            agente.memorizza(
                stato      = stato,
                azione     = {'accel': accel, 'freno': freno},
                reward     = reward,
                stato_next = stato_next,
                terminale  = False
            )

            # ── H. STATISTICHE EPISODIO 
            episodio.aggiorna(sensori, reward)

            # ── I. APPRENDIMENTO E LOG 
            if episodio.tick % FREQUENZA_UPDATE == 0:
                # Aggiornamento pesi del modello (backpropagation)
                agente.apprendi()

                # Valutazione performance e salvataggio condizionale
                agente.valuta_e_salva(reward, ep_n)

                # Log compatto su una riga con breakdown della reward
                dist = episodio.distanza_percorsa()
                print(
                    f"\r🏎️  Ep{ep_n:3d} T{episodio.tick:5d} | "
                    f"r={breakdown['totale']:+6.2f} "
                    f"[av={breakdown['avanzamento']:+5.2f} "
                    f"ct={breakdown['centro']:+4.2f} "
                    f"po={breakdown['p_uscita']:+5.2f} "
                    f"da={breakdown['p_danno']:+5.2f}] | "
                    f"v={velocita:5.1f} d={dist:6.0f}m "
                    f"ε={agente.epsilon:.3f} "
                    f"💥{episodio.n_urti} 🚧{episodio.n_uscite_pista}",
                    end='', flush=True
                )

    except KeyboardInterrupt:
        # ── Chiusura pulita 
        print("\n\n🛑  Sessione interrotta manualmente.")
        print(f"   Episodi completati:   {ep_n}")
        print(f"   Miglior reward media: {agente._miglior_reward:.3f}")

        path = agente.salva_finale(ep_n)
        print(f"   Modello salvato   →  '{path}'")

    except ConnectionError as e:
        print(f"\n❌  Errore di connessione: {e}")

    finally:
        # Sempre: chiude la socket UDP per non lasciare porte aperte
        connessione.disconnetti()
        print("   Connessione chiusa.")


if __name__ == "__main__":
    main()
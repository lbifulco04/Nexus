import os
import time
import platform
import subprocess

import snakeoil3_jm2 as snakeoil3

from config import TORCS_PORTA, TORCS_VISION


class GestoreConnessione:
    """
    Gestisce il ciclo di vita della connessione UDP con TORCS.

    """

    def __init__(self, porta: int = TORCS_PORTA, vision: bool = TORCS_VISION):
        self.porta  = porta
        self.vision = vision
        self._client = None   # sarà un oggetto snakeoil3.Client dopo connetti()

    # =================================================================
    #  Connessione e disconnessione
    # =================================================================

    def connetti(self, max_tentativi: int = 12) -> None:
        """
         Stabilisce la connessione UDP con TORCS.
        """

        #print di debug
        print(f"🔌  Connessione a TORCS su porta {self.porta}…")
        
        #chiamiamo la funzione tenta connessione
        self._client = self._tenta_connessione(max_tentativi)
        print("✅  Connesso a TORCS.")

    def disconnetti(self) -> None:
        """
        Chiude la socket UDP in modo sicuro.
        """
        if self._client is not None:
            try:
                self._client.so.close()
            except Exception:
                pass
            self._client = None

    def reset_episodio(self) -> None:
        """
        Esegue la sequenza corretta di reset dopo uno schianto o uno stallo:
          1. Invia meta=True a TORCS (richiesta di restart)
          2. Attende che TORCS riavvii la gara
          3. Chiude la vecchia socket 
          4. Crea un nuovo client 
        """
        #print di debug
        print("\n🔄  Invio reset a TORCS…")

        # Passo 1: comunica a TORCS la ripartenza
        if self._client is not None:
            #lo dico attraverso la chiave meta restituita
            self._client.R.d['meta'] = True
            #la mando al server
            self._client.respond_to_server()

        # Passo 2: attesa che TORCS riavvii internamente
        time.sleep(2.5)

        # Passo 3: chiudere la socket obsoleta
        self.disconnetti()

        # Passo 4: nuova connessione con handshake pulito
        print("🔄  Riconnessione dopo reset…")
        self._client = self._tenta_connessione(max_tentativi=12)
        print("✅  Riconnesso.")

    # =================================================================
    #  Lettura e scrittura
    # =================================================================

    def leggi_sensori(self) -> dict:
        """
        Legge il pacchetto di stato dal server TORCS e ritorna il dizionario
        dei sensori
        """
        if self._client is None:
            return {}

        self._client.get_servers_input()
        sensori = self._client.S.d

        # Validazione minima: i sensori track sono indispensabili per l'agente
        if not sensori or 'track' not in sensori:
            return {}

        return sensori

    def invia_comandi(self, sterzo: float, accel: float,
                      freno: float, marcia: int) -> None:
        """
        Invia i comandi di guida al simulatore TORCS.
        """
        if self._client is None:
            return

        #diamo i comandi al server ed inviamoli
        self._client.R.d.update({
            'steer': sterzo,
            'accel': accel,
            'brake': freno,
            'gear':  marcia,
            'meta':  False,
        })
        self._client.respond_to_server()

    # =================================================================
    #  Utilità
    # =================================================================

    @staticmethod
    def calcola_marcia(rpm: float, marcia_attuale: int) -> int:
        """
        Scaletta marce basata sugli RPM del motore.
        """
        if   rpm > 15500 and marcia_attuale < 6:
            return marcia_attuale + 1
        elif rpm < 9500  and marcia_attuale > 1:
            return marcia_attuale - 1
        return marcia_attuale

    # ──────────────────────────────────────────────────────────────────────────
    #  Metodi privati
    # ──────────────────────────────────────────────────────────────────────────

    def _tenta_connessione(self, max_tentativi: int) -> snakeoil3.Client:
        """
        Tenta di creare un nuovo Client snakeoil3, riprovando in caso di errore.
        Ogni tentativo fallito attende 2 secondi prima di riprovare.
        """
        for tentativo in range(1, max_tentativi + 1):
            try:
                return snakeoil3.Client(p=self.porta, vision=self.vision)
            except (OSError, SystemExit):
                print(f"   Tentativo {tentativo}/{max_tentativi} fallito – "
                      f"attendo 2 s…")
                time.sleep(2)

        raise ConnectionError(
            f"Impossibile connettersi a TORCS sulla porta {self.porta} "
            f"dopo {max_tentativi} tentativi.\n"
            f"Assicurati che TORCS sia avviato e in attesa di connessioni."
        )


# =================================================================
#  Funzioni di utilità per avviare/terminare TORCS (multipiattaforma)
# =================================================================

def avvia_torcs(vision: bool = False) -> None:
    """
    Avvia il processo TORCS in background in modo multipiattaforma.
    
    """

    #vediamo in che sistema è iniziata la comunciazione
    sistema    = platform.system()
    base_flags = ['-nofuel', '-nodamage', '-nolaptime'] #utilizziamo le flag di base per la visualizzazione
    if vision:
        base_flags.append('-vision') # nel caso in cui il parametro vision sia vero aggiungiamolo (sconsigliato per le prestazione--assicurato)

    if sistema == 'Windows':
        subprocess.Popen(
            ['torcs.exe'] + base_flags,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP #creiamo il nuovo sottoprocesso in Background con Popen
        )
    else:  # Linux e macOS
        subprocess.Popen(
            ['torcs'] + base_flags,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        # Su Linux, autostart.sh configura la gara automaticamente
        autostart = os.path.join(os.path.dirname(__file__), 'autostart.sh')
        if os.path.exists(autostart):
            time.sleep(1.5)
            subprocess.Popen(['sh', autostart])


def termina_torcs() -> None:
    """
    Termina il processo TORCS in modo multipiattaforma.
    Utile per fare pulizia prima di chiudere lo script.
    """
    sistema = platform.system()
    try:
        if sistema == 'Windows':
            subprocess.run(['taskkill', '/F', '/IM', 'torcs.exe'],
                           capture_output=True)
        else:
            subprocess.run(['pkill', '-f', 'torcs'],
                           capture_output=True)
    except FileNotFoundError:
        pass  # pkill/taskkill non disponibili: ignora silenziosamente
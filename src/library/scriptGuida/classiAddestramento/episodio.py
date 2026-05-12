
from config import (
    TICK_STALLO_TIMEOUT,
    DISTANZA_MINIMA_STALLO,
    TRACKPOS_FUORI,
    DELTA_DANNO_SOGLIA,
)


class Episodio:
    """
    Traccia tutte le statistiche rilevanti di un singolo episodio di guida.
    """

    def __init__(self):
        # ── Contatori di base 
        self.tick           = 0
        self.reward_totale  = 0.0

        # ── Distanza (in metri, da distRaced di TORCS
        self.dist_inizio    = None    # distRaced al primo tick valido
        self.dist_corrente  = 0.0    # distRaced aggiornato ogni tick
        self.dist_massima   = 0.0    # massima distanza dal punto di partenza

        # ── Checkpoint per il rilevamento stallo
        # Ogni TICK_STALLO_TIMEOUT tick confrontiamo la distanza attuale con
        # quella dell'ultimo checkpoint. Se non abbiamo avanzato abbastanza  stallo.
        self.dist_ck_prec   = 0.0

        # ── Contatori eventi negativi 
        self.n_uscite_pista = 0    # incrementato al primo tick fuori pista (edge rising)
        self.n_urti         = 0    # incrementato ogni volta che Δdanno > soglia

        # ── Stato interno per edge detection
        self._era_fuori     = False   # True se nel tick precedente era fuori pista
        self._danno_prec_ep = 0.0     # danno al tick precedente (per calcolare Δ)

    # =========================================================
    #  Interfaccia pubblica
    # =========================================================

    def aggiorna(self, sensori: dict, reward: float) -> None:
        """
        Aggiorna tutte le statistiche dell'episodio con i dati del tick corrente.
        Va chiamata una volta per tick, dopo aver inviato i comandi al simulatore.
        """
        self.tick          += 1
        self.reward_totale += reward

        # ── Aggiornamento distanza ─────────────────────────────────────────────
        dist  = float(sensori.get('distRaced', 0.0))
        
        if self.dist_inizio is None:
            # Primo tick: inizializziamo i riferimenti
            self.dist_inizio  = dist
            self.dist_ck_prec = dist
        
        #aggiorniamo le distanze e calcoliamo quanto abbiamo percorso
        self.dist_corrente = dist
        distanza_percorsa  = dist - self.dist_inizio
        #conserviamo la distanza massima percorsa
        self.dist_massima  = max(self.dist_massima, distanza_percorsa)

        # ── Rilevamento uscita pista 
        # Contiamo solo il momento in cui la macchina ENTRA fuori pista,

        t_pos = abs(float(sensori.get('trackPos', 0.0)))
        fuori_ora = t_pos > TRACKPOS_FUORI
        
        if fuori_ora and not self._era_fuori:
            self.n_uscite_pista += 1
        self._era_fuori = fuori_ora

        # ── Rilevamento urto 
        danno = float(sensori.get('damage', 0.0))
        if (danno - self._danno_prec_ep) > DELTA_DANNO_SOGLIA:
            self.n_urti += 1
        self._danno_prec_ep = danno

    def e_in_stallo(self) -> bool:
        """
        Controlla se la macchina è in stallo, usando la distanza percorsa
        invece della velocità istantanea.

        Motivazione: se le ruote slittano o la macchina oscilla sul posto,
        la speedX può essere > 0 anche se la macchina non avanza davvero.
        
        """
        # Controlla solo ogni TICK_STALLO_TIMEOUT tick
        if self.tick == 0 or self.tick % TICK_STALLO_TIMEOUT != 0:
            return False

        # Distanza percorsa dall'ultimo checkpoint
        percorsa          = self.dist_corrente - self.dist_ck_prec
        self.dist_ck_prec = self.dist_corrente  # aggiorna il checkpoint

        return percorsa < DISTANZA_MINIMA_STALLO

    def distanza_percorsa(self) -> float:
        """Ritorna la distanza totale percorsa dall'inizio dell'episodio."""
        return self.dist_corrente - (self.dist_inizio or 0.0)

    def reward_media(self) -> float:
        """Ritorna la reward media per tick dell'episodio corrente."""
        return self.reward_totale / max(self.tick, 1)

    def sommario(self, ep_n: int) -> None:
        """
        Stampa un riepilogo formattato dell'episodio appena concluso.
        Include distanza, reward media e contatori degli eventi negativi.
        """
        print(f"\n{'═' * 60}")
        print(f"  Episodio {ep_n:3d} terminato")
        print(f"  Tick:         {self.tick:5d}")
        print(f"  Distanza:     {self.distanza_percorsa():.1f} m  "
              f"(max: {self.dist_massima:.1f} m)")
        print(f"  Reward media: {self.reward_media():+.3f}  "
              f"(totale: {self.reward_totale:+.1f})")
        print(f"  Uscite pista: {self.n_uscite_pista:3d}  |  "
              f"Urti muro: {self.n_urti:3d}")
        print(f"{'═' * 60}")
import numpy as np
from config import (
    W_AVANZAMENTO, W_CENTRO, W_USCITA, W_DANNO, W_FOLLE,
    TRACKPOS_BORDO, TRACKPOS_FUORI, CLIFF_FUORI_PISTA,
    DELTA_DANNO_SOGLIA, ANGOLO_IMPATTO_SOGLIA, MULT_ANGOLO_IMPATTO,
    PENALITA_RETROMARCIA,
)

class CalcolatoreReward:
    """
    Calcola la reward tick per tick con memoria dello stato precedente.
    Avere stato precedente è fondamentale per calcolare Δdanno correttamente.
    """
     
    def __init__(self):
        self._danno_precedente = 0.0

    def reset_episodio(self):
        self._danno_precedente = 0.0

    def calcola(self, sensori: dict) -> tuple[float, dict]:
        v = float(sensori.get('speedX', 0.0))
        theta = float(sensori.get('angle', 0.0))
        t_pos = float(sensori.get('trackPos', 0.0))
        danno = float(sensori.get('damage', 0.0))

        #diamo un premio all'avanzamento ma una penalità allo sbandamento
        r_avanzamento = ((v * np.cos(theta)) - (v * abs(np.sin(theta)))) * W_AVANZAMENTO

        #la funzione gaussiana permette di avere massima reward al centro della pista , minore alle code
        r_centro = np.exp(-4.0 * (t_pos ** 2)) * W_CENTRO

        #utilizziamo un metodo statico privato per calcolare la penalità fuori pista
        p_uscita = self._penalita_posizione(t_pos)
        
        #calcoliamo la differenza del danno
        delta_danno = max(0.0, danno - self._danno_precedente)

        #il danno precendente ora assume il valore danno
        self._danno_precedente = danno

        #calcoliamo la penalità dovuta al danno tramite metodo statico privato
        p_danno = self._penalita_danno(delta_danno, theta)

        #per penalità folle intendiamo guida non efficiente, calcoliamo anche qui la penalità
        p_folle = self._penalita_folle(v, theta)

        #trovati tutti i pesi dei reward calcliamo il totale e ritorniamo il dizionario di reward, con anche il totale
        reward = r_avanzamento + r_centro - p_uscita - p_danno - p_folle
        return float(reward), {
            'totale': round(reward, 3), 'avanzamento': round(r_avanzamento, 3),
            'centro': round(r_centro, 3), 'p_uscita': round(-p_uscita, 3),
            'p_danno': round(-p_danno, 3), 'p_folle': round(-p_folle, 3)
        }

    @staticmethod
    def _penalita_posizione(t_pos: float) -> float:
        "Questo metodo statico calcola la penalità da dare in funzione della poszione della macchina"
        abs_pos = abs(t_pos)

        #Se la poszione è minore del bordo
        if abs_pos <= TRACKPOS_BORDO:
            #funzione quadratica che da 0 penalità se è al centro
            return W_USCITA * 0.25 * (abs_pos / TRACKPOS_BORDO) ** 2
        
        #Se la posizione è minore del fuori-pista
        elif abs_pos <= TRACKPOS_FUORI:
            #più la posizione è fuori pista, più aumenta la penalità
            t = (abs_pos - TRACKPOS_BORDO) / (TRACKPOS_FUORI - TRACKPOS_BORDO)
            return W_USCITA * (2.75 * t ** 2)
        #penalità massima
        return W_USCITA * 3.0 + CLIFF_FUORI_PISTA * (abs_pos - TRACKPOS_FUORI)

    @staticmethod
    def _penalita_danno(delta_danno: float, theta: float) -> float:
        "Questo metodo statico calcola la penalità in funzione del danno alla macchina"
        #se la differenza del danno non supera la soglia imposta da noi non diamo penalità
        if delta_danno < DELTA_DANNO_SOGLIA: 
            return 0.0
        #Calcoliamo la penalità come una frazione del danno fatto
        p = W_DANNO * (delta_danno / 100.0)
        
        #Se l'angolo di impatto è maggiore di una certa soglia sdiamo maggiore penalità
        if abs(theta) > ANGOLO_IMPATTO_SOGLIA:
            return p * MULT_ANGOLO_IMPATTO 
        else:
            return p


    @staticmethod
    def _penalita_folle(v: float, theta: float) -> float:
        "Questo metodo statico viene utilizzato per penalizzare guida inefficiente"
        
        #Se va in retromarcia diamo una penalità
        if v < 0.0:
            p = PENALITA_RETROMARCIA
        else:
            p = 0.0

        #Se l'angolo rispetto alla strata è maggiore di 60° sbanderà, dunque penalità
        if abs(theta) > np.pi / 3:
            #sommiamo alla penalità retromarcia  anche questa
            p += W_FOLLE * (abs(theta) - np.pi / 3)
        return p
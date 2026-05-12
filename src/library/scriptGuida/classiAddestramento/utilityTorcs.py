import platform
import subprocess
import time
import numpy as np

class Episodio:
    def __init__(self, trackpos_fuori, delta_danno_soglia):
        self.tick, self.reward_totale = 0, 0.0
        self.dist_inizio, self.dist_corrente, self.dist_massima = None, 0.0, 0.0
        self.n_uscite_pista, self.n_urti = 0, 0
        self._era_fuori = False
        self._danno_prec_ep = 0.0
        self.T_FUORI, self.D_SOGLIA = trackpos_fuori, delta_danno_soglia

    def aggiorna(self, sensori, reward):
        self.tick += 1
        self.reward_totale += reward
        dist = sensori.get('distRaced', 0.0)
        if self.dist_inizio is None: self.dist_inizio = dist
        self.dist_corrente = dist
        self.dist_massima = max(self.dist_massima, dist - self.dist_inizio)

        fuori_ora = abs(sensori.get('trackPos', 0.0)) > self.T_FUORI
        if fuori_ora and not self._era_fuori: self.n_uscite_pista += 1
        self._era_fuori = fuori_ora

        danno = sensori.get('damage', 0.0)
        if (danno - self._danno_prec_ep) > self.D_SOGLIA: self.n_urti += 1
        self._danno_prec_ep = danno

def normalizza_input(sensori):
    stato = np.empty((1, 23), dtype=np.float32)
    stato[0, :19] = np.array(sensori.get('track', [0]*19)) / 200.0
    stato[0, 19] = sensori.get('speedX', 0.0) / 300.0
    stato[0, 20] = sensori.get('angle', 0.0) / np.pi
    stato[0, 21] = sensori.get('trackPos', 0.0)
    stato[0, 22] = sensori.get('rpm', 0.0) / 18000.0
    return stato
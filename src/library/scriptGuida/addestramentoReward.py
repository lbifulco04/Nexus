"""
addestramentoReward.py  –  Watson RL Agent per TORCS  
===========================================================

--------------------
1. SISTEMA DI PENALITÀ COERENTE
   La reward è scomposta in 5 termini indipendenti e bilanciati:

   R = R_avanzamento          (premia la velocità pulita lungo la traiettoria)
     + R_centro_pista         (premia stare vicino alla mezzeria)
     - P_uscita_pista         (penalità progressiva + cliff quando |trackPos| > 1)
     - P_impatto_muro         (penalità proporzionale alla variazione di danno)
     - P_comportamento_folle  (penalità per angolo estremo + retromarcia)

2. REWARD SHAPING PROGRESSIVO
   - Dentro la pista  : penalità cresce con una curva quadratica dolce
   - Sul bordo        : curva molto ripida (zona "da evitare")
   - Fuori pista      : cliff fisso pesante ad ogni tick

3. MEMORIA DEI DANNI
   Il delta danno (Δdamage) viene calcolato tick per tick.
   Permette di rilevare sia schianti violenti (Δ grande in un tick)
   sia abrasioni continue contro il muro (Δ piccolo ma persistente).

4. PENALITÀ ANGOLO DI IMPATTO
   Se la macchina prende danni mentre ha un angolo > soglia rispetto
   alla pista, la penalità viene moltiplicata (sbattere di lato costa di più).

5. METRICHE DI EPISODIO DETTAGLIATE
   La classe Episodio conta uscite pista, urti e registra la distanza
   massima raggiunta per monitorare i progressi.
"""

import sys
import os
import time
import random
import platform
import subprocess
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

import library.scriptGuida.guidaManuale.snakeoil3_jm2 as snakeoil3

# ─── SILENZIA GPU E LOG TF ────────────────────────────────────────────────────
tf.config.set_visible_devices([], 'GPU')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

# ─── FILE MODELLI ─────────────────────────────────────────────────────────────
NOME_MODELLO_BASE   = "modello_keras_v1.h5"
NOME_MODELLO_BEST   = "watson_best_performer.h5"
NOME_MODELLO_BACKUP = "watson_backup_ep{}.h5"

# ─── IPERPARAMETRI RL ─────────────────────────────────────────────────────────
MEMORIA_MAX           = 8000 #il numero di episodi massimi che vogliamo registrare
GAMMA                 = 0.95 
BATCH_SIZE            = 32
EPSILON_START         = 0.12 #la possibilità che il modello esplori senza seguire le azioni che conosce
EPSILON_MIN           = 0.02 
EPSILON_DECAY         = 0.9997
FREQUENZA_UPDATE      = 100 
FINESTRA_VALUTAZ      = 400
N_ESPERIENZE_CATTIVE  = 300 #il numero di esperienze cattive da andare a svuotare prima di ricominciare
SALVATAGGIO_PERIODICO = 5 #salviamo i modelli ogni 5 episodi
TICK_STALLO_TIMEOUT   = 150
DISTANZA_MINIMA_STALLO= 0.5

# ─── PARAMETRI REWARD ─────────────────────────────────────────────────────────
# Pesi dei 5 termini della reward
W_AVANZAMENTO   = 1.0    # velocità pulita lungo la pista
W_CENTRO        = 0.5    # bonus per stare vicino alla mezzeria
W_USCITA        = 2.5    # peso penalità uscita pista
W_DANNO         = 3.0    # peso penalità danno istantaneo
W_FOLLE         = 0.8    # peso penalità comportamento folle

# Soglie pista per valutare come ricompensare il modello
TRACKPOS_BORDO  = 0.85   # oltre questo valore si è "sul bordo"
TRACKPOS_FUORI  = 1.00   # oltre questo valore si è fuori pista

# Penalità cliff (applicata ad ogni tick fuori pista)
CLIFF_FUORI_PISTA = 15.0

# Danno: variazione minima per considerare un urto "reale"
DELTA_DANNO_SOGLIA = 5.0

# Moltiplicatore penalità se l'angolo di impatto è > soglia (sbattere di lato)
ANGOLO_IMPATTO_SOGLIA = 0.3   # radianti (~17°)
MULT_ANGOLO_IMPATTO   = 2.5

# Velocità negativa (retromarcia involontaria)
PENALITA_RETROMARCIA = 8.0

# ─── TIMING ───────────────────────────────────────────────────────────────────
TICK_RATE_HZ    = 50
TICK_INTERVAL_S = 1.0 / TICK_RATE_HZ


# =======================================================================================
#  SISTEMA DI REWARD: Classe che permette di calcolare il Reward dei gesti fatti in pista
# =======================================================================================
class CalcolatoreReward:
    """
    Calcola la reward tick per tick con memoria dello stato precedente.
    Avere stato precedente è fondamentale per calcolare Δdanno correttamente.
    """

    def __init__(self):
        self._danno_precedente: float = 0.0
        self._reset()

    def _reset(self):
        self._danno_precedente = 0.0

    def reset_episodio(self):
        """Da chiamare all'inizio di ogni nuovo episodio."""
        self._reset()

    def calcola(self, sensori: dict) -> tuple[float, dict]:
        """
        Restituisce (reward_totale, breakdown_dict) dove breakdown
        contiene i singoli contributi per il logging.
        """
        v       = float(sensori.get('speedX',   0.0))
        theta   = float(sensori.get('angle',    0.0))
        t_pos   = float(sensori.get('trackPos', 0.0))
        danno   = float(sensori.get('damage',   0.0))

        # ── 1. AVANZAMENTO PULITO ─────────────────────────────────────────────
        # v·cos(θ): premia velocità lungo l'asse della pista.
        # v·|sin(θ)|: penalizza il componente laterale (driftare verso il muro).
        r_avanzamento = (v * np.cos(theta)) - (v * abs(np.sin(theta)))
        r_avanzamento *= W_AVANZAMENTO

        # ── 2. BONUS CENTRATURA ───────────────────────────────────────────────
        # Gaussiana centrata in 0: massimo quando trackPos=0, scende ai bordi.
        # Non è una penalità, è un bonus — la macchina è premiata, non solo punita.
        r_centro = np.exp(-4.0 * (t_pos ** 2)) * W_CENTRO

        # ── 3. PENALITÀ USCITA PISTA (progressiva + cliff) ───────────────────
        p_uscita = self._penalita_posizione(t_pos)

        # ── 4. PENALITÀ DANNO ISTANTANEO (Δdamage) ───────────────────────────
        delta_danno = max(0.0, danno - self._danno_precedente)
        self._danno_precedente = danno
        p_danno = self._penalita_danno(delta_danno, theta)

        # ── 5. PENALITÀ COMPORTAMENTO FOLLE ──────────────────────────────────
        p_folle = self._penalita_folle(v, theta)

        # ── COMPOSIZIONE FINALE ───────────────────────────────────────────────
        reward = r_avanzamento + r_centro - p_uscita - p_danno - p_folle

        breakdown = {
            'avanzamento': round(r_avanzamento, 3),
            'centro':      round(r_centro,      3),
            'p_uscita':    round(-p_uscita,     3),
            'p_danno':     round(-p_danno,      3),
            'p_folle':     round(-p_folle,      3),
            'totale':      round(reward,         3),
        }
        return float(reward), breakdown

    # ── Metodi privati ────────────────────────────────────────────────────────

    @staticmethod
    def _penalita_posizione(t_pos: float) -> float:
        """
        Curva di penalità in tre zone:

        |trackPos| <= BORDO   -> curva quadratica dolce (0 al centro, cresce piano)
        BORDO < |p| <= FUORI  -> curva quadratica ripida (zona di allerta)
        |p| > FUORI           -> cliff fisso pesante ad ogni tick

        Questo design insegna alla macchina a NON avvicinarsi al bordo,
        non solo a non uscire.
        """
        abs_pos = abs(t_pos)

        if abs_pos <= TRACKPOS_BORDO:
            # Zona sicura: penalità quadratica leggera
            # Vale 0 al centro, ~0.18·W al bordo
            penalita = W_USCITA * 0.25 * (abs_pos / TRACKPOS_BORDO) ** 2

        elif abs_pos <= TRACKPOS_FUORI:
            # Zona bordo: curva molto più ripida
            # Normalizzata tra 0 e 1 in questo intervallo
            t = (abs_pos - TRACKPOS_BORDO) / (TRACKPOS_FUORI - TRACKPOS_BORDO)
            penalita = W_USCITA * (0.25 + 2.75 * t ** 2)   # va da 0.25·W a 3·W

        else:
            # Fuori pista: cliff fisso ad ogni tick
            # Più si è fuori, più costa (proporzionale alla distanza dal bordo)
            distanza_fuori = abs_pos - TRACKPOS_FUORI
            penalita = W_USCITA * 3.0 + CLIFF_FUORI_PISTA * (1.0 + distanza_fuori)

        return penalita

    @staticmethod
    def _penalita_danno(delta_danno: float, theta: float) -> float:
        """
        Penalità basata sulla variazione di danno in questo tick.

        - Delta piccolo (< soglia): ignorato (rumore di simulazione)
        - Delta reale: penalità lineare scalata con W_DANNO
        - Se l'angolo di impatto è grande (sbattiamo di lato), moltiplica

        Questo è molto più reattivo della penalità sul danno assoluto:
        rileva sia schianti violenti (delta grande) sia abrasioni continue.
        """
        if delta_danno < DELTA_DANNO_SOGLIA:
            return 0.0

        penalita = W_DANNO * (delta_danno / 100.0)   # normalizzata: 100 punti danno = W_DANNO

        # Moltiplicatore angolo: sbattere di lato è peggio
        if abs(theta) > ANGOLO_IMPATTO_SOGLIA:
            penalita *= MULT_ANGOLO_IMPATTO

        return penalita

    @staticmethod
    def _penalita_folle(v: float, theta: float) -> float:
        """
        Penalizza due comportamenti degeneri:
        - Retromarcia involontaria (v < 0): la macchina è bloccata al contrario
        - Angolo molto elevato in curva (> π/3 ~60°): deriva incontrollata
        """
        p = 0.0
        if v < 0.0:
            p += PENALITA_RETROMARCIA
        if abs(theta) > np.pi / 3:
            p += W_FOLLE * (abs(theta) - np.pi / 3) / (np.pi * 2 / 3)
        return p


# ══════════════════════════════════════════════════════════════════════════════
#  RESTO DELL'INFRASTRUTTURA 
# ═════════════════════════════════════════════════════════════════════════════

# ─── TORCS LAUNCHER ───────────────────────────────────────────────────────────
def _kill_torcs():
    sistema = platform.system()
    try:
        if sistema == 'Windows':
            subprocess.run(['taskkill', '/F', '/IM', 'torcs.exe'], capture_output=True)
        else:
            subprocess.run(['pkill', '-f', 'torcs'], capture_output=True)
    except FileNotFoundError:
        pass

def _start_torcs(vision: bool = False):
    sistema    = platform.system()
    base_flags = ['-nofuel', '-nodamage', '-nolaptime']
    if vision:
        base_flags.append('-vision')
    if sistema == 'Windows':
        subprocess.Popen(['torcs.exe'] + base_flags,
                         creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        subprocess.Popen(['torcs'] + base_flags,
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        autostart = os.path.join(os.path.dirname(__file__), 'autostart.sh')
        if os.path.exists(autostart):
            time.sleep(1.5)
            subprocess.Popen(['sh', autostart])

# ─── RICONNESSIONE CLIENT ─────────────────────────────────────────────────────
def reconnect_client(porta: int = 3001, vision: bool = False,
                     max_tentativi: int = 12) -> snakeoil3.Client:
    print("\n🔄  Riconnessione a TORCS...")
    for tentativo in range(1, max_tentativi + 1):
        try:
            client = snakeoil3.Client(p=porta, vision=vision)
            print(f"✅  Connesso al tentativo {tentativo}.")
            return client
        except (OSError, SystemExit):
            print(f"   Tentativo {tentativo}/{max_tentativi} – attendo 2 s…")
            time.sleep(2)
    raise ConnectionError("Impossibile riconnettersi a TORCS.")

# ─── NORMALIZZAZIONE INPUT ────────────────────────────────────────────────────
def normalizza_input(sensori: dict) -> np.ndarray:
    stato = np.empty((1, 23), dtype=np.float32)
    track = sensori.get('track', [0] * 19)
    stato[0, :19] = np.array(track, dtype=np.float32) / 200.0
    stato[0, 19]  = sensori.get('speedX',   0.0) / 300.0
    stato[0, 20]  = sensori.get('angle',    0.0) / np.pi
    stato[0, 21]  = sensori.get('trackPos', 0.0)
    stato[0, 22]  = sensori.get('rpm',      0.0) / 18000.0
    return stato

# ─── AGGIORNAMENTO MODELLO ────────────────────────────────────────────────────
def aggiorna_modello(modello, memoria: deque):
    if len(memoria) < BATCH_SIZE:
        return
    minibatch = random.sample(memoria, BATCH_SIZE)
    s_batch   = np.vstack([e[0] for e in minibatch])
    s_next_b  = np.vstack([e[3] for e in minibatch])

    pred_st,   pred_pe  = modello(s_batch,  training=False)
    pred_next, _        = modello(s_next_b, training=False)

    t_sterzo = pred_st.numpy().copy()
    t_pedali = pred_pe.numpy().copy()

    for i, (_, azione, premio, _, fine) in enumerate(minibatch):
        target         = premio if fine else premio + GAMMA * float(pred_next[i][0])
        t_sterzo[i, 0] = target
        t_pedali[i, 0] = azione['accel']
        t_pedali[i, 1] = azione['freno']

    modello.fit(
        s_batch,
        {'uscita_sterzo': t_sterzo, 'uscita_pedali': t_pedali},
        epochs=1, verbose=0, batch_size=BATCH_SIZE
    )

# ─── CAMBIO MARCIA ────────────────────────────────────────────────────────────
def calcola_marcia(rpm: float, marcia: int) -> int:
    if   rpm > 17500 and marcia < 6: return marcia + 1
    elif rpm < 9000  and marcia > 1: return marcia - 1
    return marcia

# ─── CLASSE EPISODIO ─────────────────────────────────────────────────────────
class Episodio:
    """
    Traccia statistiche dettagliate per un singolo episodio, inclusi
    contatori di uscite pista e urti per valutare i progressi nel tempo.
    """
    def __init__(self):
        self.tick            = 0
        self.reward_totale   = 0.0
        self.dist_inizio     = None
        self.dist_corrente   = 0.0
        self.dist_ck_prec    = 0.0
        self.dist_massima    = 0.0   # record di distanza per episodio
        # contatori eventi negativi
        self.n_uscite_pista  = 0
        self.n_urti          = 0
        self._era_fuori      = False
        self._danno_prec_ep  = 0.0

    def aggiorna(self, sensori: dict, reward: float):
        self.tick          += 1
        self.reward_totale += reward
        dist  = sensori.get('distRaced', 0.0)
        t_pos = abs(sensori.get('trackPos', 0.0))
        danno = sensori.get('damage', 0.0)

        if self.dist_inizio is None:
            self.dist_inizio  = dist
            self.dist_ck_prec = dist
        self.dist_corrente = dist
        self.dist_massima  = max(self.dist_massima,
                                 dist - (self.dist_inizio or 0))

        # Conta uscite (edge rising: entra fuori pista)
        fuori_ora = t_pos > TRACKPOS_FUORI
        if fuori_ora and not self._era_fuori:
            self.n_uscite_pista += 1
        self._era_fuori = fuori_ora

        # Conta urti (Δdanno > soglia)
        delta = danno - self._danno_prec_ep
        if delta > DELTA_DANNO_SOGLIA:
            self.n_urti += 1
        self._danno_prec_ep = danno

    def e_in_stallo(self) -> bool:
        if self.tick % TICK_STALLO_TIMEOUT != 0 or self.tick == 0:
            return False
        percorsa          = self.dist_corrente - self.dist_ck_prec
        self.dist_ck_prec = self.dist_corrente
        return percorsa < DISTANZA_MINIMA_STALLO

    def sommario(self, ep_n: int):
        media = self.reward_totale / max(self.tick, 1)
        dist  = self.dist_corrente - (self.dist_inizio or 0)
        print(f"\n{'═'*60}")
        print(f"  Episodio {ep_n:3d}")
        print(f"  Tick: {self.tick:5d}  |  Distanza: {dist:.1f} m  "
              f"|  Max: {self.dist_massima:.1f} m")
        print(f"  Reward media: {media:+.3f}")
        print(f"  Uscite pista: {self.n_uscite_pista:3d}  "
              f"|  Urti muro: {self.n_urti:3d}")
        print(f"{'═'*60}")

# ─── CLASSE VALUTATORE ────────────────────────────────────────────────────────
class Valutatore:
    def __init__(self, cartella: str = '.'):
        self.miglior_reward = -999999.0
        self.reward_accum   = 0.0
        self.conteggio      = 0
        self.cartella       = cartella
        os.makedirs(cartella, exist_ok=True)

    def step(self, modello, reward: float, ep_n: int):
        self.reward_accum += reward
        self.conteggio    += 1
        if self.conteggio >= FINESTRA_VALUTAZ:
            media = self.reward_accum / self.conteggio
            print(f"\n📊  Media reward: {media:.3f}  |  Record: {self.miglior_reward:.3f}")
            if media > self.miglior_reward:
                self.miglior_reward = media
                path = os.path.join(self.cartella, NOME_MODELLO_BEST)
                modello.save(path)
                print(f"🌟  Nuovo record! Salvato → '{path}'")
            self.reward_accum = 0.0
            self.conteggio    = 0
        if ep_n > 0 and ep_n % SALVATAGGIO_PERIODICO == 0:
            nome = NOME_MODELLO_BACKUP.format(ep_n)
            path = os.path.join(self.cartella, nome)
            modello.save(path)
            print(f"\n💾  Backup episodio {ep_n} → '{path}'")

# ─── ANTI-FORGETTING ─────────────────────────────────────────────────────────
def carica_pesi_migliori(modello, cartella: str):
    path_best = os.path.join(cartella, NOME_MODELLO_BEST)
    if os.path.exists(path_best):
        try:
            tmp = keras.models.load_model(path_best, compile=False)
            modello.set_weights(tmp.get_weights())
            del tmp
            print("   ↩️  Pesi ripristinati al best performer.")
        except Exception as e:
            print(f"   ⚠️  Ripristino pesi fallito: {e}")

def pulisci_memoria_schianto(memoria: deque):
    rimossi = 0
    while memoria and rimossi < N_ESPERIENZE_CATTIVE:
        memoria.pop()
        rimossi += 1
    print(f"   🧹  Rimosse {rimossi} esperienze negative.")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    cartella_out = os.path.dirname(os.path.abspath(__file__))

    if not os.path.exists(NOME_MODELLO_BASE):
        print(f"❌  '{NOME_MODELLO_BASE}' non trovato.")
        sys.exit(1)

    print("🧠  Caricamento modello base…")
    modello = keras.models.load_model(NOME_MODELLO_BASE, compile=False)
    modello.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss={'uscita_sterzo': 'mse', 'uscita_pedali': 'mse'}
    )
    path_best = os.path.join(cartella_out, NOME_MODELLO_BEST)
    if os.path.exists(path_best):
        print(f"   Trovato '{NOME_MODELLO_BEST}' – riprendo da lì.")
        tmp = keras.models.load_model(path_best, compile=False)
        modello.set_weights(tmp.get_weights())
        del tmp

    porta        = 3001
    vision       = False
    memoria      = deque(maxlen=MEMORIA_MAX)
    valut        = Valutatore(cartella_out)
    calcolatore  = CalcolatoreReward()   # <── istanza con memoria del danno
    epsilon      = EPSILON_START
    ep_n         = 0

    print(f"🔌  Connessione a TORCS su porta {porta}…")
    client      = snakeoil3.Client(p=porta, vision=vision)
    episodio    = Episodio()
    t_prev_tick = time.monotonic()

    try:
        while True:
            # ── THROTTLE ──────────────────────────────────────────────────────
            ora        = time.monotonic()
            sleep_time = TICK_INTERVAL_S - (ora - t_prev_tick)
            if sleep_time > 0:
                time.sleep(sleep_time)
            t_prev_tick = time.monotonic()

            # ── LETTURA ───────────────────────────────────────────────────────
            client.get_servers_input()
            sensori = client.S.d
            if not sensori or 'track' not in sensori:
                continue

            danno    = sensori.get('damage',  0.0)
            velocita = sensori.get('speedX',  0.0)
            stallo   = episodio.e_in_stallo()

            # ── RESET ─────────────────────────────────────────────────────────
            if danno > 3500 or stallo:
                motivo = "danni" if danno > 3500 else "stallo"
                print(f"\n⚠️   Reset per {motivo}")
                episodio.sommario(ep_n)
                ep_n += 1

                pulisci_memoria_schianto(memoria)

                client.R.d['meta'] = True
                client.respond_to_server()
                time.sleep(2.5)

                try:
                    client.so.close()
                except Exception:
                    pass
                client = reconnect_client(porta=porta, vision=vision)

                carica_pesi_migliori(modello, cartella_out)
                calcolatore.reset_episodio()   # <── azzera Δdanno
                valut.step(modello, 0.0, ep_n)

                episodio    = Episodio()
                t_prev_tick = time.monotonic()
                continue

            # ── STATO ─────────────────────────────────────────────────────────
            stato = normalizza_input(sensori)

            # ── AZIONE ────────────────────────────────────────────────────────
            if random.random() <= epsilon:
                sterzo = random.uniform(-0.10, 0.10)
                accel  = 0.5
                freno  = 0.0
            else:
                pred_st, pred_pe = modello(stato, training=False)
                sterzo = float(pred_st[0][0])
                accel  = float(pred_pe[0][0])
                freno  = float(pred_pe[0][1])

            if epsilon > EPSILON_MIN:
                epsilon *= EPSILON_DECAY

            # ── MARCIA ────────────────────────────────────────────────────────
            rpm    = sensori.get('rpm',  0.0)
            marcia = int(sensori.get('gear', 1))
            marcia = calcola_marcia(rpm, marcia)

            # ── COMANDI ───────────────────────────────────────────────────────
            client.R.d.update({
                'steer': sterzo, 'accel': accel,
                'brake': freno,  'gear': marcia, 'meta': False
            })
            client.respond_to_server()

            # ── REWARD (nuovo sistema) ─────────────────────────────────────────
            reward, breakdown = calcolatore.calcola(sensori)
            next_stato        = normalizza_input(sensori)
            memoria.append((stato, {'accel': accel, 'freno': freno},
                            reward, next_stato, False))
            episodio.aggiorna(sensori, reward)

            # ── APPRENDIMENTO ──────────────────────────────────────────────────
            if episodio.tick % FREQUENZA_UPDATE == 0:
                aggiorna_modello(modello, memoria)
                valut.step(modello, reward, ep_n)

                # Log compatto con breakdown reward
                dist = episodio.dist_corrente - (episodio.dist_inizio or 0)
                print(
                    f"\r🏎️  Ep{ep_n:3d} T{episodio.tick:5d} | "
                    f"r={breakdown['totale']:+6.2f} "
                    f"[av={breakdown['avanzamento']:+5.2f} "
                    f"ct={breakdown['centro']:+4.2f} "
                    f"po={breakdown['p_uscita']:+5.2f} "
                    f"da={breakdown['p_danno']:+5.2f}] | "
                    f"v={velocita:.1f} d={dist:.0f}m ε={epsilon:.3f}",
                    end='', flush=True
                )

    except KeyboardInterrupt:
        print("\n\n🛑  Interruzione manuale.")
        print(f"   Episodi completati:   {ep_n}")
        print(f"   Miglior reward media: {valut.miglior_reward:.3f}")
        path_finale = os.path.join(cartella_out, f"watson_finale_ep{ep_n}.h5")
        modello.save(path_finale)
        print(f"   Salvato → '{path_finale}'")
    except ConnectionError as e:
        print(f"\n❌  {e}")
    finally:
        try:
            client.so.close()
        except Exception:
            pass

if __name__ == "__main__":
    main()
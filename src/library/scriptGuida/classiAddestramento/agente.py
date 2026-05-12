import os
import random
from collections import deque

import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import (
    NOME_MODELLO_BASE, NOME_MODELLO_BEST, NOME_MODELLO_BACKUP,
    MEMORIA_MAX, GAMMA, BATCH_SIZE,
    EPSILON_START, EPSILON_MIN, EPSILON_DECAY,
    FINESTRA_VALUTAZ, SALVATAGGIO_PERIODICO,
    N_ESPERIENZE_CATTIVE,
)

# Silenzia GPU e log TF (fatto qui perché l'agente è il primo a importare TF)
tf.config.set_visible_devices([], 'GPU')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')


class Agente:
    """
    Incapsula il modello Keras e tutta la logica di apprendimento.

    Parametri costruttore:
        cartella_out: percorso dove salvare/caricare i modelli
    """

    def __init__(self, cartella_out: str = '.'):
        self.cartella_out = cartella_out

        # ── Epsilon: probabilità di azione casuale (esplorazione) ─────────────
        self.epsilon = EPSILON_START

        # buffer circolare delle esperienze passate
        # Ogni elemento è una tupla (stato, azione, reward, stato_next, terminale)
        self._memoria = deque(maxlen=MEMORIA_MAX)

        # ── Miglior reward media vista finora 
        self._miglior_reward  = -999999.0

        # ── Accumulatori per il calcolo della reward media a finestra ─────────
        self._reward_accum  = 0.0
        self._reward_count  = 0

        # ── Caricamento modello ───────────────────────────────────────────────
        self.modello = self._carica_modello()

    
    @staticmethod
    def normalizza_input(sensori):
        """
        Converte il dizionario dei sensori TORCS in un vettore numpy (1, 23)
        con tutti i valori normalizzati nell'intervallo [0, 1] o [-1, 1].
        """
        stato = np.empty((1, 23), dtype=np.float32)
        track = sensori.get('track', [0] * 19)
        stato[0, :19] = np.array(track, dtype=np.float32) / 200.0
        stato[0, 19]  = float(sensori.get('speedX',   0.0)) / 300.0
        stato[0, 20]  = float(sensori.get('angle',    0.0)) / np.pi
        stato[0, 21]  = float(sensori.get('trackPos', 0.0))
        stato[0, 22]  = float(sensori.get('rpm',      0.0)) / 18000.0
        return stato

    def scegli_azione(self, stato: np.array):
        """
        Sceglie l'azione da eseguire usando :
          - con probabilità epsilon -> azione casuale (esplorazione)
          - altrimenti              -> azione del modello  (sfruttamento)

        Decrementa epsilon ad ogni chiamata fino a EPSILON_MIN.
        """
        if random.random() <= self.epsilon:
            # Esplorazione: azione casuale ma ragionevole
            sterzo = random.uniform(-0.10, 0.10)
            accel  = 0.5
            freno  = 0.0
        else:
            # Sfruttamento: previsione del modello
            pred_st, pred_pe = self.modello(stato, training=False)
            sterzo = float(pred_st[0][0])
            accel  = float(pred_pe[0][0])
            freno  = float(pred_pe[0][1])

        # Decay epsilon: epsilon deve diventare ad ogni chiamata con azione più piccolo
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

        return sterzo, accel, freno

    def memorizza(self, stato: np.ndarray, azione: dict,
                  reward: float, stato_next: np.ndarray,
                  terminale: bool = False):
        """
        Aggiunge un'esperienza alla replay memory. Ricordando che l'esperienza è sia lo stato (sensori)
        l' azione (sterzo e pedali) la reward.

        """
        self._memoria.append((stato, azione, reward, stato_next, terminale))

    def apprendi(self) -> None:
        """
        Esegue un passo di Q-Learning su un mini-batch estratto dalla memoria.

        L'Algoritmo è l'equazione di Bellman :
          1. Campiona BATCH_SIZE esperienze casuali dalla replay memory
          2. Per ogni esperienza, calcola il target Q:
               target = reward                         (se terminale)
               target = reward + γ · Q(s', a*)         (altrimenti)
          3. Aggiorna i pesi del modello verso i target calcolati

        """
        if len(self._memoria) < BATCH_SIZE:
            return  # non abbastanza dati per imparare qualcosa di utile

        #scegliamo a caso 12 campioni della memoria (ricordiamo sono tuple di 5 elementi)
        minibatch = random.sample(self._memoria, BATCH_SIZE)

        # Prepara i batch di stati in un unico array 
        s_batch  = np.vstack([e[0] for e in minibatch])   # impiliamo i 23 sensori in uno stack , essendo B la memoria Batch sarà una matrice (B, 23)
        sn_batch = np.vstack([e[3] for e in minibatch])   # stessa cosa qui ma con lo stato successivo (B, 23)

        # Predizione in batch unico: molto più veloce di B chiamate separate
        pred_st,   pred_pe  = self.modello(s_batch,  training=False)
        pred_next, _        = self.modello(sn_batch, training=False)

        # Calcola i target Q per sterzo e pedali
        t_sterzo = pred_st.numpy().copy()   # (B, 1) predice i valori di sterzo per i 32 episodi
        t_pedali = pred_pe.numpy().copy()   # (B, 2) predice i valori dei pedali per i 32 episodi

        for i, (_, azione, premio, _, fine) in enumerate(minibatch):
            # Equazione di Bellman
            if fine:
                target = premio
            else:
                target = premio + GAMMA * float(pred_next[i][0])
            
            # Aggiorna solo il neurone dello sterzo verso il target Q
            t_sterzo[i, 0] = target
            # Aggiorna i pedali verso l'azione effettivamente eseguita
            t_pedali[i, 0] = azione['accel']
            t_pedali[i, 1] = azione['freno']
        
        #addestriamo il modello sula base dei target calcolati dall'Equazione di Bellman
        self.modello.fit(
            s_batch,
            {'uscita_sterzo': t_sterzo, 'uscita_pedali': t_pedali},
            epochs=1, verbose=0, batch_size=BATCH_SIZE
        )

    def valuta_e_salva(self, reward: float, ep_n: int) -> None:
        """
        Accumula le reward nella finestra di valutazione e salva il modello
        se la media corrente batte il record storico.
        Salva anche un backup periodico ogni SALVATAGGIO_PERIODICO episodi.
        """
        #Accumuliamo la reward e contiamo il numero di reward accumulate
        self._reward_accum += reward
        self._reward_count += 1

        # Valutazione a fine finestra
        if self._reward_count >= FINESTRA_VALUTAZ:
            #calcoliamo la media della reward e stampiamola come debug confrontandola col miglior record
            media = self._reward_accum / self._reward_count
            print(f"\n📊  Media reward: {media:.3f}  |  "f"Record: {self._miglior_reward:.3f}")

            #se la media è migliore della miglior reward aggiorniamo lo stato dell'agente
            if media > self._miglior_reward:
                self._miglior_reward = media
                #prendiamo  il path in cui ci troviamo
                path = os.path.join(self.cartella_out, NOME_MODELLO_BEST)
                #salviamo il modello migliore
                self.modello.save(path)
                #stampa sempre di debug
                print(f"🌟  Nuovo record! Salvato → '{path}'")

            # Reset accumulatori della nostra finestra
            self._reward_accum = 0.0
            self._reward_count = 0

        # Backup periodico (dovuto a spegnimento del computer durante l'addestramento...)
        if ep_n > 0 and ep_n == SALVATAGGIO_PERIODICO:
            nome = NOME_MODELLO_BACKUP.format(ep_n)
            #prendiamo il path e il nome
            path = os.path.join(self.cartella_out, nome)
            #salviamo il modello nel path
            self.modello.save(path)
            #print di debug
            print(f"\n💾  Backup episodio {ep_n} → '{path}'")

    def ripristina_pesi_migliori(self) -> None:
        """
        Ricarica i pesi del best performer all'inizio di ogni nuovo episodio.
        """
        #recuperiamo il percorso del miglior modello
        path_best = os.path.join(self.cartella_out, NOME_MODELLO_BEST)
        if not os.path.exists(path_best):
            return  # nessun best performer ancora: lascia i pesi attuali

        try:
            #carichiamo il modello con i pesi migliori
            tmp = keras.models.load_model(path_best, compile=False)
            #aggiorniamo i pesi del nostro modello (script)
            self.modello.set_weights(tmp.get_weights())
            #eliminamo per efficienza
            del tmp
            #print di debug
            print("   ↩️  Pesi ripristinati al best performer.")
        except Exception as e:
            print(f"   ⚠️  Ripristino pesi fallito: {e}")

    def pulisci_memoria_schianto(self) -> None:
        """
        Rimuove le N_ESPERIENZE_CATTIVE più recenti dalla replay memory.
        Questo perchè al ripristino qualcosa sarà successo da non imparare
        """
        #accediamo alla memoria dell'a gente (tuple di 5 elementi) e svuotiamola dalla coda le N esperienze 
        rimossi = 0
        while self._memoria and rimossi < N_ESPERIENZE_CATTIVE:
            self._memoria.pop()
            rimossi += 1
        
        #print di debug
        print(f"   🧹  Rimosse {rimossi} esperienze negative dalla memoria.")

    def salva_finale(self, ep_n: int) -> str:
        """
        Salva il modello con un nome che include il numero di episodio.
        Usato alla chiusura del programma (Ctrl+C).
        """
        nome = f"watson_finale_ep{ep_n}.h5"
        path = os.path.join(self.cartella_out, nome)
        self.modello.save(path)
        return path

    # ==========================================================
    #  Metodi privati
    # ==========================================================

    def _carica_modello(self) -> keras.Model:
        """
        Carica il modello Keras nel seguente ordine di priorità:
          1. Se esiste watson_best_performer.h5 → carica come punto di partenza
          2. Altrimenti → carica modello_keras_v1.h5 (imitation learning)
        """
        
        # Controlliamo che il modello base esista
        if not os.path.exists(NOME_MODELLO_BASE):
            raise FileNotFoundError(
                f"'{NOME_MODELLO_BASE}' non trovato nella directory corrente.\n"
                f"Directory corrente: {os.getcwd()}"
            )

        #print di debug
        print(f"🧠  Caricamento '{NOME_MODELLO_BASE}'…")
        modello = keras.models.load_model(NOME_MODELLO_BASE, compile=False)

        # Ricompila con ottimizzatore Adam (necessario dopo load con compile=False)
        modello.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss={'uscita_sterzo': 'mse', 'uscita_pedali': 'mse'}
        )

        # Sovrascrive i pesi con il best performer se disponibile
        path_best = os.path.join(self.cartella_out, NOME_MODELLO_BEST)
        if os.path.exists(path_best):
            print(f"   Trovato '{NOME_MODELLO_BEST}' – riprendo da lì.")
            tmp = keras.models.load_model(path_best, compile=False)
            modello.set_weights(tmp.get_weights())
            del tmp

        return modello
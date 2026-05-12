"In questo file sono presenti tutti i paramtri utilizzati dalle nostre classi"
import os
 
# ==============================================================
#  PERCORSI FILE
# ==============================================================
 
# Modello originale addestrato con imitation learning (sola lettura)
NOME_MODELLO_BASE   = "modello_keras_v1.h5"
 
# Miglior modello trovato finora (aggiornato solo quando batte il record)
NOME_MODELLO_BEST   = "watson_best_performer.h5"
 
# Template per i backup periodici (es. watson_backup_ep10.h5)
NOME_MODELLO_BACKUP = "watson_backup_ep{}.h5"
 
 
# ==============================================================
#  IPERPARAMETRI REINFORCEMENT LEARNING
# ==============================================================
 
# Dimensione massima della memoria passata (esperienze passate)
MEMORIA_MAX = 8000
 
# Fattore di sconto: quanto pesano le ricompense future rispetto a quelle immediate.
# 0 = (solo reward istantanea), 1 = lungimirante (reward futura pesa quanto quella presente)
GAMMA = 0.95
 
# Numero di esperienze estratte dalla memoria per ogni aggiornamento dei pesi
BATCH_SIZE = 32
 
# Probabilità iniziale di scegliere un'azione casuale (esplorazione)
EPSILON_START = 0.12
 
# Probabilità minima di esplorazione (non scende mai sotto questo valore)
EPSILON_MIN = 0.02
 
# Moltiplicatore applicato a epsilon ad ogni tick per farlo decrescere lentamente
EPSILON_DECAY = 0.9997
 
# Ogni quanti tick eseguire un aggiornamento dei pesi (backpropagation)
FREQUENZA_UPDATE = 100
 
# Numero di tick su cui calcolare la reward media per valutare le performance
FINESTRA_VALUTAZ = 400
 
# Quante esperienze rimuovere dalla memoria dopo uno schianto
# (quelle immediatamente precedenti allo schianto, le più "tossiche")
N_ESPERIENZE_CATTIVE = 80
 
# Salva un backup del modello ogni N episodi (indipendentemente dalla performance)
SALVATAGGIO_PERIODICO = 5
 
 
# ==============================================================
#  RILEVAMENTO STALLO
# ==============================================================
 
# Ogni quanti tick controllare se la macchina è in stallo
TICK_STALLO_TIMEOUT = 150
 
# Distanza minima (in metri) che la macchina deve percorrere in TICK_STALLO_TIMEOUT tick
# Se percorre meno di questo, viene dichiarato stallo e si esegue il reset
DISTANZA_MINIMA_STALLO = 0.5
 
 
# ==============================================================
#  PARAMETRI DEL SISTEMA DI REWARD
# ==============================================================
 
# --- Pesi dei 5 termini della reward ---
 
# Peso del termine di avanzamento: v·cos(θ) − v·|sin(θ)|
# Premia la velocità pulita nella direzione della pista
W_AVANZAMENTO = 1.0
 
# Peso del bonus di centratura (gaussiana centrata in trackPos=0)
# Premia stare vicino alla mezzeria
W_CENTRO = 0.5
 
# Peso della penalità per uscita pista (curva progressiva + cliff)
# Valore alto perché vogliamo che la macchina resti in pista ad ogni costo
W_USCITA = 2.5
 
# Peso della penalità per danno istantaneo (Δdamage per tick)
# Più alto di W_USCITA perché un urto è peggio di stare sul bordo
W_DANNO = 3.0
 
# Peso della penalità per comportamento folle (angolo estremo, retromarcia)
W_FOLLE = 0.8
 
# --- Soglie di posizione sulla pista ---
 
# trackPos è normalizzato: 0 = centro, ±1 = bordo, >±1 = fuori pista
# Soglia oltre la quale si entra nella "zona di allerta" (penalità ripida)
TRACKPOS_BORDO = 0.85
 
# Soglia oltre la quale si è definitivamente fuori pista (penalità cliff)
TRACKPOS_FUORI = 1.00
 
# Penalità fissa applicata ad ogni tick in cui si è fuori pista
# Si somma alla penalità proporzionale alla distanza dal bordo
CLIFF_FUORI_PISTA = 15.0
 
# --- Parametri penalità danno ---
 
# Variazione minima di damage per considerare un urto "reale" e non rumore
DELTA_DANNO_SOGLIA = 5.0
 
# Angolo (radianti, ~17°) oltre il quale lo schianto è considerato "laterale"
# Sbattere di lato contro il muro è più penalizzato perché è meno recuperabile
ANGOLO_IMPATTO_SOGLIA = 0.3
 
# Moltiplicatore applicato alla penalità danno quando l'angolo supera la soglia
MULT_ANGOLO_IMPATTO = 2.5
 
# --- Parametri penalità comportamento folle ---
 
# Penalità applicata ogni tick in cui la macchina viaggia in retromarcia (speedX < 0)
PENALITA_RETROMARCIA = 8.0
 
 
# ==============================================================
#  TIMING E CONNESSIONE
# ==============================================================
 
# Frequenza di aggiornamento del simulatore TORCS
TICK_RATE_HZ = 50
 
# Intervallo target tra un tick e l'altro (usato dallo sleep adattivo)
TICK_INTERVAL_S = 1.0 / TICK_RATE_HZ
 
# Porta UDP su cui TORCS ascolta i comandi del client
TORCS_PORTA = 3001
 
# Se True, abilita il sensore visivo di TORCS (non usato in questa versione)
TORCS_VISION = False
 
# Danno massimo tollerato prima di forzare un reset dell'episodio
DANNO_RESET_SOGLIA = 3500
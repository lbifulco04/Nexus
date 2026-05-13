# ══════════════════════════════════════════════════════════════════════════════
#  PARAMETRI CONFIGURABILI
# ══════════════════════════════════════════════════════════════════════════════

EPOCHE_MAX        = 300      # EarlyStopping si ferma prima se necessario
BATCH_SIZE        = 64
LEARNING_RATE     = 0.0003   # più alto del base, ReduceLR lo abbasserà
TEST_SIZE         = 0.20     # 20% validazione
RANDOM_STATE      = 42

# Pesi della loss: sterzo più importante su Corkscrew
LOSS_WEIGHT_STERZO = 2.0
LOSS_WEIGHT_PEDALI = 0.5

# Soglia per considerare una sterzata "di recupero" o "di curva"
SOGLIA_STERZO_CURVA     = 0.15   # |sterzo| > 0.15 = manovra attiva
SOGLIA_STERZO_RECUPERO  = 0.35   # |sterzo| > 0.35 = recupero / emergenza

# Moltiplicatore oversampling per le manovre di recupero
# (quante volte duplicare i campioni con sterzo > SOGLIA_STERZO_RECUPERO)
OVERSAMPLE_RECUPERO = 3

#Per i 19 sensori di distanza possiamo anche aggiungere rumore per le variazioni
JITTER_SENSORI_STD = 0.015 #circa 3m se normalizziamo con /200

# EarlyStopping: quante epoche aspettare senza miglioramento
PATIENCE_EARLY_STOP = 25

#Pesi da dare ai dati del trianing 
PESO_CAMPIONE_CURVA = 2.0
PESO_CAMPIONE_RECUPERO = 3.5


# ReduceLROnPlateau
PATIENCE_LR_REDUCE  = 10
FATTORE_LR_REDUCE   = 0.5
LR_MINIMO           = 1e-6

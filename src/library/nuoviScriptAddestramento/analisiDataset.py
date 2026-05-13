from config import ( SOGLIA_STERZO_CURVA, SOGLIA_STERZO_RECUPERO,  
                    OVERSAMPLE_RECUPERO, JITTER_SENSORI_STD, PESO_CAMPIONE_CURVA, PESO_CAMPIONE_RECUPERO )
import numpy as np

def analizza_dataset(X: np.ndarray, Y: np.ndarray) -> None:
    """
    Stampa statistiche sulla distribuzione del dataset prima del training.
    Fondamentale per capire se il bilanciamento è adeguato.
    """
    sterzo = Y[:, 0]
    accel  = Y[:, 1]
    freno  = Y[:, 2]
    n      = len(sterzo) #lunghezza del dataset

    print("\n" + "=" * 55)
    print("  ANALISI DATASET")
    print("=" * 55)
    print(f"  Campioni totali  : {n:,}")
    print(f"  Feature sensori  : {X.shape[1]}")

    print("\n  STERZO")
    print(f"    Media          : {sterzo.mean():+.4f}  (0 = ideale, simmetrico)")
    print(f"    Std dev        : {sterzo.std():.4f}")
    print(f"    Min / Max      : {sterzo.min():+.4f} / {sterzo.max():+.4f}")
    
    #Verificihiamo quali campioni sono in rettilineo, quali in curva
    rett  = np.sum(np.abs(sterzo) <= SOGLIA_STERZO_CURVA)
    curva = np.sum((np.abs(sterzo) > SOGLIA_STERZO_CURVA) &
                   (np.abs(sterzo) <= SOGLIA_STERZO_RECUPERO))
    recup = np.sum(np.abs(sterzo) > SOGLIA_STERZO_RECUPERO)
    
    #Stampiamo il risultato di questa dicisione
    print(f"\n  DISTRIBUZIONE STERZO")
   #Rettilineo
    print(f"    Rettilineo   (|s|≤{SOGLIA_STERZO_CURVA})  : "
          f"{rett:,}  ({100*rett/n:.1f}%)")
    #Curva normale
    print(f"    Curva normale({SOGLIA_STERZO_CURVA}<|s|≤{SOGLIA_STERZO_RECUPERO}): "
          f"{curva:,}  ({100*curva/n:.1f}%)")
    #Recupero
    print(f"    Recupero     (|s|>{SOGLIA_STERZO_RECUPERO})  : "
          f"{recup:,}  ({100*recup/n:.1f}%)")

    # Avvisi su problemi comuni
    if sterzo.mean() > 0.05:
        print("\n  ⚠️  ATTENZIONE: sterzo medio spostato a destra. "
              "Il pilota potrebbe avere un bias. Considera di specchiare il dataset.")
    if recup / n < 0.08:
        print(f"\n  ⚠️  I campioni di recupero sono solo il {100*recup/n:.1f}% "
              f"del totale. L'oversampling x{OVERSAMPLE_RECUPERO} è consigliato.")
    if n < 10000:
        print(f"\n  ⚠️  Dataset piccolo ({n} campioni). ")
    if accel.mean() < 0.35:
        print(" ⚠️  Accelerazione media bassa: troppo prudente")
    if rett.sum() / n > 0.60:
        print(f" ⚠️  Rettilinei sono {100*rett.sum()/n} : troppi")


    #Stampiamo anche i dati sull'accelerazione
    print("\n  ACCELERAZIONE")
    print(f"    Media          : {accel.mean():.4f}")
    print(f"    % a piena accel: {100*np.sum(accel > 0.9)/n:.1f}%")
    print(f"    % in frenata   : {100*np.sum(freno > 0.1)/n:.1f}%")
    print("═" * 55 + "\n")

def specchia_dataset(X:np.ndarray, Y:np.ndarray):
    """
    Duplica il dataset specchiando i valori di sterzo trackpos
    angolo e sensori. Serve a ridurre il bias del dataset
    """

    #copiamo dapprima gli array
    X_spec = X.copy() 
    Y_spec = Y.copy()

    #Invertiamo ora i sensori
    X_spec[:, :19] = X[:, 18::-1] #track
    X_spec[: , 20] = -X[:,20] #angle
    X_spec[:,21]=-X[:,21] #trackpos

    #Invertiamo lo sterzo
    Y_spec[:,0] = -Y[:,0]

    #non invertiamo naturalmente i pedali
    #Impiliamo nello stack tutti i dati
    X_out = np.vstack([X,X_spec])
    Y_out = np.vstack([Y,Y_spec])
    
    #Eseguiamo uno shuffle dei dati: idx contiene tutti gli indici permutati
    idx = np.random.permutation(len(X_out))
    print(f" Dataset specchiato: {len(X):,} -> {len(X_out):,} campioni. "
          f"Bias sterzo: {Y[:, 0].mean():+.4f} -> {Y_out[:, 0].mean():+.4f}")
    return X_out[idx], Y_out[idx]



def bilancia_con_jitter(X: np.ndarray, Y: np.ndarray):
    """
    Oversampling dei campioni di recupero con jitter gaussiano sui sensori.
 
    Il jitter è applicato solo ai 19 sensori track (colonne 0:19),
    non ad angle, trackPos, rpm né alle azioni target.
    Rumore std=0.015 ≈ 3m su scala reale (sensori normalizzati /200).
    """

    #calcoliamo come sempre il vettore maschera che mi indica chi elemento supera quella soglia
    mask_rec = np.abs(Y[:, 0]) > SOGLIA_STERZO_RECUPERO
    #conta quante volte è stato superato
    n_orig   = mask_rec.sum()
 
    if n_orig == 0:
        print(" Nessun campione di recupero. Oversampling saltato.")
        return X, Y
 
    X_rec_base = X[mask_rec]
    Y_rec_base = Y[mask_rec]
    X_extra, Y_extra = [], []
 
    for _ in range(OVERSAMPLE_RECUPERO - 1):
        X_copy = X_rec_base.copy()
        # Jitter solo sui sensori track (prime 19 colonne)
        #creiamo un rumore "bianco" di dimensione pari a quello dei sensori
        rumore = np.random.normal(0, JITTER_SENSORI_STD, X_copy[:, :19].shape)
        X_copy[:, :19] = np.clip(X_copy[:, :19] + rumore, 0.0, 1.0)
        X_extra.append(X_copy)
        Y_extra.append(Y_rec_base.copy())
    
    #come prima impiliamo i valori e restituiamoloi
    X_out = np.vstack([X] + X_extra)
    Y_out = np.vstack([Y] + Y_extra)
    idx   = np.random.permutation(len(X_out))
 
    n_agg = len(X_out) - len(X)
    print(f"  ✅ Oversampling con jitter: +{n_agg:,} campioni recupero "
          f"(da {n_orig:,} × {OVERSAMPLE_RECUPERO - 1}). "
          f"Totale: {len(X_out):,}")
    return X_out[idx], Y_out[idx]


def calcola_sample_weights(Y_sterzo: np.ndarray) -> np.ndarray:
    """
    Assegna un peso ad ogni campione del training set.
 
    Il fit() di Keras moltiplica la loss di ogni campione per il suo peso.
    Campioni con peso alto contano di più nell'aggiornamento del gradiente.
 
    Distribuzione pesi:
      Rettilineo  (|s| <= 0.15) -> peso 1.0   (baseline)
      Curva       (|s| <= 0.35) -> peso 2.0   (2 x più importante)
      Recupero    (|s| > 0.35) -> peso 3.5   (3.5 x più importante)

    """
    #andaimo a dare una dimensione allo sterzo
    sterzo  = Y_sterzo.flatten()

    #creiamo dapprima i pesi con tutti uno
    pesi    = np.ones(len(sterzo), dtype=np.float32)
      
    #creiamo la maschera dei diversi pesi
    mask_curva = ((np.abs(sterzo) > SOGLIA_STERZO_CURVA) &
                  (np.abs(sterzo) <= SOGLIA_STERZO_RECUPERO))
    mask_recup = np.abs(sterzo) > SOGLIA_STERZO_RECUPERO
    
    #dove è vero diamo diversi pesi al modello
    pesi[mask_curva] = PESO_CAMPIONE_CURVA
    pesi[mask_recup] = PESO_CAMPIONE_RECUPERO
    
    #ritorniamo il vettore di pesi
    print(f"  ⚖️  Sample weights: rettilineo=1.0, "
          f"curva={PESO_CAMPIONE_CURVA}, "
          f"recupero={PESO_CAMPIONE_RECUPERO}")
    return pesi
 

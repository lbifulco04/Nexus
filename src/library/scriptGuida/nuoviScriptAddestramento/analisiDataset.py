from config import (LOSS_WEIGHT_STERZO , LOSS_WEIGHT_PEDALI ,
                    SOGLIA_STERZO_CURVA, SOGLIA_STERZO_RECUPERO  ,  
                    OVERSAMPLE_RECUPERO)
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

    #Stampiamo anche i dati sull'accelerazione
    print("\n  ACCELERAZIONE")
    print(f"    Media          : {accel.mean():.4f}")
    print(f"    % a piena accel: {100*np.sum(accel > 0.9)/n:.1f}%")
    print(f"    % in frenata   : {100*np.sum(freno > 0.1)/n:.1f}%")
    print("═" * 55 + "\n")



def bilancia_dataset(X: np.ndarray,
                     Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Duplica i campioni con sterzata di recupero per compensare la loro
    sottorappresentazione nel dataset.

    Il modello DEVE saper recuperare in casi critici:
    senza oversampling, la rete li tratta come outlier e li ignora.
    Il risultato è un modello che va dritto anche quando dovrebbe sterzare forte.

    Ritorna (X_bilanciato, Y_bilanciato) con i campioni di recupero duplicati.
    """
    sterzo   = Y[:, 0]
    mask_rec = np.abs(sterzo) > SOGLIA_STERZO_RECUPERO

    n_originali = np.sum(mask_rec)
    if n_originali == 0:
        print("Nessun campione di recupero trovato. Oversampling saltato.")
        return X, Y

    #X[mask-rec] recupera prima tutti i campioni che hanno maschera vera
    #np.tile prende il vettore mascherato e lo ripete N volte per sovrapopolarlo
    X_rec = np.tile(X[mask_rec], (OVERSAMPLE_RECUPERO - 1, 1))
    Y_rec = np.tile(Y[mask_rec], (OVERSAMPLE_RECUPERO - 1, 1))

    #fatto questo bisogna impilare le cose nuove con oversampling al vecchio dataset
    X_out = np.vstack([X, X_rec])
    Y_out = np.vstack([Y, Y_rec])

    # PASSAGGIO FONDAMENTALE : Shuffle finale per mescolare i duplicati con i campioni originali
    #idx è un vettore di indici permutati di lunghezza pari a tutto il dataset + oversampling
    idx    = np.random.permutation(len(X_out))
    #in questo modo possiamo mescolare i vettori
    X_out  = X_out[idx]
    Y_out  = Y_out[idx]

    n_aggiunti = len(X_rec)

    #stampa di debug
    print(f"  ✅ Oversampling: aggiunti {n_aggiunti:,} campioni di recupero "
          f"(da {n_originali:,} originali × {OVERSAMPLE_RECUPERO - 1}). "
          f"Totale: {len(X_out):,}")
    return X_out, Y_out

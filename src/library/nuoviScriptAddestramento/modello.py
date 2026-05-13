import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input, callbacks
import numpy as np
import os

from config import (SOGLIA_STERZO_CURVA, SOGLIA_STERZO_RECUPERO)

def costruisci_modello() -> keras.Model:
    """
    Costruisce il modello attraverso il modulo di tensorflow

    Struttura:
        Input(23)
           Dense(128) + BatchNorm + Dropout(0.15)     estrazione feature
           Dense(64)  + BatchNorm + Dropout(0.10)    raffinamento
           Dense(32)  + BatchNorm + Dropout(0.05)    compressione
           Dense(32)  + BatchNorm                    separazione uscite
                                 
      uscita_sterzo(1, tanh)     uscita_pedali(2, sigmoid)

    Perché BatchNormalization:
      Normalizza le uscite di ogni layer in valori con media 0 e varianza 1. 
      
    Perchè Dropout:
    Permette durante l'addestramento di spegnere [(n)*100]% neuroni affinchè il modello
    possa trovare anche altre soluzioni

    Perché due rami separati con pesi di loss diversi:
      Lo sterzo su Corkscrew è più critico dei pedali. 
    """

    inp = Input(shape=(23,), name="Ingresso_Sensori")

    # Base Comune
    x = layers.Dense(128, activation='relu')(inp) #relu trasforma i valori negativi in 0
    #il primo strato presenta ben 128 neuroni che prendono in input tutti i sensori

    #Durante l'allenamento, "spegne" casualmente il 20% dei neuroni in ogni ciclo
    #permettendo di evitare casi di overfittinig, e dunque la rete trova nuovi modi per risolvere il problema
    x = layers.Dropout(0.2)(x)

    #inseriamo poi due layers con rispettivamente 64 e 32 
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dense(32, activation='relu')(x)

    # Uscite separate
    out_sterzo = layers.Dense(1, activation='tanh', name="uscita_sterzo")(x)
    out_pedali = layers.Dense(2, activation='sigmoid', name="uscita_pedali")(x)

    #il modello è in grado di dare dati 19 input , 3 output che sono sterzo, freno e pedali
    modello = models.Model(inputs=inp, outputs=[out_sterzo, out_pedali])


    return models.Model(inputs=inp, outputs=[out_sterzo, out_pedali])



def salva_grafici(history: keras.callbacks.History,
                  cartella: str) -> None:
    """
    Salva due grafici nella cartella del modello:
      1. Curva loss training/validation per entrambi i rami
      2. Curva MAE sterzo training/validation

    Utili per diagnosticare overfitting (val_loss risale mentre train_loss scende).
    """
    nome_plot = input("Inserisci nome del grafico :")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Grafico 1: loss totale
    axes[0].plot(history.history['loss'],     label='Train loss')
    axes[0].plot(history.history['val_loss'], label='Val loss')
    axes[0].set_title('Loss totale')
    axes[0].set_xlabel('Epoca')
    axes[0].set_ylabel('MSE')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Grafico 2: MAE sterzo
    chiave_mae   = 'uscita_sterzo_mae'
    chiave_v_mae = 'val_uscita_sterzo_mae'
    if chiave_mae in history.history:
        axes[1].plot(history.history[chiave_mae],   label='Train MAE sterzo')
        axes[1].plot(history.history[chiave_v_mae], label='Val MAE sterzo')
        axes[1].set_title('MAE Sterzo')
        axes[1].set_xlabel('Epoca')
        axes[1].set_ylabel('MAE')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(cartella, nome_plot)
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"  📈 Grafici salvati → '{path}'")


def valuta_per_categoria(modello: keras.Model,
                          X_val: np.ndarray,
                          Y_val: np.ndarray) -> None:
    """
    Calcola il MAE dello sterzo separatamente per rettilineo, curva e recupero.

    Questo è molto più informativo del MAE globale: un modello può avere
    MAE=0.02 globale ma MAE=0.40 sui recuperi (che sono il 5% dei dati
    e quindi non influenzano la metrica media ma causano schianti).
    """
    pred_sterzo, _ = modello.predict(X_val, verbose=0)
    pred_sterzo    = pred_sterzo.flatten() #rendiamolo monodimensionale (Tensorflow usa i tensori )
    vero_sterzo    = Y_val[:, 0]

    #calcoliamo il MAE
    errore = np.abs(pred_sterzo - vero_sterzo)
    
    #Dividiamoli per categoria di sterzo (ci restiuisce l'array in cui la condizione è vera)
    mask_rett  = np.abs(vero_sterzo) <= SOGLIA_STERZO_CURVA
    mask_curva = ((np.abs(vero_sterzo) > SOGLIA_STERZO_CURVA) &
                  (np.abs(vero_sterzo) <= SOGLIA_STERZO_RECUPERO))
    mask_recup = np.abs(vero_sterzo) > SOGLIA_STERZO_RECUPERO

    print("\n" + "=" * 55)
    print("  VALUTAZIONE PER CATEGORIA (MAE sterzo)")
    print("=" * 55)

    #Semplice ciclo for per vedere gli errori 
    for nome, mask in [("Rettilineo  ", mask_rett),
                        ("Curva       ", mask_curva),
                        ("Recupero    ", mask_recup)]:
        if mask.sum() > 0:
            mae_cat = errore[mask].mean()
            n_cat   = mask.sum()
            #mostriamo una barra visiva di errore commesso
            barra   = "*" * int(mae_cat * 100)
            print(f"  {nome}: MAE={mae_cat:.4f}  n={n_cat:5d}  {barra}")

    print(f"\n  MAE globale sterzo: {errore.mean():.4f}")

    # Soglia di allerta
    if mask_recup.sum() > 0 and errore[mask_recup].mean() > 0.20:
        print("\n  ⚠️  MAE recupero > 0.20: il modello fa fatica con le manovre "
              "di emergenza. Considera più dati di recupero o oversampling maggiore.")
    print("=" * 55 + "\n")


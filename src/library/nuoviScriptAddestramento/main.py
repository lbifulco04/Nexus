"""
Imitation Learning per TORCS Corkscrew

SALVATAGGIO
  - Salva sia il formato .h5 (compatibilità TORCS) che SavedModel (TF2)
  - Salva anche i parametri di normalizzazione usati, per garantire
    coerenza tra training e inferenza
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input, callbacks
from analisiDataset import analizza_dataset, specchia_dataset, bilancia_con_jitter, calcola_sample_weights
from modello import costruisci_modello, salva_grafici,valuta_per_categoria
from normalizzatore import normalizza_dataset

from config import (EPOCHE_MAX, BATCH_SIZE,
                    LEARNING_RATE ,TEST_SIZE ,RANDOM_STATE , LOSS_WEIGHT_STERZO,LOSS_WEIGHT_PEDALI,
                    PATIENCE_EARLY_STOP,FATTORE_LR_REDUCE,OVERSAMPLE_RECUPERO,
                    PATIENCE_LR_REDUCE, LR_MINIMO, JITTER_SENSORI_STD,
                    PESO_CAMPIONE_CURVA,PESO_CAMPIONE_RECUPERO)


def main():
    print("🏎️  Imitation Learning per TORCS Corkscrew")
    print("=" * 55)

    # ── 1. CARICAMENTO 
    path = input("\nPercorso del dataset: ").strip()
    if not os.path.exists(path):
        print(f"❌  File non trovato: {path}")
        sys.exit(1) #usciamo dallo script nel caso non esista 

    print("\n📂  Caricamento e normalizzazione dati...")
    X, Y_tutto = normalizza_dataset(path)
    

    # ── 2. ANALISI DEL DATASET FORNITO
    bias = abs(Y_tutto[:,0].mean())
    if bias > 0.05:
        risposta = input(f"Sterzo medio = {bias}. Specchiare il dataset per correggere? [s/N]:")
        if risposta == 's':
            X,Y_tutto = specchia_dataset(X, Y_tutto)

    

    # ── 3. SPLIT 
    #ricaviamo sterzo e pedali originali del dataset
    Y_sterzo_orig = Y_tutto[:, 0:1]
    Y_pedali_orig = Y_tutto[:, 1:3]

    #per il train test split utilizziamo i valori originali del dataset
    (X_train_raw, X_val,
     y_st_train_raw, y_st_val,
     y_pe_train_raw, y_pe_val) = train_test_split(
        X, Y_sterzo_orig, Y_pedali_orig,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True
    )
    
    # 4. Bilancia solo il training set
    #impiliamo le cose splittate con train-test
    Y_train_raw = np.hstack([y_st_train_raw, y_pe_train_raw])
    #bilanciamo il dataset di training
    X_train, Y_train_bil = bilancia_con_jitter(X_train_raw, Y_train_raw)
    #ricaviamo i valori di uscita bilanciati di sterzo e pedali
    y_st_train = Y_train_bil[:, 0:1]
    y_pe_train = Y_train_bil[:, 1:3]

    print(f"\n  Train: {len(X_train):,} campioni  |  "
          f"Val: {len(X_val):,} campioni")
    
    # 6. Ora calcoliamo i valori dei pesi del nostro training set
    sample_peso = calcola_sample_weights(y_st_train)

    # 5. MODELLO 
    print("\n🧠  Costruzione modello...")
    #costruiamo il modello con la nostra libreria custom
    modello = costruisci_modello()

    modello.summary()

    #compiliamo il modello 
    modello.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=['mse', 'mse'],
        loss_weights=[LOSS_WEIGHT_STERZO, LOSS_WEIGHT_PEDALI],
        metrics=['mae', 'mae']
    )

    # 6. CALLBACKS : Tensorflow implementa anche algoritmi asincroni per richiamare il modello tramite eventi
    
    # EarlyStopping: ferma il training quando val_loss smette di migliorare
    # restore_best_weights=True ritorna ai pesi dell'epoca migliore
    cb_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE_EARLY_STOP,
        restore_best_weights=True,
        verbose=1
    )

    # ReduceLROnPlateau: dimezza il lr se val_loss è piatta per N epoche
    # Evita di "girare in tondo" intorno a un minimo locale
    cb_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=FATTORE_LR_REDUCE, #adatta il LR per fare passi più piccoli per imparare
        patience=PATIENCE_LR_REDUCE, #quante epoche aspettare se val_loss è sempre stabile
        min_lr=LR_MINIMO, #minimo lr da raggiungere
        verbose=1
    )

    # ── 7. TRAINING
    print(f"\n🏋️  Training (max {EPOCHE_MAX} epoche, EarlyStopping patience={PATIENCE_EARLY_STOP})...")
    history = modello.fit(
        X_train,
        # Passiamo i target come lista
        [y_st_train, y_pe_train],
        validation_data=(
            X_val, 
            [y_st_val, y_pe_val]
        ), 
        epochs=EPOCHE_MAX,
        batch_size=BATCH_SIZE, 
        callbacks=[cb_stop, cb_lr], 
        verbose=1,
        # Passiamo i pesi come lista
        sample_weight=[sample_peso, sample_peso] 
    )

    epoca_migliore = np.argmin(history.history['val_loss']) + 1
    print(f"\n✅  Training completato. Miglior epoca: {epoca_migliore}")

    # ── 8. VALUTAZIONE 
    Y_val_completo = np.hstack([y_st_val, y_pe_val])
    valuta_per_categoria(modello, X_val, Y_val_completo)

    # ── 9. SALVATAGGIO ─
    nome     = input("💾  Nome del modello (senza estensione): ").strip()
    cartella = os.path.dirname(os.path.abspath(path))

    # Formato .h5 per compatibilità con TORCS e snakeoil3
    path_h5 = os.path.join(cartella, f"{nome}.h5")
    modello.save(path_h5)
    print(f"  ✅ Salvato → '{path_h5}'")

    # Salva i grafici di training nella stessa cartella
    salva_grafici(history, cartella)

    # Salva i metadati del training (utili per confrontare esperimenti)
    meta = {
        "nome_modello":       nome,
        "campioni_train":     int(len(X_train)),
        "campioni_val":       int(len(X_val)),
        "epoca_migliore":     int(epoca_migliore),
        "val_loss_finale":    float(min(history.history['val_loss'])),
        "learning_rate_init": LEARNING_RATE,
        "oversample_recupero": OVERSAMPLE_RECUPERO,
        "loss_weight_sterzo": LOSS_WEIGHT_STERZO,
        "jitter_std": JITTER_SENSORI_STD,
        "peso_curva" : PESO_CAMPIONE_CURVA,
        "peso_recupero" : PESO_CAMPIONE_RECUPERO
    }
    path_meta = os.path.join(cartella, f"{nome}_meta.json")
    with open(path_meta, 'w') as f:
        json.dump(meta, f, indent=2)
    print(f"  📋 Metadati salvati → '{path_meta}'")

    print(f"\n🏁  Modello '{nome}.h5' pronto per Corkscrew!")


if __name__ == "__main__":
    main()

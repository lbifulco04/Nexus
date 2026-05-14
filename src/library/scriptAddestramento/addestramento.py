import numpy as np
import os
import json
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, Input, callbacks

tf.config.set_visible_devices([], 'GPU')

# ──────────────────────────────
def normalizza_dataset(percorso_file):
    ingressi_sensori = []
    uscite_azioni = []
    with open(percorso_file, 'r') as f:
        for i, riga in enumerate(f):
            riga = riga.strip()
            if not riga:
                continue
            try:
                dati = json.loads(riga)
                s = dati["sensors"]
                a = dati["actions"]
                pista           = np.array(s["track"]) / 200.0
                velocita        = np.array([s["speedX"] / 300.0])
                angolo          = np.array([s["angle"] / np.pi])
                posizione_pista = np.array([s["trackPos"]])
                giri_motore     = np.array([s["rpm"] / 15000.0])
                vettore_stato   = np.concatenate([pista, velocita, angolo, posizione_pista, giri_motore])
                vettore_azione  = [a["steer"], a["accel"], a["brake"], int(a["gear"])]
                ingressi_sensori.append(vettore_stato)
                uscite_azioni.append(vettore_azione)
            except KeyError as e:
                print(f"⚠️  Riga {i}: chiave mancante {e}")
            except Exception as e:
                print(f"❌  Riga {i}: {e}")
    return np.array(ingressi_sensori), np.array(uscite_azioni)

# ──────────────────────────────
SOGLIA_STERZO_RECUPERO = 0.35
OVERSAMPLE_RECUPERO    = 3
JITTER_SENSORI_STD     = 0.015

def specchia_dataset(X, Y):
    X_spec = X.copy();  Y_spec = Y.copy()
    X_spec[:, :19] = X[:, 18::-1]
    X_spec[:, 20]  = -X[:, 20]
    X_spec[:, 21]  = -X[:, 21]
    Y_spec[:, 0]   = -Y[:, 0]
    X_out = np.vstack([X, X_spec])
    Y_out = np.vstack([Y, Y_spec])
    idx   = np.random.permutation(len(X_out))
    print(f"  Dataset specchiato: {len(X)} → {len(X_out)} campioni")
    return X_out[idx], Y_out[idx]

def bilancia_con_jitter(X, Y):
    mask_rec = np.abs(Y[:, 0]) > SOGLIA_STERZO_RECUPERO
    n_orig   = mask_rec.sum()
    if n_orig == 0:
        print("  Nessun campione di recupero. Oversampling saltato.")
        return X, Y
    X_rec = X[mask_rec];  Y_rec = Y[mask_rec]
    X_extra, Y_extra = [], []
    for _ in range(OVERSAMPLE_RECUPERO - 1):
        X_copy = X_rec.copy()
        rumore = np.random.normal(0, JITTER_SENSORI_STD, X_copy[:, :19].shape)
        X_copy[:, :19] = np.clip(X_copy[:, :19] + rumore, 0.0, 1.0)
        X_extra.append(X_copy);  Y_extra.append(Y_rec.copy())
    X_out = np.vstack([X] + X_extra)
    Y_out = np.vstack([Y] + Y_extra)
    idx   = np.random.permutation(len(X_out))
    print(f"  Oversampling recuperi: +{len(X_out)-len(X)} campioni (da {n_orig})")
    return X_out[idx], Y_out[idx]

# ──────────────────────────────
def costruisci_modello(num_classi_gear):
    inp = Input(shape=(23,), name="Ingresso_Sensori")
    x   = layers.Dense(128, activation='relu')(inp)
    x   = layers.Dense(128, activation='relu')(x)
    out_sterzo = layers.Dense(1, activation='tanh',    name="uscita_sterzo")(x)
    out_pedali = layers.Dense(2, activation='sigmoid', name="uscita_pedali")(x)
    out_gear   = layers.Dense(num_classi_gear, activation='softmax', name="uscita_gear")(x)

    return models.Model(inputs=inp, outputs=[out_sterzo, out_pedali, out_gear])

# ──────────────────────────────
def crea_callbacks(nome_checkpoint):
    early = callbacks.EarlyStopping(
        monitor='val_loss', patience=25,
        restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=10, min_lr=1e-6, verbose=1
    )
    checkpoint = callbacks.ModelCheckpoint(
        f'{nome_checkpoint}_best.keras',
        monitor='val_loss', save_best_only=True, verbose=1
    )
    return [early, reduce_lr, checkpoint]

# ──────────────────────────────
def main():
    print("🏎️   Addestramento Imitation Learning — TORCS")
    print("=" * 55)

    path = input("\nPercorso del dataset: ").strip()
    if not os.path.exists(path):
        print(f"❌ File non trovato: {path}")
        return

    X, Y_tutto = normalizza_dataset(path)
    print(f"  Campioni caricati: {len(X)}")

    bias = abs(Y_tutto[:, 0].mean())
    if bias > 0.05:
        risp = input(f"  Sterzo medio = {bias:.3f}. Specchiare il dataset? [s/N]: ")
        if risp.lower() == 's':
            X, Y_tutto = specchia_dataset(X, Y_tutto)

    Y_sterzo     = Y_tutto[:, 0:1]
    Y_pedali     = Y_tutto[:, 1:3]
    Y_gear       = np.clip(Y_tutto[:, 3].astype(int), -1, 6)
    Y_gear_class = Y_gear + 1

    num_classi_gear = len(np.unique(Y_gear_class))
    print(f"  Classi gear: {num_classi_gear}  (valori: {sorted(np.unique(Y_gear_class))})")

    (X_train_raw, X_val,
     y_st_raw, y_st_val,
     y_pe_raw, y_pe_val,
     y_ge_raw, y_ge_val) = train_test_split(
        X, Y_sterzo, Y_pedali, Y_gear_class,
        test_size=0.2, random_state=42, shuffle=True
    )
    y_ge_val = np.clip(y_ge_val.astype(int), 0, num_classi_gear - 1)

    Y_train_raw = np.hstack([y_st_raw, y_pe_raw, y_ge_raw.reshape(-1, 1)])
    X_train, Y_bil = bilancia_con_jitter(X_train_raw, Y_train_raw)

    y_st_train = Y_bil[:, 0:1]
    y_pe_train = Y_bil[:, 1:3]
    y_ge_train = np.clip(Y_bil[:, 3].astype(int), 0, num_classi_gear - 1)

    print(f"  Train: {len(X_train)} | Val: {len(X_val)}")

    print("\n🧠 Costruzione modello...")
    modello = costruisci_modello(num_classi_gear)
    modello.summary()

    modello.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss={
            'uscita_sterzo': 'mse',
            'uscita_pedali': 'mse',
            'uscita_gear':   'sparse_categorical_crossentropy'
        },
        loss_weights={
            'uscita_sterzo': 2.0,
            'uscita_pedali': 0.5,
            'uscita_gear':   0.3
        },
        metrics={
            'uscita_sterzo': ['mae'],
            'uscita_pedali': ['mae'],
            'uscita_gear':   ['accuracy']
        }
    )

    nome = input("\n💾 Nome del modello (usato anche per il checkpoint): ").strip() or "modello_torcs"
    cb_list = crea_callbacks(nome)

    print(f"\n🏋️   Training (max 300 epoche)...")
    history = modello.fit(
        X_train,
        {
            'uscita_sterzo': y_st_train,
            'uscita_pedali': y_pe_train,
            'uscita_gear':   y_ge_train
        },
        validation_data=(
            X_val,
            {
                'uscita_sterzo': y_st_val,
                'uscita_pedali': y_pe_val,
                'uscita_gear':   y_ge_val
            }
        ),
        epochs=300,
        batch_size=128,
        callbacks=cb_list,
        verbose=1
    )

    miglior_epoca = np.argmin(history.history['val_loss']) + 1
    val_loss_min  = min(history.history['val_loss'])
    print(f"\n✅ Miglior epoca: {miglior_epoca}  |  val_loss: {val_loss_min:.5f}")

    modello.save(f"{nome}.keras")
    print(f"✅ Salvato '{nome}.keras'  (checkpoint: '{nome}_best.keras')")

if __name__ == "__main__":
    main()
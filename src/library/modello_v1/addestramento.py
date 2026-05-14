import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models, Input
from normalizzatore import normalizza_dataset 

# 1. CARICAMENTO DATI
# La  funzione restituisce X (sensori) e Y (tutte le azioni insieme)
path = input("fornisci il percorso del dataset: ")
X, Y_tutto = normalizza_dataset(path)

# 2. SEPARAZIONE DELLE AZIONI (Slicing)
# Y_tutto ha 3 colonne: [0] sterzo, [1] accel, [2] brake
Y_sterzo = Y_tutto[:, 0:1]  # Prendiamo solo la colonna dello sterzo (importante per la guida)
Y_pedali = Y_tutto[:, 1:3]  # Prendiamo le colonne 1 e 2 (accel e brake)

# 3. DIVISIONE TRAINING / VALIDATION
X_train, X_val, y_st_train, y_st_val, y_pe_train, y_pe_val = train_test_split(
    X, Y_sterzo, Y_pedali, test_size=0.2, random_state=42, shuffle=True
)
#notare random_state fissato per la ripetibilità dell'esperimento

# 4. ARCHITETTURA DEL MODELLO 
input_sensori = Input(shape=(23,), name="Ingresso_Sensori")

# Base Comune
x = layers.Dense(128, activation='relu')(input_sensori) #relu trasforma i valori negativi in 0
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
modello = models.Model(inputs=input_sensori, outputs=[out_sterzo, out_pedali])

tasso_apprendimento = 0.0001

# 5. COMPILAZIONE
modello.compile(
    # Creiamo l'oggetto Adam e gli passiamo il parametro 'learning_rate'
    optimizer=keras.optimizers.Adam(learning_rate=tasso_apprendimento),
    #adam è l'algoritmo di ottimizzazione utilizzato"
    
    loss={'uscita_sterzo': 'mse', 'uscita_pedali': 'mse'},
    #per la perdita si deve calcolare l'MSE
    loss_weights={'uscita_sterzo': 1.0, 'uscita_pedali': 0.5},
    #il peso dello sterzo è maggiore dei pedali già dall'inzio
    
    #Specifichiamo la metrica per ogni ramo usando un dizionario
    metrics={
        'uscita_sterzo': ['mae'],
        'uscita_pedali': ['mae']
    }
)

# 6. ADDESTRAMENTO
print("\n🏎️ Allenamento in corso con output differenziati...")
modello.fit(
    #valori dei sensori in ingresso del training set
    X_train, 

    #valori in uscita del training set
    {'uscita_sterzo': y_st_train, 'uscita_pedali': y_pe_train},
    
    #validazione dei dati sulla base del test set
    validation_data=(X_val, {'uscita_sterzo': y_st_val, 'uscita_pedali': y_pe_val}),

    #numeor di epoche (quante volte vedere il file in ingresso)
    epochs=100,

    batch_size=32,

    #vediamo visivamente l'apprendimento
    verbose=1
)

# 7. SALVATAGGIO
nome = input("\n💾 Nome del modello: ")
modello.save(f"{nome}.h5")
print(f"✅ Modello '{nome}.h5' pronto per la sfida IBM RACE!")
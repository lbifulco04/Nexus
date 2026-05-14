import json
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib

# 1. Caricamento Dati JSON Lines
def load_data(file_path):
    X = []
    y = []
    decoder = json.JSONDecoder()
    
    print("Lettura del file in corso...")
    with open(file_path, 'r') as f:
        # Leggiamo tutto il contenuto del file in una singola stringa
        content = f.read()

    pos = 0
    content_length = len(content)
    
    print("Decodifica dei campioni JSON...")
    while pos < content_length:
        # Saltiamo eventuali spazi bianchi o a capo tra un oggetto e l'altro
        while pos < content_length and content[pos].isspace():
            pos += 1
            
        if pos >= content_length:
            break
            
        try:
            # raw_decode estrae il primo JSON valido e restituisce i dati
            # e la nuova posizione (indice) da cui ripartire
            data, pos = decoder.raw_decode(content, pos)
            
            # Estrazione dei sensori nello stesso ordine
            track = data['sensors']['track']
            other_sensors = [
                data['sensors']['speedX'],
                data['sensors']['angle'],
                data['sensors']['trackPos'],
                data['sensors']['rpm'],
                data['sensors']['distFromStart']
            ]
            features = track + other_sensors # 19 + 5 = 24 features
            
            # Estrazione dei target
            targets = [
                data['actions']['accel'],
                data['actions']['brake'],
                data['actions']['steer']
            ]
            
            X.append(features)
            y.append(targets)
            
        except json.JSONDecodeError as e:
            print(f"Errore di formattazione imprevisto al carattere {pos}: {e}")
            break

    print(f"Caricati con successo {len(X)} campioni.")
    return np.array(X), np.array(y)


print("Caricamento dati...")
X_raw, y_raw = load_data('datafull_v1.json')

# 2. Preprocessing: StandardScaler e PCA
print("Normalizzazione e PCA...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# Manteniamo il 95% della varianza spiegata
pca = PCA(n_components=0.95) 
X_pca = pca.fit_transform(X_scaled)

print(f"Feature originali: {X_scaled.shape[1]} -> Feature post-PCA: {X_pca.shape[1]}")

# Salviamo gli oggetti per usarli nel Driver
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(pca, 'pca.pkl')

# Split del dataset
X_train, X_val, y_train, y_val = train_test_split(X_pca, y_raw, test_size=0.2, random_state=42)

# 3. Definizione Modello e Loss Personalizzata
# Creiamo una loss function dove lo sterzo (indice 2) pesa di più
def custom_weighted_mse(y_true, y_pred):
    # Calcolo dell'MSE per ogni componente
    mse_accel = tf.square(y_true[:, 0] - y_pred[:, 0])
    mse_brake = tf.square(y_true[:, 1] - y_pred[:, 1])
    mse_steer = tf.square(y_true[:, 2] - y_pred[:, 2])
    
    # Assegniamo un peso di 5.0 allo sterzo, 1.0 agli altri
    peso_steer = 5.0
    
    loss = tf.reduce_mean(mse_accel + mse_brake + (mse_steer * peso_steer))
    return loss

# Costruzione del modello
model = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_pca.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    # Output layer: 3 nodi (accel, brake, steer). Niente attivazione per regressione pura, 
    # o attivazioni specifiche se i range sono noti (es. tanh per sterzo tra -1 e 1)
    tf.keras.layers.Dense(3, activation='linear') 
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=custom_weighted_mse)

# 4. Callbacks e Fitting (Addestramento)
callbacks = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1),
    tf.keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)
]

print("Inizio Addestramento...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=128,
    callbacks=callbacks,
    verbose=1 # Sintassi corretta e standard di Keras
)
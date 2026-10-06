import sqlite3
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.callbacks import ModelCheckpoint, Callback

def process_mixed_columns(df):
    new_dfs = []
    drop_cols = []
    for col in df.columns:
        if df[col].dtype == 'object' and df[col].str.contains(r'\d', regex=True).any():
            text_col = df[col].str.extract(r'([a-zA-Z]+)')
            num_col = df[col].str.extract(r'(\d+\.?\d*)').astype(float)
            text_dummies = pd.get_dummies(text_col, prefix=f'{col}_text')
            new_dfs.append(text_dummies)
            new_dfs.append(pd.DataFrame(num_col, columns=[f'{col}_num']))
            drop_cols.append(col)
    df.drop(drop_cols, axis=1, inplace=True)
    df = pd.concat([df] + new_dfs, axis=1)
    return df

def positional_encoding(position, d_model):
    position = int(position)
    pos_encoding = np.zeros((1, position, d_model))
    for pos in range(position):
        for i in range(0, d_model, 2):
            angle = pos / np.power(10000, (2 * i)/np.float32(d_model))
            pos_encoding[0, pos, i] = np.sin(angle)
            pos_encoding[0, pos, i + 1] = np.cos(angle)
    return tf.constant(pos_encoding, dtype=tf.float32)

def create_sequences(X, y, sequence_length):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)

def build_transformer_model(input_dim, num_layers=4, num_heads=1, d_model=512, learning_rate=0.00001, sequence_length=4):  # Reduced learning_rate
    inputs = tf.keras.Input(shape=(sequence_length, input_dim))
    x = tf.keras.layers.Dense(d_model, kernel_initializer='lecun_normal')(inputs)  # Changed initializer
    x = tf.keras.layers.BatchNormalization()(x)  # Added BatchNormalization
    x = tf.keras.layers.LayerNormalization()(x)
    pos_enc = positional_encoding(sequence_length, d_model)
    x += pos_enc
    for _ in range(num_layers):
        qkv = tf.keras.layers.Dense(d_model * 3, kernel_initializer='lecun_normal')(x)
        qkv = tf.keras.layers.LayerNormalization()(qkv)
        query, key, value = tf.split(qkv, 3, axis=-1)
        attn_out = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)(query, key, value)
        x += attn_out
        x = tf.keras.layers.LayerNormalization()(x)
        ffnn_out = tf.keras.layers.Dense(d_model, activation='relu', kernel_initializer='lecun_normal')(x)
        x += ffnn_out
        x = tf.keras.layers.LayerNormalization()(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    outputs = tf.keras.layers.Dense(1, activation='linear', kernel_initializer='lecun_normal')(x)  # Specified activation='linear'
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipvalue=0.5)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


conn_train = sqlite3.connect('/home/thinkbe/Desktop/trading/data/streamline.db')
df_train = pd.read_sql_query("SELECT * from XAUUSD", conn_train)
conn_train.close()

conn_val = sqlite3.connect('/home/thinkbe/Desktop/trading/data/val_streamline.db')
df_val = pd.read_sql_query("SELECT * from XAUUSD", conn_val)
conn_val.close()

df_train['Target'] = df_train['Target'].astype(float)
df_val['Target'] = df_val['Target'].astype(float)

df_train_features = process_mixed_columns(df_train.drop('Target', axis=1))
df_val_features = process_mixed_columns(df_val.drop('Target', axis=1))

scaler = StandardScaler()
X_train = scaler.fit_transform(df_train_features)
X_val = scaler.transform(df_val_features)

y_train = df_train['Target']
y_val = df_val['Target']

sequence_length = 4
X_train_seq, y_train_seq = create_sequences(X_train, y_train, sequence_length)
X_val_seq, y_val_seq = create_sequences(X_val, y_val, sequence_length)

num_layers = 4
d_model = 1024
num_heads = 16
batch_size = 100
epochs = 500
learning_rate = 0.0001

train_dataset = tf.data.Dataset.from_tensor_slices((X_train_seq, y_train_seq)).batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val_seq, y_val_seq)).batch(batch_size)

input_dim = X_train_seq.shape[2]
model = build_transformer_model(input_dim, sequence_length=sequence_length)  # Added sequence_length

model.summary()

callbacks_list = [
    ModelCheckpoint('model_weights_{epoch}.h5', save_best_only=True, monitor='val_loss', mode='min')
]

history = model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, callbacks=callbacks_list)

plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_vs_val_loss.png')
plt.show()

y_pred = model.predict(X_val_seq).flatten()
mse = mean_squared_error(y_val_seq, y_pred)
print(f"Model Mean Squared Error on Validation Set: {mse}")

import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.layers import Dense, MultiHeadAttention, LayerNormalization

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

seq_len = 4
d_model = 2500
d_ffn = 1250
block_layers = 6
learning_rate = 0.00001
n_heads =  2 # multi-head attention
epochs = 200
batch_size = 128

# Mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Data Loading and Preprocessing Functions
def load_data_from_sqlite(db_path, table_name, columns=None):
    conn = sqlite3.connect(db_path)
    if columns:
        query = f"SELECT {','.join(columns)} FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_overlapping_sequences(data, seq_len):
    sequences = []
    for i in range(len(data)):
        seq = []
        idx = i
        for j in range(seq_len):
            if idx < 0:
                break
            seq.append(data[idx])
            idx -= 11  # Step back by 11 for each entry
        if len(seq) == seq_len:
            sequences.append(np.array(seq[::-1]))  # Reverse to keep the temporal order
    return np.array(sequences)

def preprocess_data(df, scaler=None, seq_len=seq_len):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")

    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values.astype('float32'))
    else:
        scaled = scaler.transform(df.values.astype('float32'))

    X = create_overlapping_sequences(scaled, seq_len)
    y = target.values[len(target) - len(X):]  # Align the lengths

    return X, y, scaler

# Before loading data
print("About to load training data...")

# Load data
columns_to_load = None
df_train = load_data_from_sqlite('/home/thinkbe/Desktop/trading/data/streamline_lag_close_(backup).db', 'XAUUSD', columns=columns_to_load)
df_val = load_data_from_sqlite('/home/thinkbe/Desktop/trading/data/val_streamline_lag_close_(backup).db', 'XAUUSD', columns=columns_to_load)

# After loading data
print("Finished loading training data...")

print("Starting to preprocess training data...")
X_train, y_train, scaler = preprocess_data(df_train, seq_len=seq_len)
print("Finished preprocessing training data...")

print("Starting to preprocess validation data...")
X_val, y_val, _ = preprocess_data(df_val, scaler, seq_len=seq_len)
print("Finished preprocessing validation data...")

# Real-time Dashboard
class RealTimeDashboard(Callback):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Real-time Dashboard')
        self.fig, self.ax = plt.subplots(1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.losses = []
        self.val_losses = []
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])
        self.epoch_count += 1

        self.ax.clear()
        self.ax.plot(range(self.epoch_count), self.losses, label='loss')
        self.ax.plot(range(self.epoch_count), self.val_losses, label='val_loss')
        self.ax.legend()
        self.ax.set_title('Epoch vs Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        
        self.canvas.draw()
        self.root.update()

# Create RealTimeDashboard callback instance
dashboard = RealTimeDashboard()

class FeatureAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, **kwargs):
        super(FeatureAttention, self).__init__(**kwargs)
        self.d_model = d_model
        self.dense = Dense(d_model, activation='tanh', kernel_initializer='glorot_normal')
        self.att_weights = Dense(d_model, activation='softmax', kernel_initializer='glorot_normal')

    def call(self, inputs):
        x = self.dense(inputs)
        att = self.att_weights(x)
        return inputs * att

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TemporalFusion(tf.keras.layers.Layer):
    def __init__(self, d_model, num_layers, **kwargs):
        super(TemporalFusion, self).__init__(**kwargs)
        self.d_model = d_model
        self.num_layers = num_layers
        self.alpha = self.add_weight(shape=(d_model, num_layers), 
                                     initializer='glorot_uniform', 
                                     trainable=True)

    def call(self, stack_of_layers):
        alpha_expanded = tf.expand_dims(tf.expand_dims(self.alpha, 0), 0)
        return tf.reduce_sum(stack_of_layers * alpha_expanded, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'num_layers': self.num_layers})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn, n_heads, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.n_heads = n_heads
        self.att = MultiHeadAttention(num_heads=n_heads, key_dim=d_model)
        self.ffn = tf.keras.Sequential(
            [Dense(d_ffn, activation='tanh', kernel_initializer='he_normal'), 
             Dense(d_model, kernel_initializer='he_normal')]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.feature_attention = FeatureAttention(d_model)

    def call(self, inputs):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return self.feature_attention(out2)

    def get_config(self):
        config = super().get_config()
        config.update({'d_model': self.d_model, 'd_ffn': self.d_ffn, 'n_heads': self.n_heads})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Define the input shape
inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

# Project the input embeddings to match d_model
projected_inputs = tf.keras.layers.Dense(d_model)(inputs)

x = projected_inputs  # Initialize x before entering the loop

outputs_of_blocks = []
for _ in range(block_layers):
    x = TransformerBlock(d_model, d_ffn, n_heads)(x)
    outputs_of_blocks.append(tf.expand_dims(x, axis=-1))

stacked_outputs = tf.concat(outputs_of_blocks, axis=-1)  # shape: [batch_size, seq_len, d_model, block_layers]
x = TemporalFusion(d_model, block_layers)(stacked_outputs)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = Dense(1, activation='linear', dtype='float32')(x)
model = tf.keras.Model(inputs, outputs)

# Print the model summary
model.summary()
print("Total number of parameters in the model:", model.count_params()) 

# Define the model checkpoint for weights only
weights_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/weights/transformer_weights_0.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/saved_models/transformer_weights_0.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

# Try loading the full model, then the weights, or use the freshly defined model if neither exists
if os.path.exists(full_model_checkpoint_path):
    print("Loading full model...")
    model = tf.keras.models.load_model(full_model_checkpoint_path, custom_objects={'FeatureAttention': FeatureAttention, 'TemporalFusion': TemporalFusion, 'TransformerBlock': TransformerBlock})
elif os.path.exists(weights_checkpoint_path):
    print("Loading model weights...")
    model.load_weights(weights_checkpoint_path)
else:
    print("Using freshly defined model...")

# Compile and fit the model
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Evaluate model immediately after loading weights (for baseline performance)
val_loss = model.evaluate(X_val, y_val)
print(f"Baseline Mean Absolute Error on Validation Set after Loading: {val_loss}")


print(f"Training on {X_train.shape[0]} examples, validating on {X_val.shape[0]} examples.")
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[weights_checkpoint, full_model_checkpoint, dashboard])

epochs = 0 # You can set this to 0 or any other value
batch_size = 1
if epochs > 0:
    print(f"Training on {X_train.shape[0]} examples, validating on {X_val.shape[0]} examples.")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[weights_checkpoint, full_model_checkpoint, dashboard])

    # Close the Tkinter window
    dashboard.root.destroy()
    print("Training complete.")

    # Plotting the loss and validation loss
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_vs_val_loss.png')
    plt.show()

# Model evaluation for both MSE and MAE
y_pred = model.predict(X_val).flatten()
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f"Model Mean Squared Error on Validation Set: {mse}")
print(f"Model Mean Absolute Error on Validation Set: {mae}")

# Obtain predictions on validation data
predictions = model.predict(X_val)

# Calculate MSE and MAE
mse = np.mean((y_val - predictions.flatten())**2)
mae = np.mean(np.abs(y_val - predictions.flatten()))

print(f"Mean Squared Error on validation data: {mse}")
print(f"Mean Absolute Error on validation data: {mae}")

# Save the actual and predicted values to a DataFrame
result_df = pd.DataFrame({"Actual": y_val, "Predicted": predictions.flatten()})

# Save result DataFrame to a CSV file
file_path = 'transformer_predictions.csv'
print("Saving file to:", file_path)
result_df.to_csv(file_path, index=False)
print("File saved successfully!")



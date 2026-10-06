import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Lambda, GlobalAveragePooling1D
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
# from tensorflow.keras.losses import MeanSquaredLogarithmicError
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

seq_len = 4
d_model = 1053
d_ffn = 3159
block_layers = 12
learning_rate = 0.00001
epochs = 500
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
        scaler = MinMaxScaler()  # Changed from MinMaxScaler to RobustScaler
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

print("About to initialize the model...")

# glorot_normal

# Define the model
class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, d_ffn, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.spatial_projection = Dense(d_ffn, activation='tanh', use_bias=True, kernel_initializer='glorot_normal')
        
        # Temporal convolution layer for temporal context
        # self.temporal_conv = tf.keras.layers.Conv1D(filters=d_ffn, kernel_size=kernel_size, padding='same', activation='relu', kernel_initializer='glorot_normal')
        
         # Forget and input gates
        self.forget_gate = Dense(d_ffn, activation='sigmoid', kernel_initializer='glorot_normal')  # Adjusted initializer
        self.input_gate = Dense(d_ffn, activation='sigmoid', kernel_initializer='glorot_normal')  # Adjusted initializer

    def build(self, input_shape):
        self.spatial_gating = self.add_weight(
            shape=(input_shape[-1],),
            initializer='he_normal',
            trainable=True
        )

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        f_gate = self.forget_gate(normalized_inputs)
        i_gate = self.input_gate(normalized_inputs)
        gated_temporal = f_gate * normalized_inputs + i_gate * self.spatial_projection(normalized_inputs)
        return gated_temporal * tf.sigmoid(self.spatial_gating) + inputs


    def get_config(self):
        config = super(SpatialGatingUnit, self).get_config()
        config.update({"d_ffn": self.d_ffn})
        return config

class gMLPBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn, **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ffn = d_ffn
        self.channel_projection_i = Dense(d_ffn, activation='tanh', kernel_initializer='glorot_normal')
        self.sgu = SpatialGatingUnit(d_ffn)
        self.channel_projection_ii = Dense(d_model, activation='tanh', kernel_initializer='glorot_normal')  # Added activation

    def call(self, inputs):
        x = self.channel_projection_i(inputs)
        x = self.sgu(x)
        x = self.channel_projection_ii(x)
        return x + inputs

    def get_config(self):
        config = super(gMLPBlock, self).get_config()
        config.update({"d_model": self.d_model, "d_ffn": self.d_ffn})
        return config

print("Model initialized!")

# Define the input shape
inputs = tf.keras.layers.Input(shape=(X_train.shape[1], X_train.shape[2]))

# Project the input embeddings to match d_model
projected_inputs = tf.keras.layers.Dense(d_model)(inputs)

x = projected_inputs  # Initialize x before entering the loop

for _ in range(block_layers):
    x = gMLPBlock(d_model, d_ffn)(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = Dense(1, activation='linear', dtype='float32')(x)
model = tf.keras.Model(inputs, outputs)

# Print the model summary
model.summary()
print("Total number of parameters in the model:", model.count_params())

# Define the model checkpoint for weights only
weights_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/weights/g_mlp_upgrade_4_weights_5_0.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/saved_models/best_model_upgrade_5_0.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

# Try loading the full model, then the weights, or use the freshly defined model if neither exists
if os.path.exists(full_model_checkpoint_path):
    print("Loading full model...")
    model = tf.keras.models.load_model(full_model_checkpoint_path, custom_objects={'SpatialGatingUnit': SpatialGatingUnit, 'gMLPBlock': gMLPBlock})
elif os.path.exists(weights_checkpoint_path):
    print("Loading model weights...")
    model.load_weights(weights_checkpoint_path)
else:
    print("Using freshly defined model...")

# Initialize optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# Compile the model
# msle = MeanSquaredLogarithmicError()
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Evaluate model immediately after loading weights (for baseline performance)
val_loss = model.evaluate(X_val, y_val)
print(f"Baseline Mean Absolute Error on Validation Set after Loading: {val_loss}")

# Fit the model
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
file_path = 'g_mlp_upgrade_3_predictions_2.csv'
print("Saving file to:", file_path)
result_df.to_csv(file_path, index=False)
print("File saved successfully!")



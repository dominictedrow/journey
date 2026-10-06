import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanSquaredLogarithmicError
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

seq_len = 8
d_model = 1048
d_ffn = 4192
block_layers = 10
learning_rate = 0.00001
epochs = 100
batch_size = 128

# TensorFlow policy to mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# New function definitions
def load_data_from_sqlite(db_path, table_name, columns=None):
    conn = sqlite3.connect(db_path)
    if columns:
        query = f"SELECT {','.join(columns)} FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def get_table_names(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    table_names = cursor.fetchall()
    conn.close()
    return table_names

# Load multiple tables
def load_all_tables_from_sqlite(db_path):
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    table_names = conn.execute(query).fetchall()
    conn.close()
    return {name[0]: load_data_from_sqlite(db_path, name[0]) for name in table_names}


def create_overlapping_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)

def preprocess_data(df, scaler=None, seq_len=seq_len):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    if scaler is None:  
        mean_values = df.mean(numeric_only=True)
        df.fillna(mean_values, inplace=True)
    else:  
        mean_dict = dict(zip(df.columns, scaler.mean_))
        df.fillna(mean_dict, inplace=True)
    
    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.values.astype('float32'))
    else:
        scaled = scaler.transform(df.values.astype('float32'))

    # Create overlapping sequences for features
    X = create_overlapping_sequences(scaled, seq_len)

    # Adjust target to align with the last element of each sequence
    y = target.values[seq_len - 1:]

    return X, y, scaler


# Load all tables into RAM
all_train_dfs = load_all_tables_from_sqlite('/home/thinkbe/Desktop/trading/data/streamline_all_close.db')
all_val_dfs = load_all_tables_from_sqlite('/home/thinkbe/Desktop/trading/data/val_streamline_all_close.db')

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

# Positional Encoding Layer
class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.position = position
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(position, d_model)

    def get_angles(self, position, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return position * angles

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(
            position=tf.range(position, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        return inputs + tf.cast(self.pos_encoding[:, :tf.shape(inputs)[1], :], inputs.dtype)
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'position': self.position,
            'd_model': self.d_model
        })
        return config


# Define the model
class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, d_ffn, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.spatial_projection = Dense(d_ffn, use_bias=True, kernel_initializer='lecun_normal')
        
        self.temporal_conv = tf.keras.layers.LSTM(1024, return_sequences=True, kernel_initializer='lecun_normal')
        self.temporal_projection = Dense(d_ffn, activation='tanh')  
        
        self.forget_gate = Dense(d_ffn, activation='sigmoid')
        self.input_gate = Dense(d_ffn, activation='sigmoid')

    def build(self, input_shape):
        self.spatial_gating = self.add_weight(
            shape=(input_shape[-1],),
            initializer='lecun_normal',
            trainable=True
        )

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        
        temporal_context = self.temporal_conv(normalized_inputs)
        temporal_context = self.temporal_projection(temporal_context)  # Project LSTM output to d_ffn dimensions
        
        f_gate = self.forget_gate(normalized_inputs)
        i_gate = self.input_gate(normalized_inputs)
        
        gated_temporal = f_gate * temporal_context + i_gate * self.spatial_projection(normalized_inputs)
        
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
        self.channel_projection_i = Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal')
        self.sgu = SpatialGatingUnit(d_ffn)
        self.channel_projection_ii = Dense(d_model, activation='tanh', kernel_initializer='lecun_normal')  

    def call(self, inputs):
        x = self.channel_projection_i(inputs)
        x = self.sgu(x)
        x = self.channel_projection_ii(x)
        return x + inputs

    def get_config(self):
        config = super(gMLPBlock, self).get_config()
        config.update({"d_model": self.d_model, "d_ffn": self.d_ffn})
        return config



# Initialize optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)


# Combine all tables into one DataFrame for training and validation
all_train_dfs_combined = pd.concat(all_train_dfs.values(), keys=all_train_dfs.keys())
all_val_dfs_combined = pd.concat(all_val_dfs.values(), keys=all_val_dfs.keys())

# Preprocess the combined DataFrame and get the scaler object
X_train_all, y_train_all, scaler = preprocess_data(all_train_dfs_combined)
X_val_all, y_val_all, _ = preprocess_data(all_val_dfs_combined, scaler)  # Use the same scaler

# Keep track of indices for splitting
indices_train = [len(df) for df in all_train_dfs.values()]
indices_val = [len(df) for df in all_val_dfs.values()]

# Separate the combined ndarray back into individual tables
train_tables_split = np.split(X_train_all, np.cumsum(indices_train)[:-1])
val_tables_split = np.split(X_val_all, np.cumsum(indices_val)[:-1])

# Convert to dictionary
train_tables_dict = {k: v for k, v in zip(all_train_dfs.keys(), train_tables_split)}
val_tables_dict = {k: v for k, v in zip(all_val_dfs.keys(), val_tables_split)}

# Create sequences for each table separately
X_train_sequenced = {key: create_overlapping_sequences(data, seq_len) for key, data in train_tables_dict.items()}
y_train_sequenced = {key: data[seq_len - 1:, -1] for key, data in train_tables_dict.items()}  # Assuming target is the last column

X_val_sequenced = {key: create_overlapping_sequences(data, seq_len) for key, data in val_tables_dict.items()}
y_val_sequenced = {key: data[seq_len - 1:, -1] for key, data in val_tables_dict.items()}  # Assuming target is the last column


# Define the input shape based on the combined data
inputs = tf.keras.layers.Input(shape=(X_train_all.shape[1], X_train_all.shape[2]))

# Project the input embeddings to match d_model
projected_inputs = tf.keras.layers.Dense(d_model)(inputs)

# Apply positional encoding to the projected embeddings
x = PositionalEncoding(X_train_all.shape[1], d_model)(projected_inputs)

for _ in range(block_layers):
    x = gMLPBlock(d_model, d_ffn)(x)

x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = tf.keras.layers.Dense(1, activation='linear', dtype='float32')(x)
model = tf.keras.Model(inputs, outputs)

# Print the model summary
model.summary()
print("Total number of parameters in the model:", model.count_params())

# Define the model checkpoint for weights only
weights_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/weights/g_mlp_all_weights_0.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/saved_models/best_model_all_0.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

# Try loading the full model, then the weights, or use the freshly defined model if neither exists
if os.path.exists(full_model_checkpoint_path):
    print("Loading full model...")
    model = tf.keras.models.load_model(full_model_checkpoint_path, custom_objects={'SpatialGatingUnit': SpatialGatingUnit, 'gMLPBlock': gMLPBlock, 'PositionalEncoding': PositionalEncoding})
elif os.path.exists(weights_checkpoint_path):
    print("Loading model weights...")
    model.load_weights(weights_checkpoint_path)
else:
    print("Using freshly defined model...")

# Compile the model
# msle = MeanSquaredLogarithmicError()
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# No need to combine again, simply assign
X_train_combined = X_train_all
y_train_combined = y_train_all
X_val_combined = X_val_all
y_val_combined = y_val_all

# Evaluate model immediately after loading weights (for baseline performance)
val_loss = model.evaluate(X_val_combined, y_val_combined)
print(f"Baseline Mean Absolute Error on Validation Set after Loading: {val_loss}")

# Use the combined data for training
print(f"Training on {X_train_combined.shape[0]} examples, validating on {X_val_combined.shape[0]} examples.")
history = model.fit(X_train_combined, y_train_combined, epochs=epochs, batch_size=batch_size, validation_data=(X_val_combined, y_val_combined), callbacks=[weights_checkpoint, full_model_checkpoint, dashboard])

epochs = 0 # You can set this to 0 or any other value
batch_size = 1
if epochs > 0:
    print(f"Training on {X_train_all.shape[0]} examples, validating on {X_val_all.shape[0]} examples.")
    history = model.fit(X_train_all, y_train_all, epochs=epochs, batch_size=batch_size, validation_data=(X_val_all, y_val_all), callbacks=[weights_checkpoint, full_model_checkpoint, dashboard])

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
y_pred = model.predict(X_val_all).flatten()
mse = mean_squared_error(y_val_all, y_pred)
mae = mean_absolute_error(y_val_all, y_pred)

print(f"Model Mean Squared Error on Validation Set: {mse}")
print(f"Model Mean Absolute Error on Validation Set: {mae}")

# Save the actual and predicted values to a DataFrame
result_df = pd.DataFrame({"Actual": y_val_all, "Predicted": y_pred})

# Save result DataFrame to a CSV file
file_path = 'g_mlp_all_predictions_0.csv'
print("Saving file to:", file_path)
result_df.to_csv(file_path, index=False)
print("File saved successfully!")




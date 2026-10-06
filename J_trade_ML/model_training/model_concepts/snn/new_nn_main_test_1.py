import numpy as np
import pandas as pd
import sqlite3
import os
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import clear_output, Image, display
import pickle

# Advanced Positional Encoding Function
def advanced_positional_encoding(X, min_freq=1e-4):
    # Ensure input is float32
    X = X.astype(np.float32)
    n_samples, n_features = X.shape
    position = np.arange(n_features, dtype=np.float32)
    frequencies = np.exp(-np.log(10000.0) * position / n_features * min_freq)
   
    pos_encoding = np.zeros((n_samples, n_features * 2), dtype=np.float32)
    pos_encoding[:, 0::2] = np.sin(position * frequencies)
    pos_encoding[:, 1::2] = np.cos(position * frequencies)
   
    # Combine original features with their positional encodings
    encoded_data = np.hstack([X, pos_encoding])
    return encoded_data.astype(np.float32)

# Data Loading Function
def load_data_from_sqlite(db_path, table_name, columns=None):
    conn = sqlite3.connect(db_path)
    if columns:
        query = f"SELECT {','.join(columns)} FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Data Preprocessing Function for Training Data
def preprocess_data(df, feature_scaler=None, target_scaler=None):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    if feature_scaler is None:
        feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(df.values.astype('float32'))

    # Add advanced positional encoding
    X_scaled = advanced_positional_encoding(X_scaled)
    target_mean = None
    target_std = None
    if target_scaler is not None:
        y_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
        target_mean = target_scaler.mean_[0]
        target_std = target_scaler.scale_[0]
    return X_scaled, y_scaled, feature_scaler, target_scaler, target_mean, target_std

# Data Preprocessing Function for Validation Data
def preprocess_validation_data(df, feature_scaler, target_scaler):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X_scaled = feature_scaler.transform(df.values.astype('float32'))
    # Add advanced positional encoding
    X_scaled = advanced_positional_encoding(X_scaled)
    y_scaled = target_scaler.transform(target.values.reshape(-1, 1)).flatten()
    return X_scaled, y_scaled

# Using the Functions for Training and Validation Data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Advanced Positional Encoding Function
def advanced_positional_encoding(X, min_freq=1e-4):
    # Ensure input is float32
    X = X.astype(np.float32)
    n_samples, n_features = X.shape
    position = np.arange(n_features, dtype=np.float32)
    frequencies = np.exp(-np.log(10000.0) * position / n_features * min_freq)
   
    pos_encoding = np.zeros((n_samples, n_features * 2), dtype=np.float32)
    pos_encoding[:, 0::2] = np.sin(position * frequencies)
    pos_encoding[:, 1::2] = np.cos(position * frequencies)
   
    # Combine original features with their positional encodings
    encoded_data = np.hstack([X, pos_encoding])
    return encoded_data.astype(np.float32)

# Data Loading Function
def load_data_from_sqlite(db_path, table_name, columns=None):
    conn = sqlite3.connect(db_path)
    if columns:
        query = f"SELECT {','.join(columns)} FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# Data Preprocessing Function for Training Data
def preprocess_data(df, feature_scaler=None, target_scaler=None):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    if feature_scaler is None:
        feature_scaler = StandardScaler()
    X_scaled = feature_scaler.fit_transform(df.values.astype('float32'))

    # Add advanced positional encoding
    X_scaled = advanced_positional_encoding(X_scaled)
    target_mean = None
    target_std = None
    if target_scaler is not None:
        y_scaled = target_scaler.fit_transform(target.values.reshape(-1, 1)).flatten()
        target_mean = target_scaler.mean_[0]
        target_std = target_scaler.scale_[0]
    return X_scaled, y_scaled, feature_scaler, target_scaler, target_mean, target_std

# Data Preprocessing Function for Validation Data
def preprocess_validation_data(df, feature_scaler, target_scaler):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X_scaled = feature_scaler.transform(df.values.astype('float32'))
    # Add advanced positional encoding
    X_scaled = advanced_positional_encoding(X_scaled)
    y_scaled = target_scaler.transform(target.values.reshape(-1, 1)).flatten()
    return X_scaled, y_scaled

# Using the Functions for Training and Validation Data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Load and preprocess training data
print("About to load training data...")
df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//streamline_lag_close_(backup).db", 'XAUUSD')
print("Finished loading training data...")
print("Starting to preprocess training data...")
X_train, y_train, feature_scaler, target_scaler, target_mean, target_std = preprocess_data(df_train, feature_scaler, target_scaler)
print("Finished preprocessing training data...")

# Load and preprocess validation data
print("About to load validation data...")
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//val_streamline_lag_close_(backup).db", 'XAUUSD')
print("Finished loading validation data...")
print("Starting to preprocess validation data...")
X_val, y_val = preprocess_validation_data(df_val, feature_scaler, target_scaler)
print("Finished preprocessing validation data...")

class CustomMLPTensorFlow:
    def __init__(self, input_size, layers, softmax_threshold, learning_rate):
        self.layers = layers
        self.softmax_threshold = softmax_threshold
        self.learning_rate = learning_rate
        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        # Initialize weights and biases for each layer
        self.weights = []
        self.biases = []
        for in_units, out_units in zip([input_size] + layers[:-1], layers):
            # Weights and biases for custom softmax layer
            self.weights.append(tf.Variable(tf.random.normal([in_units, out_units]), trainable=True))
            self.biases.append(tf.Variable(tf.random.normal([out_units]), trainable=True))
            # Weights and biases for additional dense layer
            self.weights.append(tf.Variable(tf.random.normal([out_units, out_units]), trainable=True))
            self.biases.append(tf.Variable(tf.random.normal([out_units]), trainable=True))

    def forward(self, x):
        layer_activations = [x]
        for i in range(0, len(self.weights), 2):  # Step by 2 to handle both custom and dense layers
            # Process custom softmax layer
            z = tf.matmul(layer_activations[-1], self.weights[i]) + self.biases[i]
            tanh_activation = tf.nn.tanh(z)
            softmax_outputs = tf.nn.softmax(tanh_activation, axis=-1)
            activation = self.apply_dynamic_connections(tanh_activation, softmax_outputs, i // 2)

            # Process additional dense layer
            z_dense = tf.matmul(activation, self.weights[i + 1]) + self.biases[i + 1]
            dense_activation = tf.nn.tanh(z_dense) if i < len(self.weights) - 2 else z_dense  # Output layer has no tanh

            layer_activations.append(dense_activation)

        return layer_activations[-1]

    def apply_dynamic_connections(self, activation, softmax_outputs, layer_index):
        # Calculate mask based on softmax threshold
        connection_mask = tf.cast(tf.greater(softmax_outputs, self.softmax_threshold), tf.float32)

        # Apply the mask to tanh activation outputs
        dynamic_activation = activation * connection_mask

        # Reshape dynamic_activation to match the next layer
        next_layer_units = self.layers[layer_index]  # Using the same number of units as current layer
        dynamic_activation = tf.reshape(dynamic_activation, [-1, next_layer_units])

        return dynamic_activation

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.forward(x)
            loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, predictions))  # MAE loss

        gradients = tape.gradient(loss, self.weights + self.biases)
        self.optimizer.apply_gradients(zip(gradients, self.weights + self.biases))


# Initialize your MLP
input_size = X_train.shape[1]  # Adjusted to match the feature count after positional encoding
layers = [128, 128, 128, 128, 1] # Define the number of units in each layer
softmax_threshold = 0.33

# Training loop
epochs = 20
learning_rate = 0.01
batch_size = 128

mlp = CustomMLPTensorFlow(input_size, layers, softmax_threshold, learning_rate)

# Training and validation datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)


# Assuming train_dataset and validation_dataset are already defined as batched datasets
train_batches = len(train_dataset)
val_batches = len(validation_dataset)

# Cast target_mean and target_std to float32
target_mean_tf = tf.cast(target_mean, tf.float32)
target_std_tf = tf.cast(target_std, tf.float32)

# Check if the saved plot data file exists
if os.path.exists('plot_data.pkl'):
    with open('plot_data.pkl', 'rb') as f:
        train_losses, val_losses = pickle.load(f)
else:
    train_losses = []
    val_losses = []

# Initialize the plot
plt.figure(figsize=(10, 5))

# Function to update the plot
def update_plot(epoch, train_losses, val_losses):
    plt.clf()  # Clear the current figure
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Epoch {epoch+1}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.draw()  # Redraw the current figure
    plt.pause(0.001)  # Pause to update the figure
    plt.savefig('loss_plot.png')  # Save the plot with a constant filename


# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Training loop
    with tqdm(total=train_batches, desc="Training", unit='batch') as pbar:
        total_training_mae = 0
        total_training_samples = 0

        for batch_index, (x_batch, y_batch) in enumerate(train_dataset):
            mlp.train_step(x_batch, y_batch)

            predictions = mlp.forward(x_batch)
            predictions_unscaled = predictions * target_std_tf + target_mean_tf
            y_batch_unscaled = y_batch * target_std_tf + target_mean_tf
            batch_mae = tf.reduce_mean(tf.abs(predictions_unscaled - y_batch_unscaled)).numpy()

            total_training_samples += len(x_batch)
            total_training_mae += batch_mae * len(x_batch)

            ongoing_avg_mae = total_training_mae / total_training_samples

            pbar.set_postfix(ongoing_avg_MAE=ongoing_avg_mae)
            pbar.update(1)
            if batch_index == train_batches - 1:
                pbar.close()

        avg_training_mae = total_training_mae / total_training_samples
        print(f"Average Training MAE: {avg_training_mae}")

    # Validation loop
    with tqdm(total=val_batches, desc="Validation", unit='batch') as pbar:
        total_val_mae = 0
        total_val_samples = 0

        for batch_index, (x_batch, y_batch) in enumerate(validation_dataset):
            mlp_output = mlp.forward(x_batch)

            mlp_output_unscaled = mlp_output * target_std_tf + target_mean_tf
            y_batch_unscaled = y_batch * target_std_tf + target_mean_tf

            batch_val_mae = tf.reduce_mean(tf.abs(mlp_output_unscaled - y_batch_unscaled)).numpy()

            total_val_samples += len(x_batch)
            total_val_mae += batch_val_mae * len(x_batch)

            ongoing_avg_val_mae = total_val_mae / total_val_samples

            pbar.set_postfix(ongoing_avg_MAE=ongoing_avg_val_mae)
            pbar.update(1)
            if batch_index == val_batches - 1:
                pbar.close()

        avg_val_mae = total_val_mae / total_val_samples
        print(f"Epoch {epoch + 1} Average Validation MAE: {avg_val_mae}")

    # Append losses for plotting
    train_losses.append(avg_training_mae)
    val_losses.append(avg_val_mae)

    # Update plot after each epoch
    update_plot(epoch, train_losses, val_losses)

    # Save plot data after each epoch
    with open('plot_data.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses), f)

plt.close()  # Close the plot window

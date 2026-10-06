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

    # Add advanced positional encoding (currently turned off)
    # X_scaled = advanced_positional_encoding(X_scaled)

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

    # Add advanced positional encoding (currently turned off)
    # X_scaled = advanced_positional_encoding(X_scaled)

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
print()
print("Starting to preprocess training data...")
X_train, y_train, feature_scaler, target_scaler, target_mean, target_std = preprocess_data(df_train, feature_scaler, target_scaler)
print("Finished preprocessing training data...")
print()
# Load and preprocess validation data
print("About to load validation data...")
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//val_streamline_lag_close_(backup).db", 'XAUUSD')
print("Finished loading validation data...")
print()
print("Starting to preprocess validation data...")
X_val, y_val = preprocess_validation_data(df_val, feature_scaler, target_scaler)
print("Finished preprocessing validation data...")
print()

class CustomMLPTensorFlow:
    def __init__(self, input_size, layers, end_layers, learning_rate, embeddings, mode, softmax_application):
        self.mode = mode
        self.layers = layers
        self.end_layers = end_layers
        self.learning_rate = learning_rate
        self.softmax_application = softmax_application        

        if mode == 'embeddings':
            # Set the number of embedding features to match the input size
            self.embeddings = input_size
        elif mode == 'both':
            # Use the specified number of embeddings
            self.embeddings = embeddings
        else:
            # No embeddings are used
            self.embeddings = 0

        self.optimizer = tf.optimizers.Adam(self.learning_rate)

        glorot_init = tf.keras.initializers.GlorotUniform()
        zero_init = tf.zeros_initializer()

        if self.embeddings > 0:
            # Initialize the embedding layer
            self.embedding_weights = tf.Variable(glorot_init([input_size, self.embeddings]), trainable=True)
        else:
            self.embedding_weights = None

        first_layer_input_size = input_size + self.embeddings if self.mode == 'both' else input_size
        # Initialize weights and biases for each layer in layers
        self.weights = []
        self.biases = []
        for in_units, out_units in zip([first_layer_input_size] + layers[:-1], layers):
            self.weights.append(tf.Variable(glorot_init([in_units, out_units]), trainable=True))
            self.biases.append(tf.Variable(zero_init([out_units]), trainable=True))
        
        # Initialize weights and biases for end_layers with names
        for j, (in_units, out_units) in enumerate(zip([layers[-1]] + end_layers[:-1], end_layers), start=len(layers)):
            self.weights.append(tf.Variable(glorot_init([in_units, out_units]), trainable=True, name=f"weight_{j}"))
            self.biases.append(tf.Variable(zero_init([out_units]), trainable=True, name=f"bias_{j}"))



    def forward(self, x):
        # Apply the embedding layer only if required
        if self.mode in ['embeddings', 'both']:
            embedded_x = tf.matmul(x, self.embedding_weights)
            if self.mode == 'both':
                x = tf.concat([x, embedded_x], axis=1)
            else:
                x = embedded_x

        layer_activations = [x]

        for i in range(0, len(self.layers) * 2, 2):
            # Compute the activation for the custom softmax layer
            z = tf.matmul(layer_activations[-1], self.weights[i]) + self.biases[i]
            tanh_activation = tf.nn.tanh(z)

            # Apply softmax based on the specified application mode
            if self.softmax_application == 'unit':
                softmax_outputs = tf.map_fn(tf.nn.softmax, tanh_activation)
            else:  # Default to layer-wise softmax application
                softmax_outputs = tf.nn.softmax(tanh_activation)

            # Apply dynamic connections to get continuous output
            continuous_output = self.apply_dynamic_connections(tanh_activation, softmax_outputs, i // 2)            
            
            # Compute the activation for the additional dense layer
            z_dense = tf.matmul(continuous_output, self.weights[i + 1]) + self.biases[i + 1]
            dense_activation = tf.nn.tanh(z_dense)
            dense_norm = tf.keras.layers.LayerNormalization()(dense_activation)
            layer_activations.append(dense_norm)            

        # Linear projection for residual connection if shapes are different
        if layer_activations[0].shape[-1] != layer_activations[-1].shape[-1]:
            projection_weights = tf.Variable(tf.keras.initializers.GlorotUniform()([layer_activations[0].shape[-1], layer_activations[-1].shape[-1]]), trainable=True)
            projected_input = tf.matmul(layer_activations[0], projection_weights)
        else:
            projected_input = layer_activations[0]

        # Add a residual connection to the input of the first end layer
        residual_connection = layer_activations[-1] + projected_input

        # Process through the end layers
        for i in range(len(self.layers) * 2, len(self.weights)):
            z_end = tf.matmul(residual_connection if i == len(self.layers) * 2 else layer_activations[-1], self.weights[i]) + self.biases[i]
            end_activation = tf.nn.tanh(z_end) if i < len(self.weights) - 1 else z_end
            layer_activations.append(end_activation)

        return layer_activations[-1]

    def apply_dynamic_connections(self, activation, softmax_outputs, layer_index):
        out_units = self.layers[layer_index]
        
        # Generate dynamic basis vectors for continuous values
        basis_vectors = tf.random.uniform(shape=[out_units, out_units], minval=-1, maxval=1)

        # Compute weighted sum of basis vectors using softmax outputs as weights
        continuous_output = tf.matmul(softmax_outputs, basis_vectors)

        return continuous_output

    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.forward(x)
            loss = tf.reduce_mean(tf.keras.losses.mean_absolute_error(y, predictions))

        gradients = tape.gradient(loss, self.weights + self.biases)
        self.optimizer.apply_gradients(zip(gradients, self.weights + self.biases))

    def save_weights(self, filepath):
        # Handle None for embedding_weights
        embedding_weights = self.embedding_weights.numpy() if self.embedding_weights is not None else None
        weights = [embedding_weights] + [w.numpy() for w in self.weights] + [b.numpy() for b in self.biases]
        with open(filepath, 'wb') as f:
            pickle.dump(weights, f)

    def load_weights(self, filepath):
        with open(filepath, 'rb') as f:
            weights = pickle.load(f)

        # Assign embedding weights if not None
        if weights[0] is not None:
            self.embedding_weights.assign(weights[0])

        # Correctly assign remaining weights and biases
        for i, weight in enumerate(weights[1:len(self.weights) + 1]):
            self.weights[i].assign(weight)
        for i, bias in enumerate(weights[len(self.weights) + 1:]):
            self.biases[i].assign(bias)

    def print_summary(self):
        # Print a summary of the model configuration
        print()        
        print("\nModel Summary:")
        print(f"Mode: {self.mode}")
        print(f"Learning Rate: {self.learning_rate}")
        print(f"Embeddings: {self.embeddings}")
        print(f"Layer Configuration: {self.layers}")
        print(f"End Layer Configuration: {self.end_layers}")
        print(f"Softmax Application Mode: {self.softmax_application}")        

        # Count trainable parameters
        trainable_params = sum(np.prod(v.shape) for v in self.weights + self.biases)
        if self.embedding_weights is not None:
            trainable_params += np.prod(self.embedding_weights.shape)
        non_trainable_params = 0

        print(f"Trainable Parameters: {trainable_params}")
        print(f"Non-trainable Parameters: {non_trainable_params}\n")        

# Initialize your MLP
input_size = X_train.shape[1]  # Adjusted to match the feature count after positional encoding
layers = [1200, 1200, 1200, 1200, 1200] # Define the number of units in each layer
end_layers = [1200, 1200, 1200, 1200, 1]
embeddings = 40 # activation in printed in console. If both then adjust this number, if not then ignore. 
softmax_application = 'unit'  # 'unit' Or 'layer', based on user preference
# softmax_threshold = 0.16 # old way the softmax worked

# Training loop
epochs = 50
learning_rate = 0.001
batch_size = 128

# Ask the user for the mode
mode = input("Would you like to train/run this model on inputs, embeddings or both? ").lower()
while mode not in ['inputs', 'embeddings', 'both']:
    print("Invalid choice. Please select 'inputs', 'embeddings', or 'both'.")
    mode = input("Would you like to train/run this model on inputs, embeddings or both? ").lower()
    print()

mlp = CustomMLPTensorFlow(input_size, layers, end_layers, learning_rate, embeddings, mode, softmax_application)

# Print model summary
mlp.print_summary()

print(f"Training Epochs: {epochs}")
print(f"Batch Size: {batch_size}")
print()

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
    plt.pause(0.01)  # Pause to update the figure
    plt.savefig('loss_plot.png')  # Save the plot with a constant filename

print()
load_weights = input("Do you want to load weights from a previous model? (yes/no): ").lower()
weights_path = "C://Users//crgon//OneDrive//Desktop//trading//model_testing_text//training_models//save_models_weights//new_nn_main_test_3.h5"

if load_weights == 'yes' and os.path.exists(weights_path):
    mlp.load_weights(weights_path)  # Assuming `load_weights` is a method of your model class
    print("Weights loaded successfully.")
elif load_weights == 'yes':
    print("Weights file not found. Starting training from scratch.")

best_val_mae = float('inf')

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

    # Check if this epoch has the best validation MAE
    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        mlp.save_weights(weights_path)  # Assuming `save_weights` is a method of your model class
        print(f"Saved model weights for epoch {epoch+1} with validation MAE: {avg_val_mae}")

    # Save plot data after each epoch
    with open('plot_data.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses), f)

plt.close()  # Close the plot window

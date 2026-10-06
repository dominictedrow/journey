import numpy as np
import pandas as pd
import sqlite3
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

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
    def __init__(self, input_size, layer1_units, layer2_units):
        self.input_size = input_size
        self.layer1_units = layer1_units
        self.layer2_units = layer2_units

        # LeCun normal initializer
        initializer = tf.initializers.lecun_normal()

        # Initialize weights and biases using lecun_normal initializer
        self.weights1 = tf.Variable(initializer([self.input_size, self.layer1_units]), trainable=True)
        self.biases1 = tf.Variable(tf.zeros([self.layer1_units]), trainable=True)
        self.weights2 = tf.Variable(initializer([self.layer1_units, self.layer2_units]), trainable=True)
        self.biases2 = tf.Variable(tf.zeros([self.layer2_units]), trainable=True)
        self.weights3 = tf.Variable(initializer([self.layer2_units, 1]), trainable=True)
        self.biases3 = tf.Variable(tf.zeros([1]), trainable=True)

    def forward(self, x):
        # First layer with tanh activation
        z1 = tf.matmul(x, self.weights1) + self.biases1
        a1 = tf.nn.tanh(z1)

        # Compute dynamic weights for the second layer
        dynamic_weights2 = self.compute_dynamic_weights(a1)

        # Second layer with tanh activation
        z2 = tf.matmul(a1, dynamic_weights2) + self.biases2
        a2 = tf.nn.tanh(z2)

        # Final layer with tanh activation
        z3 = tf.matmul(a2, self.weights3) + self.biases3
        final_output = z3
        return final_output

    def compute_dynamic_weights(self, a1):
        # Vectorized computation of dynamic weights
        softmax_a1 = tf.nn.softmax(a1, axis=1)
        condition = tf.greater(softmax_a1, 0.5)

        # Reshape and tile condition to match the shape of weights2
        condition_tiled = tf.tile(tf.reshape(condition, [-1, self.layer1_units, 1]), [1, 1, self.layer2_units])

        # Use broadcasting in tf.where to apply condition
        dynamic_weights = tf.where(condition_tiled, tf.broadcast_to(self.weights2, tf.shape(condition_tiled)), tf.zeros_like(self.weights2))

        # Aggregate the weights across the batch dimension
        dynamic_weights_aggregated = tf.reduce_mean(dynamic_weights, axis=0)
        return dynamic_weights_aggregated


    def update_connections(self, softmax_a1):
        # Update weights dynamically based on the softmax conditions
        softmax_a1 = tf.nn.softmax(softmax_a1, axis=1)
        condition = tf.greater(softmax_a1, 0.5)

        new_weights = []
        for i in range(self.layer1_units):
            condition_i = tf.reshape(condition[:, i], [-1, 1])
            condition_i = tf.tile(condition_i, [1, self.layer2_units])

            # Ensure that the shapes are compatible for broadcasting
            # Reshape self.weights2[i] and expand its dimensions to match the condition_i shape
            weights_i_reshaped = tf.reshape(self.weights2[i], [1, -1])
            weights_i_tiled = tf.tile(weights_i_reshaped, [tf.shape(condition_i)[0], 1])

            # Update the weights based on the condition
            updated_weights = tf.where(condition_i, weights_i_tiled, tf.zeros_like(weights_i_tiled))
            new_weights.append(updated_weights)

        # Stack the updated weights and assign them to self.weights2
        self.weights2.assign(tf.stack(new_weights, axis=0))



    def train_step(self, x, y, learning_rate):
        with tf.GradientTape() as tape:
            predictions = self.forward(x)
            loss = self.compute_loss(predictions, y)

        gradients = tape.gradient(loss, [self.weights1, self.biases1, self.weights2, self.biases2, self.weights3, self.biases3])
        self.apply_gradients(gradients, learning_rate)

    def compute_loss(self, predictions, y):
        return tf.reduce_mean(tf.square(predictions - y))

    def apply_gradients(self, gradients, learning_rate):
        optimizer = tf.optimizers.Adam(learning_rate)
        optimizer.apply_gradients(zip(gradients, [self.weights1, self.biases1, self.weights2, self.biases2, self.weights3, self.biases3]))


# Initialize your MLP
input_size = X_train.shape[1]  # Adjusted to match the feature count after positional encoding
layer1_units = 10  # Adjust as needed
layer2_units = 10  # Adjust according to your output features
batch_size = 256
mlp = CustomMLPTensorFlow(input_size, layer1_units, layer2_units)

# Training loop
epochs = 20
learning_rate = 0.001
# Training loop
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        x_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]
        mlp.train_step(x_batch, y_batch, learning_rate)
    print(f"Epoch {epoch + 1} completed")

# Prepare the validation dataset
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# Cast target_mean and target_std to float32
target_mean_tf = tf.cast(target_mean, tf.float32)
target_std_tf = tf.cast(target_std, tf.float32)

# Assuming you have a validation dataset prepared in TensorFlow format
for x_batch, y_batch in validation_dataset:
    mlp_output = mlp.forward(x_batch)

    # Unscale the outputs and targets
    mlp_output_unscaled = mlp_output * target_std_tf + target_mean_tf
    y_batch_unscaled = y_batch * target_std_tf + target_mean_tf

    """ tf.print("Shape of mlp_output_unscaled:", tf.shape(mlp_output_unscaled))
    tf.print("Shape of y_batch_unscaled:", tf.shape(y_batch_unscaled)) """

    # Ensure the shapes are compatible for subtraction
    if tf.shape(mlp_output_unscaled)[0] != tf.shape(y_batch_unscaled)[0]:
        # Reshape if necessary
        mlp_output_unscaled = tf.reshape(mlp_output_unscaled, [-1, 1])  # Adjust as needed

    # Compute MAE
    try:
        mae = tf.reduce_mean(tf.abs(mlp_output_unscaled - y_batch_unscaled))
        tf.print("MAE on Validation Data:", mae)
    except tf.errors.InvalidArgumentError as e:
        tf.print("Error in computing MAE:", e)


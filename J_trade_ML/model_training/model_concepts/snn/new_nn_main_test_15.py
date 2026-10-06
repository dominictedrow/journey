import os
import pickle
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Conv1D, LSTM, LayerNormalization, BatchNormalization, GlobalAveragePooling1D, ReLU, Multiply, AveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Reshape, PReLU, Add
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.backend import int_shape
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.config.experimental.set_visible_devices

# Mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

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

# Modified Data Preprocessing Function for Training Data
def preprocess_data(df, feature_scaler=None):
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

    return X_scaled, target, feature_scaler

# Modified Data Preprocessing Function for Validation Data
def preprocess_validation_data(df, feature_scaler):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X_scaled = feature_scaler.transform(df.values.astype('float32'))

    # Add advanced positional encoding (currently turned off)
    # X_scaled = advanced_positional_encoding(X_scaled)

    return X_scaled, target

# Using the Functions for Training and Validation Data
feature_scaler = StandardScaler()

# Load and preprocess training data
print("About to load training data...")
df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//streamline_lag_close_(backup).db", 'XAUUSD')
print("Finished loading training data...")
print()
print("Starting to preprocess training data...")
X_train, y_train, feature_scaler = preprocess_data(df_train, feature_scaler)
print("Finished preprocessing training data...")
print()

# Load and preprocess validation data
print("About to load validation data...")
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//val_streamline_lag_close_(backup).db", 'XAUUSD')
print("Finished loading validation data...")
print()
print("Starting to preprocess validation data...")
X_val, y_val = preprocess_validation_data(df_val, feature_scaler)
print("Finished preprocessing validation data...")
print()

class ReversedDense(tf.keras.layers.Layer):
    def __init__(self, dense_layer, **kwargs):
        super(ReversedDense, self).__init__(**kwargs)
        self.dense_layer = dense_layer

    def call(self, inputs):
        reversed_inputs = tf.reverse(inputs, axis=[-1])
        return self.dense_layer(reversed_inputs)    

""" class SpikingNeuronLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SpikingNeuronLayer, self).__init__(**kwargs)
        self.units = units
        self.state_size = units

    def build(self, input_shape):
        self.threshold = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
            name='threshold'
        )

    def call(self, inputs, states):
        accumulated = inputs
        fired = tf.greater_equal(accumulated, self.threshold)

        next_state = tf.where(fired, tf.zeros_like(accumulated), accumulated)
        outputs = tf.cast(fired, inputs.dtype)  # Ensure data type consistency

        return outputs, [next_state]

    def reset_state_per_sample(self):
        # Reset the state to zero for each neuron in the layer
        self.state = tf.zeros_like(self.state_size, dtype=self.dtype) """

class SpikingNeuronLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SpikingNeuronLayer, self).__init__(**kwargs)
        self.units = units
        self.state_size = (units,)

    def build(self, input_shape):
        self.threshold = self.add_weight(
            shape=(self.units,),
            initializer='random_normal',
            trainable=True,
            name='threshold'
        )
        self.state = self.add_weight(
            shape=(1, self.units),
            initializer='zeros',
            trainable=False,
            name='state'
        )

    def call(self, inputs, states):
        # Use the external state if provided, otherwise use the internal state
        accumulated = states[0] if states is not None else self.state

        # Add inputs to the accumulated state
        accumulated = tf.add(accumulated, inputs)
        
        # Determine if neurons fire
        fired = tf.greater_equal(accumulated, self.threshold)

        # Update state
        next_state = tf.where(fired, tf.zeros_like(accumulated), accumulated)

        # Cast fired to the same dtype as inputs for consistency
        outputs = tf.cast(fired, inputs.dtype)

        # Update the internal state if no external state is provided
        if states is None:
            self.state.assign(next_state)

        return outputs, [next_state]

    def reset_state_per_sample(self):
        # Reset the internal state to zero for each neuron in the layer
        self.state.assign(tf.zeros_like(self.state, dtype=self.dtype))


def create_dual_input_model(input_shape, num_blocks, num_layers_per_block, units_per_layer, RNN):
    input_normal = Input(shape=(input_shape,), name='input_normal', dtype='float32')

    # Determine the number of features based on the input shape
    num_features = int_shape(input_normal)[-1]

    # Define the Dense layer outside of the loop
    shared_dense = Dense(num_features, kernel_initializer='lecun_normal', dtype='float32')

    x_transformed = input_normal
    for _ in range(RNN):
        x_transformed_initial = shared_dense(x_transformed)
        x_transformed = tf.keras.activations.gelu(x_transformed_initial)
        x_transformed = LayerNormalization()(x_transformed)
        x_transformed = Add()([x_transformed, input_normal])  

    # New projection layer to match units_per_layer
    projection_layer = Dense(units_per_layer, kernel_initializer='lecun_normal')
    x_transformed = projection_layer(x_transformed)

    """ # Apply GELU activation function as a separate layer if not using SpikingNeuronLayer
    gelu_activation = tf.keras.layers.Activation(tf.keras.activations.gelu)
    x_transformed = gelu_activation(x_transformed) """

    # Apply SpikingNeuronLayer
    spiking_layer = SpikingNeuronLayer(units=units_per_layer)
    x_transformed, _ = spiking_layer(x_transformed, [x_transformed])

    blocks = []
    for _ in range(num_blocks):
        block_layers = [Dense(units_per_layer, kernel_initializer='lecun_normal') for _ in range(num_layers_per_block)]
        blocks.append(block_layers)

    block_outputs = []
    for block in blocks:
        x_normal = x_transformed
        for layer in block:
            x_normal = layer(x_normal)
            spiking_layer = SpikingNeuronLayer(units=units_per_layer)
            x_normal, _ = spiking_layer(x_normal, [x_normal])
            x_normal = LayerNormalization()(x_normal)
        
        # Side 2 - Reversed processing for block
        x_reversed = x_normal
        for layer in block[::-1]:
            x_reversed = tf.reverse(x_reversed, axis=[-1])
            reversed_layer = ReversedDense(layer)
            x_reversed = reversed_layer(x_reversed)             
            x_reversed, _ = spiking_layer(x_reversed, [x_reversed])  # Reuse the same SpikingNeuronLayer instance                      
            x_reversed = LayerNormalization()(x_reversed)

        # Use the final output of side 2 as the block output
        block_outputs.append(x_reversed)            

        """ # Use the final output of side 1 as the block output
        block_outputs.append(x_normal) """

    # Stack and average across all blocks
    stacked_across_blocks = tf.stack(block_outputs, axis=1) 
    x = GlobalMaxPooling1D()(stacked_across_blocks) # GlobalAveragePooling1D or GlobalMaxPooling1D    
    x_final = x

    output = Dense(1, activation='linear', kernel_initializer='lecun_normal')(x_final)
    model = Model(inputs=input_normal, outputs=output)

    # Return the model along with trainable_layers and target_layers
    return model # trainable_layers, target_layers
   
# Replace with your actual input shape and parameters
input_shape = X_train.shape[1]
num_blocks = 2
units_per_layer = 1200 
num_layers_per_block = 6 
RNN = 3  # The number of times to recycle the output

epochs = 500 
batch_size = 512
patience = 50  # Define your patience for early stopping

learning_rate = 0.001  # Matched learning rate (default = 0.001)
beta_1=0.9             # Momentum term (beta1) (default = 0.9)
beta_2=0.999           # Momentum term for the squared gradient (beta2) (default = 0.999)
epsilon=1e-07          # Small number to prevent any division by zero (default = 1e-07)
decay=0.0              # Weight decay for regularization (default = 0.0)
amsgrad=False            # Whether to apply AMSGrad variant of Adam


# Create the model and get the layers
model = create_dual_input_model(input_shape, num_blocks, num_layers_per_block, units_per_layer, RNN)

# Inspect the model architecture
model.summary()

# Customizing the Adam optimizer
custom_adam = Adam(
    learning_rate = learning_rate, 
    beta_1 = beta_1,             # Momentum term (beta1)
    beta_2 = beta_2,           # Momentum term for the squared gradient (beta2)
    epsilon = epsilon,          # Small number to prevent any division by zero
    decay = decay,              # Weight decay for regularization
    amsgrad = amsgrad           # Whether to apply AMSGrad variant of Adam
)

# Compile the model with the customized optimizer
model.compile(optimizer=custom_adam, loss='mean_absolute_error') # mean_squared_error & mean_absolute_error

# Before loading weights
weights_before = [layer.get_weights() for layer in model.layers]

# Check if 'best_model_15.h5' exists
if os.path.exists('best_model_15.h5'):
    # Load weights if the file exists
    model.load_weights('best_model_15.h5')
    print("Weights loaded from 'best_model_15.h5'")
else:
    # Skip loading and possibly print a message or take other actions
    print("'best_model_15.h5' not found. Continuing without loading weights.")


# Check if the saved plot data file exists
if os.path.exists('plot_data_15.pkl'):
    with open('plot_data_15.pkl', 'rb') as f:
        train_losses, val_losses = pickle.load(f)
else:
    train_losses = []
    val_losses = []

# Initialize the plot
plt.figure(figsize=(10, 5))

# Function to update the plot
def update_plot(epoch, train_losses, val_losses):
    plt.clf()  # Clear the current figure
    plt.plot(train_losses, label='Training MAE')
    plt.plot(val_losses, label='Validation MAE')
    plt.title(f'Epoch {epoch+1}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.draw()  # Redraw the current figure
    plt.pause(0.01)  # Pause to update the figure
    plt.savefig('mae_plot_15.png')  # Save the plot with a constant filename

# Training loop
best_val_mae = float('inf')

no_improvement_epochs = 0

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model_15.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Define a custom training step
def train_step(model, x_batch, y_batch, optimizer, loss_fn):
    total_loss = 0.0
    total_mae = 0.0
    batch_size = tf.shape(x_batch)[0]

    for i in range(batch_size):
        with tf.GradientTape() as tape:
            # Process one sample at a time
            x_sample = tf.expand_dims(x_batch[i], axis=0)
            y_sample = tf.expand_dims(y_batch[i], axis=0)

            predictions = model(x_sample, training=True)
            loss = loss_fn(y_sample, predictions)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        total_loss += loss

        # Ensure predictions and y_batch are of the same type
        predictions = tf.cast(predictions, tf.float32)
        y_sample = tf.cast(y_sample, tf.float32)

        # Calculate MAE using TensorFlow operations
        mae = tf.reduce_mean(tf.abs(y_sample - predictions))
        total_mae += mae

        # Reset states for spiking layers after processing each sample
        for layer in model.layers:
            if isinstance(layer, SpikingNeuronLayer):
                layer.reset_state_per_sample()

    # Calculate average loss and MAE for the batch
    avg_loss = total_loss / tf.cast(batch_size, dtype=total_loss.dtype)
    avg_mae = total_mae / tf.cast(batch_size, dtype=total_mae.dtype)
    return avg_loss, avg_mae

@tf.function
def validation_step(model, x_batch, y_batch, loss_fn):
    total_loss = tf.constant(0.0, dtype=tf.float32)  # Initialize total_loss
    total_mae = tf.constant(0.0, dtype=tf.float32)  # Initialize total_mae
    batch_size = tf.shape(x_batch)[0]

    for i in tf.range(batch_size):
        # Reset states for spiking layers for each validation sample
        for layer in model.layers:
            if isinstance(layer, SpikingNeuronLayer):
                layer.reset_state_per_sample()

        # Process one sample at a time
        x_sample = tf.expand_dims(x_batch[i], axis=0)
        y_sample = tf.expand_dims(y_batch[i], axis=0)

        predictions = model(x_sample, training=False)
        loss = loss_fn(y_sample, predictions)  # Calculate loss for each sample

        total_loss += loss

        # Ensure predictions and y_batch are of the same type
        predictions = tf.cast(predictions, tf.float32)
        y_sample = tf.cast(y_sample, tf.float32)

        # Calculate MAE using TensorFlow operations
        mae = tf.reduce_mean(tf.abs(y_sample - predictions))
        total_mae += mae

    # Calculate average loss and MAE for the batch
    avg_loss = total_loss / tf.cast(batch_size, dtype=total_loss.dtype)
    avg_mae = total_mae / tf.cast(batch_size, dtype=total_mae.dtype)

    return avg_loss, avg_mae


# Customizing the Adam optimizer for the custom training loop
optimizer = Adam(
    learning_rate = learning_rate, 
    beta_1 = beta_1,            # Momentum term (beta1)
    beta_2 = beta_2,            # Momentum term for the squared gradient (beta2)
    epsilon = epsilon,          # Small number to prevent any division by zero
    decay = decay,              # Weight decay for regularization
    amsgrad = amsgrad           # Whether to apply AMSGrad variant of Adam
)
# Changing the loss function to Mean Squared Error
loss_fn = tf.keras.losses.MeanAbsoluteError() 

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")

    # Initialize accumulators for loss and MAE
    total_train_loss = 0
    total_train_mae = 0
    total_val_loss = 0
    total_val_mae = 0
    train_samples = 0
    val_samples = 0

    # Training phase
    with tqdm(total=len(train_dataset), desc="Training", unit="batch") as pbar:
        for x_batch, y_batch in train_dataset:
            loss, mae = train_step(model, x_batch, y_batch, optimizer, loss_fn)
            total_train_loss += loss.numpy() * len(x_batch)
            total_train_mae += mae.numpy() * len(x_batch)
            train_samples += len(x_batch)

            # Update progress bar description
            pbar.set_description(f"Training - Loss: {total_train_loss / train_samples:.4f}, MAE: {total_train_mae / train_samples:.4f}")
            pbar.update(1)

    avg_train_loss = total_train_loss / train_samples
    avg_train_mae = total_train_mae / train_samples

    # Validation phase
    with tqdm(total=len(validation_dataset), desc="Validation", unit="batch") as pbar:
        for x_batch_val, y_batch_val in validation_dataset:
            loss, mae = validation_step(model, x_batch_val, y_batch_val, loss_fn)
            total_val_loss += loss.numpy() * len(x_batch_val)
            total_val_mae += mae.numpy() * len(x_batch_val)
            val_samples += len(x_batch_val)

            # Update progress bar description
            pbar.set_description(f"Validation - Loss: {total_val_loss / val_samples:.4f}, MAE: {total_val_mae / val_samples:.4f}")
            pbar.update(1)

    # Calculate average validation loss and MAE outside the 'with' block
    avg_val_loss = total_val_loss / val_samples if val_samples > 0 else 0
    avg_val_mae = total_val_mae / val_samples if val_samples > 0 else 0

    print(f"\nTraining Loss: {avg_train_loss}, Training MAE: {avg_train_mae}")
    print(f"Validation Loss: {avg_val_loss}, Validation MAE: {avg_val_mae}")

    # Assuming train_losses and val_losses are lists defined earlier
    train_losses.append(avg_train_mae)
    val_losses.append(avg_val_mae)

    # Update plot after each epoch
    update_plot(epoch, train_losses, val_losses)

     # Check for improvement
    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        no_improvement_epochs = 0  # Reset counter
        model.save_weights('best_model_15.h5')  # Save best model
    else:
        no_improvement_epochs += 1

    # Early stopping check
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break


    # Save plot data after each epoch
    with open('plot_data_15.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses), f)

# Close plot and load best model weights
plt.close()



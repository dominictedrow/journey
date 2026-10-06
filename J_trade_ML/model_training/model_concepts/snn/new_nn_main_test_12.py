import os
import pickle
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense, Flatten, Concatenate, Conv1D, LSTM, LayerNormalization, BatchNormalization, GlobalAveragePooling1D, ReLU, Multiply, AveragePooling1D, GlobalMaxPooling1D, MaxPooling1D, Reshape, PReLU
from tensorflow.keras.models import Model
from sklearn.metrics import mean_absolute_error
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt
from tqdm import tqdm

tf.config.experimental.set_visible_devices

# Mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

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

class ReversedDense(tf.keras.layers.Layer):
    def __init__(self, dense_layer, **kwargs):
        super(ReversedDense, self).__init__(**kwargs)
        self.dense_layer = dense_layer

    def call(self, inputs):
        reversed_inputs = tf.reverse(inputs, axis=[-1])
        return self.dense_layer(reversed_inputs)


def average_weights(blocks, num_layers_per_block, units_per_layer):
    # Function to average weights for each side (only Dense layers)
    avg_weights_side_1 = [np.zeros_like(layer.get_weights()[0]) for layer in blocks[0] if isinstance(layer, Dense)]
    avg_weights_side_2 = [np.zeros_like(layer.get_weights()[0]) for layer in blocks[0] if isinstance(layer, Dense)]

    # Averaging weights for Side 1
    for block in blocks:
        for i, layer in enumerate([l for l in block if isinstance(l, Dense)]):
            avg_weights_side_1[i] += layer.get_weights()[0]
    avg_weights_side_1 = [w / len(blocks) for w in avg_weights_side_1]

    # Averaging weights for Side 2 (reversed)
    for block in blocks:
        for i, layer in enumerate([l for l in block[::-1] if isinstance(l, Dense)]):
            avg_weights_side_2[i] += layer.get_weights()[0]
    avg_weights_side_2 = [w / len(blocks) for w in avg_weights_side_2]

    # Create new Dense layers with the correct number of units
    new_layers_side_1 = [Dense(units_per_layer, kernel_initializer='lecun_normal') for _ in range(num_layers_per_block)]
    new_layers_side_2 = [Dense(units_per_layer, kernel_initializer='lecun_normal') for _ in range(num_layers_per_block)]

    # Initialize weights by building layers with input shape
    dummy_input = np.zeros((1, units_per_layer))
    for layer in new_layers_side_1 + new_layers_side_2:
        layer(dummy_input)

    # Setting the new layers with averaged weights
    for i in range(num_layers_per_block):
        new_layers_side_1[i].set_weights([avg_weights_side_1[i], np.zeros_like(avg_weights_side_1[i][0])])
        new_layers_side_2[i].set_weights([avg_weights_side_2[i], np.zeros_like(avg_weights_side_2[i][0])])

    return new_layers_side_1, new_layers_side_2


def create_dual_input_model(input_shape, num_blocks, num_layers_per_block, units_per_layer, RNN):
    input_normal = Input(shape=(input_shape,), name='input_normal')

    # Initial transform layer without activation
    initial_transform_layer = Dense(units_per_layer, kernel_initializer='lecun_normal')(input_normal)
    x_transformed = PReLU()(initial_transform_layer)
    x_transformed = LayerNormalization()(x_transformed)

    # Shared layers for each block
    blocks = []
    for _ in range(num_blocks):
        block_layers = [Dense(units_per_layer, kernel_initializer='lecun_normal') for _ in range(num_layers_per_block)]
        blocks.append(block_layers)

    # Process each block
    block_outputs = []
    for block in blocks:
        x_input = x_transformed
        for _ in range(RNN):
            # Side 1 - Normal processing for block
            x_normal = x_input  # Start with the transformed and activated input or the previous output
            for layer in block:
                x_normal = layer(x_normal)    
                x_normal = PReLU()(x_normal)        
                x_normal = LayerNormalization()(x_normal)

            # Side 2 - Reversed processing for block
            x_reversed = x_normal
            for layer in block[::-1]:
                reversed_layer = ReversedDense(layer)
                x_reversed = reversed_layer(x_reversed)   
                x_reversed = PReLU()(x_reversed)          
                x_reversed = LayerNormalization()(x_reversed)

            # Add the output of side 2 back to the input
            x_input = x_input + x_reversed

        # Use the final output of side 2 as the block output
        block_outputs.append(x_input)

    """ # Averaging weights from all blocks
    new_layers_side_1, new_layers_side_2 = average_weights(blocks, num_layers_per_block, units_per_layer)

    # Mark new layers as non-trainable
    for layer in new_layers_side_1 + new_layers_side_2:
        layer.trainable = False """

    # Stack and average across all blocks
    stacked_across_blocks = tf.stack(block_outputs, axis=1) 
    x = GlobalMaxPooling1D()(stacked_across_blocks) # GlobalAveragePooling1D or GlobalMaxPooling1D
    x = x # + x_transformed
    
    """ # Process the input with the new layers (averaged weights)  GlobalAveragePooling1D
    # Side 1
    for layer in new_layers_side_1:
        x = layer(x)        
        x = PReLU()(x) 
        x = LayerNormalization()(x)

    # Side 2
    x_reversed = x
    for layer in new_layers_side_2:
        x_reversed = layer(x_reversed)
        x_reversed = PReLU()(x_reversed)        
        x_reversed = LayerNormalization()(x_reversed)

    # Stack and average across the outputs of the new NN structure (Second Pass)
    stacked_across_blocks_new = tf.stack([x, x_reversed], axis=1)       
    x_final = GlobalMaxPooling1D()(stacked_across_blocks_new) """
    x_final = x

    """ # Define the end_layers MLP
    mlp_output = x_final
    for i in range(end_layers):
        mlp_output = Dense(units_per_layer, kernel_initializer='lecun_normal')(mlp_output)
        mlp_output = PReLU()(mlp_output) 
        mlp_output = LayerNormalization()(mlp_output)
    # Final output
    x_final = mlp_output """

    output = Dense(1, activation='linear', kernel_initializer='lecun_normal')(x_final)
    model = Model(inputs=input_normal, outputs=output)

    """ # Identify trainable layers (assuming they are the layers in your blocks)
    trainable_layers = [layer for block in blocks for layer in block]

    # target_layers are the non-trainable layers with merged weights
    target_layers = new_layers_side_1 + new_layers_side_2 """

    # Return the model along with trainable_layers and target_layers
    return model # trainable_layers, target_layers
   
# Replace with your actual input shape and parameters
input_shape = X_train.shape[1]
num_blocks = 2
units_per_layer = 1500
num_layers_per_block = 6 
end_layers = 0
RNN = 3  # The number of times to recycle the output

epochs = 500 
batch_size = 128
patience = 50  # Define your patience for early stopping

learning_rate = 0.001  # Matched learning rate (default = 0.001)
beta_1=0.9             # Momentum term (beta1) (default = 0.9)
beta_2=0.999           # Momentum term for the squared gradient (beta2) (default = 0.999)
epsilon=1e-07          # Small number to prevent any division by zero (default = 1e-07)
decay=0.0              # Weight decay for regularization (default = 0.0)
amsgrad=True            # Whether to apply AMSGrad variant of Adam

""" # Create the model and get the layers
model, trainable_layers, target_layers = create_dual_input_model(input_shape, num_blocks, num_layers_per_block, units_per_layer) """

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

# Check if 'best_model_12.h5' exists
if os.path.exists('best_model_12.h5'):
    # Load weights if the file exists
    model.load_weights('best_model_12.h5')
    print("Weights loaded from 'best_model_12.h5'")
else:
    # Skip loading and possibly print a message or take other actions
    print("'best_model_12.h5' not found. Continuing without loading weights.")


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
    plt.plot(train_losses, label='Training MAE')
    plt.plot(val_losses, label='Validation MAE')
    plt.title(f'Epoch {epoch+1}')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.draw()  # Redraw the current figure
    plt.pause(0.01)  # Pause to update the figure
    plt.savefig('mae_plot_11.png')  # Save the plot with a constant filename

# Training loop
best_val_mae = float('inf')

no_improvement_epochs = 0

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model_12.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Cast target_mean and target_std to float32
target_mean_tf = tf.cast(target_mean, tf.float32)
target_std_tf = tf.cast(target_std, tf.float32)

def recompute_merged_weights(trainable_layers, target_layers):
    # Averaging the weights (kernels) and biases separately
    averaged_weights = np.mean([layer.get_weights()[0] for layer in trainable_layers], axis=0)  # Averaging kernels
    averaged_biases = np.mean([layer.get_weights()[1] for layer in trainable_layers], axis=0)   # Averaging biases

    # Update the target layers with these averaged weights
    for layer in target_layers:
        current_weights = layer.get_weights()
        
        # Ensure the shape of the averaged weights/biases matches the current weights/biases
        if current_weights[0].shape == averaged_weights.shape and current_weights[1].shape == averaged_biases.shape:
            layer.set_weights([averaged_weights, averaged_biases])
        else:
            raise ValueError(f"Weight shape mismatch in layer {layer.name}: expected {current_weights[0].shape}, got {averaged_weights.shape}")

# Define a custom training step
def train_step(model, x_batch, y_batch, optimizer, loss_fn, target_mean_tf, target_std_tf):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        loss = loss_fn(y_batch, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    """ # Recompute and update merged weights after each batch
    recompute_merged_weights(trainable_layers, target_layers) """

    # Cast predictions to float32
    predictions = tf.cast(predictions, tf.float32)

    # Unscale predictions and targets for MAE calculation
    predictions_unscaled = predictions * target_std_tf + target_mean_tf
    y_batch_unscaled = y_batch * target_std_tf + target_mean_tf

    # Calculate MAE using TensorFlow operations
    mae = tf.reduce_mean(tf.abs(y_batch_unscaled - predictions_unscaled))
    return loss, mae


@tf.function
def validation_step(model, x_batch, y_batch, loss_fn, target_mean_tf, target_std_tf):
    predictions = model(x_batch, training=False)
    loss = loss_fn(y_batch, predictions)

    # Cast predictions to float32
    predictions = tf.cast(predictions, tf.float32)

    # Unscale predictions and targets for MAE calculation
    predictions_unscaled = predictions * target_std_tf + target_mean_tf
    y_batch_unscaled = y_batch * target_std_tf + target_mean_tf

    # Calculate MAE using TensorFlow operations
    mae = tf.reduce_mean(tf.abs(y_batch_unscaled - predictions_unscaled))
    return loss, mae


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
            loss, mae = train_step(model, x_batch, y_batch, optimizer, loss_fn, target_mean_tf, target_std_tf)
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
            loss, mae = validation_step(model, x_batch_val, y_batch_val, loss_fn, target_mean_tf, target_std_tf)
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

    train_losses.append(avg_train_mae)
    val_losses.append(avg_val_mae)

    # Update plot after each epoch
    update_plot(epoch, train_losses, val_losses)

     # Check for improvement
    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        no_improvement_epochs = 0  # Reset counter
        model.save_weights('best_model_12.h5')  # Save best model
    else:
        no_improvement_epochs += 1

    # Early stopping check
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break


    # Save plot data after each epoch
    with open('plot_data.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses), f)

# Close plot and load best model weights
plt.close()



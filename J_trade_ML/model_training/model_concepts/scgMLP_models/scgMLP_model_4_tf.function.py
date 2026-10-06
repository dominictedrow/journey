import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm
import gc
from tensorflow.keras.models import Model
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Lambda, GlobalAveragePooling1D, GlobalAveragePooling2D, Flatten, Reshape, Concatenate, Subtract, Add, LayerNormalization, Conv1D
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from model_utils import save_model_weights, load_model_weights
import tkinter as tk

tf.config.experimental.set_visible_devices

# Mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

def is_prime(n):
    if n <= 1:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

def prime_density_around(n, range_width=10):
    lower_bound = max(2, n - range_width)
    upper_bound = n + range_width
    prime_count = sum(is_prime(x) for x in range(lower_bound, upper_bound + 1))
    total_count = upper_bound - lower_bound + 1
    return prime_count / total_count

def dynamic_min_learning_rate(prime_densities):
    min_density = min(prime_densities)
    return 0.00000001 * min_density  # Scale based on the lowest prime density

def get_largest_prime_below(n):
    """ Find the largest prime number less than or equal to n """
    for num in range(n, 1, -1):
        if is_prime(num):
            return num
    return None

def get_prime_numbers(limit):
    """ Generate a list of prime numbers less than the given limit """
    primes = []
    for num in range(2, limit):
        if is_prime(num):
            primes.append(num)
    return sorted(primes, reverse=True)  # Sort in descending order


# Adjusted function to update learning rates with a more gradual change
def update_learning_rates_advanced(current_batch_size, start_batch_size, original_lrs, prime_densities, smooth_factor):
    # Existing logic
    prime_density_current = prime_density_around(current_batch_size)
    prime_densities.append(prime_density_current)
    dynamic_min_lr = dynamic_min_learning_rate(prime_densities)
    scaling_factor = 1 / prime_density_current
    normalized_scaling_factor = (scaling_factor - 1) / (max(prime_densities) - 1) + 1

    # Adjusted scaling factor for more gradual changes
    adjusted_scaling_factor = 1 + (normalized_scaling_factor - 1) / smooth_factor

    # Apply adjusted scaling factor
    updated_lrs = [max(lr * adjusted_scaling_factor, dynamic_min_lr) for lr in original_lrs]
    return updated_lrs, prime_densities

# Initialize prime densities list
prime_densities = []

# ---Global training settings---
# If greater than 1, it acts as a multiplier for each epoch in the dynamic batch size mode.
# 1 is the default when start_batch_size > 0.
epochs =  1 # of epochs for training if batch_size > 0. If epochs = 0 & start_batch_size > 0 to skip training.
# Batch size for training. Set to 0 for dynamic batch sizing based on prime numbers less than start_batch_size.
batch_size = 0 # If greater than 0, this fixed batch size is used for all epochs.
# Initial batch size for dynamic batch sizing. This is the starting point for dynamic batch size calculation.
start_batch_size = 128 # Used only when batch_size is set to 0 for dynamic training.

# Unified number of layers and units for all 3 models
blocks = 1 # gMLP blocks
layers = 8 # dense layers per block
num_units = 600   # is like d_model
d_ffn = 1200 # Dimension of the FFN within each block of the gMLP models.

# ---Selective training settings---
smooth_factor = 100 # when start_batch_size > 0, this adjusts the scaling factor of lr vs decreasing primes (start_batch_size)
# 1st (left) gMLP model
gMLP_model_1_lr = 0.0001
# 3rd (middel) gMLP model
third_gMLP_model_lr = 0.0001
# 2nd (right) gMLP model
gMLP_model_2_lr = 0.0001

# passes are better to focus training on L/R models first, then train the middle modeles last.
model_1_2_passes = 0 # 0 skips training for model 1 & 2, > 0 creates multiple passes per epoch.
third_model_passes = 1  # 0 skips training for model 3, > 0 creates multiple passes per epoch. 

# Define the original learning rates for each model
original_lrs = [gMLP_model_1_lr, third_gMLP_model_lr, gMLP_model_2_lr]

# Set the initial current batch size to the largest prime number less than or equal to start_batch_size
current_batch_size = get_largest_prime_below(start_batch_size) if batch_size == 0 else batch_size

# Initialization of dynamic batch size and learning rates
if batch_size == 0:
    prime_numbers = get_prime_numbers(start_batch_size)
    print("Generated prime numbers:", prime_numbers)  # Debug statement
    current_batch_size = prime_numbers.pop(0)  # Start with the largest prime number less than start_batch_size
    print("Initial current_batch_size:", current_batch_size)  # Debug statement
else:
    current_batch_size = batch_size


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

# Before loading data
print("About to load training data...")
# Load and preprocess training data
df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//streamline_lag_close_(backup).db", 'XAUUSD')
# After loading data
print("Finished loading training data...")
print("Starting to preprocess training data...")
X_train, y_train, feature_scaler, target_scaler, target_mean, target_std = preprocess_data(df_train, feature_scaler, target_scaler)
print("Finished preprocessing training data...")

print("About to load validation data...")
# Load and preprocess validation data
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//val_streamline_lag_close_(backup).db", 'XAUUSD')
# After loading data
print("Finished loading training data...")
print("Starting to preprocess validation data...")
X_val, y_val = preprocess_validation_data(df_val, feature_scaler, target_scaler)
print("Finished preprocessing validation data...")

class RealTimeDashboard(Callback):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Real-time Dashboard')
        self.fig, self.ax = plt.subplots(1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.gMLP_1_losses = []
        self.gMLP_2_losses = []
        self.third_model_initial_losses = []
        self.final_model_losses = []
        self.validation_losses = []
        self.gMLP_1_val_losses = []  # Validation losses for gMLP_1
        self.gMLP_2_val_losses = []  # Validation losses for gMLP_2
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Append new loss values with error handling
        self.gMLP_1_losses.append(logs.get('gMLP_1_loss', 0))
        self.gMLP_2_losses.append(logs.get('gMLP_2_loss', 0))
        self.third_model_initial_losses.append(logs.get('third_model_initial_loss', 0))
        self.final_model_losses.append(logs.get('final_model_loss', 0))
        self.validation_losses.append(logs.get('val_loss', 0))
        self.gMLP_1_val_losses.append(logs.get('gMLP_1_val_loss', 0))
        self.gMLP_2_val_losses.append(logs.get('gMLP_2_val_loss', 0))

        self.epoch_count += 1

        self.ax.clear()
        self.ax.plot(range(self.epoch_count), self.gMLP_1_losses, label='gMLP 1 Loss')
        self.ax.plot(range(self.epoch_count), self.gMLP_2_losses, label='gMLP 2 Loss')
        self.ax.plot(range(self.epoch_count), self.third_model_initial_losses, label='Third Model Initial Loss')
        self.ax.plot(range(self.epoch_count), self.final_model_losses, label='Final Model Loss')
        self.ax.plot(range(self.epoch_count), self.validation_losses, label='Validation Loss')
        self.ax.plot(range(self.epoch_count), self.gMLP_1_val_losses, label='gMLP 1 Validation Loss')
        self.ax.plot(range(self.epoch_count), self.gMLP_2_val_losses, label='gMLP 2 Validation Loss')

        self.ax.legend()
        self.ax.set_title('Epoch vs Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')

        self.canvas.draw()
        self.root.update()

    def save_dashboard(self, filename):
        self.fig.savefig(filename)

dashboard = RealTimeDashboard()

print("About to initialize the model...")   

class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, d_ffn, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.gate = tf.keras.layers.Dense(d_ffn, activation='sigmoid', use_bias=False)

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        gated_output = self.gate(normalized_inputs)
        return gated_output * inputs

    def get_config(self):
        config = super(SpatialGatingUnit, self).get_config()
        config.update({"d_ffn": self.d_ffn})
        return config

class gMLPBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, d_ffn, layers, **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)
        self.num_units = num_units
        self.d_ffn = d_ffn
        self.layers = layers
        self.channel_projection_i = tf.keras.layers.Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal')
        self.dense_layers = [tf.keras.layers.Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal') for _ in range(self.layers)]
        self.sgu = SpatialGatingUnit(d_ffn)
        self.channel_projection_ii = tf.keras.layers.Dense(num_units, activation='tanh', kernel_initializer='lecun_normal')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        initial_inputs = inputs  # Store the initial input
        x = self.channel_projection_i(inputs)
        all_per_unit_outputs = []
        for layer in self.dense_layers:
            x = layer(x)
            x = self.sgu(x)  # Apply gating after each dense layer
            # Collect per-unit outputs
            per_unit_outputs = tf.split(x, num_or_size_splits=self.d_ffn, axis=-1)
            all_per_unit_outputs.append(per_unit_outputs)
        x = self.channel_projection_ii(x)
        return self.layer_norm(x + initial_inputs), all_per_unit_outputs  # Add the initial input here

    def get_config(self):
        config = super(gMLPBlock, self).get_config()
        config.update({
            "num_units": self.num_units,
            "d_ffn": self.d_ffn,
            "layers": self.layers
        })
        return config
   
   
class SpatialGatingUnit3(tf.keras.layers.Layer):
    def __init__(self, d_ffn, **kwargs):
        super(SpatialGatingUnit3, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.gate = tf.keras.layers.Dense(d_ffn, activation='sigmoid', use_bias=False)

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        gated_output = self.gate(normalized_inputs)
        return gated_output * inputs

    def get_config(self):
        config = super(SpatialGatingUnit3, self).get_config()
        config.update({"d_ffn": self.d_ffn})
        return config

class gMLPBlock3(tf.keras.layers.Layer):
    def __init__(self, num_units, d_ffn, layers, **kwargs):
        super(gMLPBlock3, self).__init__(**kwargs)
        self.num_units = num_units
        self.d_ffn = d_ffn
        self.layers = layers
        self.channel_projection_i = tf.keras.layers.Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal')
        self.dense_layers = [tf.keras.layers.Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal') for _ in range(self.layers)]
        self.sgu = SpatialGatingUnit3(d_ffn)
        self.channel_projection_ii = tf.keras.layers.Dense(num_units, activation='tanh', kernel_initializer='lecun_normal')
        self.layer_norm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, gMLP_1_unit_outputs, gMLP_2_unit_outputs):
        initial_inputs = inputs  # Store the initial input
        x = self.channel_projection_i(inputs)

        for i, layer in enumerate(self.dense_layers):
            # Reshape x to have three dimensions: [batch_size, 1, num_features]
            x = tf.reshape(x, [-1, 1, self.d_ffn])  # Now x has shape [batch_size, 1, num_features]

            # Process and concatenate outputs from the first two models
            combined_outputs = tf.concat([gMLP_1_unit_outputs[i], gMLP_2_unit_outputs[i]], axis=-1)
            # Ensure combined_outputs is reshaped correctly
            combined_outputs_reshaped = tf.reshape(combined_outputs, [-1, 1, self.d_ffn * 2])  # Adjust shape accordingly
            # Ensure the batch sizes are the same before concatenation
            batch_size = tf.shape(x)[0]
            combined_outputs_reshaped = combined_outputs_reshaped[:batch_size, :, :]
            
            # Concatenate along the last axis
            x = tf.concat([x, combined_outputs_reshaped], axis=2)
            x = layer(x)
            x = self.sgu(x)

        x = self.channel_projection_ii(x)
        # Reshape initial_inputs to match x's shape
        initial_inputs = tf.reshape(initial_inputs, [-1, 1, self.num_units])
        return self.layer_norm(x + initial_inputs)


    def get_config(self):
        config = super(gMLPBlock3, self).get_config()
        config.update({
            "num_units": self.num_units,
            "d_ffn": self.d_ffn,
            "layers": self.layers
        })
        return config
            

class ThirdgMLPModel(tf.keras.Model):
    def __init__(self, num_units, d_ffn, blocks, layers, **kwargs):
        super(ThirdgMLPModel, self).__init__(**kwargs)
        self.num_units = num_units
        self.d_ffn = d_ffn
        self.blocks = blocks
        self.initial_dense_layer = Dense(num_units, activation='linear', name='initial_dense_layer')        
        self.gMLP_blocks = [gMLPBlock3(num_units, d_ffn, layers, name=f'gmlp_block_{i}') for i in range(blocks)]
        self.block_layer_norms = [LayerNormalization(name=f'block_{i}_layer_norm') for i in range(blocks)]
        self.global_avg_pool = GlobalAveragePooling1D()  # Global Average Pooling layer
        self.output_layer = Dense(1, activation='linear', name='output_layer')

    def call(self, gMLP_1_output, gMLP_2_output, gMLP_1_unit_outputs, gMLP_2_unit_outputs, training=False):         
       
        # Ensure both tensors are of the same data type
        gMLP_1_output = tf.cast(gMLP_1_output, dtype=tf.float32)
        gMLP_2_output = tf.cast(gMLP_2_output, dtype=tf.float32)
       
        # Stack gMLP_1_output and gMLP_2_output as separate features
        combined_gMLP_output = tf.concat([gMLP_1_output, gMLP_2_output], axis=-1)
        
        x = self.initial_dense_layer(combined_gMLP_output)
        # x_skip = x  # Preserve the initial layer output
        
        for i, block in enumerate(self.gMLP_blocks):
            x = block(x, gMLP_1_unit_outputs[i], gMLP_2_unit_outputs[i], training=training)            
            x = self.block_layer_norms[i](x)

        #x += x_skip  # Add the preserved output of the initial layer
        x = self.global_avg_pool(x)
        final_output = self.output_layer(x)

        return final_output
       
   
def create_gMLP_model(num_units, d_ffn, blocks, input_shape, layers):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(num_units, activation='linear')(inputs)

    all_blocks_per_unit_outputs = []
    for _ in range(blocks):
        x, block_per_unit_outputs = gMLPBlock(num_units, d_ffn, layers)(x)
        all_blocks_per_unit_outputs.append(block_per_unit_outputs)
        x = tf.keras.layers.LayerNormalization()(x)    

    x = tf.keras.layers.Reshape((1, -1))(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, all_blocks_per_unit_outputs])



def create_gMLP_model_2(num_units, d_ffn, blocks, input_shape, layers):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Dense(num_units, activation='linear')(inputs)

    all_blocks_per_unit_outputs = []
    for _ in range(blocks):
        x, block_per_unit_outputs = gMLPBlock(num_units, d_ffn, layers)(x)
        all_blocks_per_unit_outputs.append(block_per_unit_outputs)
        x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Reshape((1, -1))(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    outputs = tf.keras.layers.Dense(1, activation='linear')(x)
    return tf.keras.Model(inputs=inputs, outputs=[outputs, all_blocks_per_unit_outputs])

# Instantiate and compile models
input_shape = X_train.shape[1:]  # Shape of preprocessed training data
extended_input_shape = (X_train.shape[1] + 1,)  # gMLP_model_1 outputs a single value per input

gMLP_model_1 = create_gMLP_model(num_units, d_ffn, blocks, input_shape, layers)
gMLP_model_2 = create_gMLP_model_2(num_units, d_ffn, blocks, extended_input_shape, layers)
third_gMLP_model = ThirdgMLPModel(num_units=num_units, d_ffn=d_ffn, blocks=blocks, layers=layers)

# Compile models
gMLP_model_1.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=gMLP_model_1_lr))
gMLP_model_2.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=gMLP_model_2_lr))
third_gMLP_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=third_gMLP_model_lr))

# Model summaries (third_gMLP_model cannot be summerized until training starts)
print("\nFirst gMLP Model Summary:")
gMLP_model_1.summary()

print("\nSecond gMLP Model Summary:")
gMLP_model_2.summary()

# Callback function to save the best model
gMLP_model_1_checkpoint_path = "C://Users//crgon//OneDrive//Desktop//trading//model_testing_text//training_models//save_models_weights//scgMLP_model_3//gMLP_model_1_high.h5"
gMLP_model_2_checkpoint_path = "C://Users//crgon//OneDrive//Desktop//trading//model_testing_text//training_models//save_models_weights//scgMLP_model_3//gMLP_model_2_high.h5"

def save_best_model(epoch, val_loss, gMLP_model_1, gMLP_model_2, third_gMLP_model):
    global best_val_loss
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
        best_val_loss = val_loss
        try:
            # Save the entire models
            gMLP_model_1.save(gMLP_model_1_checkpoint_path)
            print(f"gMLP Model 1 saved at {gMLP_model_1_checkpoint_path}")

            gMLP_model_2.save(gMLP_model_2_checkpoint_path)
            print(f"gMLP Model 2 saved at {gMLP_model_2_checkpoint_path}")

            # Save the subclassed model weights
            third_gMLP_model_weights_path = "C://Users//crgon//OneDrive//Desktop//trading//model_testing_text//training_models//save_models_weights//scgMLP_model_3//gMLP_model_3_high.h5"
            save_model_weights(third_gMLP_model, third_gMLP_model_weights_path)
            print(f"gMLP Model 3 (middle) saved at {third_gMLP_model_weights_path}")
       

        except Exception as e:
            print(f"Failed to save models: {e}")
            return False  # save operation failed
        return True  # successful save
    else:
        print(f"No improvement in validation loss to save the model (current: {val_loss}, best: {best_val_loss})")
        print("\n")
        return False  # Indicate that there was no improvement

# Load pre-trained weights
try:
    # Load the entire gMLP model
    custom_objects = {
        'SpatialGatingUnit': SpatialGatingUnit,
        'gMLPBlock': gMLPBlock
        # 'PositionalEncoding': PositionalEncoding                      
    }
    gMLP_model_1 = tf.keras.models.load_model(gMLP_model_1_checkpoint_path, custom_objects=custom_objects)
    print("Loaded gMLP Model 1.")

    gMLP_model_2 = tf.keras.models.load_model(gMLP_model_2_checkpoint_path, custom_objects=custom_objects)
    print("Loaded gMLP Model 2.")    

except Exception as e:
    print("Error occurred while loading models:", e)

# Define the Mean Squared Error loss function
loss_function = MeanAbsoluteError()

# Custom function to calculate MAE and MSE
def calculate_metrics(actual, predicted):
    mae = tf.keras.losses.MeanAbsoluteError()
    mse = MeanSquaredError()
    mae.update_state(actual, predicted)
    mse.update_state(actual, predicted)
    return mae.result().numpy(), mse.result().numpy()

# Define optimizers for each model
optimizer_gMLP_1 = tf.keras.optimizers.Adam(learning_rate=gMLP_model_1_lr)
optimizer_gMLP_2 = tf.keras.optimizers.Adam(learning_rate=gMLP_model_2_lr)
optimizer_third_gMLP = tf.keras.optimizers.Adam(learning_rate=third_gMLP_model_lr)

best_val_loss = float('inf')

def reshape_gMLP_outputs(gMLP_outputs, num_units):
    reshaped_outputs = []
    for block_output in gMLP_outputs:  # Each block_output is a list of lists of tensors
        block_reshaped_outputs = []
        for layer_output in block_output:  # Each layer_output is a list of tensors
            layer_reshaped_outputs = []
            for output in layer_output:  # output is a tensor
                output_rank = len(output.shape)  # Get the rank of the output tensor
                # Construct a multiples argument with 1s for all dimensions except the last
                multiples = [1] * (output_rank - 1) + [num_units]
                tiled_output = tf.tile(output, multiples)
                # Reshape the tiled output to match the desired shape
                reshaped_output = tf.reshape(tiled_output, [-1] + [num_units] * (output_rank - 1))
                layer_reshaped_outputs.append(reshaped_output)
            block_reshaped_outputs.append(layer_reshaped_outputs)
        reshaped_outputs.append(block_reshaped_outputs)
    return reshaped_outputs


@tf.function
def combined_step(X_batch, y_batch, is_training, models, optimizers, reshape_gMLP_outputs, num_units, third_model_passes, model_1_2_passes, target_mean, target_std):
    gMLP_model_1, gMLP_model_2, third_gMLP_model = models
    optimizer_gMLP_1, optimizer_gMLP_2, optimizer_third_gMLP = optimizers   

    gMLP_1_loss, gMLP_2_loss, third_model_initial_loss, final_model_loss = 0, 0, 0, 0

    if is_training:
        # Processing for gMLP_model_1
        with tf.GradientTape() as tape_gMLP_1:
            gMLP_1_output, gMLP_1_unit_outputs = gMLP_model_1(X_batch, training=True)
            gMLP_1_loss = loss_function(y_batch, gMLP_1_output)        

        # Processing for gMLP_model_2
        with tf.GradientTape() as tape_gMLP_2:
            # Modify input for gMLP_model_2
            gMLP_1_output_for_gMLP_2 = tf.squeeze(gMLP_1_output, axis=[-1, -2]) if len(gMLP_1_output.shape) == 3 else gMLP_1_output
            gMLP_1_output_for_gMLP_2 = tf.cast(gMLP_1_output_for_gMLP_2, tf.float32)
            X_batch_gMLP_2 = tf.concat([X_batch, gMLP_1_output_for_gMLP_2], axis=-1)

            gMLP_2_output, gMLP_2_unit_outputs = gMLP_model_2(X_batch_gMLP_2, training=True)
            gMLP_2_loss = loss_function(y_batch, gMLP_2_output)

        # Reshape unit outputs for both gMLP models
        gMLP_1_unit_outputs_reshaped = reshape_gMLP_outputs(gMLP_1_unit_outputs, num_units)
        gMLP_2_unit_outputs_reshaped = reshape_gMLP_outputs(gMLP_2_unit_outputs, num_units)        
        
        # Apply gradients for gMLP_model_1 and gMLP_model_2 if model_1_2_passes > 0
        if model_1_2_passes > 0:
            gradients_gMLP_1 = tape_gMLP_1.gradient(gMLP_1_loss, gMLP_model_1.trainable_variables)
            optimizer_gMLP_1.apply_gradients(zip(gradients_gMLP_1, gMLP_model_1.trainable_variables))

            gradients_gMLP_2 = tape_gMLP_2.gradient(gMLP_2_loss, gMLP_model_2.trainable_variables)
            optimizer_gMLP_2.apply_gradients(zip(gradients_gMLP_2, gMLP_model_2.trainable_variables))

        # This is a filler to avoid is used before assignment
        final_model_output = 0

        # Processing for third_gMLP_model
        if third_model_passes > 0:
            gradient_accumulator = [tf.zeros_like(variable) for variable in third_gMLP_model.trainable_variables]
            for _ in range(third_model_passes):
                with tf.GradientTape() as tape_third_model:
                    gMLP_2_output_for_third_model = tf.cast(gMLP_2_output, tf.float32)
                    gMLP_1_output_for_third_model = tf.cast(gMLP_1_output, tf.float32)
                    third_model_output = third_gMLP_model(gMLP_1_output_for_third_model, gMLP_2_output_for_third_model, gMLP_1_unit_outputs_reshaped, gMLP_2_unit_outputs_reshaped, training=True)                           
                    third_model_initial_loss = loss_function(y_batch, third_model_output)                    
                    final_model_output = third_model_output             
                    final_model_loss = loss_function(y_batch, final_model_output)

                gradients_third_model = tape_third_model.gradient(third_model_initial_loss, third_gMLP_model.trainable_variables)
                gradient_accumulator = [acc_grad + grad for acc_grad, grad in zip(gradient_accumulator, gradients_third_model)]

            # Average the accumulated gradients
            averaged_gradients = [grad / third_model_passes for grad in gradient_accumulator]
            optimizer_third_gMLP.apply_gradients(zip(averaged_gradients, third_gMLP_model.trainable_variables))  

       
        # Ensure all tensors are of the same type, preferably float32
        gMLP_1_output = tf.cast(gMLP_1_output, tf.float32)
        gMLP_2_output = tf.cast(gMLP_2_output, tf.float32)
        
        final_model_output = tf.cast(final_model_output, tf.float32)

        # Cast target_mean and target_std to float32
        target_mean_tf = tf.cast(target_mean, tf.float32)
        target_std_tf = tf.cast(target_std, tf.float32)      

        # Unscaled outputs and recalculated losses
        gMLP_1_output_unscaled = gMLP_1_output * target_std_tf + target_mean_tf
        gMLP_2_output_unscaled = gMLP_2_output * target_std_tf + target_mean_tf
        final_model_output_unscaled = final_model_output * target_std_tf + target_mean_tf        
        y_batch_unscaled = y_batch * target_std_tf + target_mean_tf

        # Recalculate losses with the unscaled outputs
        gMLP_1_loss = loss_function(y_batch_unscaled, gMLP_1_output_unscaled)
        gMLP_2_loss = loss_function(y_batch_unscaled, gMLP_2_output_unscaled)
        
        if third_model_passes > 0:
            third_model_initial_loss = loss_function(y_batch_unscaled, final_model_output_unscaled)            
            final_model_loss = loss_function(y_batch_unscaled, final_model_output_unscaled)

        del tape_third_model

        return gMLP_1_loss, gMLP_2_loss, third_model_initial_loss, final_model_loss

    else:
        # Validation logic
        gMLP_1_val_output, gMLP_1_unit_outputs_val = gMLP_model_1(X_batch, training=False)
        gMLP_1_val_output = tf.cast(gMLP_1_val_output, tf.float32)  # Cast to float32
        gMLP_1_val_output_reshaped = tf.squeeze(gMLP_1_val_output, axis=[-1, -2]) if len(gMLP_1_val_output.shape) == 3 else gMLP_1_val_output
        gMLP_1_unit_outputs_val_reshaped = reshape_gMLP_outputs(gMLP_1_unit_outputs_val, num_units)
        
        X_batch = tf.cast(X_batch, tf.float32)  # Ensure X_batch is also cast to float32
        X_batch_gMLP_2_val = tf.concat([X_batch, gMLP_1_val_output_reshaped], axis=-1)

        gMLP_2_val_output, gMLP_2_unit_outputs_val = gMLP_model_2(X_batch_gMLP_2_val, training=False)
        gMLP_2_val_output = tf.cast(gMLP_2_val_output, tf.float32)  # Cast to float32
        gMLP_2_val_output_reshaped = tf.squeeze(gMLP_2_val_output, axis=[-1, -2]) if len(gMLP_2_val_output.shape) == 3 else gMLP_2_val_output        
        gMLP_2_unit_outputs_val_reshaped = reshape_gMLP_outputs(gMLP_2_unit_outputs_val, num_units)

         # Cast target_mean and target_std to float32 for consistency
        target_mean_tf = tf.cast(target_mean, tf.float32)
        target_std_tf = tf.cast(target_std, tf.float32)

        # Unscaled outputs for validation
        y_batch_unscaled = y_batch * target_std_tf + target_mean_tf
        gMLP_1_output_unscaled = gMLP_1_val_output * target_std_tf + target_mean_tf
        gMLP_2_output_unscaled = gMLP_2_val_output * target_std_tf + target_mean_tf

        # Calculate the unscaled validation loss for gMLP_model_1 and gMLP_model_2
        gMLP_1_val_loss = loss_function(y_batch_unscaled, gMLP_1_output_unscaled)
        gMLP_2_val_loss = loss_function(y_batch_unscaled, gMLP_2_output_unscaled)
        
        # Calculate validation loss for third model if it's used
        if third_model_passes > 0:
            # Ensure gMLP_2_val_output is in the correct shape for third_gMLP_model
            third_model_val_output = third_gMLP_model(gMLP_1_val_output_reshaped, gMLP_2_val_output_reshaped, gMLP_1_unit_outputs_val_reshaped, gMLP_2_unit_outputs_val_reshaped, training=False)
            third_model_val_output = tf.cast(third_model_val_output, tf.float32)  # Cast to float32

            # Cast target_mean and target_std to float32 for consistency
            target_mean_tf = tf.cast(target_mean, tf.float32)
            target_std_tf = tf.cast(target_std, tf.float32)

            # Unscaled outputs            
            final_model_val_output_unscaled = third_model_val_output * target_std_tf + target_mean_tf

            y_batch_unscaled = y_batch * target_std_tf + target_mean_tf

            # Calculate the unscaled validation loss
            third_model_val_loss  = loss_function(y_batch_unscaled, final_model_val_output_unscaled)
            return third_model_val_loss 
        else:
            return gMLP_1_val_loss, gMLP_2_val_loss
               

third_gMLP_model_weights_path = "C://Users//crgon//OneDrive//Desktop//trading//model_testing_text//training_models//save_models_weights//scgMLP_model_3//gMLP_model_3_high.h5"

# Main training and validation loop
if epochs > 0:
    best_val_loss = float('inf')

    # Prepare training batches based on the current batch size
    training_batches = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(current_batch_size)

    # Always initialize third_gMLP_model
    X_batch, y_batch = next(iter(training_batches))  # Get the first batch
    gMLP_1_output, gMLP_1_unit_outputs = gMLP_model_1(X_batch, training=False)
    gMLP_1_output_for_gMLP_2 = tf.squeeze(gMLP_1_output, axis=[-1, -2]) if len(gMLP_1_output.shape) == 3 else gMLP_1_output
    gMLP_1_output_for_gMLP_2 = tf.cast(gMLP_1_output_for_gMLP_2, tf.float32)
    X_batch_gMLP_2 = tf.concat([X_batch, gMLP_1_output_for_gMLP_2], axis=-1)
    gMLP_2_output, gMLP_2_unit_outputs = gMLP_model_2(X_batch_gMLP_2, training=False)

    # Reshape unit outputs for both gMLP models
    gMLP_1_unit_outputs_reshaped = reshape_gMLP_outputs(gMLP_1_unit_outputs, num_units)
    gMLP_2_unit_outputs_reshaped = reshape_gMLP_outputs(gMLP_2_unit_outputs, num_units)

    # Call third_gMLP_model with reshaped outputs
    third_gMLP_model(gMLP_1_output, gMLP_2_output, gMLP_1_unit_outputs_reshaped, gMLP_2_unit_outputs_reshaped, training=False)

    # Print summary of third_gMLP_model
    print("\nSummary of third_gMLP_model:")
    third_gMLP_model.summary()

    # Load model weights if the file exists
    if os.path.exists(third_gMLP_model_weights_path):
        load_model_weights(third_gMLP_model, third_gMLP_model_weights_path)
        print("Loaded gMLP Model 3.")
    else:
        print(f"Weights file not found at {third_gMLP_model_weights_path}, skipping weight gMLP model 3 loading.")

    # Determine the total number of epochs to iterate
    total_epochs = start_batch_size * epochs if batch_size == 0 else epochs    

    # Adjusted loop for calculating batch size and updating learning rates
    for epoch in range(total_epochs):
        if batch_size == 0:
            # Decrementing the batch size each epoch and checking for prime
            potential_batch_size = start_batch_size - epoch
            if is_prime(potential_batch_size) and potential_batch_size < current_batch_size:
                current_batch_size = potential_batch_size
                print("Updated current_batch_size:", current_batch_size)

                # Update learning rates
                updated_lrs, prime_densities = update_learning_rates_advanced(
                    current_batch_size, start_batch_size, original_lrs, prime_densities, smooth_factor
                )
                optimizer_gMLP_1.learning_rate, optimizer_gMLP_2.learning_rate, optimizer_third_gMLP.learning_rate = updated_lrs
                print(f"Epoch {epoch + 1}, Dynamic Batch Size Update: Updated batch size to {current_batch_size} and learning rates to {updated_lrs}")

                # Update training batches
                training_batches = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(current_batch_size)

         # Prepare training batches based on the current batch size for each epoch
        tqdm_training_batches = tqdm(training_batches, total=len(X_train) // current_batch_size, desc=f"Epoch {epoch // epochs + 1}/{total_epochs} Training")
        print("Epoch:", epoch + 1, "Current batch size:", current_batch_size, "Total training steps:", len(X_train) // current_batch_size)

        # Reset accumulators for losses
        total_gMLP_1_loss, total_gMLP_2_loss, total_third_model_loss, total_final_model_loss = 0, 0, 0, 0
        total_val_loss = 0
        num_batches = 0

        # Training loop
        for X_batch, y_batch in tqdm_training_batches:
            losses = combined_step(
                X_batch, y_batch, True,
                [gMLP_model_1, gMLP_model_2, third_gMLP_model],
                [optimizer_gMLP_1, optimizer_gMLP_2, optimizer_third_gMLP],
                reshape_gMLP_outputs, num_units,
                third_model_passes, model_1_2_passes,  # Include model_1_2_passes here
                target_mean, target_std
            )

            # Only update gMLP 1 and gMLP 2 losses if model_1_2_passes > 0
            if model_1_2_passes > 0:
                total_gMLP_1_loss += losses[0].numpy()
                total_gMLP_2_loss += losses[1].numpy()

            if third_model_passes > 0:
                total_third_model_loss += losses[2].numpy()
                total_final_model_loss += losses[3].numpy()

            num_batches += 1

        # Validation
        validation_batches = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(current_batch_size if batch_size == 0 else batch_size)

        # Before the validation loop
        num_val_batches = len(X_val) // (current_batch_size if batch_size == 0 else batch_size)

        # Before the validation loop
        avg_gMLP_1_val_loss = 0
        avg_gMLP_2_val_loss = 0

         # Validation loop
        if third_model_passes > 0:
            total_third_model_val_loss = 0
            for X_val_batch, y_val_batch in validation_batches:
                third_model_val_loss = combined_step(
                    X_val_batch, y_val_batch, False,
                    [gMLP_model_1, gMLP_model_2, third_gMLP_model],
                    [optimizer_gMLP_1, optimizer_gMLP_2, optimizer_third_gMLP],
                    reshape_gMLP_outputs, num_units,
                    third_model_passes, model_1_2_passes,  # Include model_1_2_passes here
                    target_mean, target_std
            )
                total_third_model_val_loss += third_model_val_loss.numpy()

            # Calculate and print average losses
            average_third_model_val_loss = total_third_model_val_loss / num_val_batches
            average_val_loss = average_third_model_val_loss  # Validation loss from the third model
            print(f"Epoch {epoch // epochs + 1}, Repeat {epoch % epochs + 1}: Average Validation Loss: {average_val_loss}")

        # Inside the validation loop
        else:
            total_gMLP_1_val_loss, total_gMLP_2_val_loss = 0, 0
            for X_val_batch, y_val_batch in validation_batches:
                gMLP_1_val_loss, gMLP_2_val_loss = combined_step(
                    X_val_batch, y_val_batch, False,  # Pass the validation batch and False for is_training
                    [gMLP_model_1, gMLP_model_2, third_gMLP_model],  # List of models
                    [optimizer_gMLP_1, optimizer_gMLP_2, optimizer_third_gMLP],  # List of optimizers (not used in validation, but required by the function)
                    reshape_gMLP_outputs, num_units,  # Other necessary parameters
                    third_model_passes, model_1_2_passes,  # Passes control
                    target_mean, target_std  # Scaling parameters
                )
                total_gMLP_1_val_loss += gMLP_1_val_loss.numpy()
                total_gMLP_2_val_loss += gMLP_2_val_loss.numpy()

            avg_gMLP_1_val_loss = total_gMLP_1_val_loss / num_val_batches
            avg_gMLP_2_val_loss = total_gMLP_2_val_loss / num_val_batches
            average_val_loss = (avg_gMLP_1_val_loss + avg_gMLP_2_val_loss) / 2  # Average loss from the first two models

        # Adjust print statements
        if third_model_passes == 0:
            # Existing print statement for third_model_passes == 0
            print(f"Epoch {epoch + 1}/{epochs}, gMLP 1 Training Loss: {total_gMLP_1_loss / num_batches:.4f}, "
                f"gMLP 2 Training Loss: {total_gMLP_2_loss / num_batches:.4f}, "
                f"gMLP 1 Validation Loss: {avg_gMLP_1_val_loss:.4f}, "
                f"gMLP 2 Validation Loss: {avg_gMLP_2_val_loss:.4f}")
        elif third_model_passes > 0 and model_1_2_passes > 0:
            # New print statement when both third_model_passes and model_1_2_passes are 1 or higher
            print(f"Epoch {epoch + 1}/{epochs}, gMLP 1 Training Loss: {total_gMLP_1_loss / num_batches:.4f}, "
                f"gMLP 2 Training Loss: {total_gMLP_2_loss / num_batches:.4f}, "
                f"Third Model Training Loss: {total_third_model_loss / num_batches:.4f}, "
                f"Final Model Training Loss: {total_final_model_loss / num_batches:.4f}, "
                f"Validation Loss: {average_val_loss:.4f}")
        else:
            # Existing print statement for other cases
            print(f"Epoch {epoch + 1}/{epochs}, "                
                f"Third Model Training Loss: {total_third_model_loss / num_batches:.4f}, "
                f"Final Model Training Loss: {total_final_model_loss / num_batches:.4f}, "
                f"Third Model Validation Loss: {average_third_model_val_loss:.4f}")
    
       # At the end of each epoch in your training loop
        if third_model_passes > 0:
            dashboard.on_epoch_end(epoch, logs={
                'gMLP_1_loss': total_gMLP_1_loss / num_batches,
                'gMLP_2_loss': total_gMLP_2_loss / num_batches,                
                'final_model_loss': total_final_model_loss / num_batches,
                'val_loss': average_val_loss,  # Validation loss from the third model
                'gMLP_1_val_loss': avg_gMLP_1_val_loss,  # Validation loss for gMLP_1
                'gMLP_2_val_loss': avg_gMLP_2_val_loss   # Validation loss for gMLP_2
            })
        else:
            dashboard.on_epoch_end(epoch, logs={
                'gMLP_1_loss': total_gMLP_1_loss / num_batches,
                'gMLP_2_loss': total_gMLP_2_loss / num_batches,
                'gMLP_1_val_loss': avg_gMLP_1_val_loss,  # Validation loss for gMLP_1
                'gMLP_2_val_loss': avg_gMLP_2_val_loss   # Validation loss for gMLP_2
            })

        # At the end of the epoch, call save_best_model
        if save_best_model(epoch, average_val_loss, gMLP_model_1, gMLP_model_2, third_gMLP_model):
            print(f"Model saved successfully at epoch {epoch + 1}.")
            print("\n")
    
        # Clear session and garbage collection
        tf.keras.backend.clear_session()
        gc.collect()

    else:
        print("Skipping training loop as epochs is 0.")

# Check if weights are loaded
weights_loaded = False

# Load weights for third_gMLP_model if not already loaded
if not weights_loaded:
    if os.path.exists(third_gMLP_model_weights_path):
        # Prepare training batches based on the current batch size
        training_batches = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(current_batch_size)
        X_batch, y_batch = next(iter(training_batches))  # Get the first batch
        gMLP_1_output, gMLP_1_unit_outputs = gMLP_model_1(X_batch, training=False)
        gMLP_1_output_for_gMLP_2 = tf.squeeze(gMLP_1_output, axis=[-1, -2]) if len(gMLP_1_output.shape) == 3 else gMLP_1_output
        gMLP_1_output_for_gMLP_2 = tf.cast(gMLP_1_output_for_gMLP_2, tf.float32)
        X_batch_gMLP_2 = tf.concat([X_batch, gMLP_1_output_for_gMLP_2], axis=-1)
        gMLP_2_output, gMLP_2_unit_outputs = gMLP_model_2(X_batch_gMLP_2, training=False)

        # Reshape unit outputs for both gMLP models
        gMLP_1_unit_outputs_reshaped = reshape_gMLP_outputs(gMLP_1_unit_outputs, num_units)
        gMLP_2_unit_outputs_reshaped = reshape_gMLP_outputs(gMLP_2_unit_outputs, num_units)

        # Call third_gMLP_model with reshaped outputs
        third_gMLP_model(gMLP_1_output, gMLP_2_output, gMLP_1_unit_outputs_reshaped, gMLP_2_unit_outputs_reshaped, training=False)

        # Load model weights
        load_model_weights(third_gMLP_model, third_gMLP_model_weights_path)
        print("Loaded gMLP Model 3.")
        weights_loaded = True
    else:
        print(f"Weights file not found at {third_gMLP_model_weights_path}, skipping weight loading for gMLP model 3.")

# Initialize metrics calculators
mae_calculator = tf.keras.metrics.MeanAbsoluteError()
mse_calculator = tf.keras.metrics.MeanSquaredError()

# Cast target_mean and target_std to float32 for unscaled calculations
target_mean_tf = tf.cast(target_mean, tf.float32)
target_std_tf = tf.cast(target_std, tf.float32)

# Prepare DataFrame for storing results
result_df_columns = ["Actual", "gMLP_1_Predictions", "gMLP_2_Predictions", "Third_Model_Predictions", "Final_Model_Predictions",
                     "gMLP_1_Error", "gMLP_2_Error", "Final_Model_Error"]
result_df = pd.DataFrame(columns=result_df_columns)

# Process validation data in batches
validation_batches = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
for X_val_batch, y_val_batch in validation_batches:
    # Predict with the first gMLP model
    gMLP_1_val_output, gMLP_1_unit_outputs_val = gMLP_model_1(X_val_batch, training=False)
    gMLP_1_val_output = tf.cast(gMLP_1_val_output, tf.float32)  # Cast to float32
    gMLP_1_val_output_reshaped = tf.squeeze(gMLP_1_val_output, axis=[-1, -2]) if len(gMLP_1_val_output.shape) == 3 else gMLP_1_val_output
    X_val_batch = tf.cast(X_val_batch, tf.float32) 
    gMLP_1_unit_outputs_val_reshaped = reshape_gMLP_outputs(gMLP_1_unit_outputs_val, num_units)

    # Modify input for gMLP_model_2 prediction
    X_val_gMLP_2 = tf.concat([X_val_batch, gMLP_1_val_output_reshaped], axis=-1)

    # Predict with the second gMLP model
    gMLP_2_val_output, gMLP_2_unit_outputs_val = gMLP_model_2(X_val_gMLP_2, training=False)
    gMLP_2_val_output_reshaped = tf.squeeze(gMLP_2_val_output, axis=[-1, -2]) if len(gMLP_2_val_output.shape) == 3 else gMLP_2_val_output
    gMLP_2_unit_outputs_val_reshaped = reshape_gMLP_outputs(gMLP_2_unit_outputs_val, num_units)

    # Predict with the third gMLP model
    third_model_val_output = third_gMLP_model(gMLP_1_val_output_reshaped, gMLP_2_val_output_reshaped, gMLP_1_unit_outputs_val_reshaped, gMLP_2_unit_outputs_val_reshaped, training=False)

    # Cast outputs to float32 and compute final predictions
    gMLP_1_val_output_casted = tf.cast(gMLP_1_val_output_reshaped, tf.float32)
    gMLP_2_val_output_casted = tf.cast(gMLP_2_val_output_reshaped, tf.float32)
    third_model_val_output_casted = tf.cast(third_model_val_output, tf.float32)
    final_predictions_unscaled = third_model_val_output_casted * target_std_tf + target_mean_tf

    # Aggregate results in DataFrame with unscaled values
    batch_df = pd.DataFrame({
        "Actual": (y_val_batch.numpy() * target_std_tf + target_mean_tf).numpy().flatten(),
        "gMLP_1_Predictions": (gMLP_1_val_output_casted.numpy() * target_std_tf + target_mean_tf).numpy().flatten(),
        "gMLP_2_Predictions": (gMLP_2_val_output_casted.numpy() * target_std_tf + target_mean_tf).numpy().flatten(),
        "Third_Model_Predictions": (third_model_val_output_casted.numpy() * target_std_tf + target_mean_tf).numpy().flatten(),
        "Final_Model_Predictions": final_predictions_unscaled.numpy().flatten(),
        "gMLP_1_Error": ((y_val_batch.numpy() * target_std_tf + target_mean_tf) - (gMLP_1_val_output_casted.numpy() * target_std_tf + target_mean_tf)).numpy().flatten(),
        "gMLP_2_Error": ((y_val_batch.numpy() * target_std_tf + target_mean_tf) - (gMLP_2_val_output_casted.numpy() * target_std_tf + target_mean_tf)).numpy().flatten(),
        "Final_Model_Error": ((y_val_batch.numpy() * target_std_tf + target_mean_tf) - final_predictions_unscaled.numpy()).numpy().flatten()
    })

    result_df = pd.concat([result_df, batch_df], ignore_index=True)

     # Update MAE and MSE calculators with unscaled values
    mae_calculator.update_state(y_val_batch * target_std_tf + target_mean_tf, final_predictions_unscaled)
    mse_calculator.update_state(y_val_batch * target_std_tf + target_mean_tf, final_predictions_unscaled)

# Retrieve final MAE and MSE values
final_mae = mae_calculator.result().numpy()
final_mse = mse_calculator.result().numpy()

# Save results to CSV
result_df.to_csv('/home/thinkbe/Desktop/trading/model_testing_text/training_models/predictions_high.csv', index=False)
print("Validation predictions and summary metrics saved to validation_predictions_with_summary.csv")

print(f"Final Mean Absolute Error on Validation Set: {final_mae:.4f}")
print(f"Final Mean Squared Error on Validation Set: {final_mse:.4f}")

# After training, save the dashboard as a .png file
dashboard.save_dashboard('high_epoch_progress_graph.png')
dashboard.root.destroy()
print("Dashboard saved as high_epoch_progress_graph.png and window closed.")

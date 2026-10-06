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
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Lambda, GlobalAveragePooling1D, Flatten, Reshape, Concatenate, Subtract, Add, LayerNormalization
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from tensorflow.keras.losses import MeanAbsoluteError, MeanSquaredError
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

tf.config.experimental.set_visible_devices

# Mixed precision policy
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Unified number of layers and units for all 3 models
# sen_len = 1 (sequence length is not currently being used)
num_layers = 3
num_units = 100  
d_ffn = 200

# 1st (left) gMLP model
gMLP_model_1_lr = 0.00005

# 3rd (middel) gMLP model
third_gMLP_model_lr = 0.001
third_model_passes = 2

# 2nd (right) gMLP model
gMLP_model_2_lr = 0.00005

# Global settings
epochs = 50 # 0 skips training
batch_size = 512

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


def preprocess_data(df, scaler=None):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")

    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    if scaler is None:
        scaler = StandardScaler()
        scaled = scaler.fit_transform(df.values.astype('float32'))
    else:
        scaled = scaler.transform(df.values.astype('float32'))

    X = scaled
    y = target.values

    return X, y, scaler

# Before loading data
print("About to load training data...")

# Load data
columns_to_load = None
df_train = load_data_from_sqlite('/home/thinkbe/Desktop/trading/data/streamline_lag_high.db', 'XAUUSD', columns=columns_to_load)
df_val = load_data_from_sqlite('/home/thinkbe/Desktop/trading/data/val_streamline_lag_high.db', 'XAUUSD', columns=columns_to_load)

# After loading data
print("Finished loading training data...")

print("Starting to preprocess training data...")
X_train, y_train, scaler = preprocess_data(df_train)
print("Finished preprocessing training data...")

print("Starting to preprocess validation data...")
X_val, y_val, _ = preprocess_data(df_val, scaler)
print("Finished preprocessing validation data...")


# Real-time Dashboard
class RealTimeDashboard(Callback):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Real-time Dashboard')
        self.fig, self.ax = plt.subplots(1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.gMLP_1_losses = []
        self.gMLP_2_losses = []
        self.third_model_losses = []
        self.final_model_losses = []
        self.validation_losses = []  # Added for validation loss
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        # Append new loss values
        self.gMLP_1_losses.append(logs.get('gMLP_1_loss', 0))
        self.gMLP_2_losses.append(logs.get('gMLP_2_loss', 0))
        self.third_model_losses.append(logs.get('third_model_loss', 0))
        self.final_model_losses.append(logs.get('final_model_loss', 0))
        self.validation_losses.append(logs.get('val_loss', 0))  # Append validation loss

        # Increment epoch count
        self.epoch_count += 1

        # Clear the plot
        self.ax.clear()

        # Plot the losses
        self.ax.plot(range(self.epoch_count), self.gMLP_1_losses, label='gMLP 1 Loss')
        self.ax.plot(range(self.epoch_count), self.gMLP_2_losses, label='gMLP 2 Loss')
        self.ax.plot(range(self.epoch_count), self.third_model_losses, label='Third Model Loss')
        self.ax.plot(range(self.epoch_count), self.final_model_losses, label='Final Model Loss')
        self.ax.plot(range(self.epoch_count), self.validation_losses, label='Validation Loss')  # Plot validation loss

        # Set plot labels and title
        self.ax.legend()
        self.ax.set_title('Epoch vs Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')

        # Draw the canvas
        self.canvas.draw()
        self.root.update()
    
    def save_dashboard(self, filename):
        self.fig.savefig(filename)

# RealTimeDashboard callback instance
dashboard = RealTimeDashboard()

print("About to initialize the model...")


class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, d_ffn, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.spatial_projection = tf.keras.layers.Dense(d_ffn, activation='tanh', use_bias=True, kernel_initializer='glorot_normal')
        
        self.forget_gate = tf.keras.layers.Dense(d_ffn, activation='sigmoid', kernel_initializer='he_normal')
        self.input_gate = tf.keras.layers.Dense(d_ffn, activation='sigmoid', kernel_initializer='he_normal')

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
        spatial_output = self.spatial_projection(normalized_inputs)
        gated_output = f_gate * spatial_output + i_gate * spatial_output
        return gated_output * tf.sigmoid(self.spatial_gating) + inputs

    def get_config(self):
        config = super(SpatialGatingUnit, self).get_config()
        config.update({
            "d_ffn": self.d_ffn            
        })
        return config


class gMLPBlock(tf.keras.layers.Layer):
    def __init__(self, num_units, d_ffn, **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)
        self.num_units = num_units
        self.d_ffn = d_ffn
        self.channel_projection_i = Dense(d_ffn, activation='tanh', kernel_initializer='glorot_normal')
        self.sgu = SpatialGatingUnit(d_ffn)
        self.channel_projection_ii = Dense(num_units, activation='tanh', kernel_initializer='glorot_normal')
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.channel_projection_i(inputs)
        x = self.sgu(x)
        x = self.channel_projection_ii(x)
        per_unit_outputs = tf.split(x, num_or_size_splits=self.num_units, axis=-1)
        return self.layer_norm(x + inputs), per_unit_outputs

    def get_config(self):
        config = super(gMLPBlock, self).get_config()
        config.update({
            "num_units": self.num_units,
            "d_ffn": self.d_ffn
        })
        return config
    
class SpatialGatingUnit2(tf.keras.layers.Layer):
    def __init__(self, d_ffn, **kwargs):
        super(SpatialGatingUnit2, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.spatial_projection = tf.keras.layers.Dense(d_ffn, activation='tanh', use_bias=True, kernel_initializer='glorot_normal')
        
        self.forget_gate = tf.keras.layers.Dense(d_ffn, activation='sigmoid', kernel_initializer='he_normal')
        self.input_gate = tf.keras.layers.Dense(d_ffn, activation='sigmoid', kernel_initializer='he_normal')

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
        spatial_output = self.spatial_projection(normalized_inputs)
        gated_output = f_gate * spatial_output + i_gate * spatial_output
        return gated_output * tf.sigmoid(self.spatial_gating) + inputs

    def get_config(self):
        config = super(SpatialGatingUnit2, self).get_config()
        config.update({
            "d_ffn": self.d_ffn            
        })
        return config


class gMLPBlock2(tf.keras.layers.Layer):
    def __init__(self, num_units, d_ffn, **kwargs):
        super(gMLPBlock2, self).__init__(**kwargs)
        self.num_units = num_units
        self.d_ffn = d_ffn
        self.channel_projection_i = Dense(d_ffn, activation='tanh', kernel_initializer='glorot_normal')
        self.sgu = SpatialGatingUnit2(d_ffn)
        self.channel_projection_ii = Dense(num_units, activation='tanh', kernel_initializer='glorot_normal')
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.channel_projection_i(inputs)
        x = self.sgu(x)
        x = self.channel_projection_ii(x)
        per_unit_outputs = tf.split(x, num_or_size_splits=self.num_units, axis=-1)
        return self.layer_norm(x + inputs), per_unit_outputs

    def get_config(self):
        config = super(gMLPBlock2, self).get_config()
        config.update({
            "num_units": self.num_units,
            "d_ffn": self.d_ffn
        })
        return config

# MiddleLayer class
class MiddleLayer(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super(MiddleLayer, self).__init__(**kwargs)
        self.num_units = num_units
        self.dense_layers = [Dense(1, activation='linear') for _ in range(num_units)]

    def call(self, gMLP_1_unit_outputs, gMLP_2_unit_outputs):
        combined_outputs = [self.dense_layers[i](gMLP_1_unit_outputs[i] + gMLP_2_unit_outputs[i]) 
                            for i in range(self.num_units)]
        return tf.concat(combined_outputs, axis=-1)


# ThirdgMLPModel class
class ThirdgMLPModel(tf.keras.Model):
    def __init__(self, num_layers, num_units, **kwargs):
        super(ThirdgMLPModel, self).__init__(**kwargs)
        self.num_layers = num_layers
        self.middle_layers = [MiddleLayer(num_units) for _ in range(num_layers)]
        self.output_layer = Dense(1, activation='linear')

    def call(self, gMLP_1_unit_outputs, gMLP_2_unit_outputs, training=False):
        x = []
        for layer in self.middle_layers:
            x.append(layer(gMLP_1_unit_outputs, gMLP_2_unit_outputs))
        x = tf.concat(x, axis=-1)
        return self.output_layer(x)


def create_gMLP_model(num_units, d_ffn, num_layers, additional_input=False):
    inputs = Input(shape=(num_units,))  # Original 2D input shape

    # Reshape input to 3D - Required for SpatialGatingUnit
    x = Reshape((1, num_units))(inputs)

    all_per_unit_outputs = []
    for _ in range(num_layers):
        x, per_unit_outputs = gMLPBlock(num_units, d_ffn)(x)
        all_per_unit_outputs.extend(per_unit_outputs)

    # Reshape back to 2D format before final Dense layer
    x = Reshape((num_units,))(x)  # Keep original dimension for output
    outputs = Dense(1, activation='linear')(x)  # Output layer

    if additional_input:
        additional_inputs = Input(shape=(1,))  # Input for predictions from model 1
        # Concatenate the output of gMLP_model_1 with inputs for gMLP_model_2
        final_outputs = Concatenate(axis=-1)([outputs, additional_inputs])
        # Final output with additional input considered
        final_outputs = Dense(1, activation='linear')(final_outputs)
        return Model(inputs=[inputs, additional_inputs], outputs=[final_outputs, all_per_unit_outputs])
    else:
        return Model(inputs=inputs, outputs=[outputs, all_per_unit_outputs])
    
def create_gMLP_model_2(num_units, d_ffn, num_layers, additional_input=False):
    inputs = Input(shape=(num_units,))  # Original 2D input shape

    # Reshape input to 3D - Required for SpatialGatingUnit
    x = Reshape((1, num_units))(inputs)

    all_per_unit_outputs = []
    for _ in range(num_layers):
        x, per_unit_outputs = gMLPBlock2(num_units, d_ffn)(x)
        all_per_unit_outputs.extend(per_unit_outputs)

    # Reshape back to 2D format before final Dense layer
    x = Reshape((num_units,))(x)  # Keep original dimension for output
    outputs = Dense(1, activation='linear')(x)  # Output layer

    if additional_input:
        additional_inputs = Input(shape=(1,))  # Input for predictions from model 1
        # Concatenate the output of gMLP_model_1 with inputs for gMLP_model_2
        final_outputs = Concatenate(axis=-1)([outputs, additional_inputs])
        # Final output with additional input considered
        final_outputs = Dense(1, activation='linear')(final_outputs)
        return Model(inputs=[inputs, additional_inputs], outputs=[final_outputs, all_per_unit_outputs])
    else:
        return Model(inputs=inputs, outputs=[outputs, all_per_unit_outputs])


# Instantiate and compile models
gMLP_model_1 = create_gMLP_model(num_units, d_ffn, num_layers)
gMLP_model_2 = create_gMLP_model_2(num_units, d_ffn, num_layers, additional_input=True)
third_gMLP_model = ThirdgMLPModel(num_layers, num_units)

# Compile models
gMLP_model_1.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=gMLP_model_1_lr))
gMLP_model_2.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=gMLP_model_2_lr))
third_gMLP_model.compile(loss='mean_absolute_error', optimizer=tf.keras.optimizers.Adam(learning_rate=third_gMLP_model_lr))


# Model summaries (third_gMLP_model cannot be summerized until training starts)
print("First gMLP Model Summary:")
gMLP_model_1.summary()

print("\nSecond gMLP Model Summary:")
gMLP_model_2.summary()

# Callback function to save the best model
gMLP_model_1_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/save_models_weights/scgMLP_model_1/gMLP_model_1_high_tf_test.h5"
gMLP_model_2_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/save_models_weights/scgMLP_model_1/gMLP_model_2_high_tf_test.h5"
third_gMLP_model_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/save_models_weights/scgMLP_model_1/gMLP_model_3_high_tf_test.ckpt"

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
            third_gMLP_model.save_weights(third_gMLP_model_checkpoint_path + '.ckpt')
            print(f"Third gMLP Model weights saved at {third_gMLP_model_checkpoint_path}.ckpt")           

        except Exception as e:
            print(f"Failed to save models: {e}")
            return False  # save operation failed
        return True  # successful save
    else:
        print(f"No improvement in validation loss to save the model (current: {val_loss}, best: {best_val_loss})")
        return False  # Indicate that there was no improvement

# Function to create an instance of ThirdgMLPModel
def create_third_gmlp_model(num_layers, num_units):
    return ThirdgMLPModel(num_layers=num_layers, num_units=num_units)

# Load pre-trained weights
try:
    # Load the entire gMLP model
    custom_objects = {
        'SpatialGatingUnit': SpatialGatingUnit,
        'gMLPBlock': gMLPBlock,
        'SpatialGatingUnit2': SpatialGatingUnit2,
        'gMLPBlock2': gMLPBlock2          
    }
    gMLP_model_1 = tf.keras.models.load_model(gMLP_model_1_checkpoint_path, custom_objects=custom_objects)
    print("Loaded gMLP Model 1.")

    gMLP_model_2 = tf.keras.models.load_model(gMLP_model_2_checkpoint_path, custom_objects=custom_objects)
    print("Loaded gMLP Model 2.")
    
    # Initialize the ThirdgMLPModel and then load the weights
    third_gMLP_model = create_third_gmlp_model(num_layers=..., num_units=...)  # replace with actual values
    third_gMLP_model.load_weights(third_gMLP_model_checkpoint_path)
    print("Loaded Third gMLP Model weights.")

except Exception as e:
    print("Error occurred while loading models:", e)

# Define the Mean Squared Error loss function
loss_function = MeanAbsoluteError()

# Function to prepare input for the third model
def prepare_third_model_input(gMLP_1_unit_outputs, gMLP_2_unit_outputs):
    combined_outputs = [(gMLP_1_unit_outputs[i], gMLP_2_unit_outputs[i]) for i in range(len(gMLP_1_unit_outputs))]
    return gMLP_1_unit_outputs, gMLP_2_unit_outputs

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

# Define the adjustment layer outside the training loop
adjustment_layer = tf.keras.layers.Dense(num_units, activation='linear')

best_val_loss = float('inf')

import tensorflow as tf

@tf.function
def combined_step(X_batch, y_batch, is_training):
    """
    Performs a combined training and validation step.
    
    Args:
    - X_batch (Tensor): Input batch of features.
    - y_batch (Tensor): Input batch of labels.
    - is_training (bool): Flag to indicate training mode (True) or validation mode (False).

    Returns:
    - For training: Tuple of loss values (gMLP_1_loss, gMLP_2_loss, third_model_initial_loss, final_model_loss)
    - For validation: Validation loss.
    """
    if is_training:
        # Training logic
        X_batch_adjusted = adjustment_layer(X_batch)

        # gMLP_model_1 training
        with tf.GradientTape() as tape_gMLP_1:
            gMLP_1_output, gMLP_1_unit_outputs = gMLP_model_1(X_batch_adjusted, training=True)
            gMLP_1_loss = loss_function(y_batch, gMLP_1_output)
        gradients_gMLP_1 = tape_gMLP_1.gradient(gMLP_1_loss, gMLP_model_1.trainable_variables)
        optimizer_gMLP_1.apply_gradients(zip(gradients_gMLP_1, gMLP_model_1.trainable_variables))

        # gMLP_model_2 training
        with tf.GradientTape() as tape_gMLP_2:
            gMLP_2_input = [X_batch_adjusted, gMLP_1_output]
            gMLP_2_output, gMLP_2_unit_outputs = gMLP_model_2(gMLP_2_input, training=True)
            gMLP_2_loss = loss_function(y_batch, gMLP_2_output)
        gradients_gMLP_2 = tape_gMLP_2.gradient(gMLP_2_loss, gMLP_model_2.trainable_variables)
        optimizer_gMLP_2.apply_gradients(zip(gradients_gMLP_2, gMLP_model_2.trainable_variables))

        # Third model training
        for _ in range(third_model_passes):
            with tf.GradientTape(persistent=True) as tape_third_model:
                third_model_output = third_gMLP_model(gMLP_1_unit_outputs, gMLP_2_unit_outputs, training=True)
                gMLP_1_output_casted = tf.cast(gMLP_1_output, y_batch.dtype)
                gMLP_2_output_casted = tf.cast(gMLP_2_output, y_batch.dtype)
                error_gMLP_1 = y_batch - gMLP_1_output_casted
                error_gMLP_2 = y_batch - gMLP_2_output_casted
                combined_error = (error_gMLP_1 + error_gMLP_2) / 2
                third_model_initial_loss = loss_function(combined_error, third_model_output)
                mean_predictions = (gMLP_1_output_casted + gMLP_2_output_casted) / 2
                final_model_output = mean_predictions + tf.cast(third_model_output, mean_predictions.dtype)
                final_model_loss = loss_function(y_batch, final_model_output)
            gradients_third_model_initial = tape_third_model.gradient(third_model_initial_loss, third_gMLP_model.trainable_variables)
            optimizer_third_gMLP.apply_gradients(zip(gradients_third_model_initial, third_gMLP_model.trainable_variables))
            gradients_third_model_final = tape_third_model.gradient(final_model_loss, third_gMLP_model.trainable_variables)
            optimizer_third_gMLP.apply_gradients(zip(gradients_third_model_final, third_gMLP_model.trainable_variables))
            del tape_third_model

        return gMLP_1_loss, gMLP_2_loss, third_model_initial_loss, final_model_loss

    else:
        # Validation logic
        X_val_batch_adjusted = adjustment_layer(X_batch)
        gMLP_1_val_output, gMLP_1_unit_outputs_val = gMLP_model_1(X_val_batch_adjusted, training=False)
        gMLP_1_val_output_casted = tf.cast(gMLP_1_val_output, y_batch.dtype)
        gMLP_2_val_input = [X_val_batch_adjusted, gMLP_1_val_output_casted]
        gMLP_2_val_output, gMLP_2_unit_outputs_val = gMLP_model_2(gMLP_2_val_input, training=False)
        gMLP_2_val_output_casted = tf.cast(gMLP_2_val_output, y_batch.dtype)
        mean_predictions_val = (gMLP_1_val_output_casted + gMLP_2_val_output_casted) / 2
        third_model_val_output = third_gMLP_model(gMLP_1_unit_outputs_val, gMLP_2_unit_outputs_val, training=False)
        third_model_val_output_casted = tf.cast(third_model_val_output, y_batch.dtype)
        final_model_val_output = mean_predictions_val + third_model_val_output_casted
        val_loss = loss_function(y_batch, final_model_val_output)

        return val_loss

# Check if epochs is 0, if so, skip the training loop
if epochs > 0:
    best_val_loss = float('inf')
    for epoch in range(epochs):
        # Initialize total losses for each epoch
        total_gMLP_1_loss, total_gMLP_2_loss, total_third_model_loss, total_final_model_loss, total_val_loss = 0, 0, 0, 0, 0
        num_batches = 0

        # Combined training and validation loop
        for phase in ['train', 'validate']:
            if phase == 'train':
                data_batches = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
                tqdm_batches = tqdm(data_batches, total=len(X_train) // batch_size, desc=f"Epoch {epoch + 1}/{epochs} Training")
            else:
                data_batches = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
                tqdm_batches = tqdm(data_batches, total=len(X_val) // batch_size, desc=f"Epoch {epoch + 1}/{epochs} Validation")

            for X_batch, y_batch in tqdm_batches:
                if phase == 'train':
                    losses = combined_step(X_batch, y_batch, True)
                    gMLP_1_loss, gMLP_2_loss, third_model_initial_loss, final_model_loss = losses
                    total_gMLP_1_loss += gMLP_1_loss.numpy()
                    total_gMLP_2_loss += gMLP_2_loss.numpy()
                    total_third_model_loss += third_model_initial_loss.numpy() + final_model_loss.numpy()
                    total_final_model_loss += final_model_loss.numpy()
                else:
                    val_loss = combined_step(X_batch, y_batch, False)
                    total_val_loss += val_loss.numpy()

                num_batches += 1

        # Calculate average losses
        average_gMLP_1_loss = total_gMLP_1_loss / num_batches
        average_gMLP_2_loss = total_gMLP_2_loss / num_batches
        average_third_model_loss = total_third_model_loss / num_batches
        average_final_model_loss = total_final_model_loss / num_batches
        average_val_loss = total_val_loss / (len(X_val) // batch_size)

        # Display training progress and update dashboard
        print(f"Epoch {epoch + 1}/{epochs}, gMLP 1 Training Loss: {total_gMLP_1_loss / num_batches:.4f}, gMLP 2 Training Loss: {total_gMLP_2_loss / num_batches:.4f}, Third Model Training Loss: {total_third_model_loss / num_batches:.4f}, Final Model Training Loss: {total_final_model_loss / num_batches:.4f}, Validation Loss: {average_val_loss:.4f}")
        print("\n")

        dashboard.on_epoch_end(epoch, logs={
            'gMLP_1_loss': total_gMLP_1_loss / num_batches,
            'gMLP_2_loss': total_gMLP_2_loss / num_batches,
            'third_model_loss': total_third_model_loss / num_batches,
            'final_model_loss': total_final_model_loss / num_batches,
            'val_loss': average_val_loss  # Include the average validation loss
        })

        # At the end of the epoch, call save_best_model
        if save_best_model(epoch, average_val_loss, gMLP_model_1, gMLP_model_2, third_gMLP_model):
            print(f"Model saved successfully at epoch {epoch + 1}.")

        # Clear session and garbage collection
        tf.keras.backend.clear_session()
        gc.collect()

else:
    print("Skipping training loop as epochs is 0.")


# Preprocess X_val using the adjustment layer
X_val_adjusted = adjustment_layer(X_val)

# Predict with the first gMLP model
gMLP_1_val_predictions, gMLP_1_unit_outputs_val = gMLP_model_1.predict(X_val_adjusted, batch_size=max(batch_size, 1))

# Cast the output to the correct type
gMLP_1_val_predictions_casted = tf.cast(gMLP_1_val_predictions, y_val.dtype)

# Prepare input for the second gMLP model
gMLP_2_val_input = [X_val_adjusted, gMLP_1_val_predictions_casted]

# Predict with the second gMLP model
gMLP_2_val_predictions, gMLP_2_unit_outputs_val = gMLP_model_2.predict(gMLP_2_val_input, batch_size=max(batch_size, 1))

# Cast the output to the correct type
gMLP_2_val_predictions_casted = tf.cast(gMLP_2_val_predictions, y_val.dtype)

# Calculate mean predictions for validation
mean_predictions_val = (gMLP_1_val_predictions_casted + gMLP_2_val_predictions_casted) / 2

# Predict with the third gMLP model
third_model_val_predictions = third_gMLP_model.predict([gMLP_1_unit_outputs_val, gMLP_2_unit_outputs_val], batch_size=max(batch_size, 1))

# Calculate the final predictions
final_predictions = mean_predictions_val + third_model_val_predictions

# Calculate MAE and MSE using the actual validation targets and the final predictions
mae_calculator = tf.keras.metrics.MeanAbsoluteError()
mse_calculator = tf.keras.metrics.MeanSquaredError()

mae_calculator.update_state(y_val, final_predictions)
mse_calculator.update_state(y_val, final_predictions)

final_mae = mae_calculator.result().numpy()
final_mse = mse_calculator.result().numpy()

# Save the validation predictions and the calculated metrics
result_df = pd.DataFrame({
    "Actual": y_val.flatten(),
    "gMLP_1_Predictions": gMLP_1_val_predictions.flatten(),
    "gMLP_2_Predictions": gMLP_2_val_predictions.flatten(),
    "Third_Model_Predictions": third_model_val_predictions.flatten(),
    "Final_Model_Predictions": final_predictions.flatten(),
    "gMLP_1_Error": (y_val - gMLP_1_val_predictions).flatten(),
    "gMLP_2_Error": (y_val - gMLP_2_val_predictions).flatten(),
    "Final_Model_Error": (y_val - final_predictions).flatten()
})

result_df.to_csv('/home/thinkbe/Desktop/trading/model_testing_text/training_models/predictions_high.csv', index=False)
print("Validation predictions and summary metrics saved to validation_predictions_with_summary.csv")

print(f"Final Mean Absolute Error on Validation Set: {final_mae:.4f}")
print(f"Final Mean Squared Error on Validation Set: {final_mse:.4f}")

# After training, save the dashboard as a .png file
dashboard.save_dashboard('high_epoch_progress_graph.png')
dashboard.root.destroy()
print("Dashboard saved as high_epoch_progress_graph.png and window closed.")




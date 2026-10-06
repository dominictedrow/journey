import os
import sqlite3
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tqdm import tqdm
import gc
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Subtract, Add, MultiHeadAttention, LayerNormalization
from sklearn.impute import SimpleImputer
from tensorflow.keras.layers import Dense, LSTM, Input, concatenate, Lambda, GlobalAveragePooling1D, Flatten
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
# from tensorflow.keras.losses import MeanSquaredLogarithmicError
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

seq_len = 3

# gMLP model
d_model = 800
d_ffn = 1600
kernel_size = 3
block_layers = 12

# LSTM model
lstm_layers = 24
lstm_units = 1024
num_heads = 4 # multi-head-attentiion

learning_rate = 0.00005
epochs = 200 # 0 skips training
batch_size = 128

# Number of passes per epoch
gMLP_passes = 1
LSTM_passes = 2

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
            idx -= 11  # Step back by 11 seq for each seq (avoid overlapping of historical features)
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
        scaler = StandardScaler()  
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
        self.gMLP_losses = []
        self.LSTM_losses = []
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs.get('val_loss', 0))  # val_loss
        self.gMLP_losses.append(logs.get('gMLP_loss', 0))  # gMLP loss
        self.LSTM_losses.append(logs.get('LSTM_loss', 0))  # LSTM loss
        self.epoch_count += 1

        self.ax.clear()
        self.ax.plot(range(self.epoch_count), self.losses, label='Final Loss')
        self.ax.plot(range(self.epoch_count), self.val_losses, label='Final Val Loss')
        self.ax.plot(range(self.epoch_count), self.gMLP_losses, label='gMLP Training Loss', linestyle='--')
        self.ax.plot(range(self.epoch_count), self.LSTM_losses, label='LSTM Training Loss', linestyle='--')
        self.ax.legend()
        self.ax.set_title('Epoch vs Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        
        self.canvas.draw()
        self.root.update()
    
    def save_dashboard(self, filename):
        self.fig.savefig(filename)

# RealTimeDashboard callback instance
dashboard = RealTimeDashboard()

print("About to initialize the model...")

# Positional Encoding Layer (involves other functions andd currently messes up LSTM predictions)
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
    def __init__(self, d_ffn, kernel_size=kernel_size, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.spatial_projection = Dense(d_ffn, activation='tanh', use_bias=True, kernel_initializer='glorot_normal')
        
        # Temporal convolution layer for temporal context
        self.temporal_conv = tf.keras.layers.Conv1D(filters=d_ffn, kernel_size=kernel_size, padding='same', activation='relu')
        
        # Forget and input gates
        self.forget_gate = Dense(d_ffn, activation='sigmoid', kernel_initializer='he_normal')
        self.input_gate = Dense(d_ffn, activation='sigmoid', kernel_initializer='he_normal')

    def build(self, input_shape):
        self.spatial_gating = self.add_weight(
            shape=(input_shape[-1],),
            initializer='he_normal',
            trainable=True
        )

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        
        # Temporal context
        temporal_context = self.temporal_conv(normalized_inputs)
        
        # Forget and input gate calculations
        f_gate = self.forget_gate(normalized_inputs)
        i_gate = self.input_gate(normalized_inputs)
        
        gated_temporal = f_gate * temporal_context + i_gate * self.spatial_projection(normalized_inputs)
        
        return gated_temporal * tf.sigmoid(self.spatial_gating) + inputs  # Residual connection

    def get_config(self):
        config = super(SpatialGatingUnit, self).get_config()
        config.update({
            "d_ffn": self.d_ffn,
            "kernel_size": self.kernel_size
        })
        return config

class gMLPBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn, **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)
        self.d_model = d_model
        self.d_ffn = d_ffn

        self.channel_projection_i = Dense(d_ffn, activation='tanh', kernel_initializer='glorot_normal')
        self.sgu = SpatialGatingUnit(d_ffn)
        self.channel_projection_ii = Dense(d_model, activation='tanh', kernel_initializer='glorot_normal')

        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.channel_projection_i(inputs)
        x = self.sgu(x)
        x = self.channel_projection_ii(x)
        return self.layer_norm(x + inputs)  # Residual connection and normalization

    def get_config(self):
        config = super(gMLPBlock, self).get_config()
        config.update({
            "d_model": self.d_model,
            "d_ffn": self.d_ffn            
        })
        return config

# Corrected function definition for create_gMLP_model
def create_gMLP_model(seq_len, d_model, d_ffn, block_layers):
    inputs = Input(shape=(seq_len, X_train.shape[2]))  # Use the original number of features here
    x = Dense(d_model)(inputs)  # Project inputs to match d_model

    # Apply Positional Encoding after the input
    x = PositionalEncoding(seq_len, d_model)(x)

    for _ in range(block_layers):
        x = gMLPBlock(d_model=d_model, d_ffn=d_ffn)(x)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(1, activation='linear', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=outputs, name='gMLP_model')
    return model


def create_LSTM_model(seq_len, lstm_layers, lstm_units, d_model, num_heads, extra_features=1):
    # Adjust the input shape based on your features and the concatenated gMLP predictions
    inputs = Input(shape=(seq_len, X_train.shape[2] + extra_features))
    x = Dense(d_model)(inputs)  # Project inputs to match d_model

    for i in range(lstm_layers):
        return_sequences = i < lstm_layers - 1
        x = LSTM(lstm_units, return_sequences=return_sequences)(x)
        if return_sequences:  # Apply normalization only if returning sequences
            x = LayerNormalization()(x)
            # Apply multi-head attention at each layer
            x = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    x = LayerNormalization()(x) if lstm_layers > 1 else x
    outputs = Dense(1, activation='linear', dtype='float32')(x)
    model = Model(inputs=inputs, outputs=outputs, name='LSTM_model')
    return model


# Loss function
loss_function = tf.keras.losses.MeanAbsoluteError()

# Define and compile gMLP model
gMLP_model = create_gMLP_model(1, d_model, d_ffn, block_layers)
optimizer_gMLP = tf.keras.optimizers.Adam(learning_rate)
loss_function = tf.keras.losses.MeanAbsoluteError()
gMLP_model.compile(optimizer=optimizer_gMLP, loss=loss_function)

# Compile LSTM model - adding 1 extra feature from the gMLP model
LSTM_model = create_LSTM_model(seq_len, lstm_layers, lstm_units, d_model, num_heads, extra_features=1)
optimizer_LSTM = tf.keras.optimizers.Adam(learning_rate)
LSTM_model.compile(optimizer=optimizer_LSTM, loss=loss_function)


# Display the model's architecture
gMLP_model.summary()
LSTM_model.summary()


# Callback function to save the best model
best_val_loss = float('inf')
lstm_weights_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/ensemble_model_save/lstm_ensemble_close.h5"
gMLP_model_checkpoint_path = "/home/thinkbe/Desktop/trading/model_testing_text/training_models/ensemble_model_save/gMLP_ensemble_close.h5"

def save_best_model(epoch, val_loss, gMLP_model, LSTM_model):
    global best_val_loss
    if val_loss < best_val_loss:
        print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}")
        best_val_loss = val_loss
        try:
            # Save the entire model
            gMLP_model.save(gMLP_model_checkpoint_path)
            print(f"gMLP model saved at {gMLP_model_checkpoint_path}")

            # Save only the weights for LSTM model
            LSTM_model.save_weights(lstm_weights_checkpoint_path)
            print(f"LSTM weights saved at {lstm_weights_checkpoint_path}")

        except Exception as e:
            print(f"Failed to save models: {e}")
            return False  # save operation failed
        return True  # successful save
    else:
        print(f"No improvement in validation loss to save the model (current: {val_loss}, best: {best_val_loss})")
        return False  # Indicate that there was no improvement


# Load pre-trained weights
try:
    # Load the entire gMLP model
    custom_objects = {
        'SpatialGatingUnit': SpatialGatingUnit,
        'gMLPBlock': gMLPBlock,
        'PositionalEncoding': PositionalEncoding,
        'MultiHeadAttention': MultiHeadAttention        
    }
    gMLP_model = tf.keras.models.load_model(gMLP_model_checkpoint_path, custom_objects=custom_objects)
    print("Loaded full gMLP model with custom objects.")

    # Load saved weights for LSTM model separately
    LSTM_model.load_weights(lstm_weights_checkpoint_path)  
    print("Loaded pre-trained weights for LSTM model.")
except Exception as e:
    print("No pre-trained weights found or error occurred while loading weights:", e)



# Custom function to calculate MAE and MSE
def calculate_metrics(actual, predicted):
    mae = tf.keras.losses.MeanAbsoluteError()
    mse = tf.keras.losses.MeanSquaredError()
    mae.update_state(actual, predicted)
    mse.update_state(actual, predicted)
    return mae.result().numpy(), mse.result().numpy()

# Initialize the scaler
scaler = StandardScaler()

# Before the training loop, fit the scaler on the training residuals
gMLP_predictions_train = gMLP_model.predict(X_train[:, -1:, :], batch_size=batch_size)
gMLP_predictions_train = tf.squeeze(gMLP_predictions_train, axis=-1)
y_train_residuals = y_train - gMLP_predictions_train

# Fit the scaler on the training residuals
scaler.fit(tf.reshape(y_train_residuals, (-1, 1)))

# Extract the scaler's mean and scale (standard deviation)
scaler_mean = scaler.mean_[0]
scaler_scale = scaler.scale_[0]

# Check if epochs is 0, if so, skip the training loop
if epochs > 0:
    # Training loop with progress bar for each epoch
    for epoch in range(epochs):
        # start loss tracking variables at the start of the epoch
        total_gMLP_loss = 0
        total_unscaled_LSTM_loss = 0
        num_batches = 0


        # Stage 1: Train gMLP model for the specified number of passes
        for gMLP_pass in range(gMLP_passes):
            # Create gMLP training dataset using the most recent input only
            gMLP_dataset = tf.data.Dataset.from_tensor_slices((X_train[:, -1:, :], y_train)).shuffle(buffer_size=len(X_train)).batch(batch_size, drop_remainder=True)

            # Train gMLP model with progress bar
            print(f"Epoch {epoch + 1}/{epochs} - Training gMLP Pass {gMLP_pass + 1}:")
            for X_batch, y_batch in tqdm(gMLP_dataset, total=len(gMLP_dataset), desc='gMLP Training'):
                with tf.GradientTape() as tape:
                    gMLP_predictions_batch = gMLP_model(X_batch, training=True)
                    gMLP_loss = loss_function(y_batch, tf.squeeze(gMLP_predictions_batch, axis=-1))
                gradients_gMLP = tape.gradient(gMLP_loss, gMLP_model.trainable_variables)
                optimizer_gMLP.apply_gradients(zip(gradients_gMLP, gMLP_model.trainable_variables))
                total_gMLP_loss += gMLP_loss.numpy()
                num_batches += 1

            average_loss_gMLP = total_gMLP_loss / num_batches

        # Clear any remaining graph from previous stages before starting LSTM training
        tf.keras.backend.clear_session()
        gc.collect()

        print("\n")
        # Stage 2: Train LSTM model for the specified number of passes
        for LSTM_pass in range(LSTM_passes):
            # Reset loss tracking variables for LSTM training
            total_LSTM_loss = 0
            num_batches = 0
            
            # Create LSTM training dataset using the full sequences
            LSTM_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=len(X_train)).batch(batch_size, drop_remainder=True)

            # Train LSTM model with progress bar
            print(f"Epoch {epoch + 1}/{epochs} - Training LSTM Pass {LSTM_pass + 1}:")
            for X_batch, y_batch in tqdm(LSTM_dataset, total=len(LSTM_dataset), desc='LSTM Training'):
                with tf.GradientTape() as tape:
                    # Predict with gMLP model and calculate residuals
                    gMLP_predictions_batch = gMLP_model(X_batch[:, -1:, :], training=False)
                    gMLP_predictions_batch = tf.squeeze(gMLP_predictions_batch, axis=-1)

                    # Calculate residuals
                    residuals = y_batch - gMLP_predictions_batch

                    # Scale the residuals using the scaler's parameters
                    scaled_residuals = (residuals - scaler.mean_) / scaler.scale_

                    # Prepare LSTM input
                    gMLP_predictions_tiled = tf.tile(gMLP_predictions_batch[:, tf.newaxis, tf.newaxis], [1, seq_len, 1])
                    LSTM_input = tf.concat([X_batch, gMLP_predictions_tiled], axis=-1)

                    # Make LSTM predictions
                    LSTM_predictions_batch = LSTM_model(LSTM_input, training=True)
                    LSTM_predictions_batch = tf.squeeze(LSTM_predictions_batch, axis=-1)

                    # Calculate the loss using the LSTM predictions against the scaled residuals
                    LSTM_loss = loss_function(scaled_residuals, LSTM_predictions_batch)

                    # Invert scaling for this batch's predictions to calculate unscaled loss
                    LSTM_predictions_batch_unscaled = (LSTM_predictions_batch * scaler.scale_) + scaler.mean_
                    loss_unscaled = loss_function(y_batch, LSTM_predictions_batch_unscaled + gMLP_predictions_batch)
                    total_unscaled_LSTM_loss += loss_unscaled.numpy()

                gradients_LSTM = tape.gradient(LSTM_loss, LSTM_model.trainable_variables)
                optimizer_LSTM.apply_gradients(zip(gradients_LSTM, LSTM_model.trainable_variables))
                total_LSTM_loss += LSTM_loss.numpy()
                num_batches += 1

            average_loss_unscaled_LSTM = total_unscaled_LSTM_loss / num_batches if num_batches else 0

            # After each LSTM training pass, clear the session and manually collect garbage to free memory
            tf.keras.backend.clear_session()
            gc.collect()

        # Define the function to concatenate in smaller batches
        def concat_in_batches(tensor_a, tensor_b, batch_size, seq_len):
            concatenated = []
            for i in range(0, tensor_a.shape[0], batch_size):
                batch_a = tensor_a[i:i+batch_size]
                # Expand tensor_b to match the sequence length of tensor_a
                batch_b_expanded = tensor_b[i:i+batch_size, tf.newaxis, tf.newaxis]
                # Ensure the expanded tensor_b has the same shape as tensor_a
                batch_b_broadcast = tf.broadcast_to(batch_b_expanded, [batch_a.shape[0], seq_len, 1])
                concatenated.append(tf.concat([batch_a, batch_b_broadcast], axis=-1))
            return tf.concat(concatenated, axis=0)


        # After LSTM training is done, calculate final predictions and final training loss
        gMLP_predictions_train = gMLP_model.predict(X_train[:, -1:, :], batch_size=batch_size)
        gMLP_predictions_train = tf.squeeze(gMLP_predictions_train, axis=-1)

        # Concatenate batches for further predictions
        LSTM_input_train = concat_in_batches(X_train, gMLP_predictions_train, batch_size, seq_len)

        # Use LSTM_input_train for further predictions as before
        LSTM_predictions_train = LSTM_model.predict(LSTM_input_train, batch_size=batch_size)
        LSTM_predictions_train = tf.squeeze(LSTM_predictions_train, axis=-1)

        # Invert scaling for LSTM predictions using TensorFlow operations
        LSTM_predictions_train_unscaled = (LSTM_predictions_train * scaler_scale) + scaler_mean

        # Calculate the final predictions and training loss
        final_predictions_train = gMLP_predictions_train + LSTM_predictions_train_unscaled
        final_training_loss = loss_function(y_train, final_predictions_train).numpy()

        # Calculate Validation Loss for the combined model
        gMLP_val_predictions = gMLP_model.predict(X_val[:, -1:, :], batch_size=batch_size)
        gMLP_val_predictions = tf.squeeze(gMLP_val_predictions, axis=-1)

        # Prepare validation LSTM input
        gMLP_val_predictions_tiled = tf.tile(gMLP_val_predictions[:, tf.newaxis], [1, seq_len])
        LSTM_val_input = tf.concat([X_val, gMLP_val_predictions_tiled[..., tf.newaxis]], axis=-1)

        # Predict with LSTM model
        LSTM_val_predictions = LSTM_model.predict(LSTM_val_input, batch_size=batch_size)
        LSTM_val_predictions = tf.squeeze(LSTM_val_predictions, axis=-1)

        # Invert scaling for LSTM validation predictions using TensorFlow operations
        LSTM_val_predictions_unscaled = (LSTM_val_predictions * scaler_scale) + scaler_mean

        # Calculate the final predictions and validation loss
        final_val_predictions = gMLP_val_predictions + LSTM_val_predictions_unscaled
        val_loss = loss_function(y_val, final_val_predictions)

        # Update the dashboard with the latest losses
        dashboard.on_epoch_end(epoch, logs={
            'loss': final_training_loss,
            'val_loss': val_loss.numpy(),
            'gMLP_loss': average_loss_gMLP,
            'LSTM_loss': average_loss_unscaled_LSTM
        })

        # At the end of each epoch after calculating val_loss
        model_saved = save_best_model(epoch, val_loss.numpy(), gMLP_model, LSTM_model)
        if model_saved:
            print(f"Model saved successfully at epoch {epoch + 1}.")
        else:
            print(f"Model not saved at epoch {epoch + 1}.")

        # Print losses
        print(f"Epoch {epoch + 1}/{epochs}, gMLP Training Loss: {average_loss_gMLP:.4f}, LSTM Training Loss: {average_loss_unscaled_LSTM:.4f}, Final Training Loss: {final_training_loss:.4f}, Final Validation Loss: {val_loss:.4f}")
        print("\n")

else:
    print("Skipping training loop as epochs is 0.")


# Predict with gMLP model on the validation data
gMLP_val_predictions = gMLP_model.predict(X_val[:, -1:, :], batch_size=max(batch_size, 1))  # Ensure batch_size is at least 1 for prediction
gMLP_val_predictions = tf.squeeze(gMLP_val_predictions, axis=-1)

# Calculate residuals between gMLP predictions and actual validation targets
residuals_val = y_val - gMLP_val_predictions

# Convert scaler parameters to TensorFlow tensors
scaler_mean = tf.constant(scaler.mean_, dtype=tf.float32)
scaler_scale = tf.constant(scaler.scale_, dtype=tf.float32)

# Ensure the scale is not zero to avoid division by zero
scaler_scale = tf.where(tf.equal(scaler_scale, 0), tf.ones_like(scaler_scale), scaler_scale)

# Scale the residuals using the scaler's parameters
scaled_residuals_val = (residuals_val - scaler_mean) / scaler_scale
scaled_residuals_val = tf.cast(scaled_residuals_val, dtype=tf.float32)

# Prepare validation LSTM input
gMLP_val_predictions_tiled = tf.tile(gMLP_val_predictions[:, tf.newaxis], [1, seq_len])
LSTM_val_input = tf.concat([X_val, gMLP_val_predictions_tiled[..., tf.newaxis]], axis=-1)

# Predict with LSTM model
LSTM_val_predictions = LSTM_model.predict(LSTM_val_input, batch_size=max(batch_size, 1))  # Ensure batch_size is at least 1 for prediction
LSTM_val_predictions = tf.squeeze(LSTM_val_predictions, axis=-1)

# Invert the scaling of the LSTM predictions to get the predicted errors on the original scale
# This operation assumes that the scaler can inverse transform using TensorFlow operations
LSTM_val_predictions_unscaled = (LSTM_val_predictions * scaler_scale) + scaler_mean

# Calculate the final predictions on the validation data
final_val_predictions = gMLP_val_predictions + LSTM_val_predictions_unscaled

# Calculate MAE and MSE using the actual validation targets and final predictions
mae_calculator = tf.keras.metrics.MeanAbsoluteError()
mse_calculator = tf.keras.metrics.MeanSquaredError()

# Update the metrics with the actual and predicted values
mae_calculator.update_state(y_val, final_val_predictions)
mse_calculator.update_state(y_val, final_val_predictions)

# Calculate final MAE and MSE
final_mae = mae_calculator.result().numpy()
final_mse = mse_calculator.result().numpy()

# Save the validation predictions and the calculated metrics
result_df = pd.DataFrame({
    "Actual": y_val,
    "gMLP_Predictions": gMLP_val_predictions,
    "LSTM_Error_Predictions": LSTM_val_predictions_unscaled,
    "Final_Predictions": final_val_predictions
})

# Calculate errors
result_df['gMLP_Error'] = result_df['Actual'] - result_df['gMLP_Predictions']
result_df['LSTM_Error'] = result_df['Actual'] - result_df['Final_Predictions']

# Save the DataFrame to a CSV file
result_df.to_csv('/home/thinkbe/Desktop/trading/model_testing_text/training_models/predictions_close.csv', index=False)
print("Predictions and summary metrics saved to ensemble_predictions_with_summary.csv")

# Output final MAE and MSE
print(f"Final Mean Absolute Error on Validation Set: {final_mae:.4f}")
print(f"Final Mean Squared Error on Validation Set: {final_mse:.4f}")

# After training, save the dashboard as a .png file
dashboard.save_dashboard('dashboard.png')
dashboard.root.destroy()
print("Dashboard saved as dashboard.png and Tkinter window closed.")




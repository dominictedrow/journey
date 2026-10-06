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

# Unified number of layers and units for all 3 models
# sen_len = 1 (sequence length is not currently being used)
blocks = 8
layers = 6 # blocks not layers
num_units = 400   # is like d_model
d_ffn = 800

# 1st (left) gMLP model
gMLP_model_1_lr = 0.000001

# 3rd (middel) gMLP model
third_gMLP_model_lr = 0.000001
third_model_passes = 2

# 2nd (right) gMLP model
gMLP_model_2_lr = 0.000001

# Global settings
epochs =  150 # 0 skips training
batch_size = 64

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
    y_scaled = target_scaler.transform(target.values.reshape(-1, 1)).flatten()

    return X_scaled, y_scaled

# Using the Functions for Training and Validation Data
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Before loading data
print("About to load training data...")
# Load and preprocess training data
df_train = load_data_from_sqlite("C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\streamline_lag_close_(backup).db", 'XAUUSD')
# After loading data
print("Finished loading training data...")
print("Starting to preprocess training data...")
X_train, y_train, feature_scaler, target_scaler, target_mean, target_std = preprocess_data(df_train, feature_scaler, target_scaler)
print("Finished preprocessing training data...")

print("About to load validation data...")
# Load and preprocess validation data
df_val = load_data_from_sqlite("C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\data\\val_streamline_lag_close_(backup).db", 'XAUUSD')
# After loading data
print("Finished loading training data...")
print("Starting to preprocess validation data...")
X_val, y_val = preprocess_validation_data(df_val, feature_scaler, target_scaler)
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
        self.third_model_initial_losses = []
        self.final_model_losses = []
        self.validation_losses = []  # Added for validation loss
        self.epoch_count = 0

    def on_epoch_end(self, epoch, logs=None):
        # Append new loss values
        self.gMLP_1_losses.append(logs.get('gMLP_1_loss', 0))
        self.gMLP_2_losses.append(logs.get('gMLP_2_loss', 0))
        self.third_model_initial_losses.append(logs.get('third_model_initial_loss', 0))
        self.final_model_losses.append(logs.get('final_model_loss', 0))
        self.validation_losses.append(logs.get('val_loss', 0))  # Append validation loss

        # Increment epoch count
        self.epoch_count += 1

        # Clear the plot
        self.ax.clear()

        # Plot the losses
        self.ax.plot(range(self.epoch_count), self.gMLP_1_losses, label='gMLP 1 Loss')
        self.ax.plot(range(self.epoch_count), self.gMLP_2_losses, label='gMLP 2 Loss')
        self.ax.plot(range(self.epoch_count), self.third_model_initial_losses, label='Third model initial loss')
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


""" class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, num_units, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.num_units = num_units

    def get_angles(self, position, i, num_units):
        angles = 1 / tf.pow(10000.0, (2 * (i // 2)) / tf.cast(num_units, tf.float32))
        return position * angles

    def positional_encoding(self, feature_len):
        angle_rads = self.get_angles(
            position=tf.range(feature_len, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(self.num_units, dtype=tf.float32)[tf.newaxis, :],
            num_units=self.num_units
        )
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)
        return pos_encoding

    def call(self, inputs):
        feature_len = tf.shape(inputs)[1]
        pos_encoding = self.positional_encoding(feature_len)

        # Ensure pos_encoding has the same data type as inputs
        pos_encoding = tf.cast(pos_encoding, dtype=inputs.dtype)

        # Expand the dimensions of inputs to match pos_encoding
        inputs_expanded = tf.expand_dims(inputs, -1) # New shape: [batch_size, feature_length, 1]

        # Tile the inputs to match the last dimension of pos_encoding
        inputs_tiled = tf.tile(inputs_expanded, [1, 1, self.num_units]) # New shape: [batch_size, feature_length, num_units]

        return inputs_tiled + pos_encoding

    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({'num_units': self.num_units})
        return config """

    

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
    def __init__(self, num_units, d_ffn, **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)
        self.num_units = num_units
        self.d_ffn = d_ffn
        self.channel_projection_i = Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal')
        # Additional Dense layers
        self.dense_layers = [Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal') for _ in range(layers)]
        self.sgu = SpatialGatingUnit(d_ffn)
        self.channel_projection_ii = Dense(num_units, activation='tanh', kernel_initializer='lecun_normal')
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, inputs):
        x = self.channel_projection_i(inputs)
        for layer in self.dense_layers:
            x = layer(x)
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
    def __init__(self, num_units, d_ffn, **kwargs):
        super(gMLPBlock3, self).__init__(**kwargs)
        self.num_units = num_units
        self.d_ffn = d_ffn
        self.channel_projection_i = Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal')
        # Additional Dense layers
        self.dense_layers = [Dense(d_ffn, activation='tanh', kernel_initializer='lecun_normal') for _ in range(layers)]
        self.sgu = SpatialGatingUnit3(d_ffn)
        self.channel_projection_ii = Dense(num_units, activation='tanh', kernel_initializer='lecun_normal')
        self.layer_norm = LayerNormalization(epsilon=1e-6)

    def call(self, gMLP_1_unit_output, gMLP_2_unit_output):        
        combined_input = gMLP_1_unit_output + gMLP_2_unit_output        

        x = self.channel_projection_i(combined_input)
        for layer in self.dense_layers:
            x = layer(x)
        x = self.sgu(x)
        x = self.channel_projection_ii(x)
        return self.layer_norm(x + combined_input)

    
    def get_config(self):
        config = super(gMLPBlock3, self).get_config()
        config.update({
            "num_units": self.num_units,
            "d_ffn": self.d_ffn
        })
        return config
        

class ThirdgMLPModel(tf.keras.Model):
    def __init__(self, num_units, d_ffn, blocks, **kwargs):
        super(ThirdgMLPModel, self).__init__(**kwargs)
        self.num_units = num_units
        self.d_ffn = d_ffn
        self.blocks = blocks
        self.initial_dense_layer = Dense(num_units, activation='linear', name='initial_dense_layer')
        self.global_avg_pool = GlobalAveragePooling1D()  # Global Average Pooling layer
        self.gMLP_blocks = [gMLPBlock3(num_units, d_ffn, name=f'gmlp_block_{i}') for i in range(blocks)]
        self.block_layer_norms = [LayerNormalization(name=f'block_{i}_layer_norm') for i in range(blocks)]
        self.output_layer = Dense(1, activation='linear', name='output_layer')

    def call(self, gMLP_1_output, gMLP_2_output, gMLP_1_unit_outputs, gMLP_2_unit_outputs, training=False):        
        # Ensure both tensors are of the same data type
        gMLP_1_output = tf.cast(gMLP_1_output, dtype=tf.float32)
        gMLP_2_output = tf.cast(gMLP_2_output, dtype=tf.float32)
        
        # Stack gMLP_1_output and gMLP_2_output as separate features
        combined_gMLP_output = tf.stack([gMLP_1_output, gMLP_2_output], axis=-1)
        x = self.initial_dense_layer(combined_gMLP_output)
        x_skip = x  # Preserve the initial layer output

        batch_size = tf.shape(x)[0]  # Get the batch size from x

        for i, block in enumerate(self.gMLP_blocks):
            
            combined_outputs = gMLP_1_unit_outputs[i] + gMLP_2_unit_outputs[i]            
            combined_outputs_reshaped = tf.reshape(combined_outputs, [batch_size, -1, num_units])                        
            
            x = block(x, combined_outputs_reshaped, training=training)            

            x = tf.reshape(x, [batch_size, -1, num_units])            
            x = self.block_layer_norms[i](x)

        x += x_skip  # Add the preserved output of the initial layer
        # Apply GlobalAveragePooling1D before the final output layer
        x = self.global_avg_pool(x)
        final_output = self.output_layer(x)

        return final_output
        
    
def create_gMLP_model(num_units, d_ffn, blocks, input_shape):
    inputs = Input(shape=input_shape)

    # First layer to match num_units
    x = Dense(num_units, activation='linear')(inputs)

    all_per_unit_outputs = []
    for _ in range(blocks):
        x, per_unit_outputs = gMLPBlock(num_units, d_ffn)(x)
        all_per_unit_outputs.extend(per_unit_outputs)

    # Reshape x to add a temporal dimension, making it 3D (batch_size, 1, features)
    x = Reshape((1, -1))(x)

    # Now, apply GlobalAveragePooling1D
    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1, activation='linear')(x)
    return Model(inputs=inputs, outputs=[outputs, all_per_unit_outputs])



def create_gMLP_model_2(num_units, d_ffn, blocks, input_shape):
    inputs = Input(shape=input_shape)

    # First layer to match num_units
    x = Dense(num_units, activation='linear')(inputs)

    
    all_per_unit_outputs = []
    for _ in range(blocks):
        x, per_unit_outputs = gMLPBlock(num_units, d_ffn)(x)
        all_per_unit_outputs.extend(per_unit_outputs)

    # Reshape x to add a temporal dimension, making it 3D (batch_size, 1, features)
    x = Reshape((1, -1))(x)

    # Apply GlobalAveragePooling1D directly on the 2D output
    x = GlobalAveragePooling1D()(x)

    outputs = Dense(1, activation='linear')(x)
    return Model(inputs=inputs, outputs=[outputs, all_per_unit_outputs])


# Instantiate and compile models
input_shape = X_train.shape[1:]  # Shape of preprocessed training data
extended_input_shape = (X_train.shape[1] + 1,)  # gMLP_model_1 outputs a single value per input

gMLP_model_1 = create_gMLP_model(num_units, d_ffn, blocks, input_shape)
gMLP_model_2 = create_gMLP_model_2(num_units, d_ffn, blocks, extended_input_shape)
third_gMLP_model = ThirdgMLPModel(num_units=num_units, d_ffn=d_ffn, blocks=blocks)

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
gMLP_model_1_checkpoint_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\model_testing_text\\training_models\\save_models_weights\\scgMLP_model_3\\gMLP_model_1_high.h5"
gMLP_model_2_checkpoint_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\model_testing_text\\training_models\\save_models_weights\\scgMLP_model_3\\gMLP_model_2_high.h5"

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
            third_gMLP_model_weights_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\model_testing_text\\training_models\\save_models_weights\\scgMLP_model_3\\gMLP_model_3_high.h5"
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
    for output in gMLP_outputs:
        # Assuming the output is rank-2 based on your model's design
        tiled_output = tf.tile(output, [1, num_units])
        reshaped_output = tf.reshape(tiled_output, [-1, num_units])
        reshaped_outputs.append(reshaped_output)
    return reshaped_outputs


@tf.function
def combined_step(X_batch, y_batch, is_training, models, optimizers, reshape_gMLP_outputs, num_units, third_model_passes, target_mean, target_std):
    gMLP_model_1, gMLP_model_2, third_gMLP_model = models
    optimizer_gMLP_1, optimizer_gMLP_2, optimizer_third_gMLP = optimizers    

    if is_training:
        # Processing for gMLP_model_1
        with tf.GradientTape() as tape_gMLP_1:
            gMLP_1_output, gMLP_1_unit_outputs = gMLP_model_1(X_batch, training=True)
            gMLP_1_loss = loss_function(y_batch, gMLP_1_output)        

        # Modify input for gMLP_model_2
        gMLP_1_output_for_gMLP_2 = tf.squeeze(gMLP_1_output, axis=[-1, -2]) if len(gMLP_1_output.shape) == 3 else gMLP_1_output
        gMLP_1_output_for_gMLP_2 = tf.cast(gMLP_1_output_for_gMLP_2, tf.float32)
        X_batch_gMLP_2 = tf.concat([X_batch, gMLP_1_output_for_gMLP_2], axis=-1)

        # Processing for gMLP_model_2
        with tf.GradientTape() as tape_gMLP_2:
            gMLP_2_output, gMLP_2_unit_outputs = gMLP_model_2(X_batch_gMLP_2, training=True)
            gMLP_2_error = y_batch - gMLP_1_output_for_gMLP_2
            gMLP_2_loss = loss_function(gMLP_2_error, gMLP_2_output)

        # Reshape unit outputs for both gMLP models
        gMLP_1_unit_outputs_reshaped = reshape_gMLP_outputs(gMLP_1_unit_outputs, num_units)
        gMLP_2_unit_outputs_reshaped = reshape_gMLP_outputs(gMLP_2_unit_outputs, num_units)

        # Processing for third gMLP model
        with tf.GradientTape(persistent=True) as tape_third_model:
            for _ in range(third_model_passes):
                gMLP_2_output_for_third_model = tf.cast(gMLP_2_output, tf.float32)
                gMLP_1_output_for_third_model = tf.cast(gMLP_1_output, tf.float32)
                third_model_output = third_gMLP_model(gMLP_1_output_for_third_model, gMLP_2_output_for_third_model, gMLP_1_unit_outputs_reshaped, gMLP_2_unit_outputs_reshaped, training=True)
                combined_gMLP_output = gMLP_1_output_for_third_model + gMLP_2_output_for_third_model
                third_model_error = y_batch - combined_gMLP_output        
                third_model_initial_loss = loss_function(third_model_error, third_model_output)                
                final_model_output = third_model_output + gMLP_1_output            
                final_model_loss = loss_function(y_batch, final_model_output)

        # Apply gradients for all models
        gradients_gMLP_1 = tape_gMLP_1.gradient(gMLP_1_loss, gMLP_model_1.trainable_variables)
        optimizer_gMLP_1.apply_gradients(zip(gradients_gMLP_1, gMLP_model_1.trainable_variables))

        gradients_gMLP_2 = tape_gMLP_2.gradient(gMLP_2_loss, gMLP_model_2.trainable_variables)
        optimizer_gMLP_2.apply_gradients(zip(gradients_gMLP_2, gMLP_model_2.trainable_variables))

        gradients_third_model_initial = tape_third_model.gradient(third_model_initial_loss, third_gMLP_model.trainable_variables)
        optimizer_third_gMLP.apply_gradients(zip(gradients_third_model_initial, third_gMLP_model.trainable_variables))         
      

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
        gMLP_2_error_unscaled = y_batch_unscaled - gMLP_1_output_unscaled
        gMLP_2_loss = loss_function(gMLP_2_error_unscaled, gMLP_2_output_unscaled)
        combined_gMLP_output = gMLP_1_output_unscaled + gMLP_2_output_unscaled
        third_model_error = y_batch_unscaled - combined_gMLP_output  
        third_model_initial_loss = loss_function(third_model_error, final_model_output_unscaled)        
        final_model_output = final_model_output_unscaled + gMLP_1_output_unscaled            
        final_model_loss = loss_function(y_batch, final_model_output)

        del tape_third_model

        return gMLP_1_loss, gMLP_2_loss, third_model_initial_loss, final_model_loss

    else:
        # Validation logic
        gMLP_1_val_output, gMLP_1_unit_outputs_val = gMLP_model_1(X_batch, training=False)
        gMLP_1_val_output = tf.cast(gMLP_1_val_output, tf.float32)  # Cast to float32
        gMLP_1_val_output_reshaped = tf.squeeze(gMLP_1_val_output, axis=[-1, -2]) if len(gMLP_1_val_output.shape) == 3 else gMLP_1_val_output
        gMLP_1_unit_outputs_val_reshaped = reshape_gMLP_outputs(gMLP_1_unit_outputs_val, num_units)

        gMLP_1_val_output_reshaped = tf.squeeze(gMLP_1_val_output, axis=[-1, -2]) if len(gMLP_1_val_output.shape) == 3 else gMLP_1_val_output
        X_batch = tf.cast(X_batch, tf.float32)  # Ensure X_batch is also cast to float32
        X_batch_gMLP_2_val = tf.concat([X_batch, gMLP_1_val_output_reshaped], axis=-1)

        gMLP_2_val_output, gMLP_2_unit_outputs_val = gMLP_model_2(X_batch_gMLP_2_val, training=False)
        gMLP_2_val_output = tf.cast(gMLP_2_val_output, tf.float32)  # Cast to float32
        gMLP_1_val_output_reshaped = tf.squeeze(gMLP_1_val_output, axis=[-1, -2]) if len(gMLP_1_val_output.shape) == 3 else gMLP_1_val_output
        gMLP_2_val_output_reshaped = tf.squeeze(gMLP_2_val_output, axis=[-1, -2]) if len(gMLP_2_val_output.shape) == 3 else gMLP_2_val_output
        gMLP_2_unit_outputs_val_reshaped = reshape_gMLP_outputs(gMLP_2_unit_outputs_val, num_units)
        
        # Ensure gMLP_2_val_output is in the correct shape for third_gMLP_model
        third_model_val_output = third_gMLP_model(gMLP_1_val_output_reshaped, gMLP_2_val_output_reshaped, gMLP_1_unit_outputs_val_reshaped, gMLP_2_unit_outputs_val_reshaped, training=False)
        third_model_val_output = tf.cast(third_model_val_output, tf.float32)  # Cast to float32

        # Cast target_mean and target_std to float32 for consistency
        target_mean_tf = tf.cast(target_mean, tf.float32)
        target_std_tf = tf.cast(target_std, tf.float32)

        # Unscaled outputs
        gMLP_1_val_output_unscaled = gMLP_1_val_output_reshaped * target_std_tf + target_mean_tf
        #gMLP_2_val_output_unscaled = gMLP_2_val_output_reshaped * target_std_tf + target_mean_tf
        final_model_val_output_unscaled = third_model_val_output * target_std_tf + target_mean_tf

        y_batch_unscaled = y_batch * target_std_tf + target_mean_tf

        # Recalculate loss with unscaled outputs
        # mean_predictions_unscaled = gMLP_2_val_output_unscaled
        # final_val_output = mean_predictions_unscaled + tf.cast(final_model_val_output_unscaled, mean_predictions_unscaled.dtype)

        # Calculate the unscaled validation loss
        final_model_output = final_model_val_output_unscaled + gMLP_1_val_output_unscaled 
        val_loss = loss_function(y_batch_unscaled, final_model_val_output_unscaled)
        return val_loss


third_gMLP_model_weights_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\model_testing_text\\training_models\\save_models_weights\\scgMLP_model_3\\gMLP_model_3_high.h5"

# Main training and validation loop
if epochs > 0:
    best_val_loss = float('inf')
    weights_loaded = False  # Initialize flag for weight loading

    training_batches = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)

    # Initialize and load weights for third_gMLP_model with actual data
    if not weights_loaded:
        if os.path.exists(third_gMLP_model_weights_path):
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
            print(f"Weights file not found at {third_gMLP_model_weights_path}, skipping weight gMLP model 3 loading.")

    for epoch in range(epochs):
        total_gMLP_1_loss, total_gMLP_2_loss, total_third_model_loss, total_final_model_loss = 0, 0, 0, 0
        total_val_loss = 0
        num_batches = 0

        tqdm_training_batches = tqdm(training_batches, total=len(X_train) // batch_size, desc=f"Epoch {epoch + 1}/{epochs} Training")

        # Training loop
        for X_batch, y_batch in tqdm_training_batches:
            losses = combined_step(
                X_batch, y_batch, True,
                [gMLP_model_1, gMLP_model_2, third_gMLP_model],
                [optimizer_gMLP_1, optimizer_gMLP_2, optimizer_third_gMLP],
                reshape_gMLP_outputs, num_units,
                third_model_passes,
                target_mean, target_std
            )
            total_gMLP_1_loss += losses[0].numpy()
            total_gMLP_2_loss += losses[1].numpy()
            total_third_model_loss += losses[2].numpy()
            total_final_model_loss += losses[3].numpy()
            num_batches += 1 

        # Validation
        validation_batches = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)

        # Validation loop
        for X_val_batch, y_val_batch in validation_batches:
            val_loss = combined_step(
                X_val_batch, y_val_batch, False, 
                [gMLP_model_1, gMLP_model_2, third_gMLP_model], 
                [optimizer_gMLP_1, optimizer_gMLP_2, optimizer_third_gMLP], 
                reshape_gMLP_outputs, num_units, 
                third_model_passes,
                target_mean, target_std
            )
            total_val_loss += val_loss.numpy()

        average_val_loss = total_val_loss / (len(X_val) // batch_size)

        # Logging and other end-of-epoch processes
        print(f"Epoch {epoch + 1}/{epochs}, gMLP 1 Training Loss: {total_gMLP_1_loss / num_batches:.4f}, "
              f"gMLP 2 Training Loss: {total_gMLP_2_loss / num_batches:.4f}, "
              f"Third Model Training Loss: {total_third_model_loss / num_batches:.4f}, "
              f"Final Model Training Loss: {total_final_model_loss / num_batches:.4f}, "
              f"Validation Loss: {average_val_loss:.4f}")
        
        dashboard.on_epoch_end(epoch, logs={
            'gMLP_1_loss': total_gMLP_1_loss / num_batches,
            'gMLP_2_loss': total_gMLP_2_loss / num_batches,
            'third_model_initial_loss': total_third_model_loss / num_batches,
            'final_model_loss': total_final_model_loss / num_batches,
            'val_loss': average_val_loss  # Include the average validation loss
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

# Prepare DataFrame for storing results
result_df_columns = ["Actual", "gMLP_1_Predictions", "gMLP_2_Predictions", "Third_Model_Predictions", "Final_Model_Predictions", 
                     "gMLP_1_Error", "gMLP_2_Error", "Final_Model_Error"]
result_df = pd.DataFrame(columns=result_df_columns)

# Process validation data in batches
validation_batches = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size)
for X_val_batch, y_val_batch in validation_batches:
    # Predict with the first gMLP model
    gMLP_1_val_output, gMLP_1_unit_outputs_val = gMLP_model_1(X_val_batch, training=False)
    gMLP_1_val_output_reshaped = tf.squeeze(gMLP_1_val_output, axis=[-1, -2]) if len(gMLP_1_val_output.shape) == 3 else gMLP_1_val_output
    gMLP_1_unit_outputs_val_reshaped = reshape_gMLP_outputs(gMLP_1_unit_outputs_val, num_units)

    # Modify input for gMLP_model_2 prediction
    X_val_gMLP_2 = tf.concat([X_val_batch, gMLP_1_val_output_reshaped], axis=-1)

    # Predict with the second gMLP model
    gMLP_2_val_output, gMLP_2_unit_outputs_val = gMLP_model_2(X_val_gMLP_2, training=False)
    gMLP_2_val_output_reshaped = tf.squeeze(gMLP_2_val_output, axis=[-1, -2]) if len(gMLP_2_val_output.shape) == 3 else gMLP_2_val_output
    gMLP_2_unit_outputs_val_reshaped = reshape_gMLP_outputs(gMLP_2_unit_outputs_val, num_units)

    # Predict with the third gMLP model
    third_model_val_output = third_gMLP_model(gMLP_1_val_output_reshaped, gMLP_2_val_output_reshaped, gMLP_1_unit_outputs_val_reshaped, gMLP_2_unit_outputs_val_reshaped, training=False)

    # Cast the model outputs to float32
    gMLP_1_val_output_casted = tf.cast(gMLP_1_val_output_reshaped, tf.float32)
    gMLP_2_val_output_casted = tf.cast(gMLP_2_val_output_reshaped, tf.float32)
    third_model_val_output_casted = tf.cast(third_model_val_output, tf.float32)

    # Calculate the final predictions
    final_predictions = third_model_val_output_casted + gMLP_1_val_output_casted

    # Cast target_mean and target_std to float32
    target_mean_tf = tf.cast(target_mean, tf.float32)
    target_std_tf = tf.cast(target_std, tf.float32)

    # Unscaled final predictions
    final_predictions_unscaled = final_predictions * target_std_tf + target_mean_tf

    # Aggregate results in DataFrame
    batch_df = pd.DataFrame({
        "Actual": y_val_batch.numpy().flatten(),
        "gMLP_1_Predictions": gMLP_1_val_output_casted.numpy().flatten(),
        "gMLP_2_Predictions": gMLP_2_val_output_casted.numpy().flatten(),
        "Third_Model_Predictions": third_model_val_output_casted.numpy().flatten(),
        "Final_Model_Predictions": final_predictions_unscaled.numpy().flatten(),
        "gMLP_1_Error": (y_val_batch.numpy().flatten() - gMLP_1_val_output_casted.numpy().flatten()),
        "gMLP_2_Error": (y_val_batch.numpy().flatten() - gMLP_2_val_output_casted.numpy().flatten()),
        "Final_Model_Error": (y_val_batch.numpy().flatten() - final_predictions_unscaled.numpy().flatten())
    })
    result_df = pd.concat([result_df, batch_df], ignore_index=True)

    # Update MAE and MSE calculators
    mae_calculator.update_state(y_val_batch, final_predictions_unscaled)
    mse_calculator.update_state(y_val_batch, final_predictions_unscaled)

# Retrieve final MAE and MSE values
final_mae = mae_calculator.result().numpy()
final_mse = mse_calculator.result().numpy()

# Save results to CSV
result_df.to_csv("C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\model_testing_text\\training_models\\predictions_high.csv", index=False)
print("Validation predictions and summary metrics saved to validation_predictions_with_summary.csv")

print(f"Final Mean Absolute Error on Validation Set: {final_mae:.4f}")
print(f"Final Mean Squared Error on Validation Set: {final_mse:.4f}")

# After training, save the dashboard as a .png file
dashboard.save_dashboard('high_epoch_progress_graph.png')
dashboard.root.destroy()
print("Dashboard saved as high_epoch_progress_graph.png and window closed.")




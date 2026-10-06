"""
  GGG     M    M    L       QQQ      PPPP .......................
 G        MM  MM    L      Q    Q    P   P .................    .
 G  GG    M M  M    L     Q      Q   PPPP ............ .  .   .    .     .      .       .        .         .          .
 G   G   M      M   L       Q   Q    P .....................    .
  GGG   M        M  LLLLL    QQQQ    P ..........................    
                                  Q
               
:Welcome to Gated Multi-Layer Perceptron FFT KAN Regressive (float32): (Final Version 2 Experiments)

Version 2: Each Quantum Layer output feeds forward through each QuantumEquation layer. (New features) - Currently adding optimal 
and suboptimal dataframes for dynamic learning. Changed the entire gMLQPBlock formula that uses different techniques and removed 
Markov_Chain function from the gMLQPBlock. Added a dynamic learning rate decay to help smooth out training. correctly applied 
Spatial Gating, where before it was not correctly implemented.

Inspired by Kolmogorov-Arnold Networks but using 1d fourier coefficients instead of splines coefficients, as fourier are more 
dense than spline (global vs local).

Next Steps: Quantum Self Attention, being that attention is still the best case for building contexr of inputs for LLMs, the focus with this
model is to remove the need of multi-head attention. Convolutional NNs will replace convolutional 
layers with CQNN (Convolutional Quantum Neural Netwrok). 

By: JD
"""  
import os
import time
import sqlite3
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.utils import Progbar
from tensorflow.keras.optimizers import Adam 
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.mixed_precision import experimental
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error 
from tensorflow.keras.layers import Dense, LayerNormalization, Embedding, Flatten, Concatenate

"""                                              .
TTTTT  H   H  IIIII  N   N  K   K  B B B  EEEE ...   .
  T    H   H    I    NN  N  K  K   B   B  E    .......     .
  T    HHHHH    I    N N N  KKK    BBBB   EEE  .............        .
  T    H   H    I    N  NN  K  K   B   B  E    ......................        .
  T    H   H  IIIII  N   N  K   K  B B B  EEEE ...............................

"""

# ::Main Parameters::
seq_len = 12        # Sequences are reshaped on same axis
q_ffn = 500         # gMLQPBlock Units
grid_size=12
sg_ffn= 1000         # Spatial Gating Units
block_layers = 2    # gMLQPBlock, Gating, Dense
epochs = 1500
batch_size = 512

# ::Dynamic Training Dataset:: 
""" 
A batch with error greater than the optimal_max in optimal_dataframe, copies to suboptimal_dataframe.
Optimal_dataframe always stays the same, while suboptimal_data_frame either grows, or loses data on 
retraining if the error go back below optimal_max. This allows training to double down on hard to 
learn batches. 
"""
dynamic_train = False       # Toggle dynamic train adjustment
optimal_factor = 2.5        # 0 to use default avg error
optimal_max = 0             # self adjusting
suboptimal_batch_size = 1
suboptimal_lr = .0000001    # suboptimal_lr not implemented yet.

# ::Dynamic Learning_rate Parameters::
lr_enable = True            # Toggle learning rate adjustment
print_enable = True 
initial_lr = 0.00005         # starting learning rate  
min_lr = 0.000001111111     # minimum learning rate
decay_step = 0.000000065    # amount to decrease learning rate each epoch

# ::Adam Optimizer parameters::
learning_rate = initial_lr
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7
amsgrad = False

loss_func = 'mean_absolute_error'

# ::gMLQP activations:; tf.nn.relu, tf.nn.sigmoid, tf.tanh
activation = tf.tanh

# ::gMLQP Initilizers::
ini='glorot_normal'

tf.config.experimental.set_visible_devices
print("Eager execution:", tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e) 

# Data Loading and Preprocessing Functions
def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    # Drop the first column regardless of its name
    df.drop(df.columns[0], axis=1, inplace=True)
    return df

def preprocess_data(df, scaler=None, seq_len=seq_len, 
                    num_unique_days=None, num_unique_months=None,
                    num_unique_hours=None, num_unique_minutes=None):
    # Drop index column explicitly if it's present
    if df.index.name is not None:
        df = df.reset_index(drop=True)

    # Ensure required columns are present
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    
    # Separate target column and remove it from the DataFrame
    target = df.pop('Target').astype('float32')

    # Extract embedding columns separately
    day_col = df.pop('Day').values
    month_col = df.pop('Month').values
    hour_col = df.pop('Hour').values
    minute_col = df.pop('Minute').values

    # Scale remaining numerical features, excluding index and target columns
    remaining_columns = df.columns  # Verify remaining columns
    # print(f"Remaining columns: {remaining_columns}")
    
    non_embedded_features = df.values.astype('float32')
    num_non_embedded_features = non_embedded_features.shape[1]
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(non_embedded_features)
    else:
        scaled_features = scaler.transform(non_embedded_features)

    # Create overlapping sequences
    def create_overlapping_sequences(values, seq_len):
        sequences = []
        for i in range(seq_len - 1, len(values)):
            seq = values[i - (seq_len - 1):i + 1]
            sequences.append(seq)
        return np.array(sequences)

    day_seqs = create_overlapping_sequences(day_col, seq_len)
    month_seqs = create_overlapping_sequences(month_col, seq_len)
    hour_seqs = create_overlapping_sequences(hour_col, seq_len)
    minute_seqs = create_overlapping_sequences(minute_col, seq_len)
    num_seqs = create_overlapping_sequences(scaled_features, seq_len)

    # Prepare dictionary for model input
    X = {f'day_{i}': day_seqs[:, i] for i in range(seq_len)}
    X.update({f'month_{i}': month_seqs[:, i] for i in range(seq_len)})
    X.update({f'hour_{i}': hour_seqs[:, i] for i in range(seq_len)})
    X.update({f'minute_{i}': minute_seqs[:, i] for i in range(seq_len)})

    # Non-embedded features should be the whole sequence
    X['other_features'] = num_seqs

    # Target sequence
    y = target.values[seq_len - 1:]

    return X, y, scaler, num_non_embedded_features


# Before loading data
print("About to load training data...")

# Load data
columns_to_load = None
df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_High_Train.db", 'XAUUSD' )
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_High_Val.db", 'XAUUSD' )

def flatten_with_position_encoding(X, seq_len):
    feature_names = [key for key in X.keys() if key != 'other_features']
    num_non_embedded_features = X['other_features'].shape[-1]

    # Flatten time-based embedding sequences
    for name in feature_names:
        X[name] = np.reshape(X[name], (-1,))

    # Apply positional encoding to non-embedded features
    non_embedded_seqs = X['other_features']
    reshaped_non_embedded = np.array([np.array(x).flatten() for x in non_embedded_seqs[:, ::-1]])

    pos_encoded_non_embedded = []
    for i in range(seq_len):
        step_data = reshaped_non_embedded[:, i * num_non_embedded_features: (i + 1) * num_non_embedded_features]
        pos_data = np.full((reshaped_non_embedded.shape[0], 1), i)  # Positional encoding for this time step
        pos_encoded_non_embedded.append(np.concatenate([pos_data, step_data], axis=1))

    X['other_features'] = np.concatenate(pos_encoded_non_embedded, axis=1)

    # Generate feature names
    flattened_feature_names = []
    for name in feature_names:
        for i in range(seq_len):
            flattened_feature_names.append(f'{name}_{i}')
    for i in range(seq_len):
        for j in range(num_non_embedded_features + 1):  # +1 for positional encoding
            flattened_feature_names.append(f'other_features_{i}_{j}')

    return X, flattened_feature_names

# After loading data
print("Finished loading training data...")

# Calculate the number of unique values for each column
num_unique_days = len(df_train['Day'].unique())
num_unique_months = len(df_train['Month'].unique())
num_unique_hours = len(df_train['Hour'].unique())
num_unique_minutes = len(df_train['Minute'].unique())

print("Starting to preprocess training data...")
X_train, y_train, scaler, num_non_embedded_features = preprocess_data(df_train, seq_len=seq_len,
                                                                      num_unique_days=num_unique_days,
                                                                      num_unique_months=num_unique_months,
                                                                      num_unique_hours=num_unique_hours,
                                                                      num_unique_minutes=num_unique_minutes)
print("Finished preprocessing training data...")
X_train, train_feature_names = flatten_with_position_encoding(X_train, seq_len)
print("starting X_train shape:", X_train['other_features'].shape)
# print("New feature names in training data after reshape:", train_feature_names) # print train feature names after processing

print("Starting to preprocess validation data...")
X_val, y_val, _, _ = preprocess_data(df_val, scaler, seq_len=seq_len,
                                    num_unique_days=num_unique_days,
                                    num_unique_months=num_unique_months,
                                    num_unique_hours=num_unique_hours,
                                    num_unique_minutes=num_unique_minutes)
X_val, val_feature_names = flatten_with_position_encoding(X_val, seq_len)
print("Finished preprocessing validation data...")
print("starting X_val shape:", X_val['other_features'].shape)
#print("New feature names in valuation data after reshape:", val_feature_names) # print val feature names after processing

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
        plt.savefig("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/Qg_MLP_XAUUSD_high_train_loss_v2_experiments.png") 
        self.root.update()

# Create RealTimeDashboard callback instance
dashboard = RealTimeDashboard()

print("About to initialize the model...")

# Define the model
class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, q_ffn, sg_ffn, grid_size, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.sg_ffn = sg_ffn
        self.grid_size = grid_size
        self.q_ffn = q_ffn

    def build(self, input_shape):
        input_dim = input_shape[-1] // 2

        self.alpha_beta_ini = 1 / (np.sqrt(input_dim) * np.sqrt(self.grid_size))

        # Define the weights for fft_KAN
        self.alpha_beta_weights = self.add_weight(shape=(input_dim, self.grid_size, self.sg_ffn // 2),
                                               initializer=tf.keras.initializers.RandomNormal(stddev=self.alpha_beta_ini),
                                               trainable=True, name='alpha_alpha_weights')
        self.bias = self.add_weight(name='bias', shape=(sg_ffn // 2,),
                                      initializer='zeros', trainable=True)

        self.layer_norm = LayerNormalization()
        self.layer_norm_output = LayerNormalization()

    def fft_KAN(self, inputs, alpha_beta, bias):
        k = tf.reshape(tf.range(1, self.grid_size + 1, dtype=tf.float32), (self.grid_size,))
        x_expanded = tf.expand_dims(inputs, axis=-1)

        cos_values = tf.cos(tf.matmul(x_expanded, tf.reshape(k, (1, -1))))
        sin_values = tf.sin(tf.matmul(x_expanded, tf.reshape(k, (1, -1))))

        y_cos = tf.einsum('bij,jki->bk', cos_values, tf.transpose(alpha_beta, [1, 2, 0]))
        y_sin = tf.einsum('bij,jki->bk', sin_values, tf.transpose(alpha_beta, [1, 2, 0]))

        y = y_cos + y_sin
        y = tf.nn.bias_add(y, bias)
        return y

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        split_half = tf.split(normalized_inputs, 2, axis=-1)

        gating_signal = tf.abs(tf.cos(split_half[0]))
        gating_signal = self.fft_KAN(gating_signal, self.alpha_beta_weights, self.bias)

        transformed_signal = tf.abs(tf.cos(split_half[1]))
        transformed_signal = self.fft_KAN(transformed_signal, self.alpha_beta_weights, self.bias)

        gated_output = gating_signal * transformed_signal
        gated_output = self.layer_norm_output(gated_output)
        
        return gated_output
    
    def get_config(self):
        config = super(SpatialGatingUnit, self).get_config()
        config.update({
            "sg_ffn": self.sg_ffn,
            "q_ffn": self.q_ffn,
            "grid_size": self.grid_size,
            "alpha_beta_ini": self.alpha_beta_ini
        })
        return config

class gMLQPBlock(tf.keras.layers.Layer):
    def __init__(self, q_ffn, grid_size): 
        super().__init__() 
        self.q_ffn = q_ffn
        self.grid_size = grid_size
        self.x_norm = LayerNormalization()

    def build(self, input_shape):
        input_dim = input_shape[-1]

        # Alpha/Beta + Bias Weights Group #0
        alpha_beta_ini_2 = 1 / (np.sqrt(input_dim) * np.sqrt(self.grid_size))
        self.alpha_beta_weights_1 = self.add_weight(shape=(2, input_dim, self.grid_size),
                                                    initializer=tf.keras.initializers.RandomNormal(stddev=alpha_beta_ini_2), 
                                                    trainable=True, name='alpha_beta_weights_1')
        self.bias_1 = self.add_weight(name='bias_1', shape=(self.q_ffn,), initializer='zeros', trainable=True)

        super().build(input_shape)
    
    def fft_KAN(self, inputs, alpha_beta, bias):
        batch_size = tf.shape(inputs)[0]
        input_dim = tf.shape(inputs)[1]
        
        inputs = tf.expand_dims(inputs, -1)  # Shape: (batch_size, input_dim, 1)
        k = tf.reshape(tf.range(1, self.grid_size + 1, dtype=tf.float32), (1, 1, self.grid_size))  # Shape: (1, 1, grid_size)

        cos_values = tf.cos(inputs * k)  # Shape: (batch_size, input_dim, grid_size)
        sin_values = tf.sin(inputs * k)  # Shape: (batch_size, input_dim, grid_size)

        y_cos = tf.reduce_sum(cos_values * alpha_beta[0], axis=[2])  # Shape: (batch_size, input_dim)
        y_sin = tf.reduce_sum(sin_values * alpha_beta[1], axis=[2])  # Shape: (batch_size, input_dim)
        y = y_cos + y_sin  # Shape: (batch_size, input_dim)

        y = tf.reshape(y, [batch_size, self.q_ffn])  # Ensure the reshape matches batch_size and q_ffn
        y = tf.nn.bias_add(y, bias)
        
        return y

    
    def QuantumEquation(self, inputs):
        x = self.fft_KAN(inputs, self.alpha_beta_weights_1, self.bias_1) 
        x1_norm_output = self.x_norm(x)

        output = x1_norm_output
        return output

    def call(self, inputs):
        return self.QuantumEquation(inputs)

    def get_config(self):
        config = super(gMLQPBlock, self).get_config()
        config.update({"q_ffn": self.q_ffn})
        return config

print("Model initialized!")

expected_shape = seq_len * (num_non_embedded_features + 1)

# Define individual input layers for each feature with embedding
day_inputs = [Input(shape=(1,), name=f'day_{i}') for i in range(seq_len)]
month_inputs = [Input(shape=(1,), name=f'month_{i}') for i in range(seq_len)]
hour_inputs = [Input(shape=(1,), name=f'hour_{i}') for i in range(seq_len)]
minute_inputs = [Input(shape=(1,), name=f'minute_{i}') for i in range(seq_len)]
other_features_input = Input(shape=(expected_shape,), name='other_features')

# Define embeddings and flatten them
day_embeddings = [Embedding(input_dim=num_unique_days, output_dim=1)(day_input) for day_input in day_inputs]
month_embeddings = [Embedding(input_dim=num_unique_months, output_dim=1)(month_input) for month_input in month_inputs]
hour_embeddings = [Embedding(input_dim=num_unique_hours, output_dim=1)(hour_input) for hour_input in hour_inputs]
minute_embeddings = [Embedding(input_dim=num_unique_minutes, output_dim=1)(minute_input) for minute_input in minute_inputs]

day_flat = [Flatten()(embedding) for embedding in day_embeddings]
month_flat = [Flatten()(embedding) for embedding in month_embeddings]
hour_flat = [Flatten()(embedding) for embedding in hour_embeddings]
minute_flat = [Flatten()(embedding) for embedding in minute_embeddings]

# Concatenate embedded inputs
embedded = Concatenate(axis=-1)(day_flat + month_flat + hour_flat + minute_flat)

# Combine embedded inputs with other features
combined_input = Concatenate(axis=-1)([embedded, other_features_input])

# Project the input to match q_ffn (units)
input = combined_input

# Apply block layers
block_outputs = []
x = input
for _ in range(block_layers):
    projected_inputs = Dense(q_ffn, activation='linear', kernel_initializer=ini)(x)
    projected_norm = LayerNormalization()(projected_inputs)
    qnn_block = gMLQPBlock(q_ffn, grid_size)(projected_norm)  # Assuming you have defined gMLQPBlock
    #x_gate = SpatialGatingUnit(q_ffn, sg_ffn, grid_size)(qnn_block)
    x_block = gMLQPBlock(q_ffn, grid_size)(qnn_block) 
    output = x_block + projected_norm
    block_outputs.append(output)

# Concatenate outputs of block layers
concatenated_outputs = Concatenate()(block_outputs)

outputs = Dense(1, activation='linear', kernel_initializer=ini, dtype='float32')(concatenated_outputs)
inputs = day_inputs + month_inputs + hour_inputs + minute_inputs + [other_features_input]
model = Model(inputs=inputs, outputs=outputs)

# model summary
model.summary()
print("Total number of parameters in the model:", model.count_params())

# Define the model checkpoint for weights only
weights_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/Qg_MLP_XAUUSD_high_weights_v2_experiments.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/Qg_MLP_XAUUSD_high_model_v2_experiments.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

# Try loading the full model, then the weights, or use the freshly defined model if neither exists
if os.path.exists(full_model_checkpoint_path):
    print("Loading full model...")
    model = tf.keras.models.load_model(full_model_checkpoint_path, custom_objects={'SpatialGatingUnit': SpatialGatingUnit, 'gMLQPBlock': gMLQPBlock})
elif os.path.exists(weights_checkpoint_path):
    print("Loading model weights...")
    model.load_weights(weights_checkpoint_path)
else:
    print("Using freshly defined model...")

class PrintLR(Callback):
    def __init__(self, print_enable=print_enable):
        super().__init__()
        self.print_enable = print_enable

    def on_epoch_end(self, epoch, logs=None):
        if self.print_enable:
            current_lr = self.model.optimizer.lr.read_value()
            print(f"Epoch {epoch + 1}, Current Learning Rate: {current_lr:.10f}")


class CustomLearningRateScheduler(Callback):
    """Custom learning rate scheduler."""
    def __init__(self, initial_lr, min_lr, decay_step, lr_enable=lr_enable):
        super().__init__()
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.decay_step = decay_step
        self.lr_enable = lr_enable
        self.current_lr = initial_lr

    def on_epoch_end(self, epoch, logs=None):
        if self.lr_enable:
            new_lr = max(self.current_lr - self.decay_step, self.min_lr)
            self.model.optimizer.lr.assign(new_lr)
            self.current_lr = new_lr
            print(f"Epoch {epoch + 1}, New Learning Rate: {new_lr:.10f}")

# Initialize optimizer
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_func)

# Evaluate model immediately after loading weights (for baseline performance)
""" val_loss = model.evaluate(X_val, y_val)
print(f"Baseline Mean Absolute Error on Validation Set after Loading: {val_loss}") """

lr_scheduler = CustomLearningRateScheduler(initial_lr, min_lr, decay_step, lr_enable)

# Initialize the custom PrintLR callback
print_lr = PrintLR()

if dynamic_train:
    callbacks = [weights_checkpoint, dashboard, lr_scheduler, print_lr]

    # Initialize DataFrames for custom training loop
    optimal = pd.DataFrame(X_train, columns=train_feature_names)
    optimal['label'] = y_train
    suboptimal = pd.DataFrame(columns=optimal.columns)

    # Calculate total steps per epoch accounting for all data
    steps_per_epoch = (len(optimal) + batch_size - 1) // batch_size  

    # Initialize callbacks
    for callback in callbacks:
        callback.set_model(model)
        callback.set_params({
            'epochs': epochs,
            'steps': steps_per_epoch,
            'verbose': 1,
            'metrics': ['loss', 'val_loss'],
        })

    # Initialize time tracking and overall training logic
    epoch_times = []
    for epoch in range(epochs):
        start_time = time.time()  # Start timing the epoch
        print(f"Epoch {epoch+1}/{epochs}")

        # Initialize list to store losses for this epoch
        # loss = []

        for callback in callbacks:
            callback.on_epoch_begin(epoch)

        # At the beginning of each epoch
        if epoch == 0:
            optimal['batch_error'] = np.nan

        # Handling of optimal batches
        steps_per_epoch = (len(optimal) + batch_size - 1) // batch_size
        progbar = Progbar(steps_per_epoch)
        for index, batch in optimal.groupby(np.arange(len(optimal)) // batch_size):
            batch_X = batch.drop(['label', 'batch_error'], axis=1)
            batch_y = batch['label']
            model.train_on_batch(batch_X, batch_y)
            batch_error = model.evaluate(batch_X, batch_y, verbose=0)
            optimal.loc[batch.index, 'batch_error'] = batch_error

            if batch_error > optimal_max and index not in suboptimal.index:
                suboptimal = pd.concat([suboptimal, batch], ignore_index=True)

            progbar.update(index + 1, values=[("loss", batch_error)])

        if not optimal['batch_error'].isna().all():
            optimal_mean_error = optimal['batch_error'].mean()
            optimal_max = optimal_mean_error * (optimal_factor if optimal_factor != 0 else 1)

        # Handling of suboptimal batches
        suboptimal_indexes_to_drop = []
        suboptimal_steps = (len(suboptimal) + suboptimal_batch_size - 1) // suboptimal_batch_size
        suboptimal_progbar = Progbar(suboptimal_steps)
        for index in range(0, len(suboptimal), suboptimal_batch_size):
            batch = suboptimal.iloc[index:index+suboptimal_batch_size]
            if not batch.empty:
                batch_X = batch.drop(['label', 'batch_error'], axis=1)
                batch_y = batch['label']
                model.train_on_batch(batch_X, batch_y)
                batch_error = model.evaluate(batch_X, batch_y, verbose=0, batch_size=suboptimal_batch_size)
                suboptimal.loc[index:index+suboptimal_batch_size-1, 'batch_error'] = batch_error

                if batch_error <= optimal_max:
                    # optimal = pd.concat([optimal, batch])
                    suboptimal_indexes_to_drop.extend(batch.index.tolist())

                suboptimal_progbar.update((index // suboptimal_batch_size) + 1, values=[("loss", batch_error)])

        # Remove the improved batches from suboptimal
        suboptimal.drop(suboptimal_indexes_to_drop, inplace=True)

        end_time = time.time()  # End timing the epoch
        epoch_duration = end_time - start_time
        epoch_times.append(epoch_duration)
        print(f"Epoch {epoch+1} completed in {epoch_duration:.2f} seconds.")

        # Validation and logging
        val_loss = model.evaluate(X_val, y_val, verbose=0)
        loss = optimal_mean_error
        logs = {'loss': loss, 'val_loss': val_loss}
        total_records = len(suboptimal)
        suboptimal_batches = (total_records + suboptimal_batch_size - 1) // suboptimal_batch_size

        # Callbacks for epoch end
        for callback in callbacks:
            callback.on_epoch_end(epoch, logs=logs)

        print(f"Validation Loss: {val_loss}")
        print(f"Optimal max for next epoch: {optimal_max}")
        print(f"Number of batches in suboptimal: {suboptimal_batches}")

    print("Training complete.")

else:
    # Standard .fit() method
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[weights_checkpoint, dashboard, lr_scheduler, print_lr])
    print("Training complete.")

epochs = 0 
batch_size = 1
if epochs > 0:
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[weights_checkpoint, dashboard])

    # Close the Tkinter window
    dashboard.root.destroy()
    print("Training complete.") 

# Model evaluation for both MSE and MAE
y_pred = model.predict(X_val).flatten()
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f"Model Mean Squared Error on Validation Set: {mse}")
print(f"Model Mean Absolute Error on Validation Set: {mae}")

# Obtain predictions on validation data
predictions = model.predict(X_val)

# Calculate MSE and MAE
mse = np.mean((y_val - predictions.flatten())**2)
mae = np.mean(np.abs(y_val - predictions.flatten()))

print(f"Mean Squared Error on validation data: {mse}")
print(f"Mean Absolute Error on validation data: {mae}")

# Save the actual and predicted values to a DataFrame
result_df = pd.DataFrame({"Actual": y_val, "Predicted": predictions.flatten()})

# Save result DataFrame to a CSV file
file_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/Qg_MLP_XAUUSD_high_predictions_v2_experiments.csv"
print("Saving file to:", file_path)
result_df.to_csv(file_path, index=False)
print("File saved successfully!")



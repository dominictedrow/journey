import os
import pickle
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Add, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.config.experimental.set_visible_devices

""" # Mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy) """

# parameters
num_blocks = 1 
units_per_layer = 1800 
num_layers_per_block = 3
RNN = num_blocks  # number of times to recycle the output by adding inputs for continus data transformation for each main block's input
""" 
:param this will be for LLM configuration with Softmax final layer.

This will be dynamically adjusted based on the statistical accuracy of the final tokenized output.
If the most accurate prediction is say 70% confident, and the threshold is 80%, the output of
Side 1 - Forward processing for each block, will recycle back as inputs for each block, each adding.
a residual connection from rnn_outputs, each matching the number of blocks from num_blocks. 

Since this is a completely custom Spiking NN design, units that fire stay inactive on the next cycle.
when the model does not feel confident about its prediction, and the cycle will keep repeating until
the threshold is reached. It's a form of automatic thinking. This model can be adapted to any problem,
including CNN's, temporal data and 3D data.

thinking_threshold= .80 
"""
epochs = 50 
batch_size = 16
patience = 50                       # Define your patience for early stopping

""" 
:STDP parameters
:param A_plus: STDP learning rate for potentiation.
:param A_minus: STDP learning rate for depression.
:param tau_plus: Time constant for potentiation.
:param tau_minus: Time constant for depression. 
"""
A_plus = 0.001                       # default range = 0.001 - 0.1
A_minus = 0.002                      # default range = 0.001 - 0.02
tau_plus = 20                       # default range = 10 - 100ms+
tau_minus = 20                      # default range = 10 - 100ms+

""" 
:param max_rate: The upper limit on the firing rate of neurons in the network, typically set between 10.0 and 200Hz.
:param base_max_rate: The starting value for the max_rate, usually within the range of 20.0 to 100Hz.
:param current_max_rate: A dynamic parameter representing the current maximum firing rate, initially set to the value of base_max_rate.
:param scale_factor: A coefficient used to scale adjustments in max_rate based on the average neuronal threshold, with a default range between 0 and 1. 
"""

base_max_rate = 15.0                # default range = 20.0 - 100Hz+
current_max_rate = base_max_rate    # dynamic parameter
max_rate = current_max_rate         # default range = 10.0 - 200Hz+
scale_factor = 0.05                  # default range = 0 - 1

""" This is the final output layer, that uses a custom Adam optimizer """
learning_rate = 0.001               # Matched learning rate (default = 0.001)
beta_1=0.9                          # Momentum term (beta1) (default = 0.9)
beta_2=0.999                        # Momentum term for the squared gradient (beta2) (default = 0.999)
epsilon=1e-07                       # Small number to prevent any division by zero (default = 1e-07)
decay=0.0                           # Weight decay for regularization (default = 0.0)
amsgrad=True

def rate_encoding(X, max_rate=max_rate):
    """
    Converts feature values to spike rates, normalized and scaled by max_rate.

    :param X: Input feature matrix.
    :param max_rate: The maximum spike rate.
    :return: Spike rates corresponding to the feature values.
    """
    X_normalized = np.divide(X - X.min(axis=0), X.max(axis=0) - X.min(axis=0), out=np.zeros_like(X), where=(X.max(axis=0) - X.min(axis=0)) != 0)
    spike_rates = X_normalized * max_rate
    return spike_rates


def encode_target_to_spikes(target_values, max_rate):
    """
    Encodes target values into spike rates, consistent with the feature encoding scheme.

    :param target_values: Array of target values to encode.
    :param max_rate: The maximum spike rate, consistent with feature encoding.
    :return: Spike rates corresponding to the target values.
    """
    # Normalize the target values based on their range
    min_target, max_target = target_values.min(), target_values.max()
    normalized_targets = (target_values - min_target) / (max_target - min_target)

    # Convert normalized target values to spike rates
    spike_rates = normalized_targets * max_rate
    return spike_rates

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

def preprocess_data(df, feature_scaler=None, encoder=rate_encoding, target_encoder=encode_target_to_spikes):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    
    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
    X_scaled = feature_scaler.fit_transform(df.values.astype('float32'))

    # advanced positional encoding (currently turned off)
    # X_scaled = advanced_positional_encoding(X_scaled)

    # Reshape the data to make each feature a separate time step in the sequence
    X_reshaped = np.expand_dims(X_scaled, axis=2)  # Shape becomes [n_samples, n_features, 1]

    # Encode features into spike rates
    X_encoded = encoder(X_reshaped)
   # Encode the target into spike rates
    target_encoded = target_encoder(target, max_rate)

    return X_encoded, target_encoded, feature_scaler

def preprocess_validation_data(df, feature_scaler, encoder=rate_encoding, target_encoder=encode_target_to_spikes):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)
    X_scaled = feature_scaler.transform(df.values.astype('float32'))

    # advanced positional encoding (currently turned off)
    # X_scaled = advanced_positional_encoding(X_scaled)

    # Reshape the data to make each feature a separate time step in the sequence
    X_reshaped = np.expand_dims(X_scaled, axis=2)  # Shape becomes [n_samples, n_features, 1]

    # Encode features into spike rates
    X_encoded = encoder(X_reshaped)
    # Encode the target into spike rates
    target_encoded = target_encoder(target, max_rate)

    return X_encoded, target_encoded

# Initialize the feature scaler with MinMaxScaler
feature_scaler = MinMaxScaler()

# Load and preprocess training data
print("About to load training data...")
df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//streamline_lag_close_(backup).db", 'XAUUSD')
print("Finished loading training data...")
print()
print("Starting to preprocess training data...")
# Preprocess training data
X_train_encoded, y_train_encoded, _ = preprocess_data(df_train, feature_scaler, lambda x: rate_encoding(x, max_rate=max_rate), encode_target_to_spikes)
print("Finished preprocessing training data...")
print()

# Load and preprocess validation data
print("About to load validation data...")
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//val_streamline_lag_close_(backup).db", 'XAUUSD')
print("Finished loading validation data...")
print()
print("Starting to preprocess validation data...")
# Preprocess validation data
X_val_encoded, y_val_encoded = preprocess_validation_data(df_val, feature_scaler, lambda x: rate_encoding(x, max_rate=max_rate), encode_target_to_spikes)
print("Finished preprocessing validation data...")
print()

    
def adjust_max_rate(model, base_max_rate=base_max_rate, scale_factor=scale_factor):
    """
    Adjusts the max_rate based on the current thresholds of the spiking layers in the model.

    :param model: The SNN model.
    :param base_max_rate: The base value for max_rate.
    :param scale_factor: Factor to scale the adjustment.
    :return: Adjusted max_rate.
    """
    thresholds = [layer.threshold for layer in model.layers if isinstance(layer, SpikingNeuronLayer)]
    if not thresholds:
        return base_max_rate

    # Concatenate and calculate the average threshold using TensorFlow operations
    all_thresholds = tf.concat(thresholds, axis=0)
    avg_threshold = tf.reduce_mean(all_thresholds)

    # Adjust max_rate based on the average threshold
    adjusted_max_rate = base_max_rate + scale_factor * avg_threshold
    return adjusted_max_rate

def stdp_threshold_update(pre_spike_train, post_spike_train, threshold, A_plus, A_minus, tau_plus, tau_minus, global_error_factor):
    """
    Compute STDP threshold update for a single neuron.

    :param pre_spike_train: Spike train of the presynaptic neuron.
    :param post_spike_train: Spike train of the postsynaptic neuron.
    :param threshold: Current threshold value.
    :param A_plus: STDP learning rate for threshold potentiation (similar to synaptic potentiation).
    :param A_minus: STDP learning rate for threshold depression (similar to synaptic depression).
    :param tau_plus: Time constant for threshold potentiation (similar to synaptic potentiation).
    :param tau_minus: Time constant for threshold depression (similar to synaptic depression).
    :return: Updated threshold value.
    """
    pre_spike_train = tf.cast(pre_spike_train, tf.float32)
    post_spike_train = tf.cast(post_spike_train, tf.float32)
    threshold = tf.cast(threshold, tf.float32)

    t_pre_expanded = tf.expand_dims(pre_spike_train, -1)
    t_post_expanded = tf.expand_dims(post_spike_train, 0)
    spike_diff = t_post_expanded - t_pre_expanded

    potentiation = tf.where(spike_diff > 0, A_plus * tf.exp(-spike_diff / tau_plus), 0.0)
    depression = tf.where(spike_diff < 0, A_minus * tf.exp(spike_diff / tau_minus), 0.0)

    delta_thresh = tf.reduce_sum(potentiation + depression)
    # Scale the STDP updates by the global error factor
    delta_thresh *= global_error_factor

    # Update and clip the threshold
    new_threshold = tf.clip_by_value(threshold + delta_thresh, clip_value_min=0, clip_value_max=1)
    return new_threshold



def stdp_update(pre_spike_train, post_spike_train, w, A_plus, A_minus, tau_plus, tau_minus, global_error_factor):
    """
    Compute STDP weight update for a single synapse, modulated by global error factor.

    :param pre_spike_train: Spike train of the presynaptic neuron.
    :param post_spike_train: Spike train of the postsynaptic neuron.
    :param w: Current synaptic weight.
    :param A_plus: STDP learning rate for potentiation.
    :param A_minus: STDP learning rate for depression.
    :param tau_plus: Time constant for potentiation.
    :param tau_minus: Time constant for depression.
    :param global_error_factor: Scaling factor based on the global error.
    :return: Updated synaptic weight.
    """

    pre_spike_train = tf.cast(pre_spike_train, tf.float32)
    post_spike_train = tf.cast(post_spike_train, tf.float32)
    w = tf.cast(w, tf.float32)

    # Expand dimensions to allow broadcasting
    t_pre_expanded = tf.expand_dims(pre_spike_train, -1)
    t_post_expanded = tf.expand_dims(post_spike_train, 0)

    # Compute the differences between all pairs of pre- and post-synaptic spikes
    spike_diff = t_post_expanded - t_pre_expanded

    # Calculate potentiation and depression updates
    potentiation = tf.where(spike_diff > 0, A_plus * tf.exp(-spike_diff / tau_plus), 0.0)
    depression = tf.where(spike_diff < 0, A_minus * tf.exp(spike_diff / tau_minus), 0.0)

    # Sum the updates over all spike pairs
    delta_w = tf.reduce_sum(potentiation + depression)

    # Scale the STDP updates by the global error factor
    delta_w *= global_error_factor

    # Update and clip the weight
    new_w = tf.clip_by_value(w + delta_w, clip_value_min=0, clip_value_max=1)
    return new_w

class SpikingNeuronLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(SpikingNeuronLayer, self).__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.threshold = self.add_weight(
            shape=(self.units,),
            initializer='LecunNormal',
            trainable=True,
            name='threshold'
        )

    def call(self, inputs):
        # Initialize state for the entire batch
        state = tf.zeros_like(inputs)

        # Vectorized accumulation
        accumulated = state + inputs

        # Firing condition applied across the entire batch
        fired = tf.greater_equal(accumulated, self.threshold)

        # Reset accumulation based on firing
        state = tf.where(fired, tf.zeros_like(accumulated), accumulated)

        # Cast fired to float for further processing
        fired_cast = tf.cast(fired, tf.float32)

        """ # Record the firing data
        self.spike_record = fired_cast """

        return fired_cast

    def get_spike_record(self):
        # Modify as per your implementation requirements
        return self.spike_record

    def get_thresholds(self):
        return self.threshold
    
    def stdp_threshold_update(self, pre_spike_train, post_spike_train):
        new_threshold = stdp_threshold_update(pre_spike_train, post_spike_train, self.threshold, A_plus, A_minus, tau_plus, tau_minus)
        self.threshold.assign(new_threshold)

class SpikeToContinuousLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpikeToContinuousLayer, self).__init__(**kwargs)

    def call(self, inputs):
        # Convert spike counts to continuous values
        return tf.reduce_sum(inputs, axis=1)  # Summing spikes

class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units
        self.is_connected_to_spiking_layer = False        

    def build(self, input_shape):
        # Initialize the weights of the layer
        initializer = tf.keras.initializers.LecunNormal()  # LecunNormal initializer
        print("Input shape to CustomDenseLayer:", input_shape)  # Debug print statement
        self.kernel = self.add_weight(shape=(int(input_shape[-1]), self.units),
                                      initializer=initializer,
                                      name='kernel')
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    name='bias')

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel) + self.bias

    def set_connected_to_spiking_layer(self, connected):
        self.is_connected_to_spiking_layer = connected

    def stdp_weight_update(self, pre_spike_train, post_spike_train):
        new_weights = stdp_update(pre_spike_train, post_spike_train, self.kernel, A_plus, A_minus, tau_plus, tau_minus)
        self.kernel.assign(new_weights)


def create_dual_input_model(input_shape, num_blocks, num_layers_per_block, units_per_layer, RNN):
    assert num_blocks == RNN, "Number of RNN iterations must match number of blocks"

    # Adjust input shape for sequential data
    input_normal = Input(shape=input_shape, name='input_normal', dtype='float32')
   
    num_features = int_shape(input_normal)[-1]

    # Type 1 & 2 RNN loop
    shared_custom_dense = CustomDenseLayer(num_features)
    shared_custom_dense.set_connected_to_spiking_layer(True)

    # Type 2: RNN loop
    shared_spiking_layer = SpikingNeuronLayer(num_features)

    """ # Type 1: Use the shared CustomDenseLayer, but creates new SpikingNeuronLayer each RNN loop
    rnn_outputs = []
    x_transformed = input_normal
    for _ in range(RNN):
        x_transformed_initial = shared_custom_dense(x_transformed)
        spiking_layer = SpikingNeuronLayer(num_features)
        x_spiked = spiking_layer(x_transformed_initial)
        x_transformed = Add()([x_spiked, input_normal])
        rnn_outputs.append(x_transformed) """

    # Type 2: Use the shared SpikingNeuronLayer & CustomDenseLayer for RNN loop
    rnn_outputs = []
    x_transformed = input_normal
    for _ in range(RNN):
        x_transformed_initial = shared_custom_dense(x_transformed)        
        x_spiked = shared_spiking_layer(x_transformed_initial)
        x_transformed = Add()([x_spiked, input_normal])
        rnn_outputs.append(x_transformed)

    projection_layer = CustomDenseLayer(units_per_layer)
    projection_layer.set_connected_to_spiking_layer(True)

    block_outputs = []
    for i in range(num_blocks):
        x_block_input = projection_layer(rnn_outputs[i])
        x_block_input = SpikingNeuronLayer(units=units_per_layer)(x_block_input)

        for j in range(num_layers_per_block):
            layer = CustomDenseLayer(units_per_layer)
            x_block_input = layer(x_block_input)
            x_block_input = SpikingNeuronLayer(units=units_per_layer)(x_block_input)
            layer.set_connected_to_spiking_layer(True)
        
        block_outputs.append(x_block_input)   

    continuous_outputs = []

    for block_output in block_outputs:
        # Convert each block's output to continuous values
        spike_to_continuous = SpikeToContinuousLayer()
        continuous_output = spike_to_continuous(block_output)
        continuous_outputs.append(continuous_output)

    # Concatenate the continuous outputs from all blocks
    concatenated = Concatenate(axis=-1)(continuous_outputs)

    # Final dense layer for prediction
    custom_output_layer = CustomDenseLayer(1)
    output = custom_output_layer(concatenated)
    model = Model(inputs=input_normal, outputs=output)

    return model

input_shape = X_train_encoded.shape[1:] 

# Create the model and get the layers
model = create_dual_input_model(input_shape, num_blocks, num_layers_per_block, units_per_layer, RNN)

# Inspect the model architecture
model.summary()

# Customizing the Adam optimizer
custom_adam = Adam(
    learning_rate = learning_rate, 
    beta_1 = beta_1,             # Momentum term (beta1)
    beta_2 = beta_2,             # Momentum term for the squared gradient (beta2)
    epsilon = epsilon,           # Small number to prevent any division by zero
    decay = decay,               # Weight decay for regularization
    amsgrad = amsgrad            # Whether to apply AMSGrad variant of Adam
)

# Compile the model with the customized adam optimizer. Spiking units aare handeled in train_ste
model.compile(optimizer=custom_adam, loss='mean_absolute_error') # mean_squared_error & mean_absolute_error

# Before loading weights
weights_before = [layer.get_weights() for layer in model.layers]

# Check if 'best_model_22.h5' exists
if os.path.exists('best_model_22.h5'):
    # Load weights if the file exists
    model.load_weights('best_model_22.h5')
    print("Weights loaded from 'best_model_22.h5'")
else:
    # Skip loading and possibly print a message or take other actions
    print("'best_model_22.h5' not found. Continuing without loading weights.")


# Check if the saved plot data file exists
if os.path.exists('best_model_22.pkl'):
    with open('best_model_22.pkl', 'rb') as f:
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
    plt.savefig('best_model_20.png')  # Save the plot with a constant filename

""" def animate_firing_pattern(firing_data, layer_idx=0, interval=100):
    
    Animate the firing pattern for a specific layer.
    
    :param firing_data: Collected firing data from the SpikingNeuronLayer.
    :param layer_idx: Index of the layer to visualize.
    :param interval: Time interval (in milliseconds) between frames.
   
    layer_data = firing_data[layer_idx]
    num_neurons, time_steps = layer_data.shape

    fig, ax = plt.subplots()
    mat = ax.matshow(layer_data[:, [0]], aspect='auto', cmap='binary')
    ax.set_xlabel('Time Steps')
    ax.set_ylabel('Neurons')
    ax.set_title(f'Layer {layer_idx+1} Firing Pattern Over Time')

    def update(frame):
        mat.set_data(layer_data[:, :frame])
        return [mat]

    ani = FuncAnimation(fig, update, frames=time_steps, interval=interval, blit=True)
    plt.show()

def quick_visualize_firing_pattern(firing_data, layer_idx=0):
    plt.figure(figsize=(10, 3))
    layer_data = firing_data[layer_idx]
    plt.imshow(layer_data.T, aspect='auto', cmap='binary')
    plt.title(f'Layer {layer_idx+1} Firing Pattern')
    plt.ylabel('Neurons')
    plt.xlabel('Time Steps')
    plt.colorbar(label='Firing (1: Fired, 0: Not Fired)')
    plt.show()

def visualize_firing_pattern(firing_data):
    num_layers = len(firing_data)
    plt.figure(figsize=(10, num_layers * 2))
    
    for i, layer_data in enumerate(firing_data):
        plt.subplot(num_layers, 1, i + 1)
        plt.imshow(layer_data.T, aspect='auto', cmap='binary')
        plt.colorbar(label='Firing (1: Fired, 0: Not Fired)')
        plt.ylabel(f'Layer {i+1} Neurons')
        plt.xlabel('Time Steps')
    
    plt.tight_layout()
    plt.show() """

def calculate_global_error(output, target):
    """
    Calculate the global error signal.

    :param output: The output of the network.
    :param target: The target values.
    :return: Global error signal.
    """
    # Example: Mean Squared Error
    global_error = tf.reduce_mean(tf.square(output - target))
    return global_error

def calculate_global_error_factor(global_error):
    """
    Transform the global error into a scaling factor for STDP updates.

    :param global_error: The calculated global error.
    :return: A scaling factor based on the global error.
    """
    # Example transformation: inverse scaling
    # Prevent division by zero with a small epsilon
    epsilon = 1e-7
    return 1 / (global_error + epsilon)

# Training loop
best_val_mae = float('inf')

no_improvement_epochs = 0

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model_22.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# @tf.function
def train_step(model, x_batch, y_batch, optimizer, loss_fn):
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)
        batch_loss = loss_fn(y_batch, predictions)

    gradients = tape.gradient(batch_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Calculate global error
    global_error = calculate_global_error(predictions, y_batch)
    global_error_factor = calculate_global_error_factor(global_error)

    batch_size = tf.shape(x_batch)[0]
    for i in tf.range(batch_size):
        x_sample = tf.expand_dims(x_batch[i], axis=0)
        model(x_sample, training=True)  # Forward pass for STDP update

        for layer in model.layers:
            if isinstance(layer, CustomDenseLayer) and layer.is_connected_to_spiking_layer:
                previous_layer_index = model.layers.index(layer) - 1
                previous_layer = model.layers[previous_layer_index]

                if isinstance(previous_layer, SpikingNeuronLayer) and hasattr(previous_layer, 'spike_record'):
                    pre_spike_train = previous_layer.spike_record
                    post_spike_train = layer.output

                    if len(pre_spike_train.shape) == 2:
                        pre_spike_train = tf.squeeze(pre_spike_train, axis=0)
                    if len(post_spike_train.shape) == 2:
                        post_spike_train = tf.squeeze(post_spike_train, axis=0)

                    # STDP weight update modulated by global error
                    layer.stdp_weight_update(pre_spike_train, post_spike_train, global_error_factor)

                    # STDP threshold update modulated by global error
                    new_threshold = stdp_threshold_update(pre_spike_train, post_spike_train, previous_layer.threshold, A_plus, A_minus, tau_plus, tau_minus, global_error_factor)
                    previous_layer.threshold.assign(new_threshold)

    avg_loss = tf.reduce_mean(batch_loss)
    avg_mae = tf.reduce_mean(tf.abs(y_batch - predictions))

    return avg_loss, avg_mae


# @tf.function
def validation_step(model, x_batch, y_batch, loss_fn):
    # Process the entire batch at once
    predictions = model(x_batch, training=False)
    loss = loss_fn(y_batch, predictions)

    # Calculate average loss for the batch
    avg_loss = tf.reduce_mean(loss)

    # Calculate mean absolute error for the batch
    avg_mae = tf.reduce_mean(tf.abs(y_batch - predictions))

    return avg_loss, avg_mae


# Adam optimizer for the custom training loop
optimizer = Adam(
    learning_rate = learning_rate, 
    beta_1 = beta_1,            # Momentum term (beta1)
    beta_2 = beta_2,            # Momentum term for the squared gradient (beta2)
    epsilon = epsilon,          # Small number to prevent any division by zero
    decay = decay,              # Weight decay for regularization
    amsgrad = amsgrad           # Whether to apply AMSGrad variant of Adam
)
loss_fn = tf.keras.losses.MeanAbsoluteError() 

# Create TensorFlow datasets using encoded targets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_encoded, y_train_encoded)).batch(batch_size)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val_encoded, y_val_encoded)).batch(batch_size)

def print_model_parameters(model):
    for layer in model.layers:
        if isinstance(layer, (CustomDenseLayer, SpikingNeuronLayer)):
            weights = layer.get_weights()
            print(f"Layer: {layer.name}, Weights: {weights}")

            if isinstance(layer, SpikingNeuronLayer):
                threshold = layer.threshold.numpy()
                print(f"Layer: {layer.name}, Threshold: {threshold}")

# visualization_interval = 1  # Define how often to visualize (e.g., visualize every epoch)

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

    # Dynamically adjust max_rate based on the model's current state
    current_max_rate = adjust_max_rate(model, base_max_rate=current_max_rate)

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

    train_losses.append(avg_train_mae)
    val_losses.append(avg_val_mae)

    """ # Periodic Visualization of Firing Patterns
    if epoch % visualization_interval == 0:
        # Collect firing data for visualization
        sample_x_batch, _ = next(iter(train_dataset))  # Get a sample batch from the training dataset
        _ = model(sample_x_batch, training=False)
        
        firing_data = []
        for layer in model.layers:
            if isinstance(layer, SpikingNeuronLayer):
                firing_data.append(layer.get_spike_record().numpy())

        # Visualize the comprehensive firing pattern across all layers
        visualize_firing_pattern(firing_data) """

    """ some_interval = 1
    # Optionally print parameters at the end of an epoch
    if epoch % some_interval == 0:
        print(f"Epoch: {epoch}")
        print_model_parameters(model) """

    # Update plot after each epoch
    update_plot(epoch, train_losses, val_losses)

     # Check for improvement
    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        no_improvement_epochs = 0  # Reset counter
        model.save_weights('best_model_22.h5')  # Save best model
    else:
        no_improvement_epochs += 1

    # Early stopping check
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    # Save plot data after each epoch
    with open('plot_data_22.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses), f)

# Close plot and load best model weights
plt.close()

# After the training loop
print("Training completed")

# Loading best model weights for evaluation
model.load_weights('best_model_22.h5')

""" # Choose a batch of data for visualization
x_visualize, _ = next(iter(train_dataset))  # or use a different dataset

# Forward pass for visualization
_ = model(x_visualize, training=False)

# Collect firing data from each SpikingNeuronLayer
firing_data = []
for layer in model.layers:
    if isinstance(layer, SpikingNeuronLayer):
        firing_data.append(layer.get_spike_record().numpy())

# Using the visualization functions
animate_firing_pattern(firing_data, layer_idx=0, interval=100)
quick_visualize_firing_pattern(firing_data, layer_idx=0)
visualize_firing_pattern(firing_data) """

# Model evaluation after training completion
if epochs == 0 or no_improvement_epochs >= patience:
    # Evaluate model
    y_pred = model.predict(X_val_encoded).flatten()
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)

    print(f"Model Mean Squared Error on Validation Set: {mse}")
    print(f"Model Mean Absolute Error on Validation Set: {mae}")

    # Save the actual and predicted values to a DataFrame
    result_df = pd.DataFrame({"Actual": y_val, "Predicted": y_pred})
    
    # Save result DataFrame to a CSV file
    file_path = 'best_model_predictions_22.csv'
    print("Saving file to:", file_path)
    result_df.to_csv(file_path, index=False)
    print("File saved successfully!")

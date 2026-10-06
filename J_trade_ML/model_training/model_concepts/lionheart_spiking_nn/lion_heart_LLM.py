import os
import pickle
import time
import numpy as np
import json
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, LayerNormalization, Layer 
from tensorflow.keras.models import Model
from tensorflow.keras.backend import int_shape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress TensorFlow warnings (due to custom learning with synaptic modulation)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.config.experimental.set_visible_devices
print("Eager execution:", tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

""" # Mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy) """

""" 
:Welcome to Lion Heart Regression. A 4D custom built neural network inspired by the human brain.
This NN converts data into 4D shapes [batch, seq_len, features, spike_trains], spike train 
encoding is a custom encoding method that converts data into a spiking frequency of 0's and 1's, 
which is represented as a temporal sequence of 0's and 1's for each feature in a sample. You can
control its frequency of spikes and it's duration. Each spike train sequence across features is
also tied to a sequence length of past samples each broken up into a spike train sequence across 
features, which considers historical time to predict the most current sample target. 

This data runs to the model in slices of spike train sequences across features one at a time
for each sample within its sequence length. the first layer is the synaptic connections, 
like any dense layer, but it does not sum the inputs, it processes each slice of data at a time
across its weights (leaving zero's zero and 1 * weight), without summing nor any form of activation. 
Each output slice goes into the spiking layer, where it has a membrane that accumulates all incoming
values at each unit separately across time from all slices in a sample tied to its entire sequence 
length. Each membrane is tied to a learnable threshold, in which when a unit's membrane exceeds
it's unit threshold, it fires a 0 or a 1, and resets the membrane accumulation to a resting value,
which also has a natural decay during accumulation. This model is not the final design, so how in 
terms of how the model takes these sliced outputs from this spiking layer and interprets that,
is not yet designed with the best intentions.

This model completely changes how NN's function. This model currently is only 1 layer deep, which
in the next version, these modifications will allow multiple layers. This model is designed 
for regression task. Soon future models moderations will include Auto Regressive, and 
classification with SoftMax outputs, leading to an LLM design, and later for continuous state 
machines, like in robotics for continuous motor activators to simulate movement through reactions. 

Currently this model does not include any recurring Neural Network mechanisms, due to carful 
required consideration. Future models will include a biological concept called mirroring neuros, 
which will help provide a better working memory separate from needing to solve never before seen 
task (aka it will be able to feed itself data (compress relationships to important samples from 
it's training) to generate a dream state for deeper level learning), and lateral back and 
forth movement between units within each layer and the layer themselves.


By: JD (ThinkBe)
"""

# Load training and validation data
train_data_loc = {"path": "C://Users//crgon//OneDrive//Desktop//trading//data//streamline_snn.db", "dataset": "XAUUSD"}
val_data_loc = {"path": "C://Users//crgon//OneDrive//Desktop//trading//data//val_streamline_snn.db", "dataset": "XAUUSD"}

# Custom SNN parameters
train_data_cutoff = 0.03            # % of training data to load (0.001 - 1)

num_blocks = 1                      # Not currently in use
units = 2500 
num_layers = 2                      # Not currently in use
RNN = num_blocks                    # number of times to recycle the output by adding inputs for continus data transformation for each main block's input

"""
:Population Encoding Parameters: These parameters are used for the Population Encoding Layer in the Spiking Neural Network.

:population_neurons: The number of neurons in the Population Encoding Layer. This value represents the dimensionality of the population code and should be chosen based on the complexity of the data being encoded. 
:decoded_dim: The dimensionality of the output after reverse population encoding. This should match the dimensionality of the original target data. 
"""
population_neurons = 2500           # The dimensionality of the population code, adjust based on data complexity
""" 
:param this will be for LLM configuration with Softmax final layer.

This will be dynamically adjusted based on the statistical accuracy of the final tokenized output.
If the most accurate prediction is say 70% confident, and the threshold is 80%, the output of
Side 1 - Forward processing for each block, will recycle back as inputs for each block, each adding.
a residual connection from rnn_outputs, each matching the number of blocks from num_blocks. 

Since this is a completely custom Spiking NN design, units that fire stay inactive on the next cycle.
when the model does not feel confident about its prediction, and the cycle will keep repeating until
the threshold is reached. It's a form of automatic thinking. 

thinking_threshold= .80 
"""
epochs = 30 
batch_size = 1                        # Batch_size must be low (1 - 3) for the model to learn
patience = 10                         # Define patience for early stopping

""" 
:STDP parameters: this is all but the final output dense layer, for STDP backpropagation

:param A_plus: STDP learning rate for potentiation.
:param A_minus: STDP learning rate for depression.
:param tau_plus: Time constant for potentiation.
:param tau_minus: Time constant for depression. 
"""
A_plus = 0.01                        # default range = 0.001 - 0.1
A_minus = 0.05                        # default range = 0.001 - 0.02
tau_plus = 15                         # default range = 10 - 100ms+
tau_minus = 15                        # default range = 10 - 100ms+

"""
:param tau_mem: Membrane time constant of the Leaky Integrate-and-Fire (LIF) neuron model, determining the rate of decay of the membrane potential.
:param v_rest: Resting membrane potential for the LIF neurons, representing the baseline potential when the neuron is not activated.
"""
tau_mem = 20.0                        # Default range = 10 - 100ms
v_rest = 0.0001                        # Typical range = -1 to 1 for normalized potentials, 0 for non-negative activations

"""
:Threshold initializer weight range

:param minval: min weight initializer Threshold distribution.
:param maxval: max weight initializer Threshold distribution
"""
minval = 1.0                          # Default range based on data
maxval = 20.0                         # Default range based on data

""" 
:param max_rate: The upper limit on the firing rate of neurons in the network, typically set between 10.0 and 200Hz.
:param base_max_rate: The starting value for the max_rate, usually within the range of 20.0 to 100Hz.
:param current_max_rate: A dynamic parameter representing the current maximum firing rate, initially set to the value of base_max_rate.
:param scale_factor: A coefficient used to scale adjustments in max_rate based on the average neuronal threshold, with a default range between 0 and 1. 
"""
base_max_rate = 20.0                # default range = 20.0 - 100Hz+
current_max_rate = base_max_rate     # dynamic parameter
max_rate = 200                       # default range = 10.0 - 200Hz+
scale_factor = 0.50                  # default range = 0 - 1

"""
:Duration and Time Step of Each Spike Train:

:param duration: The total time span of each spike train in seconds. This parameter determines the length of time over which spikes can occur.
:param time_step: The time resolution for each step in the spike train, measured in seconds. This parameter defines the granularity or precision of the spike train.

The total number of time steps (and hence, the length of the spike train array) is determined by dividing the duration by the time_step. Each entry in the 
spike train array represents whether a spike occurs at that time step (1 for spike, 0 for no spike). A smaller time_step results in more granular spike 
trains, capturing more precise spike timings.
"""
seq_len = 5
duration = 2.00                      # 1 is = to 1 second
time_step = 0.01                     # 1 / max_rate  # This ensures time_step is appropriately scaled
                  
record_spikes = True                 # This is for either custom visualization of NN or to improve learning
exclude_columns = True               # Remove unused columns in the data set (currently 12:25)

# Ensure that max_rate * duration is sufficient for the precision
# assert max_rate * duration >= 10**2, "The product of max_rate and duration is not sufficient to maintain the required precision."

""" 
Secondary learning parameters 
"""
learning_rate = 0.01                # Matched learning rate (default = 0.001)
beta_1=0.9                          # Momentum term (beta1) (default = 0.9)
beta_2=0.999                        # Momentum term for the squared gradient (beta2) (default = 0.999)
epsilon=1e-07                       # Small number to prevent any division by zero (default = 1e-07)
decay=0.0                           # Weight decay for regularization (default = 0.0)
amsgrad=False

def generate_spike_train(rate, duration, time_step):
    """
    Generate a spike train for a given rate over a specified duration and time step.

    :param rate: Spike rate.
    :param duration: Duration of the spike train.
    :param time_step: Time step for the spike train.
    :return: Generated spike train.
    """
    num_time_steps = int(duration / time_step)
    spikes = np.random.rand(num_time_steps) < (rate * time_step)
    return spikes.astype(int)

def rate_encoding_to_spike_train(X, max_rate, duration, time_step):
    """
    Converts feature values to spike trains using rate encoding.

    :param X: Input feature matrix.
    :param max_rate: The maximum spike rate.
    :param duration: Duration of each spike train.
    :param time_step: Time step for each spike train.
    :return: Spike trains corresponding to the feature values.
    """
    epsilon = 1e-7  # Small constant to prevent division by zero
    X_min = X.min(axis=0)
    X_max = X.max(axis=0)

    # Normalization with a small constant to avoid division by zero
    X_normalized = (X - X_min) / (X_max - X_min + epsilon)
    X_normalized = np.nan_to_num(X_normalized)  # Handling potential NaNs

    spike_rates = X_normalized * max_rate

    # Use generate_spike_train to create spike trains
    spike_trains = np.array([[generate_spike_train(rate, duration, time_step) 
                              for rate in sample.flatten()] 
                             for sample in spike_rates])

    return np.array(spike_trains)

# Data Loading Function
def load_data_from_sqlite(data_loc, columns=None):
    """    
    Parameters:
    - data_loc: A dictionary with 'path' to the database file and 'dataset' as the table name.
    - columns: Optional list of columns to select from the dataset.
    
    Returns:
    - df: A pandas DataFrame containing the loaded data.
    """
    conn = sqlite3.connect(data_loc['path'])
    if columns:
        query = f"SELECT {','.join(columns)} FROM {data_loc['dataset']}"
    else:
        query = f"SELECT * FROM {data_loc['dataset']}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_data(df, max_rate, duration, time_step, seq_len, train_data_cutoff, exclude_columns, is_training):
    """
    Preprocess data for SNN, treating each row as a sequence.

    :param df: DataFrame containing data.
    :param max_rate: Maximum spike rate for encoding.
    :param duration: Duration of spike train.
    :param time_step: Time step for spike train.
    :param seq_len: Length of the sequence to be considered.
    :param exclude_columns: Boolean to control exclusion of columns 5 to 25.
    :return: Processed spike train data and targets.
    """
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")    
    
    # Drop the 'index' column if it's present
    if 'index' in df.columns:
        df = df.drop('index', axis=1)

    # Exclude columns if specified
    if exclude_columns:
        df = df.drop(df.columns[4:-1], axis=1)   
        
        # Print remaining feature names and their count
        print("Remaining features:", df.columns[:-1].tolist())  # Exclude 'Target' column from feature names
        print("Number of features:", len(df.columns) - 1)  # Subtract 1 to exclude 'Target' column from count 

    # Replace infinities and fill NaNs before converting to NumPy array
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    total_sequences = len(df) - seq_len + 1

    if is_training:
        # For training data, select sequences from the bottom up based on train_data_cutoff
        start_index = total_sequences - int(total_sequences * train_data_cutoff)
        sampled_start_indices = np.arange(start_index, total_sequences)
    else:
        # For validation data, use all sequences
        sampled_start_indices = np.arange(total_sequences)

    df_values = df.values.astype('float32')
    sampled_sequences = np.array([df_values[idx:idx + seq_len] for idx in sampled_start_indices])

    # Separate features and target
    features = sampled_sequences[:, :, :-1]
    target = df_values[sampled_start_indices, -1]

    # Assume rate_encoding_to_spike_train is a predefined function
    encoded_sequences = np.array([rate_encoding_to_spike_train(sequence, max_rate, duration, time_step) for sequence in features])

    batch_size, seq_len, features, spike_train_sequences = encoded_sequences.shape

    return encoded_sequences, np.array(target)

# Load and preprocess training data
print("About to load training data...")
df_train = load_data_from_sqlite(train_data_loc)
print("Finished loading training data...")
print("Starting to preprocess training data...")
X_train_encoded, y_train_original = preprocess_data(df_train, max_rate, duration, time_step, seq_len, train_data_cutoff, exclude_columns, is_training=True)
print("Finished preprocessing training data...")
print()

# Load and preprocess validation data
print("About to load validation data...")
df_val = load_data_from_sqlite(val_data_loc)
print("Finished loading validation data...")
print("Starting to preprocess validation data...")
X_val_encoded, y_val_original = preprocess_data(df_val, max_rate, duration, time_step, seq_len, train_data_cutoff, exclude_columns, is_training=True)
print("Finished preprocessing validation data...")
print()

# Identify unique targets across both datasets
all_unique_targets = np.unique(np.concatenate((y_train_original, y_val_original)))

# Create target to integer mapping for decimal targets
target_to_int_map = {target: i for i, target in enumerate(all_unique_targets)}

# Calculate the number of unique classes - this remains unchanged
num_classes = len(all_unique_targets)

# Function to apply mapping to original decimal targets
def apply_target_mapping(y_original, mapping):
    return np.array([mapping[target] for target in y_original])

# Apply mapping to training and validation original targets
y_train = apply_target_mapping(y_train_original, target_to_int_map)
y_val = apply_target_mapping(y_val_original, target_to_int_map)

# Convert keys to strings if they are not already in a JSON serializable format
target_to_int_map_str_keys = {str(key): value for key, value in target_to_int_map.items()}

# Now, save this dictionary with string keys to a file
with open('target_to_int_map.json', 'w') as f:
    json.dump(target_to_int_map_str_keys, f)

class PopulationEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, population_neurons, **kwargs):
        super(PopulationEncodingLayer, self).__init__(**kwargs)
        self.population_neurons = population_neurons

    def build(self, input_shape):
        # Adjusted for the potentially new shape after pooling
        self.encoding_weights = self.add_weight(
            shape=(input_shape[-1], self.population_neurons),
            initializer=tf.keras.initializers.LecunNormal(),
            trainable=True,
            name='encoding_weights'
        )
        self.biases = self.add_weight(
            shape=(self.population_neurons,),
            initializer='zeros',
            trainable=True,
            name='biases'
        )

    def call(self, inputs):
        # Assuming inputs are already pooled and reshaped appropriately
        encoded = tf.matmul(inputs, self.encoding_weights) + self.biases
        return tf.nn.relu(encoded)

    
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

    # Concatenate and calculate the average threshold
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
    def __init__(self, units, tau_mem, v_rest, minval, maxval, record_spikes, **kwargs):
        super(SpikingNeuronLayer, self).__init__(**kwargs)
        self.units = units
        self.tau_mem = tau_mem
        self.v_rest = v_rest
        self.minval = minval
        self.maxval = maxval
        self.record_spikes = record_spikes  # Flag to control spike recording
        self.threshold = None
        self.V_mem = None
        self.spike_record = [] if record_spikes else None

    def build(self, input_shape):
        self.threshold = self.add_weight(
            shape=(1, self.units),
            initializer=tf.random_uniform_initializer(self.minval, self.maxval),
            trainable=True,
            name='threshold')
        super().build(input_shape)

    def call(self, inputs, training=None, reset_state=False):
        if self.V_mem is None or reset_state:
            batch_size = tf.shape(inputs)[0]
            # Initialize V_mem for each neuron in each batch
            self.V_mem = tf.fill([batch_size, self.units], self.v_rest)

        # inputs shape: [batch_size, seq_len, features]
        # Expand V_mem to match inputs shape for broadcasting (fuly connected)
        V_mem_expanded = tf.expand_dims(self.V_mem, axis=1)  # Now [batch_size, 1, units]

        # Calculate membrane potential
        new_V_mem = V_mem_expanded + inputs - (V_mem_expanded - self.v_rest) / self.tau_mem
        
        # Determine firing based on threshold
        fired = tf.greater_equal(new_V_mem, self.threshold)
        spikes = tf.cast(fired, tf.float32)

        # Update V_mem based on firing
        self.V_mem = tf.where(fired[:, -1, :], self.v_rest, new_V_mem[:, -1, :])

        # Handle spike recording if enabled
        if self.record_spikes:
            self.spike_record.append(spikes)

        return spikes

    def reset_states(self, batch_size=None):
        if batch_size is None:
            batch_size = tf.shape(self.V_mem)[0]
        self.V_mem = tf.fill([batch_size, self.units], self.v_rest)
        if self.record_spikes:
            self.spike_record = []

    def get_config(self):
        config = super(SpikingNeuronLayer, self).get_config()
        config.update({
            "units": self.units,
            "tau_mem": self.tau_mem,
            "v_rest": self.v_rest,
            "minval": self.minval,
            "maxval": self.maxval
        })
        return config

    def get_spike_record(self):
        if not self.record_spikes:
            raise ValueError("Spike recording is disabled. Enable spike recording to access the spike record.")
        return tf.concat(self.spike_record, axis=0) if self.spike_record else tf.zeros((0, self.units), dtype=tf.float32)    
    
    def reset_spike_record(self):
        # Reset the spike record
        self.spike_record = []
    
    def stdp_threshold_update(self, pre_spike_train, post_spike_train, global_error_factor):
        new_threshold = stdp_threshold_update(pre_spike_train, post_spike_train, self.threshold, A_plus, A_minus, tau_plus, tau_minus, global_error_factor)
        self.threshold.assign(new_threshold)

class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=False, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units  
        self.is_connected_to_spiking_layer = False      
        self.use_bias = use_bias
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(
            name='kernel', 
            shape=(input_dim, self.units),
            initializer='glorot_uniform', 
            trainable=True
        )
        
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias', 
                shape=(self.units,),
                initializer='zeros', 
                trainable=True
            )
        super(CustomDenseLayer, self).build(input_shape)  # Mark the layer as built

    def call(self, inputs):
        inputs = tf.cast(inputs, tf.float32)

        # Matrix multiplication
        outputs = tf.matmul(inputs, self.kernel)

        if self.use_bias:
            outputs += self.bias

        # Conditional activation
        if not self.is_connected_to_spiking_layer:
            # Use linear activation for layers not connected to a spiking layer
            return outputs
        else:
            # Use activation for layers connected to a spiking layer
            return tf.nn.relu(outputs) # tf.nn.relu, tf.nn.sigmoid, tf.tanh

    def get_config(self):
        config = super(CustomDenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
        })
        return config

    def set_connected_to_spiking_layer(self, connected):
        self.is_connected_to_spiking_layer = connected

    def stdp_weight_update(self, pre_spike_train, post_spike_train, global_error_factor):
        new_weights = stdp_update(pre_spike_train, post_spike_train, self.kernel, A_plus, A_minus, tau_plus, tau_minus, global_error_factor)
        self.kernel.assign(new_weights)


class CustomPoolingLayer(Layer):
    def __init__(self, **kwargs):
        super(CustomPoolingLayer, self).__init__(**kwargs)
    
    def call(self, inputs):
        # [batch, seq_len, features, spike_trains]
        if len(inputs.shape) != 4:
            raise ValueError(f"CustomPoolingLayer expects inputs with 4 dimensions [batch, seq_len, features, spike_trains], got {inputs.shape}")

        # Apply max pooling across the seq_len dimension (axis=1)
        pooled_outputs = tf.reduce_max(inputs, axis=1, keepdims=True)
        
        # Flatten the pooled outputs to prepare for the PopulationEncodingLayer
        # [batch, 1, features, spike_trains] to [batch, features * spike_trains]
        # removed the redundant dimension resulting from keepdims=True in the pooling step
        reshaped_outputs = tf.reshape(pooled_outputs, [tf.shape(pooled_outputs)[0], -1])
        
        return reshaped_outputs

    
class AccumulateSpikeTrainsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.accumulated_outputs = []

    def call(self, inputs):
        # [batch_size, seq_len, units]
        self.accumulated_outputs.append(inputs)
        return inputs

    def get_accumulated_outputs(self):
        # [batch, seq_len, features, spike_trains]        
        accumulated_tensor = tf.stack(self.accumulated_outputs, axis=1)  # Now [batch_size, accumulated_seq_len, seq_len, units]
        # If accumulated_seq_len == seq_len and you're treating each unit's output as a "spike train", adjust accordingly
        accumulated_tensor = tf.reshape(accumulated_tensor, [tf.shape(accumulated_tensor)[0], tf.shape(accumulated_tensor)[1], -1, 1])  
        # Adjust to [batch, seq_len, features, 1] if treating units as features
        return accumulated_tensor

    def reset_states(self):
        self.accumulated_outputs = []

class SequentialSpikeTrainModel(Model):
    def __init__(self, units, population_neurons, tau_mem, v_rest, minval, maxval, record_spikes, num_classes, input_shape=None):
        super(SequentialSpikeTrainModel, self).__init__()
        self._input_shape = input_shape
        self.units = units
        self.population_neurons = population_neurons
        self.num_classes =num_classes
        
        # Initialize layers
        self.custom_dense_layer = CustomDenseLayer(self.units)
        self.custom_dense_layer.set_connected_to_spiking_layer(True)
        self.spiking_neuron_layer = SpikingNeuronLayer(self.units, tau_mem, v_rest, minval, maxval, record_spikes)
        self.accumulate_layer = AccumulateSpikeTrainsLayer()
        self.custom_pooling_layer = CustomPoolingLayer()        
        self.population_encoding_layer = PopulationEncodingLayer(self.population_neurons)
        self.layer_normalization = LayerNormalization(axis=-1)
        self.final_dense_layer = CustomDenseLayer(self.num_classes)

    def call(self, inputs, training=False):
        # Ensure inputs have a seq_len dimension
        if len(inputs.shape) == 3:
            # [batch_size, features, spike_trains]
            inputs = tf.expand_dims(inputs, axis=1)
        
        # [batch_size, seq_len, features, spike_trains]
        for i in range(inputs.shape[1]):
            input_slice = inputs[:, i, :, :]
            slice_output = self.custom_dense_layer(input_slice)            
            spiking_output = self.spiking_neuron_layer(slice_output)
            self.accumulate_layer(spiking_output)

        accumulated_output = self.accumulate_layer.get_accumulated_outputs()

        # Process the accumulated outputs through the subsequent layers
        pooled_output = self.custom_pooling_layer(accumulated_output)
        encoded_output = self.population_encoding_layer(pooled_output)
        normalized_output = self.layer_normalization(encoded_output)
        logits = self.final_dense_layer(normalized_output)
        final_output = tf.nn.softmax(logits)

        return final_output
    
    def reset_states(self):
        # Explicitly reset states of stateful layers
        self.accumulate_layer.reset_states()
        self.spiking_neuron_layer.reset_states()
        self.spiking_neuron_layer.reset_spike_record()

""" class SequentialSpikeTrainModel(Model):
    def __init__(self, units, population_neurons, num_layers, tau_mem, v_rest, minval, maxval, record_spikes, input_shape=None):
        super(SequentialSpikeTrainModel, self).__init__()
        self._input_shape = input_shape
        self.units = units
        self.population_neurons = population_neurons
        self.num_layers = num_layers
        
        # Initialize custom dense and spiking neuron layers as lists
        self.custom_dense_layers = []
        self.spiking_neuron_layers = []
        for _ in range(self.num_layers):
            custom_dense_layer = CustomDenseLayer(self.units)
            custom_dense_layer.set_connected_to_spiking_layer(True) 
            self.custom_dense_layers.append(custom_dense_layer)

            spiking_neuron_layer = SpikingNeuronLayer(self.units, tau_mem, v_rest, minval, maxval, record_spikes)
            self.spiking_neuron_layers.append(spiking_neuron_layer)

        # Other layers
        self.accumulate_layer = AccumulateSpikeTrainsLayer()
        self.custom_pooling_layer = CustomPoolingLayer()
        self.layer_normalization = LayerNormalization(axis=-1)
        self.population_encoding_layer = PopulationEncodingLayer(self.population_neurons)
        self.final_dense_layer = CustomDenseLayer(1)

    def call(self, inputs, training=False):
        if len(inputs.shape) == 3:
            inputs = tf.expand_dims(inputs, axis=1)
        
        # Initialize an empty list to collect outputs from the last layer of each timestep
        final_outputs = []

        for i in range(inputs.shape[1]):  # Loop over time steps
            input_slice = inputs[:, i, :, :]  # Shape: [batch_size, features, spike_trains]
            for j in range(self.num_layers):  # Loop over layers
                # Process input_slice through j-th CustomDenseLayer and SpikingNeuronLayer
                slice_output = self.custom_dense_layers[j](input_slice)
                input_slice = self.spiking_neuron_layers[j](slice_output)  # Update input_slice for the next layer

            # accumulate output of the last layer
            self.accumulate_layer(input_slice)  # Note: input_slice now holds the output of the last SpikingNeuronLayer

        accumulated_output = self.accumulate_layer.get_accumulated_outputs()
        pooled_output = self.custom_pooling_layer(accumulated_output)
        normalized_output = self.layer_normalization(pooled_output)
        encoded_output = self.population_encoding_layer(normalized_output)
        final_output = self.final_dense_layer(encoded_output)

        return final_output


    def reset_states(self):
        self.accumulate_layer.reset_states()
        for layer in self.spiking_neuron_layers:
            layer.reset_states()
            layer.reset_spike_record() """


seq_len = X_train_encoded.shape[1]
features = X_train_encoded.shape[2]
spike_trains = X_train_encoded.shape[3]

input_shape = (features, spike_trains)

# Create the model and get the layers
model = SequentialSpikeTrainModel(units=units,
                                  population_neurons=population_neurons,                                 
                                  tau_mem=tau_mem, 
                                  v_rest=v_rest, 
                                  minval=minval, 
                                  maxval=maxval, 
                                  record_spikes=record_spikes,
                                  num_classes=num_classes,
                                  input_shape=(features, spike_trains)) 

# Creates a dummy input 
dummy_input = tf.random.normal([batch_size, seq_len, features, spike_trains])

model(dummy_input)

# call summary() 
model.summary()

# Adam optimizer
optimizer = Adam(
    learning_rate = learning_rate, 
    beta_1 = beta_1,             # Momentum term (beta1)
    beta_2 = beta_2,             # Momentum term for the squared gradient (beta2)
    epsilon = epsilon,           # Small number to prevent any division by zero
    decay = decay,               # Weight decay for regularization
    amsgrad = amsgrad            # Whether to apply AMSGrad variant of Adam
)

loss_fn = loss_fn = SparseCategoricalCrossentropy(from_logits=False) 
model.compile(optimizer=optimizer, loss=loss_fn, metrics=['accuracy']) 

# Before loading weights
weights_before = [layer.get_weights() for layer in model.layers]

# Check if 'lion_heart_classification_2.h5' exists
if os.path.exists('lion_heart_classification_2.h5'):
    # Load weights if the file exists
    model.load_weights('lion_heart_classification_2.h5')
    print("Weights loaded from 'lion_heart_classification_2.h5'")
else:
    # Skip loading and print 
    print("'lion_heart_classification_2.h5' not found. Continuing without loading weights.")


# Check if the saved plot data file exists
if os.path.exists('lion_heart_classification_2.pkl'):
    with open('lion_heart_classification_2.pkl', 'rb') as f:
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
    plt.draw()  
    plt.pause(0.01)  # Pause to update the figure
    plt.savefig('lion_heart_classification_2.png')  

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
    target_int = tf.cast(target, tf.int32)
    # Since output is probabilities (after softmax), set from_logits=False
    return tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target_int, output, from_logits=False))


def calculate_global_error_factor(global_error):
    """
    Transform the global error into a scaling factor for STDP updates.

    :param global_error: The calculated global error.
    :return: A scaling factor based on the global error.
    """
    # inverse scaling
    # Prevent division by zero with a small epsilon
    epsilon = 1e-7
    return 1 / (global_error + epsilon)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('lion_heart_classification_2.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Now, use y_train and y_val for dataset creation
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_encoded, y_train)).batch(batch_size)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val_encoded, y_val)).batch(batch_size)

# Your batch counts remain unchanged
train_batch_count = np.ceil(len(X_train_encoded) / batch_size).astype(int)  
validation_batch_count = np.ceil(len(X_val_encoded) / batch_size).astype(int)

""" visualization_interval = 1  # Define how often to visualize (e.g., visualize every epoch) """

# Initialize metrics tracking
train_losses = []
train_accuracies = [] 
val_losses = []
val_accuracies = []
best_val_accuracy = -np.inf
no_improvement_epochs = 0

# Placeholder for debugging callback
class DebuggingCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        print(f"Starting Epoch {epoch+1}")
    
    def on_epoch_end(self, epoch, logs=None):
        print(f"Ending Epoch {epoch+1}")
    
    def on_train_batch_end(self, batch, logs=None):
        print(f"Finished training batch {batch}")
    
    def on_test_batch_end(self, batch, logs=None):
        print(f"Finished validation batch {batch}")

debugging_callback = DebuggingCallback()

def calculate_accuracy(predictions, labels):
    predicted_classes = tf.argmax(predictions, axis=1)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_classes, tf.cast(labels, tf.int64)), tf.float32))
    return accuracy

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    debugging_callback.on_epoch_begin(epoch)
    total_train_loss = 0
    total_train_accuracy = 0
    train_samples = 0

    total_val_loss = 0
    total_val_accuracy = 0    
    val_samples = 0

    current_max_rate = adjust_max_rate(model, base_max_rate=current_max_rate)  # Adjust learning rate

    # Initialize the progress bar
    with tqdm(total=len(list(train_dataset)), desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar_train:
        for x_batch, y_batch in train_dataset:
            actual_batch_size = x_batch.shape[0]

            if actual_batch_size != batch_size:
                print(f"Skipping last batch of size {actual_batch_size}.")
                continue

            with tf.GradientTape() as tape:
                model.reset_states()
                predictions = model(x_batch, training=True) 
                global_error = calculate_global_error(predictions, y_batch)
                global_error_factor = calculate_global_error_factor(global_error)

                activations = {}  # Dictionary to store activations for each layer

                # Manually forward pass to capture activations
                for i in range(seq_len):
                    x_slice = x_batch[:, i, :, :]
                    for layer in model.layers:
                        if isinstance(layer, (CustomDenseLayer, SpikingNeuronLayer)):
                            x_slice = layer(x_slice, training=True)
                            activations[layer.name] = x_slice

                loss = loss_fn(y_batch, predictions) 

                # selective gradient application and STDP updates
                stdp_layers = [(layer, model.layers[index - 1]) for index, layer in enumerate(model.layers) 
                            if isinstance(layer, CustomDenseLayer) and layer.is_connected_to_spiking_layer 
                            and index > 0 and isinstance(model.layers[index - 1], SpikingNeuronLayer)]

                non_stdp_variables = [var for var in model.trainable_variables if all(var not in layer.trainable_variables for layer, _ in stdp_layers)]
                gradients = tape.gradient(loss, non_stdp_variables)
                optimizer.apply_gradients(zip(gradients, non_stdp_variables))

                for layer in model.layers:
                    if isinstance(layer, CustomDenseLayer) and layer.is_connected_to_spiking_layer:
                        previous_layer_index = model.layers.index(layer) - 1
                        previous_layer = model.layers[previous_layer_index]

                        if isinstance(previous_layer, SpikingNeuronLayer):
                            pre_spike_train = previous_layer.get_spike_record()
                            post_spike_train = activations[layer.name]

                            if pre_spike_train.shape[0] != post_spike_train.shape[0]:
                                pre_spike_train = tf.reshape(pre_spike_train, [post_spike_train.shape[0], *pre_spike_train.shape[1:]])

                            layer.stdp_weight_update(pre_spike_train, post_spike_train, global_error_factor)
                            previous_layer.stdp_threshold_update(pre_spike_train, post_spike_train, global_error_factor)

                # For classification
                accuracy = calculate_accuracy(predictions, y_batch)
                total_train_loss += loss.numpy()
                # Track accuracy as a sum to average later
                total_train_accuracy += accuracy.numpy() * actual_batch_size  
                train_samples += actual_batch_size

            pbar_train.set_postfix(Loss=f"{total_train_loss / train_samples:.4f}", Accuracy=f"{total_train_accuracy / train_samples:.4f}")
            pbar_train.update(1)

    pbar_train.close()
    avg_train_accuracy = total_train_accuracy / train_samples
    avg_train_loss = total_train_loss / train_samples

    print(f"Training Complete - Avg Loss: {avg_train_loss:.4f}, Avg Accuracy: {avg_train_accuracy:.4f}")
    print("Starting Validation Phase")

    # Validation phase, simplified without storing activations
    with tqdm(total=len(list(validation_dataset)), desc=f"Validation Epoch {epoch + 1}", unit="batch") as pbar_val:
        for x_batch_val, y_batch_val in validation_dataset:
            actual_batch_size_val = x_batch_val.shape[0]

            if actual_batch_size != batch_size:
                print(f"Skipping last batch of size {actual_batch_size}.")
                continue
            
            model.reset_states()  # Reset model state at the start of each new batch

            predictions_val = []  # List to accumulate predictions for each time step
            for i in range(seq_len):
                x_slice_val = x_batch_val[:, i, :, :]
                for layer in model.layers:
                    if isinstance(layer, (CustomDenseLayer, SpikingNeuronLayer)):
                        x_slice_val = layer(x_slice_val, training=False)  # 'training=False' for validation
                        
                # Collect predictions after processing all time steps, assuming last step holds final predictions
                if i == seq_len - 1:  # If at the last step, collect predictions
                    predictions_val.append(x_slice_val)

            # Convert list of predictions to a tensor (adjust shape/concatenation logic as needed)
            predictions_val = tf.concat(predictions_val, axis=0)

            # Calculate validation loss and accuracy
            loss_val = loss_fn(y_batch_val, predictions_val)
            accuracy_val = calculate_accuracy(predictions_val, y_batch_val)  # Ensure this is defined to suit your classification task

            # Update validation totals
            total_val_loss += loss_val.numpy() * actual_batch_size_val  # Adjust for actual batch size
            total_val_accuracy += accuracy_val.numpy() * actual_batch_size_val
            val_samples += actual_batch_size_val

            pbar_val.set_postfix(Loss=f"{total_val_loss / val_samples:.4f}", Accuracy=f"{total_val_accuracy / val_samples:.4f}")
            pbar_val.update(1)

    pbar_val.close()  # Ensure the progress bar is closed after validation

    # Summarize and log validation results
    if val_samples > 0:
        avg_val_loss = total_val_loss / val_samples
        avg_val_accuracy = total_val_accuracy / val_samples
        print(f"Validation Complete - Avg Loss: {avg_val_loss:.4f}, Avg Accuracy: {avg_val_accuracy:.4f}")
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_accuracy)
    else:
        print("No validation samples were processed.")

    train_losses.append(avg_train_loss)
    debugging_callback.on_epoch_end(epoch)

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
    best_val_accuracy = 0  # or float('-inf')

    # Check for improvement based on validation accuracy
    if avg_val_accuracy > best_val_accuracy:
        best_val_accuracy = avg_val_accuracy
        no_improvement_epochs = 0  # Reset counter
        model.save_weights('lion_heart_classification_2.h5')  # Save best model
        print("New best model saved with accuracy: {:.4f}".format(best_val_accuracy))
    else:
        no_improvement_epochs += 1

    # Early stopping check
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    # Save plot data after each epoch
    with open('lion_heart_classification_2.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses), f)

# Close plot and load best model weights
plt.close()

print("Training completed")

# Loading best model weights for evaluation
model.load_weights('lion_heart_classification_2.h5')

"""
x_visualize, _ = next(iter(train_dataset)) 

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
    mse = mean_squared_error(y_val_original, y_pred)
    mae = mean_absolute_error(y_val_original, y_pred)

    print(f"Model Mean Squared Error on Validation Set: {mse}")
    print(f"Model Mean Absolute Error on Validation Set: {mae}")

    # Save the actual and predicted values to DataFrame
    result_df = pd.DataFrame({"Actual": y_val_original, "Predicted": y_pred})
    
    # DataFrame to CSV file
    file_path = 'lion_heart_classification_2.csv'
    print("Saving file to:", file_path)
    result_df.to_csv(file_path, index=False)
    print("File saved successfully!")

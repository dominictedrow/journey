import os
import pickle
import numpy as np
import pandas as pd
import sqlite3
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Concatenate, LayerNormalization, GlobalAveragePooling1D
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
print("Eager execution:", tf.executing_eagerly())

""" # Mixed precision policy
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy) """

# Custom SNN parameters
train_data_cutoff = 0.05            # % of training data to load (0.001 - 1)

num_blocks = 1 
units_per_layer = 100 
num_layers_per_block = 2
RNN = num_blocks                    # number of times to recycle the output by adding inputs for continus data transformation for each main block's input

"""
:Population Encoding Parameters: These parameters are used for the Population Encoding Layer in the Spiking Neural Network.

:population_neurons: The number of neurons in the Population Encoding Layer. This value represents the dimensionality of the population code and should be chosen based on the complexity of the data being encoded. 
:decoded_dim: The dimensionality of the output after reverse population encoding. This should match the dimensionality of the original target data. 
"""
population_neurons = 1800           # The dimensionality of the population code, adjust based on data complexity
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
epochs = 30 
batch_size = 10
patience = 5                        # Define patience for early stopping

""" 
:STDP parameters: this is all but the final output dense layer, for STDP backpropagation

:param A_plus: STDP learning rate for potentiation.
:param A_minus: STDP learning rate for depression.
:param tau_plus: Time constant for potentiation.
:param tau_minus: Time constant for depression. 
"""
A_plus = 0.01                       # default range = 0.001 - 0.1
A_minus = 0.01                      # default range = 0.001 - 0.02
tau_plus = 10                         # default range = 10 - 100ms+
tau_minus = 10                        # default range = 10 - 100ms+

"""
:param tau_mem: Membrane time constant of the Leaky Integrate-and-Fire (LIF) neuron model, determining the rate of decay of the membrane potential.
:param v_rest: Resting membrane potential for the LIF neurons, representing the baseline potential when the neuron is not activated.
"""
tau_mem = 20.0                       # Default range = 10 - 100ms
v_rest = 0.0                        # Typical range = -1 to 1 for normalized potentials, 0 for non-negative activations

"""
:Threshold initializer weight range

:param minval: min weight initializer Threshold distribution.
:param maxval: max weight initializer Threshold distribution
"""
minval = 0.0                         # Default range based on data
maxval = 10.0                         # Default range based on data

""" 
:param max_rate: The upper limit on the firing rate of neurons in the network, typically set between 10.0 and 200Hz.
:param base_max_rate: The starting value for the max_rate, usually within the range of 20.0 to 100Hz.
:param current_max_rate: A dynamic parameter representing the current maximum firing rate, initially set to the value of base_max_rate.
:param scale_factor: A coefficient used to scale adjustments in max_rate based on the average neuronal threshold, with a default range between 0 and 1. 
"""
base_max_rate = 100.0                # default range = 20.0 - 100Hz+
current_max_rate = base_max_rate    # dynamic parameter
max_rate = 500                      # default range = 10.0 - 200Hz+
scale_factor = 0.20                  # default range = 0 - 1

"""
:Duration and Time Step of Each Spike Train:

:param duration: The total time span of each spike train in seconds. This parameter determines the length of time over which spikes can occur.
:param time_step: The time resolution for each step in the spike train, measured in seconds. This parameter defines the granularity or precision of the spike train.

The total number of time steps (and hence, the length of the spike train array) is determined by dividing the duration by the time_step. Each entry in the 
spike train array represents whether a spike occurs at that time step (1 for spike, 0 for no spike). A smaller time_step results in more granular spike 
trains, capturing more precise spike timings.
"""
seq_len = 2
duration = 1.0                      # 1 is = to 1 second
time_step = 0.1                    # 1 / max_rate  # This ensures time_step is appropriately scaled

exclude_columns = True              # Remove unused columns in the data set (currently 12:25)

# Ensure that max_rate * duration is sufficient for the precision
# assert max_rate * duration >= 10**2, "The product of max_rate and duration is not sufficient to maintain the required precision."

""" 
This is the final output dense layer, that uses a custom Adam optimizer for backpropagation
"""
learning_rate = 0.01                # Matched learning rate (default = 0.001)
beta_1=0.9                          # Momentum term (beta1) (default = 0.9)
beta_2=0.999                        # Momentum term for the squared gradient (beta2) (default = 0.999)
epsilon=1e-07                       # Small number to prevent any division by zero (default = 1e-07)
decay=0.0                           # Weight decay for regularization (default = 0.0)
amsgrad=True

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
def load_data_from_sqlite(db_path, table_name, columns=None):
    conn = sqlite3.connect(db_path)
    if columns:
        query = f"SELECT {','.join(columns)} FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def preprocess_data(df, max_rate, duration, time_step, seq_len, train_data_cutoff, exclude_columns):
    """
    Preprocess data for SNN, treating each row as a sequence.

    :param df: DataFrame containing data.
    :param max_rate: Maximum spike rate for encoding.
    :param duration: Duration of spike train.
    :param time_step: Time step for spike train.
    :param seq_len: Length of the sequence to be considered.
    :param exclude_columns: Boolean to control exclusion of columns 12 to 25.
    :return: Processed spike train data and targets.
    """
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")    
    
    # Drop the 'index' column if it's present
    if 'index' in df.columns:
        df = df.drop('index', axis=1)

    # Exclude columns if specified
    if exclude_columns:
        df = df.drop(df.columns[5:-1], axis=1)    

    # Replace infinities and fill NaNs before converting to NumPy array
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    # Calculate the number of sequences to include based on train_data_cutoff
    total_sequences = len(df) - seq_len + 1
    num_sequences_to_sample = int(total_sequences * train_data_cutoff)

    # Sample the start indices for sequences
    sampled_start_indices = np.random.choice(range(total_sequences), size=num_sequences_to_sample, replace=False)
    sampled_start_indices.sort()  # Sort the indices to maintain time series order

    df_values = df.values.astype('float32')
    sampled_sequences = np.array([df_values[idx:idx + seq_len] for idx in sampled_start_indices])

    # Separate features and target
    features = sampled_sequences[:, :, :-1]  # All features excluding the 'Target' column
    target = df_values[sampled_start_indices, -1]  # Target from the first row of each sequence

    # Encode features into spike trains
    encoded_sequences = np.array([rate_encoding_to_spike_train(sequence, max_rate, duration, time_step) for sequence in features])

    batch_size, seq_len, features, spike_train_sequences = encoded_sequences.shape

    return encoded_sequences, np.array(target)

# Load and preprocess training data
print("About to load training data...")
df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//streamline_snn.db", 'XAUUSD')
print("Finished loading training data...")
print("Starting to preprocess training data...")
X_train_encoded, y_train_original = preprocess_data(df_train, max_rate, duration, time_step, seq_len, train_data_cutoff, exclude_columns)
print("Finished preprocessing training data...")
print()

# Load and preprocess validation data
print("About to load validation data...")
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//val_streamline_snn.db", 'XAUUSD')
print("Finished loading validation data...")
print("Starting to preprocess validation data...")
X_val_encoded, y_val_original = preprocess_data(df_val, max_rate, duration, time_step, seq_len, train_data_cutoff, exclude_columns)
print("Finished preprocessing validation data...")
print()

class PopulationEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons, **kwargs):
        super(PopulationEncodingLayer, self).__init__(**kwargs)
        self.num_neurons = num_neurons

    def build(self, input_shape):
        # Assuming input_shape is [batch, features] from flattened and pooled inputs
        self.encoding_weights = self.add_weight(
            shape=(input_shape[-1], self.num_neurons),
            initializer=tf.keras.initializers.LecunNormal(),
            trainable=True,
            name='encoding_weights'
        )
        self.biases = self.add_weight(
            shape=(self.num_neurons,),
            initializer='zeros',
            trainable=True,
            name='biases'
        )

    def call(self, inputs):
        # Directly use inputs assuming they are properly pooled and reshaped before
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
    def __init__(self, units, tau_mem, v_rest, **kwargs):
        super(SpikingNeuronLayer, self).__init__(**kwargs)
        self.units = units
        self.tau_mem = tau_mem
        self.v_rest = v_rest
        self.threshold = None
        self.V_mem = None

    def build(self, input_shape):
        self.threshold = self.add_weight(shape=(self.units,),
                                         initializer=tf.random_uniform_initializer(minval, maxval),
                                         trainable=True,
                                         name='threshold')
        # Membrane potential initialized here but will be dynamically resized later
        self.V_mem = self.add_weight(shape=(1, self.units),
                                     initializer=tf.constant_initializer(self.v_rest),
                                     trainable=False,
                                     name='V_mem',
                                     dtype=tf.float32)
        super(SpikingNeuronLayer, self).build(input_shape)

    def call(self, inputs, training=None, reset_state=False):
        batch_size = tf.shape(inputs)[0]

        # Dynamically adjust V_mem for the current batch size
        V_mem = tf.broadcast_to(self.V_mem, [batch_size, self.units])
        
        # Update membrane potential
        new_V_mem = V_mem + inputs - (V_mem - self.v_rest) / self.tau_mem

        # Spike generation
        fired = tf.greater_equal(new_V_mem, self.threshold)
        spikes = tf.cast(fired, tf.float32)

        # Reset membrane potential where neurons have fired to v_rest
        new_V_mem = tf.where(fired, tf.fill([batch_size, self.units], self.v_rest), new_V_mem)

        # Optionally reset states at the end of each sequence
        if reset_state:
            new_V_mem = tf.fill([batch_size, self.units], self.v_rest)

        # Update the membrane potential for the next step
        self.V_mem.assign(tf.reduce_mean(new_V_mem, axis=0, keepdims=True))

        return spikes

    def reset_states(self):
        # Reset V_mem to the resting state for all units, across all samples in the batch
        self.V_mem.assign(tf.fill([1, self.units], self.v_rest))

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
        # Return the recorded spikes
        return tf.concat(self.spike_record, axis=0)

    def get_thresholds(self):
        # Return the current thresholds
        return self.threshold
    
    def reset_spike_record(self):
        # Reset the spike record
        self.spike_record = []
    
    def stdp_threshold_update(self, pre_spike_train, post_spike_train, global_error_factor):
        new_threshold = stdp_threshold_update(pre_spike_train, post_spike_train, self.threshold, A_plus, A_minus, tau_plus, tau_minus, global_error_factor)
        self.threshold.assign(new_threshold)

class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units        
        self.use_bias = use_bias
        self.is_connected_to_spiking_layer = False
        self.kernel = None
        self.bias = None

    def build(self, input_shape):
        # Determine input dimension based on the shape of the inputs
        if len(input_shape) > 2:  # Assuming 4D input for the first layer
            input_dim = input_shape[-1] * input_shape[-2] * input_shape[-3]
        else:  # Assuming 2D input for subsequent layers
            input_dim = input_shape[-1]
        
        self.kernel = self.add_weight(name='kernel', shape=(input_dim, self.units),
                                      initializer='glorot_uniform', trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias', shape=(self.units,),
                                        initializer='zeros', trainable=True)

    def call(self, inputs):
        if len(inputs.shape) == 2:  # Input is already 2D
            # No need to flatten, process it directly
            flat_inputs = inputs
        else:  # Assuming the input is 4D: [batch_size, seq_len, features, spike_trains]
            input_shape = tf.shape(inputs)
            batch_size = input_shape[0]
            flattened_dim = input_shape[1] * input_shape[2] * input_shape[3]
            flat_inputs = tf.reshape(inputs, [batch_size, flattened_dim])
        
        outputs = tf.matmul(flat_inputs, self.kernel)
        if self.use_bias:
            outputs += self.bias
        
        return outputs

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

class CustomGlobalMaxPooling(tf.keras.layers.Layer):
    def call(self, inputs):
        # Assumes inputs is 2D: [batch_size, features]
        # Returns max-pooled output as [batch_size, 1], compressing all features
        return tf.reduce_max(inputs, axis=1, keepdims=True)
    
class AccumulateSpikeTrainsLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AccumulateSpikeTrainsLayer, self).__init__(**kwargs)
        self.accumulated_outputs = []

    def call(self, inputs):
        self.accumulated_outputs.append(inputs)
        return inputs

    def get_accumulated_outputs(self):
        # Stack the accumulated outputs along a new dimension to maintain their sequence
        accumulated_tensor = tf.stack(self.accumulated_outputs, axis=0)
        return accumulated_tensor

    def reset_states(self):
        # Clear the accumulated outputs to reset the layer's state
        self.accumulated_outputs = []


def create_dual_input_model(input_shape, units_per_layer, population_neurons):
    input_normal = Input(shape=input_shape, name='input_normal', dtype='float32')

    custom_dense_layer = CustomDenseLayer(units=units_per_layer)
    custom_dense_layer.set_connected_to_spiking_layer(True)
    x_transformed = custom_dense_layer(input_normal)

    spiking_layer = SpikingNeuronLayer(units=units_per_layer, tau_mem=tau_mem, v_rest=v_rest)
    x_spiked = spiking_layer(x_transformed)

    # Add the accumulation layer directly after the spiking layer
    accumulate_layer = AccumulateSpikeTrainsLayer(name='accumulate_spike_trains')
    accumulated_output = accumulate_layer(x_spiked)

    population_encoding_layer = PopulationEncodingLayer(num_neurons=population_neurons)
    encoded_output = population_encoding_layer(accumulated_output)

    custom_output_layer = CustomDenseLayer(1)
    output = custom_output_layer(encoded_output)

    model = Model(inputs=input_normal, outputs=output)
    return model

""" def create_dual_input_model(input_shape, num_blocks, num_layers_per_block, units_per_layer, RNN, population_neurons):
    assert num_blocks == RNN, "Number of RNN iterations must match number of blocks"
    input_normal = Input(shape=input_shape, name='input_normal', dtype='float32')    

    # RNN loop
    rnn_outputs = []
    x_transformed = input_normal

    print(f"Input shape: {input_normal.shape}")
        
    for i in range(RNN):
        custom_dense_layer = CustomDenseLayer(units_per_layer)
        custom_dense_layer.set_connected_to_spiking_layer(True)
        x_transformed_initial = custom_dense_layer(x_transformed)
        print(f"After CustomDenseLayer {i+1}, shape: {x_transformed_initial.shape}")

        spiking_layer = SpikingNeuronLayer(units=units_per_layer, tau_mem=tau_mem, v_rest=v_rest)
        x_spiked = spiking_layer(x_transformed_initial)
        print(f"After SpikingNeuronLayer {i+1}, shape: {x_spiked.shape}")
        x_transformed = x_spiked  
        
        rnn_outputs.append(x_transformed)

    projection_layer = CustomDenseLayer(units_per_layer)
    projection_layer.set_connected_to_spiking_layer(True)
       
    # Initialize the list to hold block outputs
    block_outputs = []
    for i in range(num_blocks):        
        spiking_layer = SpikingNeuronLayer(units=units_per_layer, tau_mem=tau_mem, v_rest=v_rest)
        x_spiked = spiking_layer(rnn_outputs[i])
        
        
        # Initialize x_block_input with the output from the SpikingNeuronLayer
        x_block_input = x_spiked

        for j in range(num_layers_per_block):
            layer = CustomDenseLayer(units_per_layer)
            layer.set_connected_to_spiking_layer(True)
            x_block_input = layer(x_block_input)
            

            spiking_layer = SpikingNeuronLayer(units=units_per_layer, tau_mem=tau_mem, v_rest=v_rest)
            x_spiked = spiking_layer(x_block_input)            
            x_block_input = x_spiked            

        block_outputs.append(x_block_input)
    block_outputs_encoded = []

    population_encoding_layer = PopulationEncodingLayer(num_neurons=population_neurons)

    for block_output in block_outputs:
        encoded_output = population_encoding_layer(block_output)
        block_outputs_encoded.append(encoded_output)
        print("Shape after PopulationEncodingLayer:", encoded_output.shape)

    concatenated_encoded = Concatenate(axis=-1)(block_outputs_encoded)
    print("Shape after concatenation:", concatenated_encoded.shape)

    normalized_encoded = LayerNormalization()(concatenated_encoded)
    print("Shape after LayerNormalization:", normalized_encoded.shape)

    # Apply Global Max Pooling
    pooled_output = GlobalMaxPooling2D()(normalized_encoded)
    print("Shape after GlobalMaxPooling2D:", pooled_output.shape)

    # Final dense layer for prediction
    custom_output_layer = CustomDenseLayer(1)
    output = custom_output_layer(pooled_output)
    print("Shape of final output:", output.shape)

    model = Model(inputs=input_normal, outputs=output)

    return model """

## Assuming X_train_encoded is now available after preprocessing
seq_len = X_train_encoded.shape[1]
features = X_train_encoded.shape[2]
spike_trains = X_train_encoded.shape[3]

# Correct input shape for the model, excluding the batch size
input_shape = (seq_len, features, spike_trains)


""" # Create the model and get the layers
model = create_dual_input_model(input_shape, num_blocks, num_layers_per_block, units_per_layer, RNN, population_neurons) """
# Create the model and get the layers
model = create_dual_input_model(input_shape, units_per_layer, population_neurons)

# Inspect the model architecture
model.summary()

# Customizing the Adam optimizer
optimizer = Adam(
    learning_rate = learning_rate, 
    beta_1 = beta_1,             # Momentum term (beta1)
    beta_2 = beta_2,             # Momentum term for the squared gradient (beta2)
    epsilon = epsilon,           # Small number to prevent any division by zero
    decay = decay,               # Weight decay for regularization
    amsgrad = amsgrad            # Whether to apply AMSGrad variant of Adam
)

loss_fn = tf.keras.losses.MeanSquaredError() # MeanAbsoluteError & MeanSquaredError
# Compile the model with the customized adam optimizer. Spiking units are handeled in train_step
model.compile(optimizer=optimizer, loss=loss_fn) 

""" # Before loading weights
weights_before = [layer.get_weights() for layer in model.layers] """

# Check if 'best_model_25.4.h5' exists
if os.path.exists('best_model_25.4.h5'):
    # Load weights if the file exists
    model.load_weights('best_model_25.4.h5')
    print("Weights loaded from 'best_model_25.4.h5'")
else:
    # Skip loading and possibly print a message or take other actions
    print("'best_model_25.4.h5' not found. Continuing without loading weights.")


# Check if the saved plot data file exists
if os.path.exists('best_model_25.4.pkl'):
    with open('best_model_25.4.pkl', 'rb') as f:
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
    plt.savefig('best_model_25.4.png')  # Save the plot with a constant filename

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
    # Modify error calculation based on the final model output
    return tf.reduce_mean(tf.square(output - target))
    # return tf.reduce_mean(tf.abs(output - target))

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

# Training loop
best_val_mae = float('inf')

no_improvement_epochs = 0

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint('best_model_25.4.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

def train_step(model, x_batch, y_batch, optimizer, loss_fn):
    # (batch_size, seq_len, features, spike_train)
    with tf.GradientTape() as tape:
        predictions = model(x_batch, training=True)  # Model handles 4D input correctly now
        loss = loss_fn(y_batch, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    # Calculate global error
    global_error = calculate_global_error(predictions, y_batch_original)
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

                    # Updated call to include global_error_factor
                    previous_layer.stdp_threshold_update(pre_spike_train, post_spike_train, global_error_factor)

    avg_loss = tf.reduce_mean(loss)
    avg_mae = tf.reduce_mean(tf.abs(y_batch_original  - predictions))

    return avg_loss, avg_mae

def validation_step(model, x_batch, y_batch, loss_fn):
    # Process the entire batch at once
    predictions = model(x_batch, training=False)
    loss = loss_fn(y_batch, predictions)

    # Calculate average loss for the batch
    avg_loss = tf.reduce_mean(loss)

    # Calculate mean absolute error for the batch
    avg_mae = tf.reduce_mean(tf.abs(y_batch - predictions))

    return avg_loss, avg_mae

""" def validation_step(model, x_batch, y_batch, loss_fn):
    batch_size = x_batch.shape[0]
    # Process each sample in the batch
    for i in range(batch_size):       
        # Process each sequence and spike train within the sample
        for seq_idx in range(x_batch.shape[1]):
            for train_idx in range(x_batch.shape[3]):
                # Extract and reshape the spike train for the current sequence and train
                spike_train = x_batch[i, seq_idx, :, train_idx]
                spike_train_reshaped = tf.reshape(spike_train, (1, -1))
                
                # Forward pass through the model for the reshaped spike train
                # Note: `training=False` since we're in validation phase
                model_output = model(spike_train_reshaped, training=False)

    # Compute loss for the batch
    loss = loss_fn(y_batch, model_output)

    avg_loss = tf.reduce_mean(loss)
    avg_mae = tf.reduce_mean(tf.abs(y_batch - model_output))

    return avg_loss, avg_mae """

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_encoded, y_train_original)).batch(batch_size)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val_encoded, y_val_original)).batch(batch_size)

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
    membrane_potentials_data = []  # Store membrane potentials

    # Dynamically adjust max_rate based on the model's current state
    current_max_rate = adjust_max_rate(model, base_max_rate=current_max_rate)

    """ # Enable spike recording and reset spike records for SpikingNeuronLayer layers
    for layer in model.layers:
        if isinstance(layer, SpikingNeuronLayer):
            layer.track_spike_record = False  # Enable spike recording
            layer.reset_spike_record()        # Reset spike record at the start """

    # Training phase
    with tqdm(total=len(train_dataset), desc="Training", unit="batch") as pbar:
        for x_batch, y_batch_original in train_dataset:
            loss, mae = train_step(model, x_batch, y_batch_original, optimizer, loss_fn)
            
            # Accumulate and log metrics
            total_train_loss += loss.numpy() * len(x_batch)
            total_train_mae += mae.numpy() * len(x_batch)
            train_samples += len(x_batch)

            pbar.set_description(f"Training - Loss: {total_train_loss / train_samples:.4f}, MAE: {total_train_mae / train_samples:.4f}")
            pbar.update(1)

    # Calculate average training loss and MAE
    avg_train_loss = total_train_loss / train_samples
    avg_train_mae = total_train_mae / train_samples

    # Validation phase
    with tqdm(total=len(validation_dataset), desc="Validation", unit="batch") as pbar:
        for x_batch_val, y_batch_val_original in validation_dataset:
            loss, mae = validation_step(model, x_batch_val, y_batch_val_original, loss_fn)
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
        model.save_weights('best_model_25.4.h5')  # Save best model
    else:
        no_improvement_epochs += 1

    # Early stopping check
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    # Save plot data after each epoch
    with open('best_model_25.4.pkl', 'wb') as f:
        pickle.dump((train_losses, val_losses), f)

# Close plot and load best model weights
plt.close()

# After the training loop
print("Training completed")

# Loading best model weights for evaluation
model.load_weights('best_model_25.4.h5')

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
    mse = mean_squared_error(y_val_original, y_pred)
    mae = mean_absolute_error(y_val_original, y_pred)

    print(f"Model Mean Squared Error on Validation Set: {mse}")
    print(f"Model Mean Absolute Error on Validation Set: {mae}")

    # Save the actual and predicted values to DataFrame
    result_df = pd.DataFrame({"Actual": y_val_original, "Predicted": y_pred})
    
    # Save result DataFrame to CSV file
    file_path = 'best_model_predictions_25.4.csv'
    print("Saving file to:", file_path)
    result_df.to_csv(file_path, index=False)
    print("File saved successfully!")

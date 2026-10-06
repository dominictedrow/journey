""" 
L       IIIII   OOO   N   N  H   H  EEEE    A     RRRR   TTTTT......
L         I    O   O  NN  N  H   H  E      A A    R   R    T  ...........
L         I    O   O  N N N  HHHHH  EEE    AAAA   RRRR     T  .................
L         I    O   O  N  NN  H   H  E     A    A  R   R    T  ........................
LLLLL   IIIII   OOO   N   N  H   H  EEEE A      A R   RR   T  ...............................

:Welcome to LionHeart Regression, the TRANSFORMER KILLER. (float32) Binary encoding schema

A 4D custom built neural network inspired by the human brain.
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

This model completely changes how NN's function. This model allows continuous state 
machines, like in robotics for continuous motor activators to simulate movement through reactions
and reaal-time learning. 

Future models will include a biological concept called mirroring neuros, which will help provide 
a better working memory separate from needing to solve never before seen task (aka it will be able
to feed itself data (compress relationships to important samples from it's training) to generate a
dream state for deeper level learning), and lateral back and forth movement between units within 
each layer and the layer themselves.

By: JD (ThinkBe)
"""

import os
import pickle
import sqlite3
import numpy as np
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.backend import int_shape
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Input, LayerNormalization, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.mixed_precision import experimental

# Suppress TensorFlow warnings (due to custom learning with synaptic modulation outside off gradients)
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

# Load training and validation data
train_data_loc = {"path": "C://Users//crgon//OneDrive//Desktop//trading//data//streamline_snn.db", "dataset": "XAUUSD"}
val_data_loc = {"path": "C://Users//crgon//OneDrive//Desktop//trading//data//val_streamline_snn.db", "dataset": "XAUUSD"}

"""
TTTTT  H   H  IIIII  N   N  K   K  B B B  EEEE ...
  T    H   H    I    NN  N  K  K   B   B  E    .......
  T    HHHHH    I    N N N  KKK    BBBB   EEE  .............
  T    H   H    I    N  NN  K  K   B   B  E    ......................
  T    H   H  IIIII  N   N  K   K  B B B  EEEE ...............................

:parameters: Below are the model settings. Read each one carfully!

"""

# Lion Heart Model settings
spiking_rnn = False                 # Set to False to disable recurrent spiking units
dense_rnn = False                   # Set to False to disable recurrent dense units
unit_activation_dense = tf.nn.relu  # tf.keras.activations.linear, tf.nn.relu, tf.nn.sigmoid, tf.tanh, (non-spiking layer)
unit_activation_pop = tf.tanh       # tf.nn.relu, tf.nn.sigmoid, tf.tanh (population layer)

units = 512                         # Number of units for all CustomDense and Spiking layers  
population_units = units            # The dimensionality of the population code, adjust based on data complexity 
num_layers = 6                      # Number of CustomDense & Spiking layers pairs 
pop_layers = 2                      # Number of population dense layers

seq_len = 50                        # historical time-steps

epochs = 20                         # BaNumber of training runs
batch_size = 64                     # Batch_size must be low (1 - 3) for the model to learn
patience = 20                       # Define patience for early stopping

record_spikes = False               # record spiking layer spikes for learning purposes
aggregated_spikes = False           # True for neuronal_activities_viewer visualization (high_memory - reduce train_data_cutoff = 0.001 if True)

track_membrane_potential = True     # This is for either custom visualization of NN or to improve learning

train_data_cutoff = 0.001           # % of training data to load (0.001 - 1)
exclude_columns = True              # Remove unused columns
excluded_columns_L_R = slice(10,-1) # Select columns to remove but leaving the last column target
excluded_columns_R_L = slice(4, 10)  # Select columns to remove for further feature control

"""
:param tau_mem: Membrane time constant of the Leaky Integrate-and-Fire (LIF) neuron model, determining the rate of decay of the membrane potential.
:param v_rest: Resting membrane potential for the LIF neurons, representing the baseline potential when the neuron is not activated.
"""

tau_mem = 1                          # Default range = 10 - 100ms (1 produces zero decay and should be default)
v_rest = -1.0                         # Typical range = -1 to 1 for normalized potentials, 0 for non-negative activations

"""
:unit thresholds connections initializer weight ranges

:param minval_thresholds: min weight range for unit Thresholds
:param maxval_thresholds: max weight range for unit Thresholds
"""

minval_thresholds = 1.0              # Spiking unit threshold weight range (min)
maxval_thresholds = 1.00             # Spiking unit threshold weight range (max)

""" 
Custom Adam Optimizer 
:param Non-spiking/custom layers learning parameters except for learning_rate shared with threshold updates
"""
learning_rate = 0.01               # Matched learning rate (default = 0.001)
beta_1=0.9                          # Momentum term (beta1) (default = 0.9)
beta_2=0.999                        # Momentum term for the squared gradient (beta2) (default = 0.999)
epsilon=1e-07                       # Small number to prevent any division by zero (default = 1e-07)
decay=0.0                           # Weight decay for regularization (default = 0.0)
amsgrad=False                       # A learnable parameter

class Preprocessor:
    def __init__(self):
        self.buffer_length = None
        self.max_decimals = None 

    def calculate_max_decimals(self, X):
        """
        Calculate the maximum number of decimal places in the dataset.

        :param X: Input feature matrix.
        :return: Maximum number of decimal places found in X.
        """
        decimal_places = np.vectorize(lambda x: len(str(x).split('.')[1]) if '.' in str(x) else 0)
        self.max_decimals = np.max(decimal_places(X))
        return self.max_decimals

    def calculate_buffer_length(self, max_value):
        """
        Calculate buffer length considering the maximum decimal precision detected.

        :param max_value: The maximum value after normalization and scaling.
        :return: Required length of the binary representation.
        """
        if self.buffer_length is None:
            if self.max_decimals is None:
                raise ValueError("Maximum decimals not calculated. Please run calculate_max_decimals first.")
            scale_factor = 10 ** self.max_decimals
            self.buffer_length = np.floor(np.log2(max_value * scale_factor)) + 1 if max_value > 0 else 1
        return int(self.buffer_length)

    def binary_conversion_to_spike_train(self, value):
        """
        Convert a numerical value to a binary spike train.

        :param value: The value to convert.
        :return: Spike train representing the binary conversion of the value.
        """
        if self.buffer_length is None:
            raise ValueError("Buffer length has not been set.")
        
        width = int(self.buffer_length)
        value = int(value)

        binary_representation = np.binary_repr(value, width=width)
        spike_train = np.array([int(bit) for bit in binary_representation])

        return spike_train

    def rate_encoding_to_spike_train(self, X, max_value):
        """
        Converts feature values to spike trains automatically detecting decimal precision.

        :param X: Input feature matrix.
        :param max_value: Maximum value to normalize data.
        :return: Spike trains corresponding to the feature values.
        """
        if self.max_decimals is None:
            self.calculate_max_decimals(X)
        scale_factor = 10 ** self.max_decimals
        X_normalized = np.clip(X / max_value, 0, 1) * max_value
        X_scaled = np.round(X_normalized * scale_factor).astype(int)

        spike_trains = np.array([[self.binary_conversion_to_spike_train(value)
                                  for value in sample] for sample in X_scaled])

        return spike_trains

    def load_data_from_sqlite(self, data_loc, exclude_columns, excluded_columns_L_R, excluded_columns_R_L):
        """
        Load data from an SQLite database and optionally exclude specified columns, including the 'index' column if present.

        Parameters:
        - data_loc: A dictionary with 'path' to the database file and 'dataset' as the table name.
        - excluded_columns_L_R: Optional list of specific column names to exclude (identified typically from left to right).
        - excluded_columns_R_L: Optional list of specific column names to exclude (identified typically from right to left).

        Returns:
        - df: A pandas DataFrame containing the loaded data, with specified exclusions applied.
        """
        conn = sqlite3.connect(data_loc['path'])
        query = f"SELECT * FROM {data_loc['dataset']}"
        df = pd.read_sql_query(query, conn)
        conn.close()

        # Drop the 'index' column if it's present
        if 'index' in df.columns:
            df = df.drop('index', axis=1)

        # Exclude specified columns if requested
        if exclude_columns:
            if excluded_columns_L_R:
                df = df.drop(df.columns[excluded_columns_L_R], axis=1)
            if excluded_columns_R_L:
                df = df.drop(df.columns[excluded_columns_R_L], axis=1)

                # Print remaining feature names and their count
                print("Remaining features:", df.columns[:-1].tolist())  # Exclude 'Target' column from feature names
                print("Number of features:", len(df.columns) - 1)  # Subtract 1 to exclude 'Target' column from count

        return df

    def preprocess_data(self, df, seq_len, min_target, max_target, train_data_cutoff, is_training):
        """
        Preprocess data for Lion Heart, supporting both rate and temporal encoding.

        :param df: DataFrame containing data.
        :param seq_len: Length of the sequence to be considered.
        :return: Processed spike train data and targets.
        """
        if 'Target' not in df.columns:
            raise KeyError("The DataFrame does not contain a 'Target' column.")    

        # Replace infinities and fill NaNs before converting to NumPy array
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(df.mean(numeric_only=True), inplace=True)

        total_sequences = len(df) - seq_len + 1
        if is_training:
            # (True) For training data, select sequences from the bottom up based on train_data_cutoff
            start_index = total_sequences - int(total_sequences * train_data_cutoff)
            sampled_start_indices = np.arange(start_index, total_sequences)
        else:
            # (False) use all sequences
            sampled_start_indices = np.arange(total_sequences)

        # Normalize target column using Min-Max scaling
        df['Target'] = (df['Target'] - min_target) / (max_target - min_target)

        self.calculate_max_decimals(df.drop('Target', axis=1).values.flatten())

        df_values = df.values.astype('float32')
        sampled_sequences = np.array([df_values[idx:idx + seq_len] for idx in sampled_start_indices])

        features = sampled_sequences[:, :, :-1]
        target = df_values[sampled_start_indices, -1]

        max_value = np.max(df.values) 
        self.calculate_buffer_length(max_value)
        
        encoded_sequences = np.array([
            self.rate_encoding_to_spike_train(sequence, max_value) for sequence in features
        ])
        print("Encoded sequences shape:", encoded_sequences.shape)

        batch_size, seq_len, features, spike_train_sequences = encoded_sequences.shape
        print("Encoded sequences shape:", encoded_sequences.shape)

        # Return encoded data along with the min and max targets used for normalization
        return encoded_sequences, np.array(target), max_value, min_target, max_target  
    
preprocessor = Preprocessor()

# Load and preprocess training data
print("About to load training data...")
df_train = preprocessor.load_data_from_sqlite(train_data_loc, exclude_columns, excluded_columns_L_R, excluded_columns_R_L)
print("Finished loading training data...")

# Calculate normalization parameters from training data
min_target_train = df_train['Target'].min()
max_target_train = df_train['Target'].max()

print("Starting to preprocess training data...")
X_train_encoded, y_train_original, max_value_train, min_target_train, max_target_train = preprocessor.preprocess_data(df_train, seq_len, min_target_train, max_target_train, train_data_cutoff, is_training=True) # True loads data based on train_data_cutoff
print("Finished preprocessing training data...")
print()

# Load and preprocess validation data
print("About to load validation data...")
df_val = preprocessor.load_data_from_sqlite(val_data_loc, exclude_columns, excluded_columns_L_R, excluded_columns_R_L)
print("Finished loading validation data...")
print("Starting to preprocess validation data...")
X_val_encoded, y_val_original, max_value_val, min_target_val, max_target_val = preprocessor.preprocess_data(df_val, seq_len, min_target_train, max_target_train, train_data_cutoff, is_training=False)  # False loads all data
print("Finished preprocessing validation data...")
print()

class SpikingNeuronLayer(tf.keras.layers.Layer):
    def __init__(self, units, tau_mem, v_rest, minval_thresholds, maxval_thresholds, record_spikes, 
                 track_membrane_potential, spiking_rnn, **kwargs):
        super(SpikingNeuronLayer, self).__init__(**kwargs)
        self.units = units
        self.tau_mem = tau_mem
        self.v_rest = v_rest
        self.minval_thresholds = minval_thresholds
        self.maxval_thresholds = maxval_thresholds
        self.record_spikes = record_spikes
        self.track_membrane_potential = track_membrane_potential
        self.threshold = None
        self.V_mem = None
        self.spike_record = [] if record_spikes else None
        self.membrane_potential_sum = None
        self.min_membrane_potential = None
        self.max_membrane_potential = None
        self.batch_count = 0
        self.recurrent_weight = None
        self.use_recurrent_weight = spiking_rnn
        self.previous_spikes = None
        self.min_membrane_potential = None
        self.max_membrane_potential = None

    def build(self, input_shape):
        if self.V_mem is None:  # Check to prevent reinitialization
            self.V_mem = self.add_weight(name='V_mem', shape=(input_shape[0], self.units), initializer=tf.constant_initializer(self.v_rest), trainable=False)

        initializer_1 = tf.random_uniform_initializer(self.minval_thresholds, self.maxval_thresholds)

        self.threshold = self.add_weight(shape=(self.units,),
                                         initializer=initializer_1,
                                         trainable=True, 
                                         name='threshold')
        if self.use_recurrent_weight:
            self.recurrent_weight = self.add_weight(shape=(self.units, self.units),
                                                    initializer=tf.keras.initializers.HeNormal(),
                                                    trainable=True, 
                                                    name='recurrent_weight')
            
        if self.track_membrane_potential:
            # Initialize as variables for tracking within the TensorFlow graph
            self.min_membrane_potential = self.add_weight(name='min_membrane_potential',
                                                          shape=(self.units,),
                                                          initializer=tf.keras.initializers.Constant(value=float('inf')),
                                                          trainable=False)
            self.max_membrane_potential = self.add_weight(name='max_membrane_potential',
                                                          shape=(self.units,),
                                                          initializer=tf.keras.initializers.Constant(value=float('-inf')),
                                                          trainable=False)
        super().build(input_shape)

    def call(self, inputs, training=None, reset_state=False):
        def spike_function(V_mem, threshold):
            fired = tf.greater_equal(V_mem, threshold)
            return tf.cast(fired, tf.float32)

        @tf.custom_gradient
        def spiking_operation(V_mem, threshold):
            spikes = spike_function(V_mem, threshold)

            def grad(dy):
                # Surrogate gradient: fast sigmoid approximation.
                dy = tf.reshape(dy, tf.shape(V_mem))  # Ensure dy has the same shape as V_mem
                V_mem_grad = dy * tf.sigmoid(0.1 * (V_mem - threshold))
                threshold_grad = tf.reduce_sum(-V_mem_grad, axis=0)  # Aggregate gradients over the batch dimension
                return V_mem_grad, threshold_grad

            return spikes, grad

        # Apply recurrent weight if enabled and previous spikes are not None
        if self.use_recurrent_weight and self.previous_spikes is not None:
            recurrent_input = tf.matmul(self.previous_spikes, self.recurrent_weight)
            inputs += recurrent_input

        # Direct one-to-one connections
        new_V_mem = self.V_mem + inputs - (self.V_mem - self.v_rest) / self.tau_mem

        # Use spiking_operation with surrogate gradient
        spikes = spiking_operation(new_V_mem, self.threshold)

        self.V_mem.assign(tf.where(spikes > 0, tf.fill(tf.shape(self.V_mem), self.v_rest), new_V_mem))
        self.previous_spikes = spikes

        if self.record_spikes:
            self.spike_record.append(spikes.numpy())
        
        if self.track_membrane_potential:
            current_min = tf.reduce_min(self.V_mem, axis=0)
            current_max = tf.reduce_max(self.V_mem, axis=0)
            self.min_membrane_potential.assign(tf.minimum(self.min_membrane_potential, current_min))
            self.max_membrane_potential.assign(tf.maximum(self.max_membrane_potential, current_max))

        return spikes

    def reset_states(self):
        if self.V_mem is not None:
            self.V_mem.assign(tf.fill(tf.shape(self.V_mem), self.v_rest))

        self.previous_spikes = None  # Reset previous spikes
        
        if self.record_spikes:
            self.spike_record = []  # Clear spike records

        self.batch_count = 0

    def get_config(self):
        config = super(SpikingNeuronLayer, self).get_config()
        config.update({
            "units": self.units,
            "tau_mem": self.tau_mem,
            "v_rest": self.v_rest,
            "minval_thresholds": self.minval_thresholds,
            "maxval_thresholds": self.maxval_thresholds,
            "track_membrane_potential": self.track_membrane_potential,
            "record_spikes": self.record_spikes,    
            "use_recurrent_weight": self.use_recurrent_weight,
        })
        return config


    def get_spike_record(self):
        if not self.record_spikes:
            raise ValueError("Spike recording is disabled. Enable spike recording to access the spike record.")
        return tf.concat(self.spike_record, axis=0) if self.spike_record else tf.zeros((0, self.units), dtype=tf.float32)    

    def get_dynamic_membrane_potential_range(self):
        if self.track_membrane_potential:
            return self.min_membrane_potential, self.max_membrane_potential
        else:
            return None    
    
    def stdp_threshold_update(self, global_error_factor, learning_rate):
        if not self.track_membrane_potential:
            print("Membrane potential tracking is not enabled or no data has been processed.")
            return

        # Calculate the current range of membrane potentials
        min_potential, max_potential = self.get_dynamic_membrane_potential_range()
        potential_range = max_potential - min_potential

        # Normalize the potential range to [0, 1] for consistent adjustment scaling
        normalized_range = (potential_range - min_potential) / (max_potential - min_potential)

        adjustment_factor = tf.math.tanh(0.5 - normalized_range)

        threshold_update = learning_rate * global_error_factor * adjustment_factor

        # Apply the update to thresholds
        new_thresholds = self.threshold + threshold_update
        new_thresholds_clipped = tf.clip_by_value(new_thresholds, self.minval_thresholds, self.maxval_thresholds)

        # Update the thresholds
        self.threshold.assign(new_thresholds_clipped)

      

class CustomDenseLayer(tf.keras.layers.Layer):
    def __init__(self, units, dense_rnn, use_bias=False, **kwargs):
        super(CustomDenseLayer, self).__init__(**kwargs)
        self.units = units    
        self.use_bias = use_bias
        self.dense_rnn = dense_rnn
        self.recurrent_weight = None
        self.previous_outputs = None

    def build(self, input_shape):
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=tf.keras.initializers.HeNormal(),
            trainable=True
        )

        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )

        if self.dense_rnn:
            self.recurrent_weight = self.add_weight(
                name='recurrent_kernel',
                shape=(self.units, self.units),
                initializer=tf.keras.initializers.HeNormal(),
                trainable=True
            )
        super(CustomDenseLayer, self).build(input_shape)
    
    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32)
        outputs = tf.matmul(inputs, self.kernel)

        if self.dense_rnn and self.previous_outputs is not None:
            recurrent_input = tf.matmul(self.previous_outputs, self.recurrent_weight)
            outputs += recurrent_input

        if self.use_bias:
            outputs += self.bias

        # Activation is not required but can be + or - with unit_activation_dense
        outputs = unit_activation_dense(outputs)

        # Store the current outputs for the next call if RNN is enabled
        if self.dense_rnn:
            self.previous_outputs = outputs

        return outputs 

    def get_config(self):
        config = super(CustomDenseLayer, self).get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
            'dense_rnn': self.dense_rnn,
        })
        return config
     

class DenseOutLayer(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, **kwargs):
        super(DenseOutLayer, self).__init__(**kwargs)
        self.units = units     
        self.use_bias = use_bias

    def build(self, input_shape):
        input_dim = input_shape[-1]       
        initializer = tf.keras.initializers.GlorotNormal()

        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_dim, self.units),
            initializer=initializer,
            trainable=True
        )
        if self.use_bias:
            self.bias = self.add_weight(
                name='bias',
                shape=(self.units,),
                initializer='zeros',
                trainable=True
            )
        super(DenseOutLayer, self).build(input_shape)

    def call(self, inputs, training=False):
        inputs = tf.cast(inputs, tf.float32) 
        z = tf.matmul(inputs, self.kernel)  # Linear transformation

        if self.use_bias:
            z += self.bias 

        # linear activation function
        outputs = tf.keras.activations.linear(z)
        return outputs

    def get_config(self):
        config = super(DenseOutLayer, self).get_config()
        config.update({
            'units': self.units,
            'use_bias': self.use_bias,
        })
        return config
    
class PopulationEncodingLayer(tf.keras.layers.Layer):
    def __init__(self, population_units, **kwargs):
        super(PopulationEncodingLayer, self).__init__(**kwargs)
        self.population_units = population_units

    def build(self, input_shape):
        self.encoding_weights = self.add_weight(
            shape=(input_shape[-1], self.population_units),
            initializer=tf.keras.initializers.GlorotNormal(),
            trainable=True,
            name='encoding_weights'
        )
        self.biases = self.add_weight(
            shape=(self.population_units,),
            initializer='zeros',
            trainable=True,
            name='biases'
        )

    def call(self, inputs, training=False):
        # inputs: [batch_size, feature_dimension]
        encoded = tf.matmul(inputs, self.encoding_weights) + self.biases
        return unit_activation_pop(encoded)
    
    def get_config(self):
        config = super(PopulationEncodingLayer, self).get_config()
        config.update({
            'population_units': self.population_units,
        })
        return config

class ReverseEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, units, max_value, max_decimals, **kwargs):
        super(ReverseEncoding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.units = units
        self.max_value = max_value
        self.max_decimals = max_decimals  # Store the maximum decimal places
        self.rnn_layer = tf.keras.layers.LSTM(units, return_sequences=False)

    def call(self, inputs, training=False):
        # [batch_size, seq_len, units, spike_duration]
        continuous_values = self.spike_trains_to_continuous(inputs)
        # [batch_size, seq_len, features]
        rnn_output = self.rnn_layer(continuous_values)

        return rnn_output

    def spike_trains_to_continuous(self, spike_trains):
        # Convert binary spike trains back to numerical values
        binary_length = spike_trains.shape[-1]
        multiplier = tf.cast(2 ** tf.range(binary_length - 1, -1, -1), tf.float32)
        continuous_values = tf.reduce_sum(spike_trains * multiplier, axis=-1)  # Sum along the binary_length axis

        # Scale the decoded values to their original range, considering max_decimals
        scale_factor = 10 ** self.max_decimals
        continuous_values = (continuous_values / (2**binary_length - 1) * self.max_value) / scale_factor

        return continuous_values

    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'units': self.units,
            'max_value': self.max_value,
            'max_decimals': self.max_decimals, 
        })
        return config
    
class AccumulateSpikeTrainsLayer(tf.keras.layers.Layer):
    def __init__(self, seq_len, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = seq_len
        self.accumulated_outputs = []

    def call(self, inputs):
        # Inputs should be [batch_size, features, binary_representation_length] per call
        self.accumulated_outputs.append(inputs)
        return inputs

    def get_accumulated_outputs(self):
        accumulated_tensor = tf.stack(self.accumulated_outputs, axis=1)
        
        # Determine binary_length by dividing the total accumulated count by seq_len
        total_accumulation = len(self.accumulated_outputs)
        binary_length = total_accumulation // self.seq_len

        # Shape target: [batch_size, seq_len, features, binary_length]
        target_shape = (-1, self.seq_len, accumulated_tensor.shape[2], binary_length)
        accumulated_tensor = tf.reshape(accumulated_tensor, target_shape)
        
        #print("Shape of accumulated_tensor:", accumulated_tensor.shape)
        return accumulated_tensor

    def reset_states(self):
        self.accumulated_outputs = []

    def get_config(self):
        config = super().get_config()
        config.update({"seq_len": self.seq_len})
        return config    

class SequentialSpikeTrainModel(tf.keras.Model):
    def __init__(self, units, seq_len, population_units, tau_mem, v_rest, minval_thresholds, 
                 maxval_thresholds, record_spikes, track_membrane_potential,
                 num_layers, pop_layers, spiking_rnn, dense_rnn, max_decimals, max_value, input_shape=None):
        super(SequentialSpikeTrainModel, self).__init__()
        self.units = units
        self.seq_len = seq_len
        self.population_units = population_units
        self.tau_mem = tau_mem
        self.v_rest = v_rest
        self.minval_thresholds = minval_thresholds
        self.maxval_thresholds = maxval_thresholds
        self.record_spikes = record_spikes
        self.track_membrane_potential = track_membrane_potential
        self.num_layers = num_layers
        self.spiking_rnn = spiking_rnn
        self.dense_rnn = dense_rnn
        self.pop_layers = pop_layers
        self.max_decimals = max_decimals

        # Initialize lists to hold layers
        self.custom_dense_layers = []
        self.spiking_neuron_layers = []
        
        # Initialize pairs of CustomDenseLayer and SpikingNeuronLayer        
        for _ in range(num_layers):
            self.custom_dense_layers.append(CustomDenseLayer(self.units, self.dense_rnn))
            
            self.spiking_neuron_layers.append(SpikingNeuronLayer(self.units, self.tau_mem, self.v_rest, 
                                                                self.minval_thresholds, self.maxval_thresholds, 
                                                                self.record_spikes, self.track_membrane_potential, 
                                                                self.spiking_rnn))
            # self.layer_normalizations.append(LayerNormalization(axis=-1))

        self.accumulate_layer = AccumulateSpikeTrainsLayer(seq_len)
        self.custom_reverse_encoding = ReverseEncoding(seq_len, units, max_value, max_decimals) 
        self.reverse_layer_normalization = LayerNormalization(axis=-1)   

        self.population_encoding_layers = [PopulationEncodingLayer(population_units) for _ in range(pop_layers)]
        self.population_layer_normalizations = [LayerNormalization(axis=-1) for _ in range(pop_layers)]

        self.final_dense_layer = DenseOutLayer(1)

    def call(self, inputs, training=False):
        self.accumulate_layer.reset_states()
        for spiking_neuron_layer in self.spiking_neuron_layers:
            spiking_neuron_layer.reset_states()

        # inputs shape: [batch_size, seq_len, features, binary_length]
        for seq_index in range(inputs.shape[1]):  # Iterate over each sequence
            for spike_train_index in range(inputs.shape[3]):  # Iterate over each spike train slice
                input_slice = inputs[:, seq_index, :, spike_train_index]

                # (batch_size, features) for each slice of binary_length
                input_slice = tf.reshape(input_slice, (input_slice.shape[0], -1))

                # Process through the custom layers
                for custom_dense_layer, spiking_neuron_layer in zip(self.custom_dense_layers, self.spiking_neuron_layers):
                    input_slice = custom_dense_layer(input_slice)
                    input_slice = spiking_neuron_layer(input_slice)
                
                self.accumulate_layer(input_slice)

        accumulated_output = self.accumulate_layer.get_accumulated_outputs()

        # Process the accumulated outputs through the subsequent layers
        reverse_encoding = self.custom_reverse_encoding(accumulated_output)
        reverse_encoding = self.reverse_layer_normalization(reverse_encoding) 
        encoded_output = reverse_encoding

        for pop_layer, layer_norm in zip(self.population_encoding_layers, self.population_layer_normalizations):
            encoded_output = pop_layer(encoded_output)
            encoded_output = layer_norm(encoded_output)

        final_output = self.final_dense_layer(encoded_output)

        return final_output

    def get_config(self):
        return {
            "units": self.units,
            "seq_len": self.seq_len,
            "population_units": self.population_units,
            "tau_mem": self.tau_mem,
            "v_rest": self.v_rest,
            "minval_thresholds": self.minval_thresholds,
            "maxval_thresholds": self.maxval_thresholds,
            "record_spikes": self.record_spikes,
            "track_membrane_potential": self.track_membrane_potential,
            "num_layers": self.num_layers,
            "spiking_rnn": self.spiking_rnn,
            "dense_rnn": self.dense_rnn,
        }

    @classmethod
    def from_config(cls, config):
        return cls(**config)

seq_len = X_train_encoded.shape[1]
features = X_train_encoded.shape[2]
spike_trains = X_train_encoded.shape[3]

input_shape = (features, spike_trains)

model = SequentialSpikeTrainModel(units, seq_len,
                                  population_units,                                 
                                  tau_mem, 
                                  v_rest, 
                                  minval_thresholds, maxval_thresholds,
                                  record_spikes,
                                  track_membrane_potential,
                                  num_layers, pop_layers,
                                  spiking_rnn,
                                  dense_rnn, max_decimals=preprocessor.max_decimals, max_value=max_value_train, input_shape=(features, spike_trains)) 

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

loss_fn = tf.keras.losses.MeanAbsoluteError() # MeanAbsoluteError & MeanSquaredError
model.compile(optimizer=optimizer, loss=loss_fn) 

# Before loading weights
weights_before = [layer.get_weights() for layer in model.layers]

""" # Define custom objects for loading the model
custom_objects = {
    'SpikingNeuronLayer': SpikingNeuronLayer,
    'CustomDenseLayer': CustomDenseLayer,
    'DenseOutLayer': DenseOutLayer,
    'PopulationEncodingLayer': PopulationEncodingLayer,
    'ReverseEncoding': ReverseEncoding,
    'AccumulateSpikeTrainsLayer': AccumulateSpikeTrainsLayer,
    'SequentialSpikeTrainModel' : SequentialSpikeTrainModel,
} """

# Paths to model and weights
model_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/lion_heart_hybrid_regression_full_model"
weights_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/lion_heart_hybrid_regression.h5"

""" # Attempt to load the full model first
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects=custom_objects)
    print(f"Full model loaded from '{model_path}'")
else:
    print(f"Full model at '{model_path}' not found.") """

# After initializing your model, try loading weights
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print(f"Weights loaded from '{weights_path}'")
else:
    print(f"Weights file '{weights_path}' not found. Continuing with initialized model.")


# Check if the saved plot data file exists
if os.path.exists("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/lion_heart_hybrid_regression.pkl"):
    with open("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/lion_heart_hybrid_regression.pkl", 'rb') as f:
        train_losses, val_losses = pickle.load(f)
else:
    train_losses = []
    val_losses = []

# Initialize the plot
plt.figure(figsize=(10, 5))

# Update the plot Function
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
    plt.savefig("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/lion_heart_hybrid_regression.png") 

def calculate_global_error(output, target):
    # return tf.reduce_mean(tf.square(output - target))
    # return tf.reduce_mean(tf.abs(output - target))
    return tf.reduce_mean(tf.abs(output - target))

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
model_checkpoint = ModelCheckpoint("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/lion_heart_hybrid_regression.h5",
                                    monitor='val_loss', verbose=1, save_best_only=True, mode='min')

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_encoded, y_train_original)).batch(batch_size)
validation_dataset = tf.data.Dataset.from_tensor_slices((X_val_encoded, y_val_original)).batch(batch_size)

train_batch_count = np.ceil(len(X_train_encoded) / batch_size).astype(int)  
validation_batch_count = np.ceil(len(X_val_encoded) / batch_size).astype(int)

train_losses = []
val_losses = []
best_val_mae = np.inf
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

# Directory for membrane potentials
membrane_potentials_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/membrane_potentials/lion_heart_regression"
if not os.path.exists(membrane_potentials_dir):
    os.makedirs(membrane_potentials_dir)

# Directory for neuronal activities
neuronal_activities_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/neuronal_activities/lion_heart_regression"
if not os.path.exists(neuronal_activities_dir):
    os.makedirs(neuronal_activities_dir)

def convert_to_original_scale(norm_value, min_target, max_target):
        return norm_value * (max_target - min_target) + min_target

# Training loop
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    debugging_callback.on_epoch_begin(epoch)
    total_train_loss = 0
    total_train_mae = 0
    total_val_loss = 0
    total_val_mae = 0
    train_samples = 0
    val_samples = 0

    if aggregated_spikes:
        # Initialize a structure to hold spikes for all SpikingNeuronLayers for the entire epoch
        epoch_spike_records = {layer.name: [] for layer in model.layers if isinstance(layer, SpikingNeuronLayer)}

    # Initialize structures to hold min and max membrane potentials for all SpikingNeuronLayers for the entire epoch
    epoch_membrane_potentials = {layer.name: {'min': [], 'max': []} for layer in model.layers if isinstance(layer, SpikingNeuronLayer)}

    # Initialize the progress bar
    with tqdm(total=len(list(train_dataset)), desc=f"Training Epoch {epoch + 1}", unit="batch") as pbar_train:
        for x_batch, y_batch in train_dataset:
            actual_batch_size = x_batch.shape[0]

            if actual_batch_size != batch_size:
                print(f"Skipping last batch of size {actual_batch_size}.")
                continue

            with tf.GradientTape() as tape:
                predictions = model(x_batch, training=True)
                loss = loss_fn(y_batch, predictions)

                # Calculate gradients for all trainable variables
                gradients = tape.gradient(loss, model.trainable_variables)
                """ for var, grad in zip(model.trainable_variables, gradients):
                    print(f"{var.name}: Gradient is None? {grad is None}") """
                
                # To exclude threshold gradients from training loop for stdp_threshold_update
                variables = model.trainable_variables
                # Filter out gradients for 'threshold' variables
                filtered_gradients = []
                filtered_variables = []
                for grad, var in zip(gradients, variables):
                    if 'threshold' in var.name:
                        continue  # Skip applying gradients to threshold variables
                    filtered_gradients.append(grad)
                    filtered_variables.append(var)

                # Apply the filtered gradients
                optimizer.apply_gradients(zip(filtered_gradients, filtered_variables))
                
                """ # Apply gradients using the Adam optimizer for all layers (toggle comment above for use)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables)) """

                for i, layer in enumerate(model.layers):
                    if isinstance(layer, SpikingNeuronLayer):
                        global_error = calculate_global_error(predictions, y_batch)
                        global_error_factor = calculate_global_error_factor(global_error)
                        layer.stdp_threshold_update(global_error_factor, learning_rate)

            # Update metrics and progress bar as before
            mae = tf.reduce_mean(tf.abs(predictions - y_batch)).numpy()
            total_train_loss += loss.numpy() * actual_batch_size
            total_train_mae += mae * actual_batch_size
            train_samples += actual_batch_size
            pbar_train.set_postfix(Loss=f"{total_train_loss / train_samples:.4f}", MAE=f"{total_train_mae / train_samples:.4f}")
            pbar_train.update(1)

    pbar_train.close()

    # Training metrics conversion
    avg_train_loss = total_train_loss / train_samples
    avg_train_mae = total_train_mae / train_samples
    avg_train_mae_original = convert_to_original_scale(avg_train_mae, min_target_train, max_target_train)

    print(f"Training Complete - Avg Loss: {avg_train_loss:.4f}, Avg MAE (Original Scale): {avg_train_mae_original:.4f}")
    print("Starting Validation Phase")

    # Validation phase
    with tqdm(total=len(list(validation_dataset)), desc=f"Validation Epoch {epoch + 1}", unit="batch") as pbar_val:
        for x_batch_val, y_batch_val in validation_dataset:
            actual_batch_size_val = x_batch_val.shape[0]

            if actual_batch_size_val != batch_size:
                print(f"Skipping last batch of size {actual_batch_size_val}.")
                continue

            model.reset_states()
            
            predictions_val = model(x_batch_val, training=False)
            loss_val = loss_fn(y_batch_val, predictions_val)

            mae_val = tf.reduce_mean(tf.abs(predictions_val - y_batch_val)).numpy()
            total_val_loss += loss_val.numpy() * actual_batch_size_val
            total_val_mae += mae_val * actual_batch_size_val
            val_samples += actual_batch_size_val

            pbar_val.set_postfix(Loss=f"{total_val_loss / val_samples:.4f}", MAE=f"{total_val_mae / val_samples:.4f}")
            pbar_val.update(1)

    pbar_val.close()

    if val_samples > 0:
        # Validation metrics conversion
        avg_val_loss = total_val_loss / val_samples
        avg_val_mae = total_val_mae / val_samples
        avg_val_mae_original = convert_to_original_scale(avg_val_mae, min_target_val, max_target_val)

        print(f"Validation Complete - Avg Loss: {avg_val_loss:.4f}, Avg MAE (Original Scale): {avg_val_mae_original:.4f}")
    else:
        print("No validation samples were processed.")

    train_losses.append(avg_train_loss)
    if val_samples > 0:
        val_losses.append(avg_val_loss)

    # calculate and save the membrane potentials for each SpikingNeuronLayer
    if track_membrane_potential:
        for layer_index, layer in enumerate(model.layers):
            if isinstance(layer, SpikingNeuronLayer):
                membrane_potential_range = layer.get_dynamic_membrane_potential_range()
                if membrane_potential_range is not None:
                    min_potentials, max_potentials = membrane_potential_range
                    filename = os.path.join(membrane_potentials_dir, f"layer_{layer_index}_epoch_{epoch+1}.npz")
                    np.savez_compressed(filename, min_potentials=min_potentials.numpy(), max_potentials=max_potentials.numpy())
                else:
                    print(f"Layer {layer_index} did not provide membrane potential range data.")

                # Reset membrane potential tracking
                layer.min_membrane_potential.assign(tf.fill([layer.units], float('inf')))
                layer.max_membrane_potential.assign(tf.fill([layer.units], float('-inf')))

    debugging_callback.on_epoch_end(epoch)

    # Update plot after each epoch
    update_plot(epoch, train_losses, val_losses)

    # Check for improvement
    if avg_val_mae < best_val_mae:
        best_val_mae = avg_val_mae
        no_improvement_epochs = 0  # Reset counter

        """ try:
            # Attempt to save the full model
            model.save("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/lion_heart_hybrid_regression_full_model")
            print("Model saved to 'lion_heart_hybrid_regression_full_model'")
        except Exception as e:
            print(f"Failed to save full model due to error: {e}. Saving weights instead.")
            # Fallback: Save just the weights if saving the full model fails """
        
        model.save_weights("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/lion_heart_hybrid_regression.h5")
        print("Weights saved to 'lion_heart_hybrid_regression.h5'")
    else:
        no_improvement_epochs += 1

    # Early stopping check
    if no_improvement_epochs >= patience:
        print(f"Early stopping triggered at epoch {epoch + 1}")
        break

    # Save plot data after each epoch
    with open("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/lion_heart_hybrid_regression.pkl", 'wb') as f:
        pickle.dump((train_losses, val_losses), f)

    if aggregated_spikes:
        for layer_name, spikes_list in epoch_spike_records.items():
            # Concatenate spikes for all batches in this epoch
            aggregated_spikes = tf.concat(spikes_list, axis=0)
            epoch_spike_records[layer_name] = aggregated_spikes.numpy()  

        # Save the aggregated spike records for the epoch
        epoch_file_name = f"spike_records_epoch_{epoch+1}.pkl"
        epoch_file_path = os.path.join(neuronal_activities_dir, epoch_file_name)
        with open(epoch_file_path, "wb") as f:
            pickle.dump(epoch_spike_records, f)

# Close plot
plt.close()

print("Training completed")

# Before loading weights
weights_before = [layer.get_weights() for layer in model.layers]

# Paths to the full model and weights
model_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/lion_heart_hybrid_regression_full_model"
weights_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/lion_heart_hybrid_regression.h5"

""" # Attempt to load the full model first
if os.path.exists(model_path):
    model = load_model(model_path, custom_objects=custom_objects)
    print(f"Full model loaded from '{model_path}' for evaluation.")
else:
    print(f"Full model at '{model_path}' not found for evaluation.")
    print("Initializing model for weights loading.") """

# After initializing your model, try loading weights
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print(f"Weights loaded from '{weights_path}' for evaluation.")
else:
    print(f"Weights file '{weights_path}' not found. Proceeding with model initialized for evaluation.")

def convert_to_original_scale(norm_value, min_target, max_target):
    return norm_value * (max_target - min_target) + min_target

# Model evaluation after training
total_mse = 0.0
total_mae = 0.0
total_samples = 0
total_val_loss = 0
results = []

with tqdm(total=len(list(validation_dataset)), desc="Model Evaluation", unit="batch") as pbar:
    for x_batch_val, y_batch_val in validation_dataset:
        actual_batch_size_val = x_batch_val.shape[0]

        if actual_batch_size_val != batch_size:
            print(f"Skipping last batch of size {actual_batch_size_val}.")
            continue

        model.reset_states()
        predictions_val = model(x_batch_val, training=False)
        loss_val = loss_fn(y_batch_val, predictions_val)

        # Calculate MAE and MSE
        mae_val = tf.reduce_mean(tf.abs(predictions_val - y_batch_val))
        mse_val = tf.reduce_mean(tf.square(predictions_val - y_batch_val))
        total_val_loss += loss_val.numpy() * actual_batch_size_val
        total_mae += mae_val.numpy() * actual_batch_size_val
        total_mse += mse_val.numpy() * actual_batch_size_val
        total_samples += actual_batch_size_val
        
        # Decode and store results for this batch
        decoded_actuals = [convert_to_original_scale(val.numpy(), min_target_val, max_target_val) for val in y_batch_val]
        decoded_predictions = [convert_to_original_scale(val.numpy()[0], min_target_val, max_target_val) for val in predictions_val]
        results.extend([
            {'Actual': act, 'Predicted': pred}
            for act, pred in zip(decoded_actuals, decoded_predictions)
        ])

        pbar.update()

# Calculate average MAE and MSE
decoded_average_mae = convert_to_original_scale(total_mae / total_samples, min_target_val, max_target_val)
decoded_average_mse = convert_to_original_scale(total_mse / total_samples, min_target_val, max_target_val)
decoded_avg_val_loss = convert_to_original_scale(total_val_loss / total_samples, min_target_val, max_target_val)

print(f"Model evaluation complete - Avg Loss: {decoded_avg_val_loss:.4f}, Avg MAE: {decoded_average_mae:.4f}")
print(f"Average MAE: {decoded_average_mae}")
print(f"Average MSE: {decoded_average_mse}")

# Save the decoded results to a CSV file
result_df = pd.DataFrame(results)
csv_file_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/lion_heart_hybrid_regression_predictions.csv"
result_df.to_csv(csv_file_path, index=False)
print(f"Results saved to {csv_file_path}")
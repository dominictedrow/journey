"""
DDDD   Y   Y  N   N    AAA     |   M   M    L     PPPP
D   D   Y Y   NN  N   A   A    |   MM MM    L     P   P
D   D    Y    N N N   AAAAA    |   M M M    L     PPPP
D   D    Y    N  NN  A     A   |  M     M   L     P
DDDD     Y    N   N A       A  |  M     M   LLLLL P
_______________________________________________________ 
:Welcome to Custom Temporal Spartial Gating Multi Output 2 Stage Positional Encoding | Multi-Layer Perceptron (float32): (Version 1)


Version 1: Designed for sequenced or non sequenced data, which can be used for predicting continuous values, classes and even LLMs.

By: JD
"""  
import os 

"""
C++ minimum log level to filter out warnings for mixed precision 16bit 
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import joblib
import sqlite3
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
import keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras import Model, Input
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam 
from sklearn.preprocessing import StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.initializers import GlorotUniform, LecunNormal, GlorotNormal
from tensorflow.keras.layers import Dense, LayerNormalization, Layer, Conv1D, Dropout, Lambda, LSTM, GlobalAveragePooling1D, Concatenate, Layer, GlobalMaxPooling1D
"""                                                                           ++            ++++++
 \\\\\\\\\\\\\\\\\____________________________________________________________++++++++++++++++++++
|.. ........ ....... ...... ..... .... ... .. .   .   .     .        .        .  DYNA | MLP  .  ||||
|-----___   ___-----___   ____   ___-----__----    .       .     .        .           .         ||||
|TTTTT  H   H  IIIII  N   N  K   K  B B B  EEEE ..    .     .        .        .    Trading      ||||
|  T    H   H    I    NN  N  K  K   B   B  E       ... ...     .        .          .             ||
|/////  /////  /////  /////  ////   /////  ////     .....  ......        .        .  ||||-----|||-----||  | 
|  T    H   H    I    N  NN  K  K   B   B  E       ........    ..........         .               |
|  T    H   H  IIIII  N   N  K   K  B B B  EEEE ............      .............                  ||
|\\\\\\\     \\\\\\\\\     \\\\\\\\\     ___________________      /////////////      AI/ML       ||
__---__-----___---___-----___---___-----__--__--__--__--__-------__--__--_-_-_---_-_-__--___-_--_|
"""

# :::: Main Parameters: scroll toline 438 <Model Start> to change unit dimentions ::::

seq_len = 1                    # Currently: 15 Minute data: with positional encoding has 396 features per seq_len
layers = 1                     # Number of custom dense layers with special gating
epochs = 1000                  # Training iterations
batch_size = 512               # Samples per batch

# :::: learning warmup & Adam Optimizer ::::

custom_lr = False              # Activate warmup_steps
warmup_steps = int(979*3.0886) # Number of batches for lr warmup_steps  
lr_scale = 1                   # 1 is default for no change (increases starting lr 4 warmup_steps)

fixed_learning_rate = 0.0001   # Used if custom_lr=False
clipnorm = 1.0
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7
amsgrad = True

# :::: Training Target Predictions ::::
""" target_columns should be the name of any Target columns and output_names
should be the the targets you want to use, which requires adjusting drop_targets
to match the output_names you selected """
drop_targets = -1             # exclude the last target columns
target_columns = ['Target1', 'Target2', 'Target3']
output_names =   ['High', 'Low']

# :::: Training Loss Function ::::

loss_func = mean_absolute_error
""" 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'
'mean_squared_logarithmic_error', 'huber_loss', 'log_cosh', 'cosine_similarity' """

# :::: Dual Activation ::::

activation_1 = tf.tanh        # MLP Activation
activation_2 = tf.tanh        # Spartial Gating Activation
""" 'relu', 'sigmoid', 'tanh', 'gelu', 'softplus', 'elu', LeakyReLU """

# :::: Dense Layer Initilizers ::::

ini_1 = GlorotNormal()         # MLP Initilizers
ini_2 = GlorotNormal()        # Spartial Gating Activation
""" 'zeros', 'ones', 'constant', 'random_normal', 'random_uniform', 'truncated_normal',
'variance_scaling', 'orthogonal', 'identity', 'glorot_normal', 'glorot_uniform',
'lecun_normal', 'lecun_uniform', 'he_normal', 'he_uniform' """

# :::::::::::::::::::::::::::::TensorFlow setup:::::::::::::::::::::::::::::: #

# Suppress TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.config.experimental.set_visible_devices
print("Eager execution:", tf.executing_eagerly())

# Set device configuration early in the script
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Only use the first GPU and set memory growth
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)

print("Eager execution:", tf.executing_eagerly()) 
print()

# ::::::::::::::::::::::Data Processing Transformation::::::::::::::::::::::: #

# Data Loading and Preprocessing Functions
def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name} ORDER BY ROWID DESC LIMIT 1"
    df = pd.read_sql_query(query, conn)
    conn.close()
    #df.drop(df.columns[0], axis=1, inplace=True)
    return df

def create_overlapping_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len + 1):
        seq = data[i:i + seq_len]
        sequences.append(seq)
    return np.array(sequences)

def preprocess_data(df, seq_len, feature_scaler=None):
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Initialize, fill missing values, and scale features
    if feature_scaler is None:        
        feature_scaler = StandardScaler()
        mean_values = df.mean(numeric_only=True)
        df.fillna(mean_values, inplace=True)
        scaled_features = feature_scaler.fit_transform(df.values.astype('float32'))
    else:        
        mean_dict = dict(zip(df.columns, feature_scaler.mean_))
        df.fillna(mean_dict, inplace=True)
        scaled_features = feature_scaler.transform(df.values.astype('float32'))

    # Ensure the feature set has an even number of features
    if scaled_features.shape[1] % 2 != 0:
        scaled_features = np.hstack([scaled_features, np.zeros((scaled_features.shape[0], 1), dtype=scaled_features.dtype)])

    # Creating overlapping sequences for inputs
    X = create_overlapping_sequences(scaled_features, seq_len)

    return X, feature_scaler

class ExtendedPositionalEncoding(Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(ExtendedPositionalEncoding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.extended_pos_encoding = self.create_positional_encoding(seq_len, d_model)

    def create_positional_encoding(self, seq_len, d_model):
        position = tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * -(tf.math.log(10000.0) / d_model))
        pos_encoding = tf.concat([
            tf.sin(position * div_term),
            tf.cos(position * div_term)], axis=-1)
        pos_encoding = tf.concat([pos_encoding, pos_encoding], axis=-1)  # Double the features
        return pos_encoding[tf.newaxis, ...]

    def call(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size = input_shape[0]
        seq_len = input_shape[1]

        # Extend the positional encoding based on the current input sequence length
        extended_encoding = self.extended_pos_encoding[:, :seq_len, :]
        extended_encoding = tf.tile(extended_encoding, [batch_size, 1, 1])
        extended_encoding = tf.cast(extended_encoding, dtype=inputs.dtype)

        # Create the new feature next to its target feature
        inputs_extended = tf.TensorArray(inputs.dtype, size=0, dynamic_size=True)

        # Double the number of features by interleaving the input features and positional encodings
        for i in range(self.d_model):
            inputs_extended = inputs_extended.write(i * 2, inputs[:, :, i])
            inputs_extended = inputs_extended.write(i * 2 + 1, extended_encoding[:, :, i])

        inputs_extended = inputs_extended.stack()
        inputs_extended = tf.transpose(inputs_extended, perm=[1, 2, 0])
        inputs_extended.set_shape([None, self.seq_len, self.d_model * 2])

        return inputs_extended
    
# Integrate positional encoding
def integrate_positional_encoding(X, seq_len, d_model):
    layer = ExtendedPositionalEncoding(seq_len, d_model)
    X_tensor = tf.convert_to_tensor(X, dtype=tf.float16)
    X_encoded = layer(X_tensor)
    return X_encoded.numpy()

# Load the scaler
feature_scaler = joblib.load('scalers/scaler_15m.pkl')

print(f"\n") 
print(f"Loading AI Model...") 
# Load and preprocess data
df = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//metatrader//dwxconnect-main//python//data//databases//XAUUSD_15m_OHLC_Live.db", 'XAUUSD')

# Store feature names 
feature_names = df.columns

# Starting to preprocess data
X, _ = preprocess_data(df, seq_len, feature_scaler)
X = integrate_positional_encoding(X, seq_len, X.shape[-1])
print(f"Finished loading AI Model...")                

# :::::::::::::::::::::::::::Custom Positional Encoding Layer::::::::::::::::::::::::::::::: #

class CustomActivation(Layer):
    def __init__(self, alpha=0.1, max_neg=-1.2, **kwargs):
        super().__init__(**kwargs)
        # Alpha controls the negative slope
        self.alpha = alpha
        # max_neg allows the negative output to go beyond -1
        self.max_neg = max_neg

    def call(self, inputs):
        tanh_out = tf.tanh(inputs)   
        return tf.where(tanh_out < -1, tanh_out * self.alpha + (1 - self.alpha) * self.max_neg, tanh_out)

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(PositionalEncoding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self.positional_encoding(seq_len, d_model)

    def get_angles(self, pos, i, d_model):
        angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
        return pos * angles

    def positional_encoding(self, seq_len, d_model):
        angle_rads = self.get_angles(
            pos=tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=d_model
        )
        # Sines and cosines calculated on appropriate indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Concatenate sines and cosines, reshape for easy addition
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # Add new axis for batch size
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
        # Add positional encoding to inputs, handling different input lengths
        return inputs + tf.cast(self.pos_encoding[:, :tf.shape(inputs)[1], :], inputs.dtype)
    
    def get_config(self):
        config = super(PositionalEncoding, self).get_config()
        config.update({
            'seq_len': self.seq_len,
            'd_model': self.d_model
        })
        return config

# :::::::::::::::::::::::::::Custom Spatial Gating MLP::::::::::::::::::::::::::::::: #

class SpatialGatingUnit(Layer):
    def __init__(self, q_ffn, activation_1=activation_1, activation_2=activation_2, initializer_1=ini_1, initializer_2=ini_2, kernel_size=6, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.q_ffn = q_ffn
        self.kernel_size = kernel_size
        self.initializer_1 = initializer_1
        self.initializer_2 = initializer_2
        self.activation_1 = activation_1
        self.activation_2 = activation_2
        
        self.normalize_1 = LayerNormalization()
        self.normalize_2 = LayerNormalization()

        self.spatial_projection = Dense(q_ffn, activation=self.activation_2, kernel_initializer=self.initializer_2, use_bias=True)
        
        self.temporal_1 = LSTM(int(q_ffn), activation=self.activation_1, kernel_initializer=self.initializer_1, return_sequences=True)
        self.temporal_2 = Conv1D(filters=q_ffn, kernel_size=self.kernel_size, padding='same', activation='relu')
        self.temporal_projection = Dense(self.q_ffn, activation=self.activation_2, kernel_initializer=self.initializer_2)
        
        # Forget and input gates
        self.forget_gate = Dense(self.q_ffn, activation='sigmoid', kernel_initializer=GlorotNormal())
        self.input_gate = Dense(self.q_ffn, activation='sigmoid', kernel_initializer=GlorotNormal())

    def build(self, input_shape):
        self.spatial_gating = self.add_weight(
            name="spatial_gating",
            shape=(input_shape[-1],),
            initializer=GlorotNormal(),
            trainable=True
        )

    def call(self, inputs):
        normalized_inputs = self.normalize_1(inputs)

        temporal_context = self.temporal_1(normalized_inputs)
        temporal_context = self.temporal_2(temporal_context)
        temporal_context = self.normalize_2(temporal_context)
        temporal_context = self.temporal_projection(temporal_context)
        
        f_gate = self.forget_gate(normalized_inputs)
        i_gate = self.input_gate(normalized_inputs)
        
        gated_temporal = f_gate * temporal_context + i_gate * self.spatial_projection(normalized_inputs)
        
        return gated_temporal * tf.sigmoid(self.spatial_gating) + inputs

    def get_config(self):
        config = super(SpatialGatingUnit, self).get_config()
        config.update({
            "q_ffn": self.q_ffn,
            "activation_1": tf.keras.activations.serialize(self.activation_1),
            "activation_2": tf.keras.activations.serialize(self.activation_2),
            "initializer_1": tf.keras.initializers.serialize(self.initializer_1),
            "initializer_2": tf.keras.initializers.serialize(self.initializer_2)
        })
        return config


class gMLPBlock(Layer):
    def __init__(self, d_ffn, q_ffn, activation_1=activation_1, activation_2=activation_2, initializer_1=ini_1, initializer_2=ini_2, dropout_rate=.01, regularizer=l2(0.01), **kwargs):
        super(gMLPBlock, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.q_ffn = q_ffn
        self.initializer_1 = initializer_1
        self.initializer_2 = initializer_2
        self.activation_1 = activation_1
        self.activation_2 = activation_2
        self.regularizer = regularizer

        self.dropout_1 = Dropout(dropout_rate)
        self.dropout_2 = Dropout(dropout_rate)
 
       
        self.channel_projection_i = Dense(self.q_ffn, activation=self.activation_2, kernel_initializer=initializer_2)
        self.sgu_1 = SpatialGatingUnit(self.q_ffn)
        self.channel_projection_ii =  Dense(self.d_ffn, activation=self.activation_1, kernel_initializer=initializer_1)

    def call(self, inputs, training=False):
        x1 = self.dropout_1(inputs, training=training)
        x1 = self.channel_projection_i(x1)

        x2 = self.sgu_1(x1)

        x3 = self.dropout_2(x2, training=training)
        x3 = self.channel_projection_ii(x3)
        x3 += inputs                            
        return x3    
       

    def get_config(self):
        config = super(gMLPBlock, self).get_config()
        config.update({ 
            "q_ffn": self.q_ffn, 
            "d_ffn": self.d_ffn,
            "activation_1": tf.keras.activations.serialize(self.activation_1),
            "activation_2": tf.keras.activations.serialize(self.activation_2),
            "initializer_1": tf.keras.initializers.serialize(self.initializer_1),
            "initializer_2": tf.keras.initializers.serialize(self.initializer_2),
            "regularizer": tf.keras.regularizers.serialize(self.regularizer)
        })
        return config
    
class DynamicDenseLayers(Layer):
    def __init__(self, feature_groups, unit_e, kernel_initializer='he_normal', **kwargs):
        super().__init__(**kwargs)
        self.feature_groups = feature_groups
        self.unit_e = unit_e
        self.kernel_initializer = kernel_initializer
        # Dictionary to store layers for each sequence length
        self.dense_layers_by_seq_len = {}

    def build(self, input_shape):
        self.d_model = input_shape[-1] 
        super().build(input_shape)

    def call(self, inputs):
        seq_len = inputs.shape[1]

        if seq_len not in self.dense_layers_by_seq_len:
            self.dense_layers_by_seq_len[seq_len] = [
                Dense(self.unit_e, activation='linear', kernel_initializer=self.kernel_initializer,
                      name=f'dense_layer_{seq_len}_{i}') for i in range(self.d_model // self.feature_groups)
            ]
        
        # Split the input tensor into groups
        num_groups = self.d_model // self.feature_groups
        grouped_inputs = tf.split(inputs, num_or_size_splits=num_groups, axis=2)

        dense_outputs = []
        for i, group in enumerate(grouped_inputs):
            x = self.dense_layers_by_seq_len[seq_len][i](group)
            dense_outputs.append(x)
        
        # Concatenate outputs
        concatenated_outputs = Concatenate(name=f'concat_layer_{seq_len}')(dense_outputs)

        return concatenated_outputs
    
# :::::::::::::::::::::::Custom Layer (undefined)::::::::::::::::::::::::::: #

""" Space for later"""

# ::::::::::::::::::::::::Main Model Parameters::::::::::::::::::::::::::::: #

# ::Data Model Transformation::              
d_model = X.shape[2]                  # Input shape in units
inputs = Input(shape=(seq_len, d_model))    # Input seq_len & features

d_ffn = int(d_model*5)                      # 1nd unit exspansion/supression
q_ffn = int(d_ffn*2)                        # 2nd unit exspansion/supression

# :::::::::::::::::::::::::::Main Model Start::::::::::::::::::::::::::::::: #

#dynamic_layer = DynamicDenseLayers(feature_groups, unit_e, ini_1)
#outputs = dynamic_layer(inputs)
#d_ffn = int(outputs.shape[2]*8)

#projected = Dense(d_ffn, activation='linear', kernel_initializer=ini_1)(inputs)
#pos_layer = PositionalEncoding(seq_len, d_ffn)
#pos = pos_layer(projected)
#x = pos

# ::::::::::::::Main Model Training (Multi Target Prediction):::::::::::::: #

""" Add averaged_output only if wanting to avg seq_len predictions to a single output,
     per output_names, but must change loss_function in training loop to loss_func. loss_function
     averages each seq_len prediction after the final output rather than in the model before the
     final output where you would use loss_func"""

outputs = []
for name in output_names:
    projected = Dense(d_ffn, activation='linear', kernel_initializer=ini_1)(inputs)
    pos_layer = PositionalEncoding(seq_len, d_ffn)
    pos = pos_layer(projected)
    x = pos        
    for _ in range(layers):
        x = gMLPBlock(d_ffn, q_ffn)(x, training=True)      
    output = Dense(1, activation='linear', dtype='float32')(x)
    pooling_layer = GlobalMaxPooling1D(name=name, dtype='float32')
    output = pooling_layer(output)

    "Average the seq_len predictions to a single prediction"
    #output = Lambda(lambda x: tf.reduce_mean(x, axis=1), dtype='float32')(output)
    outputs.append(output) 

    """Apply SoftMax to each seq_len prediction and average the seq_len single unit predictions
    for each output_names for single value output. Or apply SoftMax to gMLPBlock output, 
    multiplied by the output, and either average the seq_len dimention to make single prediction,
    or make a prediction for each seq_len and average the final output together."""

    #weighted_output = Lambda(lambda x: tf.reduce_sum(x, axis=1), name=f"{name}_avg")(weighted_output)
    #final_output = Dense(1, activation='linear', name=name, dtype='float32')(weighted_output)
    #final_output = Lambda(lambda x: tf.reduce_sum(x, axis=1), name=name, dtype='float32')(final_output)
    #outputs.append(final_output)   

model = Model(inputs=inputs, outputs=outputs)

#model.summary()
print("Model initialized!")
print("Total number of parameters in the model:", model.count_params())

# :::::::::::::::::::::Model Checkpoint For Weights::::::::::::::::::::::: #

# Define the model checkpoint for weights only
weights_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/models/model_weights/XAUUSD_high_weights_v1.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/models/model_weights/XAUUSD_high_model_v1_FULL.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', 
                                        verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

# Try loading the full model, then the weights, or use the freshly defined model if neither exists
if os.path.exists(weights_checkpoint_path):
    print("Loading model weights...")
    model.load_weights(weights_checkpoint_path)
elif os.path.exists(full_model_checkpoint_path):
    print("Loading full model...")
    model = tf.keras.models.load_model(full_model_checkpoint_path, custom_objects={'PositionalEncoding': PositionalEncoding,                                                                              
                                                                                   #'DynamicDenseLayers': DynamicDenseLayers, 
                                                                                   'SpatialGatingUnit': SpatialGatingUnit, 
                                                                                   'gMLPBlock': gMLPBlock})
else:
    print("Using freshly defined model...")


# ::::::::::::::::::::::::Training & Validation Loop:::::::::::::::::::::::::::: #

# Make a single prediction
predictions = model.predict(X)
print("Detailed prediction outputs:", predictions)
for pred in predictions:
    print("Flattened prediction:", pred.flatten())


# Save predictions to text file
with open('predictions/predictions_15m.txt', 'a') as f:
    if len(predictions) >= 2:
        # Assume the first element in predictions is High and the second is Low
        high_pred = predictions[0][0, 0]  # Access the first and only element in the array
        low_pred = predictions[1][0, 0]  # Access the first and only element in the array
        f.write(f"Predicted_High: {high_pred}, Predicted_Low: {low_pred}\n")
    else:
        print("Insufficient prediction outputs:", predictions)
print(f"AI Model Predictions Complete...\n") 





# ::::::::::::::::::Evaluation 2nd Model (Save Predictions):::::::::::::::::::::: #





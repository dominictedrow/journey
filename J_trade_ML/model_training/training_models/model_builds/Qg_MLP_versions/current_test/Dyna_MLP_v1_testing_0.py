"""
TTTTT   GGG   IIIII      AAA     |   M   M    L     PPPP
  T    G        I       A   A    |   MM MM    L     P   P
  T    G  GG    I       AAAAA    |   M M M    L     PPPP
  T    G   G    I      A     A   |  M     M   L     P
  T     GGG   IIIII   A       A  |  M     M   LLLLL P
_________________________________________________       

:Welcome to Temproal Gating Information Theory | Flash-Attention(v1) Multi-Layer Perceptron (float32): (Version 1)

Version 1:dfgsd fgsdfgsdfgsdfgsd fgs df gsd fgs df gsd fgsfgsdfg sfgs dfg sdfg sdfg dsfg sdf gd fg sdf sdf
sdf gsdfgsdfg sdfg sdf gsdfgs dfgsdfg sdfg dfgsdf gdfg sdfg sdf gsdf gsd fgs dfgsdfgsfd gsdfgdfg sdfgsdfsd
dsf gsdfg sdfg sdfgsdfgsdfgsdfgsdf
sdf gdf gf gsdfgsdfg 
fd gdsfg sdfg  

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
import tensorflow.keras.backend as K
from tensorflow.keras import Model, Input
from tensorflow.keras.optimizers import Adam, SGD 
from tensorflow.keras.mixed_precision import experimental
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.layers import Dense, LayerNormalization, Layer, Dropout, GlobalAveragePooling1D, Conv1D, LSTM, Reshape, Concatenate, Bidirectional, Lambda, SimpleRNN

"""                                              .
TTTTT  H   H  IIIII  N   N  K   K  B B B  EEEE ...   .
  T    H   H    I    NN  N  K  K   B   B  E    .......     .
  T    HHHHH    I    N N N  KKK    BBBB   EEE  .............        .
  T    H   H    I    N  NN  K  K   B   B  E    ......................        .
  T    H   H  IIIII  N   N  K   K  B B B  EEEE ...............................

"""

# ::Preprocessing Parameters::
flat_seq_len = 2            # Sequences are reshaped on same axis

layers_1 = 6                # Sequences are reshaped on same axis
seq_len = 4                 # Sequences are reshaped on same axis

epochs_1 = 5                # Sequences are reshaped on same axis
batch_size_1 = 32           # Sequences are reshaped on same axis

# ::Main Parameters::
max_feature_length = 1280   # Sequences are reshaped on same axis
memory_size = 100           # Sequences are reshaped on same axis

d_ffn = 792                 # 1st unit exspansion 1x
q_ffn = int(d_ffn*2)        # 2nd unit exspansion 2x
layers_2 = 1                # Sequences are reshaped on same axis
block_layers = 2            # gMLQPBlock, Gating, Dense

epochs_2 = 2000             # Sequences are reshaped on same axis
batch_size_2 = 32           # Sequences are reshaped on same axis

scale_target = False        # normilize target
scale_features = True       # normilize features

# ::Dynamic Learning_rate Parameters::
lr_enable = False           # Toggle learning rate adjustment
print_enable = True 
initial_lr = 0.0001         # starting learning rate  
min_lr = 0.000001111111     # minimum learning rate
decay_step = 0.00000065     # amount to decrease learning rate each epoch

# ::Adam Optimizer parameters::
learning_rate = initial_lr
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7
amsgrad = True

# ::gMLQP loss function::
""" 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'
'mean_squared_logarithmic_error', 'huber_loss', 'log_cosh', 'cosine_similarity' """

loss_func = 'mean_absolute_error'

# ::gMLQP activations & Initilizers::
activation_1 = tf.nn.tanh
activation_2 = tf.nn.tanh 
ini='lecun_normal'

# TensorFlow policy to mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

tf.config.experimental.set_visible_devices
print("Eager execution:", tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e) 

# ::::::::::::::::::::::::::::Real-time Dashboard::::::::::::::::::::::::::::::: #

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
        plt.savefig("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/XAUUSD_high_train_loss_v1_testing.png") 
        self.root.update()

print("About to initialize the model...")

# ::::::::::::::::::::::Data Processing Transformation #1::::::::::::::::::::::: #

def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.drop(df.columns[0], axis=1, inplace=True)
    return df

def create_overlapping_sequences(data, flat_seq_len):
    sequences = []
    for i in range(flat_seq_len - 1, len(data)):
        seq = data[i - (flat_seq_len - 1):i + 1] 
        sequences.append(np.array(seq))
    return np.array(sequences)

def preprocess_data(df, flat_seq_len, scale_features=False, feature_scaler=None):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")

    # Separate target from features
    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    if scale_features:
        if feature_scaler is None:
            feature_scaler = StandardScaler()
            mean_values = df.mean(numeric_only=True)
            df.fillna(mean_values, inplace=True)
            scaled_features = feature_scaler.fit_transform(df.values.astype('float32'))
        else:
            mean_dict = {column: np.nanmean(df[column]) for column in df.columns}
            df.fillna(mean_dict, inplace=True)
            scaled_features = feature_scaler.transform(df.values.astype('float32'))
    else:
        df.fillna(df.mean(numeric_only=True), inplace=True)
        scaled_features = df.values

    target.fillna(target.mean(), inplace=True)
    target_scaled = target.values

    # Create sequences
    X = create_overlapping_sequences(scaled_features, flat_seq_len)
    y = target_scaled[flat_seq_len - 1:]

    return X, y

def sinusoidal_positional_encoding(flat_seq_len, num_features):
    """Generates a sinusoidal positional encoding matrix based on sequence length and number of features."""
    pos_encoding = np.zeros((flat_seq_len, num_features))
    for pos in range(flat_seq_len):
        for i in range(num_features):
            angle_rate = 1 / np.power(10000, (2 * (i // 2)) / num_features)
            if i % 2 == 0:
                pos_encoding[pos, i] = np.sin(pos * angle_rate)
            else:
                pos_encoding[pos, i] = np.cos(pos * angle_rate)
    return pos_encoding

def flatten_with_position_encoding(data, feature_names, flat_seq_len):
    """Flattens data and applies sinusoidal positional encoding."""
    num_features = len(feature_names)
    reshaped_data = data.reshape(data.shape[0], -1)  # Flatten the flat_seq_len and features

    # Generate positional encoding for each sequence and feature
    pos_encoding = sinusoidal_positional_encoding(flat_seq_len, num_features)
    expanded_pos_encoding = np.repeat(pos_encoding[np.newaxis, :, :], data.shape[0], axis=0).reshape(data.shape[0], -1)

    # Concatenate the original data with positional encoding
    enhanced_data = np.concatenate([reshaped_data, expanded_pos_encoding], axis=1)

    # Adjust feature names to include positional encoding
    new_feature_names = [f"{name}_{i}" for i in range(flat_seq_len) for name in feature_names]
    new_feature_names += [f"pos_enc_{j}_{i}" for i in range(flat_seq_len) for j in range(num_features)]  # Positional features names

    return enhanced_data, new_feature_names

# Load and preprocess data
print("About to load and preprocess training data...")
df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_High_Train.db", 'XAUUSD')
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_High_Val.db", 'XAUUSD')
common_features = df_train.columns.intersection(df_val.columns)
df_train = df_train[common_features]
df_val = df_val[common_features]

# Store feature names before modifying dataframes
feature_names = common_features.drop('Target') 
#feature_names = df_train.columns.tolist() # Make sure 'Target' is part of the dataframe before this operation

# Initialize and potentially fit scalers
feature_scaler = StandardScaler() if scale_features else None

if feature_scaler:
    df_train_filtered = df_train.drop(columns='Target').replace([np.inf, -np.inf], np.nan).fillna(df_train.mean(numeric_only=True))
    feature_scaler.fit(df_train_filtered.values.astype('float32'))

# After loading data
print("Finished loading training data...")

print("Starting to preprocess training data...")
X_train, y_train = preprocess_data(df_train, flat_seq_len, scale_features, feature_scaler)
print("Finished preprocessing training data...")
X_train, train_feature_names = flatten_with_position_encoding(X_train, feature_names, flat_seq_len)
print("starting X_train shape:", X_train.shape)
# print("New feature names in training data after reshape:", train_feature_names) # print train feature names after processing

print("Starting to preprocess validation data...")
X_val, y_val = preprocess_data(df_val, flat_seq_len, scale_features, feature_scaler)
X_val, val_feature_names = flatten_with_position_encoding(X_val, feature_names, flat_seq_len)
print("Finished preprocessing validation data...")
print("starting X_val shape:", X_val.shape)
#print("New feature names in valuation data after reshape:", val_feature_names) # print val feature names after processing

# ::::::::::::::::::::::Data Processing Transformation #2::::::::::::::::::::::: #

class DenseBlock(Layer):
    def __init__(self, feature_dim, dropout_rate=0.2):
        super(DenseBlock, self).__init__()
        self.dense = Dense(feature_dim, kernel_initializer=ini, use_bias=False)
        #self.norm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)
    
    def call(self, inputs):
        x = self.dense(inputs)
        #x = self.norm(x)
        x = activation_1(x)
        x = self.dropout(x)
        return x

def build_transformation_model(input_dim, seq_len, feature_dim):
    inputs = tf.keras.Input(shape=(input_dim,))  # Input dimension
    #x = LayerNormalization()(inputs) 
    x = inputs

    outputs = [x]  # Start the output sequence with the normalized input
    prediction_outputs = []
    
    for i in range(1, seq_len):  # Start from 1 as the first element is already handled
        for _ in range(layers_1):
            x = DenseBlock(feature_dim)(x)
        
        """ # residual connection every few layers, adjust "3" to your preferred frequency
        if i % 3 == 0 and len(outputs) > 3:
            x = tf.add(x, outputs[-3])  """

        prediction = Dense(1, activation=None)(x)
        prediction_outputs.append(prediction)
        outputs.append(x)

    stacked_output = Lambda(lambda x: tf.stack(x, axis=1))(outputs)
    predictions_output = Lambda(lambda x: tf.stack(x, axis=1))(prediction_outputs)

    model = Model(inputs=inputs, outputs=[stacked_output, predictions_output])
    return model

# :::::::::::::::::::::::::::Saving Model weights::::::::::::::::::::::::::::::: #

# Define the model checkpoint for weights only
weights_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/XAUUSD_high_weights_v1_data_prep.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/XAUUSD_high_model_v1_data_prep.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')
    
# Define model parameters
input_dim = X_train.shape[1]  
feature_dim = input_dim 

# Build the transformation model
if os.path.exists(weights_checkpoint_path):
    print("Loading model weights...")
    transformation_model = build_transformation_model(input_dim, seq_len, feature_dim)
    transformation_model.load_weights(weights_checkpoint_path)
else:
    print("Using freshly defined model...")
    transformation_model = build_transformation_model(input_dim, seq_len, feature_dim)

transformation_model.summary()
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)
transformation_model.compile(optimizer=optimizer, loss=loss_func)

y_train_expanded = np.tile(y_train[:, np.newaxis], (1, seq_len - 1))  # Repeat target across sequence length
y_val_expanded = np.tile(y_val[:, np.newaxis], (1, seq_len - 1))  # Repeat target across sequence length

dashboard = RealTimeDashboard()
history = transformation_model.fit(
    X_train, [np.tile(X_train[:, np.newaxis, :], (1, seq_len, 1)), y_train_expanded],
    epochs=epochs_1,
    batch_size=batch_size_1,
    validation_data=(X_val, [np.tile(X_val[:, np.newaxis, :], (1, seq_len, 1)), y_val_expanded]),
    callbacks=[weights_checkpoint, dashboard]
)

# Predictions
X_train_transformed, X_train_predictions = transformation_model.predict(X_train)
X_val_transformed, X_val_predictions = transformation_model.predict(X_val)

print("Data pre-processing complete.")
dashboard.root.destroy()

# Check shapes
print("Transformed X_train shape:", X_train_transformed.shape)
print("Prediction X_train shape:", X_train_predictions.shape)
print("Transformed X_val shape:", X_val_transformed.shape)
print("Prediction X_val shape:", X_val_predictions.shape)

# ::::::::::::::::::::::::::::Custom DynaDense Layer Class::::::::::::::::::::::::::::::: #

class DynaDense(Layer):
    def __init__(self, units, activation, memory_size, **kwargs):
        super(DynaDense, self).__init__(**kwargs)
        self.units = units
        self.memory_size = memory_size
        self.activation = activation

    def build(self, input_shape):
        assert len(input_shape) >= 2, "Input must have at least two dimensions (batch size, features)"
        self.w = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        """ self.alpha = self.add_weight(shape=(1,),
                                     initializer='random_normal',
                                     trainable=True)
        self.beta = self.add_weight(shape=(1,),
                                    initializer='random_normal',
                                    trainable=True) """
        self.thresholds = self.add_weight(shape=(self.units,),
                                          initializer='random_normal',
                                          trainable=True)
        self.threshold_memory = tf.Variable(initial_value=tf.zeros((self.units, self.memory_size)),
                                            trainable=False)
        self.input_memory = tf.Variable(initial_value=tf.zeros((self.memory_size, input_shape[-1])),
                                        trainable=False)
        super(DynaDense, self).build(input_shape)

    def call(self, inputs):
        inputs = tf.expand_dims(inputs, 0) if len(inputs.shape) == 1 else inputs

        # Roll input memory and update
        rolled_input_memory = tf.roll(self.input_memory, shift=1, axis=0)
        updates = tf.concat([tf.reduce_mean(inputs, axis=0, keepdims=True), rolled_input_memory[1:]], axis=0)
        self.input_memory.assign(updates)

        z = K.dot(inputs, self.w) + self.b

        # Roll threshold memory and prepare for update
        rolled_threshold_memory = tf.roll(self.threshold_memory, shift=1, axis=1)
        # Compute new threshold values, reshape them to match the required dimensions for concatenation
        new_thresholds = tf.reduce_mean(z, axis=0)  # Get the mean for each unit across batch
        new_thresholds_reshaped = tf.reshape(new_thresholds, (self.units, 1))  # Reshape to [160, 1] to match the other dimension
        # Concatenate along the second axis (columns)
        updated_threshold_memory = tf.concat([new_thresholds_reshaped, rolled_threshold_memory[:, :-1]], axis=1)
        self.threshold_memory.assign(updated_threshold_memory)

        mask = K.greater(z, self.thresholds)
        z_masked = tf.cast(mask, tf.float32) * z
        output = self.activation(z_masked)

        """ update_value = tf.reduce_mean([self.beta * K.dot(K.transpose(self.input_memory[k]), output) for k in range(self.memory_size)], axis=0)
        self.w.assign(self.w + (update_value - self.alpha * self.w) * 0.01) """

        return output, tf.cast(mask, tf.float32)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

class DynamicOutputModel(Model):
    def __init__(self, d_model, activation, memory_size, max_feature_length):
        super(DynamicOutputModel, self).__init__()
        self.d_model = d_model
        self.activation = activation
        self.memory_size = memory_size

        self.layer1 = DynaDense(d_model, activation, memory_size)
        self.layer2 = DynaDense(d_model, activation, memory_size)
        self.max_feature_length = max_feature_length

    def call(self, inputs):
        x1, mask1 = self.layer1(inputs)
        x2, mask2 = self.layer2(x1)

        active_outputs = tf.concat([x1 * mask1, x2 * mask2], axis=1)
        print("Active outputs shape:", active_outputs.shape)  # Print shape after concatenation

        active_feature_count = tf.shape(active_outputs)[1]
        padding_size = tf.maximum(0, self.max_feature_length - active_feature_count)  # Ensure padding size is not negative
        padded_output = tf.pad(active_outputs, [[0, 0], [0, padding_size]], "CONSTANT")

        final_output = padded_output[:, :self.max_feature_length]
        #final_output += inputs
        final_output.set_shape([None, self.max_feature_length])  # Explicitly set the shape to prevent None dimensions
        print("Final output shape:", final_output.shape)  # Print final output shape before layer return

        return final_output
    
# ::::::::::::::::::::::::::::SpatialGatingUnit Layer Class::::::::::::::::::::::::::::::: #
    
class SpatialGatingUnit(Layer):
    def __init__(self, d_ffn, q_ffn, kernel_size=3, dropout_rate=0.1, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.q_ffn = q_ffn
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        self.layer_norm = LayerNormalization()
        
        # Replace Dense with Bidirectional RNN
        rnn_layer_spatial = SimpleRNN(d_ffn, activation=activation_2, kernel_initializer=ini, use_bias=True, return_sequences=True)
        self.spatial_projection = Bidirectional(rnn_layer_spatial)
        
        # Adapt Conv1D to handle new shape (None, seq_len, d_ffn)
        self.temporal_conv = Conv1D(filters=d_ffn, kernel_size=self.kernel_size, 
                                    activation='linear', padding='same', 
                                    kernel_initializer=ini)
        
        # Replace Dense with Bidirectional RNN for temporal projection
        rnn_layer_temporal = SimpleRNN(d_ffn, activation=activation_2, return_sequences=True)
        self.temporal_projection = Bidirectional(rnn_layer_temporal)
        
        # Forget and input gates replaced with RNN
        rnn_layer_forget = SimpleRNN(d_ffn, activation='sigmoid', kernel_initializer=ini, return_sequences=True)
        self.forget_gate = Bidirectional(rnn_layer_forget)
        
        rnn_layer_input = SimpleRNN(d_ffn, activation='sigmoid', kernel_initializer=ini, return_sequences=True)
        self.input_gate = Bidirectional(rnn_layer_input)

        self.dropout = Dropout(dropout_rate)

    def build(self, input_shape):
        self.spatial_gating = self.add_weight(
            shape=(input_shape[-1],),
            initializer='he_normal',
            trainable=True
        )

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        normalized_inputs = self.dropout(normalized_inputs)
        spatial_context = self.spatial_projection(normalized_inputs)
        
        # Conv1D processing, handling sequence length
        inputs_expanded = tf.expand_dims(normalized_inputs, axis=1)
        temporal_context = self.temporal_conv(inputs_expanded)
        temporal_context = tf.squeeze(temporal_context, axis=1)

        temporal_context = self.temporal_projection(temporal_context)
        
        f_gate = self.forget_gate(normalized_inputs)
        i_gate = self.input_gate(normalized_inputs)
        
        # Element-wise gating operation
        gated_temporal = f_gate * temporal_context + i_gate * spatial_context
        
        return gated_temporal * tf.sigmoid(self.spatial_gating) + inputs

    def get_config(self):
        config = super(SpatialGatingUnit, self).get_config()
        config.update({"d_ffn": self.d_ffn, "kernel_size": self.kernel_size,
                       "activation": activation_1, "ini": ini, "dropout_rate": self.dropout_rate})
        return config


class gMLQPBlock(Layer):
    def __init__(self, d_ffn, q_ffn, feature_dim, dropout_rate=0.1):
        super().__init__()
        self.q_ffn = q_ffn
        self.d_ffn = d_ffn
        self.feature_dim = feature_dim
        self.dropout_rate = dropout_rate 
       
        rnn_layer_i = SimpleRNN(d_ffn, activation=None, kernel_initializer=ini, use_bias=True, return_sequences=True)
        self.rnn_bi_i = Bidirectional(rnn_layer_i)
        
        #self.rnn_layer_norm_i = LayerNormalization()
        
        self.sgu = SpatialGatingUnit(d_ffn, q_ffn)
        
        rnn_layer_ii = SimpleRNN(d_ffn, activation=None, kernel_initializer=ini, use_bias=True, return_sequences=True)
        self.rnn_bi_ii = Bidirectional(rnn_layer_ii)
        
        #self.rnn_layer_norm_ii = LayerNormalization()

        self.dense_resize = Dense(feature_dim, activation=None, use_bias=False)       
        self.dropout_layer = Dropout(dropout_rate)

    def call(self, inputs):
        x = self.dropout_layer(inputs) 
        x = self.rnn_bi_i(x)
        #x = self.rnn_layer_norm_i(x) 
        x = activation_2(x)
        
        x = self.sgu(x)
        
        x = self.rnn_bi_ii(x)
        #x = self.rnn_layer_norm_ii(x) 
        x = activation_2(x)

        x = self.dense_resize(x)        
        return x + inputs

    def get_config(self):
        config = super(gMLQPBlock, self).get_config()
        config.update({"q_ffn": self.q_ffn, "d_ffn": self.d_ffn, "dropout_rate": self.dropout_rate})
        return config
    
# ::::::::::::::::::::::::::::Main Model Loop::::::::::::::::::::::::::::::: #

inputs = Input(shape=(seq_len, feature_dim))
normilized_inputs = LayerNormalization()(inputs)
# Initialize variational dropout
dropout_layer = Dropout(0.2)

# Define the block structure
block_outputs = []
for _ in range(block_layers): 
    x = normilized_inputs
    for _ in range(layers_2): 
        x = gMLQPBlock(d_ffn, q_ffn, feature_dim)(x)

    block_outputs.append(x)

# Concatenate outputs from different blocks
concatenated_outputs = Concatenate(axis=1)(block_outputs)

# Final RNN layer after concatenating outputs
final_rnn_layer = SimpleRNN(d_ffn, activation=activation_2, kernel_initializer=ini, return_sequences=False)
x = final_rnn_layer(concatenated_outputs)

# Output layer
outputs = Dense(1)(x) 
model = Model(inputs=inputs, outputs=outputs)

model.summary()
print("Total number of parameters in the model:", model.count_params())

# :::::::::::::::::::::::::::Saving Model weights::::::::::::::::::::::::::::::: #

# Define the model checkpoint for weights only
weights_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/XAUUSD_high_weights_v1_testing.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/XAUUSD_high_model_v1_testing.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

# Try loading the full model, then the weights, or use the freshly defined model if neither exists
if os.path.exists(full_model_checkpoint_path):
    print("Loading full model...")
    model = tf.keras.models.load_model(full_model_checkpoint_path, custom_objects={})
elif os.path.exists(weights_checkpoint_path):
    print("Loading model weights...")
    build_transformation_model.load_weights(weights_checkpoint_path)
else:
    print("Using freshly defined model...")

# :::::::::::::::::::::::::::Training & Validation Functions::::::::::::::::::::::::::::::: #

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

class PrintPredictionsAndLoss_0(Callback):
    def __init__(self, validation_data, batch_size):
        super().__init__()
        self.validation_data = validation_data
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val = self.validation_data
        # Make predictions on the validation data
        predictions = self.model.predict(X_val, batch_size=self.batch_size)
        # Flatten predictions if they're in the shape of [[value]]
        predictions_flattened = np.squeeze(predictions)[:5]
        # Print the first 5 predictions and real values, ensuring predictions are flattened
        print("First 5 Validation Predictions:", predictions_flattened)
        print("First 5 Validation Real Values:", y_val[:5])

# :::::::::::::::::::::::::::Training & Validation Start::::::::::::::::::::::::::::::: #

# Initialize optimizer
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)

# Compile the model
model.compile(optimizer=optimizer, loss=loss_func)

# Evaluate model immediately after loading weights (for baseline performance)
#val_loss = model.evaluate(X_val, y_val)
#print(f"Baseline Mean Absolute Error on Validation Set after Loading: {val_loss}")

lr_scheduler = CustomLearningRateScheduler(initial_lr, min_lr, decay_step, lr_enable)

# Initialize the custom PrintLR callback
print_lr = PrintLR()

# callback
print_predictions_and_loss_0 = PrintPredictionsAndLoss_0((X_val_transformed, y_val), batch_size_2)
dashboard = RealTimeDashboard()

# Standard .fit() method

history = model.fit(
    X_train_transformed, y_train, 
    epochs=epochs_2, 
    batch_size=batch_size_2, 
    validation_data=(X_val_transformed, y_val), 
    callbacks=[weights_checkpoint, dashboard, lr_scheduler, print_lr, print_predictions_and_loss_0])#weights_checkpoint
    
dashboard.root.destroy()
print("Training complete.")

# :::::::::::::::::::::::::::Evaluation 2nd Model (Save Predictions)::::::::::::::::::::::::::::::: #

epochs = 0 
batch_size = 1
if epochs > 0:
    history = model.fit(
    X_train, y_train, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_val, y_val), 
    callbacks=[weights_checkpoint]
)
    # Close the Tkinter window
    dashboard.root.destroy()
    print("Training complete.") 

# Assuming 'model', 'X_val', 'y_val', and 'scale_target' are defined elsewhere
# Model evaluation for both MSE and MAE
if scale_target:
    y_pred_scaled = model.predict(X_val).flatten()
    # Unscaled predictions
    y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    # Unscaled actual values
    y_real = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
else:
    y_pred = model.predict(X_val).flatten()
    y_real = y_val.flatten()

# Calculate MSE and MAE with unscaled data
mse = mean_squared_error(y_real, y_pred)
mae = mean_absolute_error(y_real, y_pred)

print(f"Model Mean Squared Error on Validation Set: {mse}")
print(f"Model Mean Absolute Error on Validation Set: {mae}")

# Save the actual and predicted values to a DataFrame using unscaled data
result_df = pd.DataFrame({"Actual": y_real, "Predicted": y_pred})

# Save result DataFrame to a CSV file
file_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/XAUUSD_high_predictions_v1_testing.csv"
print("Saving file to:", file_path)
result_df.to_csv(file_path, index=False)
print("File saved successfully!")

# :::::::::::::::::::::::::::The End::::::::::::::::::::::::::::::: #


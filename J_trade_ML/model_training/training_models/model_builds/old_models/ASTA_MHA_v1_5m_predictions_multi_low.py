"""
   AA     SSSS  TTTTT    AAA      |  SSS   EEEEE   QQQ
  A  A    S       T     A   A     |  S     E      Q   Q
  AAAA    SSSS    T     AAAAA     |  SSS   EEEEE  Q   Q
 A    A      S    T    A     A    |    S   E      Q  QQ
A      A  SSSS    T   A       A   |  SSS   EEEEE   QQQ Q
_______________________________________________________ 
:Welcome to Custom ASTA (Adaptive Sphere Transformation) Sequencer LSTM Multi Target Prediction | (float32): (Version 1)

Version 1: Designed for sequenced, time series data, which can be used for predicting continuous values or classes for stock/currency dat. 

Model: Model accepts multiple Target columns. Features use MaxMin Scaler and Targets MaxMin Scaler. The model consist of Multi-Head Attention
with encoding and decoding, Followed by A Dense layer to project the feature dimention linearly into a higher dimention, then the remaining
layers are LSTMs, with a combination of skipping layers and residual connections, and the final LSTM layer only returns the last sequence. before
the final layer the original inputs are added to the last layer output, for final single unit sense layer for predictions.

Adaptive Sphere Transformation is used 3 seperate times as layers between layer transitions, which is exsplained below. 

Info: Adaptive Sphere Transformation

Imagine you have a dataset made of (None, seq_len, feature dimension). We start by finding the feature count, for each seq_len in the sequence, 
and we want to find the center cell value. so if there features = 10, the center feature is 5, and if the seq_len = 16, then we go to the center
seq_len of 8 in each seq_len of 16, which is the center cell/value of that sequence at column 5 row 8. It starts at the center cell and 
transforms the data outwards in a radial 3D spherical geometric shape, transforming every value to bend the feature and seq_len dimension values
into a 3d spherical object, where it takes the top and bottom of seq_len, and the far left and and right columns and connects them together to form
a numerical geometric shape. 

The more features or the longer the seq_len, the higher the resolution of the data's geometric curviture, as mathematically it becomes closer to a 
sphere. When it has less features and a smaller seq_len, it become more of a geometric shape with shaper edges. The difference between a 3D hexogon VS 
a sphere, which is more curved.

The transformations start at the center value for each prediction sequence, but the center starting point for the data transformation changes as a 
learned mechanism, where the model changes the center starting point with each batch as the model learns. As it learns, it moves in the x, y & z axis 
between columns and sequenced rows for each sequence. The ultimate goal here (not fully implemented), is to use the data shape landscape as the topographical 
landscape for gradient decent during the updating of the weights.

This learning mechanism could be imagined as the starting center at a point on a 3D surface, moveing over the surface to to optimize loss represented be the 
data rather than gradient landscape. Added an mechanism that moves the center during learning, but has the ability based on this changing learnable center, 
to stay at a certain center during training, when the model is performing well. Once a certain threshold is reached, the center changes. So as it learns it 
can decide which center is the best, lock on to that center, then learns additional data transformations regarding that center.  

By: JD
"""  
import os 

"""
C++ minimum log level to filter out warnings for mixed precision 16bit 
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import joblib
import sqlite3
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.initializers import HeNormal, RandomNormal
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.layers import MultiHeadAttention, Add, Conv2D, Concatenate
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, LayerNormalization, Flatten

"""                                                                           ++            ++++++
 \\\\\\\\\\\\\\\\\____________________________________________________________++++++++++++++++++++
|.. ........ ....... ...... ..... .... ... .. .   .   .     .        .        .  DYNA | MLP  .  |||||
|-----___   ___-----___   ____   ___-----__----    .       .     .        .           .         |||||
|TTTTT  H   H  IIIII  N   N  K   K  B B B  EEEE ..    .     .        .        .    Trading      |||||
|  T    H   H    I    NN  N  K  K   B   B  E       ... ...     .        .          .             ||
|/////  /////  /////  /////  ////   /////  ////     .....  ......        .        .  ||||-----|||-----||
|  T    H   H    I    N  NN  K  K   B   B  E       ........    ..........         .               |
|  T    H   H  IIIII  N   N  K   K  B B B  EEEE ............      .............                  ||
|\\\\\\\     \\\\\\\\\     \\\\\\\\\     ___________________      /////////////      AI/ML       ||
__---__-----___---___-----___---___-----__--__--__--__--__-------__--__--_-_-_---_-_-__--___-_--_|
"""

# :::: Main Parameters: scroll toline 683 <Model Start> to change unit dimentions ::::

seq_len = 8                     # Currently: 15 Minute data: with positional encoding has 80 features per seq_len     
seq_step = 1                    # Number of steps per dequence
epochs = 100                   # Training iterations
batch_size = 64                 # Samples per batch (last was 32)

# :::: Adaptive Sphere Transformation ::::

asta_threshold_1 = 0.1          # Determines when to consider the model's performance stable.   
asta_threshold_2 = 0.1          # Defines the performance metric value above which the centers are locked.             
asta_preformance = 0.01          # Controls how quickly the model perceives stability. (0.01 default)

# :::: learning warmup & Adam Optimizer ::::

custom_lr = False               # Activate warmup_steps
warmup_steps = int(1000)         # Number of batches for lr warmup_steps  
lr_scale = 1                    # 1 is default for no change (increases starting lr 4 warmup_steps)

fixed_learning_rate = 0.00005 # Used if custom_lr=False
clipnorm = 1.0
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7
amsgrad = False

# :::: Training Target Predictions ::::
""" 
target_columns should be the name of any Target columns and output_names
should be the the targets you want to use, which requires adjusting drop_targets
to match the output_names you selected: ['High', 'Low', 'Close']. Curretly only
using 1 Target of 3 possible in dataset 
"""
#drop_targets = -2            # exclude the last target columns
target_columns = ['Target1', 'Target2', 'Target3']
output_names =   ['Low']       

# :::: Training Target Predictions ::::

Scaler_features = MinMaxScaler()
Scaler_targets = MinMaxScaler()   

# :::: Training Loss Function ::::

loss_func = 'mean_squared_error'
""" 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'
'mean_squared_logarithmic_error', 'huber_loss', 'log_cosh', 'cosine_similarity' """

# :::::::::::::::::::::::::::::TensorFlow setup:::::::::::::::::::::::::::::: #

# Suppresses TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

#tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Set environment variables for NCCL debugging
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["NCCL_DEBUG_SUBSYS"] = "ALL"
os.environ["NCCL_LAUNCH_MODE"] = "PARALLEL"

# Set device configuration early in the script
def configure_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                tf.config.experimental.set_virtual_device_configuration(
                    gpu,
                    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=80000)])  # Example memory limit
        except RuntimeError as e:
            print(e)

""" # Set up the first group of GPUs (0, 1)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
configure_gpus() """

# Use MirroredStrategy for multi-GPU training
strategy4 = tf.distribute.MirroredStrategy(["GPU:1"])
print('Number of devices: {}'.format(strategy4.num_replicas_in_sync))
print("Eager execution:", tf.executing_eagerly()) 
print()

# ::::::::::::::::::::::Data Processing Transformation::::::::::::::::::::::: #

def load_data_from_sqlite(db_path):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    print("Executing query to get table names...")
    tables = pd.read_sql_query(query, conn)
    all_data = []
    for table_name in tables['name']:
        print(f"Loading data from table: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        df['source_table'] = table_name  # Add source column
        all_data.append(df)
    conn.close()
    print("Data loading complete.")
    return all_data

def create_overlapping_sequences(data, seq_len, step):
    print(f"Creating overlapping sequences with seq_len={seq_len} and step={step}")
    return np.array([data[i:i + seq_len] for i in range(0, len(data) - seq_len + 1, step)])

def preprocess_and_combine_data(all_data, seq_len, target_columns, columns_to_encode, encoder=None):
    print("Preprocessing and combining data...")
    all_dfs = []
    all_targets = []
    feature_scalers = {}  # Dictionary to store feature scalers for each table
    target_scalers = {}

    combined_df = pd.concat(all_data, ignore_index=True)

    # Encoding specified columns
    columns_indices = sorted(columns_to_encode.keys())
    if encoder is None:
        encoder = OrdinalEncoder(categories=[columns_to_encode[i] for i in columns_indices])
        encoded_features = encoder.fit_transform(combined_df.iloc[:, columns_indices])
    else:
        encoded_features = encoder.transform(combined_df.iloc[:, columns_indices])

    encoded_df = pd.DataFrame(encoded_features, columns=[f'encoded_{i}' for i in columns_indices])
    combined_df.drop(columns=[combined_df.columns[i] for i in columns_indices], inplace=True)
    combined_df = pd.concat([combined_df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    for idx, df in enumerate(all_data):
        table_name = df['source_table'].iloc[0]
        print(f"Processing data from table: {table_name}")
        if not set(target_columns).issubset(df.columns):
            raise KeyError(f"DataFrame does not contain the required target columns: {target_columns}")

        # Determine which target columns to use
        #used_targets = target_columns[:drop_targets]
        used_targets = [target_columns[1]]
        df_targets = df[used_targets].astype('float32')
        df.drop(columns=target_columns, inplace=True)
        df.replace([np.inf, -np.inf], np.nan, inplace=True)

        # Scale the target columns separately
        if table_name not in target_scalers:
            target_scalers[table_name] = {name: MinMaxScaler() for name in output_names}
        df_targets_scaled = np.column_stack([
            target_scalers[table_name][name].fit_transform(df_targets.iloc[:, i].values.reshape(-1, 1))
            for i, name in enumerate(output_names)
        ])
        # Scale columns 4 to 27 separately for each source table
        if table_name not in feature_scalers:
            feature_scalers[table_name] = MinMaxScaler()
        df.iloc[:, 4:28] = feature_scalers[table_name].fit_transform(df.iloc[:, 4:28])

        df['table_id'] = idx  # Add unique numeric feature based on table order
        all_dfs.append(df)
        all_targets.append(pd.DataFrame(df_targets_scaled, columns=used_targets))

    merged_df = pd.concat(all_dfs, ignore_index=True)
    merged_targets = pd.concat(all_targets, ignore_index=True)

    # Dropping non-numeric columns except source_table
    merged_df['source_table'] = merged_df['source_table'].astype(str)
    non_numeric_columns = merged_df.select_dtypes(exclude=[np.number]).columns.tolist()
    non_numeric_columns.remove('source_table')  # Do not drop source_table
    print(f"Dropping non-numeric columns: {non_numeric_columns}")
    merged_df.drop(columns=non_numeric_columns, inplace=True)

    merged_df.fillna(merged_df.mean(numeric_only=True), inplace=True)  # Fill NaN values in a single operation

    # Separate columns 4 to 27 before scaling combined data
    scaled_columns_4_to_27 = merged_df.iloc[:, 4:28]
    features = merged_df.drop(columns=merged_df.columns[4:28].tolist() + ['source_table'])
    source_table_column = merged_df['source_table']

    # Creating overlapping sequences for features, source table ids, and targets
    X_features = create_overlapping_sequences(features.values, seq_len, seq_step)
    X_columns_4_to_27 = create_overlapping_sequences(scaled_columns_4_to_27.values, seq_len, seq_step)
    source_tables = create_overlapping_sequences(source_table_column.values, seq_len, seq_step)
    target_dict = {output_names[idx]: create_overlapping_sequences(merged_targets.values[:, idx], seq_len, seq_step)[:, -1] for idx in range(len(output_names))}

    print("Preprocessing complete.")
    return X_features, X_columns_4_to_27, target_dict, feature_scalers, target_scalers, source_tables, encoder

def scale_combined_data(X, X_columns_4_to_27, y_dict, feature_scaler=None):
    print("Scaling combined data...")
    cols_to_scale = X.shape[-1]

    # Here we scale the combined data excluding columns 4 to 27
    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
        X = X.reshape(-1, cols_to_scale)
        scaled_features = feature_scaler.fit_transform(X).reshape(-1, seq_len, cols_to_scale)
    else:
        X = X.reshape(-1, cols_to_scale)
        scaled_features = feature_scaler.transform(X).reshape(-1, seq_len, cols_to_scale)

    # Combine back with columns 4 to 27
    scaled_features_combined = np.concatenate((scaled_features, X_columns_4_to_27), axis=-1)

    print("Scaling complete.")
    return scaled_features_combined, y_dict, feature_scaler

def randomize_data(X, y, source_tables):
    print("Randomizing data...")
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = {key: value[perm] for key, value in y.items()}
    source_tables_shuffled = source_tables[perm]
    print("Randomization complete.")
    return X_shuffled, y_shuffled, source_tables_shuffled

# Load and preprocess data
print("About to load and preprocess training data...")
all_train_data = load_data_from_sqlite("/home/jd/Desktop/trading/data/multi_5m_OHLC_Train_5_5_new.db")
all_val_data = load_data_from_sqlite("/home/jd/Desktop/trading/data/multi_5m_OHLC_Val_5_5_new.db")

columns_to_encode = {
    0: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55],
    1: list(range(1, 32)),
    2: list(range(1, 13)),
    3: list(range(0, 24)),
    **{i: [0.0, 1.0] for i in range(28, 40)},
    40: list(range(2, 24972))
}

# Preprocess and combine data
print("Preprocessing and combining training data...")
X_train_features, X_train_columns_4_to_27, y_train_dict, feature_scalers, target_scalers, source_tables_train, encoder = preprocess_and_combine_data(all_train_data, seq_len, target_columns, columns_to_encode)
print("Scaling training data...")
X_train, y_train_dict, feature_scaler = scale_combined_data(X_train_features, X_train_columns_4_to_27, y_train_dict)
print("Randomizing training data...")
X_train, y_train_dict, source_tables_train = randomize_data(X_train, y_train_dict, source_tables_train)

# Save scalers and encoder
print("Saving scalers and encoder...")
joblib.dump({
    'feature_scalers': feature_scalers,
    'target_scalers': target_scalers,
    'encoder': encoder,
    'feature_scaler': feature_scaler,
    'columns_to_encode': columns_to_encode,
    'seq_len': seq_len,
    'seq_step': seq_step
}, 'scalers_5m_multi_low.pkl')

# Load scalers and encoder for validation
print("Loading scalers and encoder for validation...")
scalers_encoder = joblib.load('scalers_5m_multi_low.pkl')
feature_scalers = scalers_encoder['feature_scalers']
target_scalers = scalers_encoder['target_scalers']
encoder = scalers_encoder['encoder']
feature_scaler = scalers_encoder['feature_scaler']
columns_to_encode = scalers_encoder['columns_to_encode']
seq_len = scalers_encoder['seq_len']
seq_step = scalers_encoder['seq_step']

# Preprocess and combine validation data
print("Preprocessing and combining validation data...")
X_val_features, X_val_columns_4_to_27, y_val_dict, feature_scalers, target_scalers, source_tables_val, _ = preprocess_and_combine_data(all_val_data, seq_len, target_columns, columns_to_encode, encoder)
print("Scaling validation data...")
X_val, y_val_dict, _ = scale_combined_data(X_val_features, X_val_columns_4_to_27, y_val_dict, feature_scaler)

print(f"Training data shapes - X: {X_train.shape}, y: {[y.shape for y in y_train_dict.values()]}, source tables: {source_tables_train.shape}")
print(f"Validation data shapes - X: {X_val.shape}, y: {[y.shape for y in y_val_dict.values()]}, source tables: {source_tables_val.shape}")

# :::::::::::::::::::::::::::Real-time Dashboard::::::::::::::::::::::::::::::: #

class RealTimeDashboard(Callback):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('5m Multi Predictions Low')
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
        plt.savefig("/home/jd/Desktop/trading/model_testing_text/training_models/training_progress/multi_high_train_loss_v1_5m_low.png") 
        self.root.update()

print("About to initialize the model...")

with strategy4.scope():
    
    # :::::::::::::::::::::::::::Custom Positional Encoding Layer::::::::::::::::::::::::::::::: #
    
    class PositionalEncoding(Layer):
        def __init__(self, seq_len, d_model, **kwargs):
            super(PositionalEncoding, self).__init__(**kwargs)
            self.seq_len = seq_len
            self.d_model = d_model
            self.pos_encoding = self.positional_encoding(seq_len, d_model)

        def get_angles(self, pos, i, d_model):
            # Compute the angles for the positional encoding
            angles = 1 / tf.pow(10000, (2 * (i // 2)) / tf.cast(d_model, tf.float32))
            return pos * angles

        def positional_encoding(self, seq_len, d_model):
            # Calculate the angle rates for the position
            angle_rads = self.get_angles(
                pos=tf.range(seq_len, dtype=tf.float32)[:, tf.newaxis],
                i=tf.range(d_model, dtype=tf.float32)[tf.newaxis, :],
                d_model=d_model
            )

            # Apply sin to even indices and cos to odd indices
            sines = tf.math.sin(angle_rads[:, 0::2])
            cosines = tf.math.cos(angle_rads[:, 1::2])

            # Concatenate sines and cosines along the last axis
            pos_encoding = tf.concat([sines, cosines], axis=-1)
            pos_encoding = pos_encoding[tf.newaxis, ...]  # Add batch dimension
            return tf.cast(pos_encoding, tf.float32)

        def call(self, inputs):
            # Ensure that the positional encoding matches the input length
            seq_len = tf.shape(inputs)[1]
            pos_encoding = self.pos_encoding[:, :seq_len, :]
            return inputs + tf.cast(pos_encoding, inputs.dtype)
        
        def get_config(self):
            config = super(PositionalEncoding, self).get_config()
            config.update({
                'seq_len': self.seq_len,
                'd_model': self.d_model
            })
            return config

        @classmethod
        def from_config(cls, config):
            return cls(
                seq_len=config['seq_len'],
                d_model=config['d_model']
            )
        
    # :::::::::::::::::::::::::::Custom Positional Encoding Layer::::::::::::::::::::::::::::::: #
        
    class ConvolutionalPositionalEncoding(Layer):
        def __init__(self, **kwargs):
            super(ConvolutionalPositionalEncoding, self).__init__(**kwargs)
            self.positional_encoding_conv = None

        def build(self, input_shape):
            # Create a learnable positional encoding with the same shape as the input (excluding batch size)
            self.positional_encoding_conv = self.add_weight(
                name='positional_encoding_conv',
                shape=(1, input_shape[1], input_shape[2], input_shape[3]),
                initializer=RandomNormal(),
                trainable=True
            )
            super(ConvolutionalPositionalEncoding, self).build(input_shape)

        def call(self, inputs):
            # Add the learnable positional encoding to the inputs
            return inputs + self.positional_encoding_conv

        def compute_output_shape(self, input_shape):
            return input_shape

        def get_config(self):
            config = super(ConvolutionalPositionalEncoding, self).get_config()
            return config

        @classmethod
        def from_config(cls, config):
            return cls()
        
    # :::::::::::::::::::::::Custom Layer (undefined)::::::::::::::::::::::::::: #

    class EncoderLayer(Layer):
        def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
            super(EncoderLayer, self).__init__(**kwargs)
            self.num_heads = num_heads
            self.key_dim = d_model // num_heads
            self.mha = MultiHeadAttention(num_heads=num_heads, key_dim=self.key_dim)
            self.dropout = Dropout(dropout_rate)
            self.norm1 = LayerNormalization(epsilon=1e-9)
            self.ffn = Sequential([
                Dense(1024, activation='relu', use_bias=True, kernel_initializer=HeNormal()),
                Dense(d_model, use_bias=True, kernel_initializer=HeNormal())
            ])
            self.norm2 = LayerNormalization(epsilon=1e-9)

        def call(self, x, training=False):
            attn_output = self.mha(x, x, x)  # Attention
            attn_output = self.dropout(attn_output, training=training)
            out1 = self.norm1(x + attn_output)  # Residual connection
            
            ffn_output = self.ffn(out1)
            ffn_output = self.dropout(ffn_output, training=training)
            return self.norm2(out1 + ffn_output)  # Residual connection
        
        def get_config(self):
            config = super(EncoderLayer, self).get_config()
            config.update({
                "d_model": self.key_dim * self.num_heads,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout.rate
            })
            return config

    class DecoderLayer(Layer):
        def __init__(self, d_model, num_heads, dropout_rate, **kwargs):
            super(DecoderLayer, self).__init__(**kwargs)
            self.num_heads = num_heads
            self.key_dim = d_model // num_heads
            self.mha1 = MultiHeadAttention(num_heads=num_heads, key_dim=self.key_dim)
            self.dropout1 = Dropout(dropout_rate)
            self.norm1 = LayerNormalization(epsilon=1e-9)
            self.mha2 = MultiHeadAttention(num_heads=num_heads, key_dim=self.key_dim)
            self.dropout2 = Dropout(dropout_rate)
            self.norm2 = LayerNormalization(epsilon=1e-9)
            self.ffn = Sequential([
                Dense(1024, activation='relu', use_bias=True, kernel_initializer=HeNormal()),
                Dense(d_model, use_bias=True, kernel_initializer=HeNormal())
            ])
            self.norm3 = LayerNormalization(epsilon=1e-9)

        def call(self, x, enc_output, training=False):
            attn_output1 = self.mha1(x, x, x)  # Self-attention
            attn_output1 = self.dropout1(attn_output1, training=training)
            out1 = self.norm1(x + attn_output1)  # Residual connection
            
            attn_output2 = self.mha2(out1, enc_output, enc_output)  # Cross-attention
            attn_output2 = self.dropout2(attn_output2, training=training)
            out2 = self.norm2(out1 + attn_output2)  # Residual connection
            
            ffn_output = self.ffn(out2)
            ffn_output = self.dropout2(ffn_output, training=training)
            return self.norm3(out2 + ffn_output)  # Residual connection
        
        def get_config(self):
            config = super(DecoderLayer, self).get_config()
            config.update({
                "d_model": self.key_dim * self.num_heads,
                "num_heads": self.num_heads,
                "dropout_rate": self.dropout1.rate
            })
            return config

    class CustomMultiHeadAttentionModel(Model):
        def __init__(self, seq_len, d_model, num_heads, num_enc_layers, num_dec_layers, dropout_rate, **kwargs):
            super(CustomMultiHeadAttentionModel, self).__init__(**kwargs)
            self.seq_len = seq_len
            self.d_model = d_model
            self.num_heads = num_heads
            self.num_enc_layers = num_enc_layers
            self.num_dec_layers = num_dec_layers
            self.dropout_rate = dropout_rate
            
            self.enc_layers = [EncoderLayer(d_model, num_heads, dropout_rate) for _ in range(num_enc_layers)]
            self.dec_layers = [DecoderLayer(d_model, num_heads, dropout_rate) for _ in range(num_dec_layers)]

        def call(self, inputs, training=False):
            enc_output = inputs
            for enc_layer in self.enc_layers:
                enc_output = enc_layer(enc_output, training=training)
            
            dec_output = enc_output
            for dec_layer in self.dec_layers:
                dec_output = dec_layer(dec_output, enc_output, training=training)
            
            return dec_output

        def get_config(self):
            return {
                'seq_len': self.seq_len,
                'd_model': self.d_model,
                'num_heads': self.num_heads,
                'num_enc_layers': self.num_enc_layers,
                'num_dec_layers': self.num_dec_layers,
                'dropout_rate': self.dropout_rate,
            }

        @classmethod
        def from_config(cls, config):
            return cls(
                seq_len=config['seq_len'],
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                num_enc_layers=config['num_enc_layers'],
                num_dec_layers=config['num_dec_layers'],
                dropout_rate=config['dropout_rate']
            )
        
    # ::::::::::::::::::::::::Main Model Parameters::::::::::::::::::::::::::::: #

    # ::Data Model Transformation::
    seq_len = X_train.shape[1]               
    d_model = X_train.shape[2]            # Input shape in units
    inputs = Input(shape=(seq_len, d_model), name='main_input_features')

    num_heads_1 = 8                       # Heads for Attention
    num_heads_2 = 2                       # Heads for convolutions
    num_enc_layers = 8                    # Encoding Attention layers
    num_dec_layers = 8                   # Decoding Attention layers
    dropout_rate = 0.00                   # Dropout rate

    conv_layers = 2                       # 1st convolutional layers
    initial_filters = int(256)            # 1nd unit hidden layer exspansion (MHA)
    d_ffn = int(initial_filters)       # 1nd unit exspansion/supression (LSTM)

    # ::::::::::::::Main Model Training (Multi Target Prediction):::::::::::::: #

    outputs = []
    for name in output_names: 
        q_ffn = initial_filters
        d_ffn= d_ffn*2
        pos = PositionalEncoding(seq_len, d_model)(inputs)    
        mha_model = CustomMultiHeadAttentionModel(seq_len, d_model, num_heads_1, num_enc_layers, 
                                                num_dec_layers, dropout_rate)
        mha_model = mha_model(pos)   
        mha_model = Add()([mha_model, inputs])  
        mha_model = tf.expand_dims(mha_model, axis=-1)

        conv_positional_encoding_layer = ConvolutionalPositionalEncoding()
        mha_model = conv_positional_encoding_layer(mha_model)

        for i in range(conv_layers):
            mha_model = Conv2D(filters=q_ffn, kernel_size=(8, 1), activation='relu',
                            kernel_initializer=HeNormal(), padding='same', use_bias=True)(mha_model)
            q_ffn *= 2

        height = mha_model.shape[1]
        width = mha_model.shape[2]
        channels = mha_model.shape[3]

        mha_model = tf.reshape(mha_model, (-1, height, width * channels))
        #mha_model = LayerNormalization(epsilon=1e-9)(mha_model)
        mha_model = Dense(d_ffn, activation='relu', kernel_initializer=HeNormal(), use_bias=True)(mha_model)
        mha_model = Flatten()(mha_model)

    # ::::::::::::::----------------------------------------------:::::::::::::: #

        output = Dense(1, activation='linear', kernel_initializer=HeNormal(), use_bias=True, name=name, dtype='float32')(mha_model)
        outputs.append(output)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    print("Model initialized!")
    print("Total number of parameters in the model:", model.count_params())

    # :::::::::::::::::::::Model Checkpoint For Weights::::::::::::::::::::::: #

    # Define the model checkpoint for weights only
    weights_checkpoint_path = "/home/jd/Desktop/trading/model_testing_text/training_models/saved_weights/multi_high_weights_v1_5m_low.h5"
    weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', 
                                        verbose=1, save_best_only=True, mode='min', save_weights_only=True)

    # Define the model checkpoint for the full model
    full_model_checkpoint_path = "/home/jd/Desktop/trading/model_testing_text/training_models/saved_weights/multi_high_model_v1_FULL_5m_low.h5"
    full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', 
                                            verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

    custom_objects={
        'CustomMultiHeadAttentionModel': CustomMultiHeadAttentionModel,
        'EncoderLayer': EncoderLayer,
        'DecoderLayer': DecoderLayer,
        'PositionalEncoding': PositionalEncoding,
        'ConvolutionalPositionalEncoding': ConvolutionalPositionalEncoding,
    }

    # Try loading the full model, then the weights, or use the freshly defined model if neither exists
    if os.path.exists(weights_checkpoint_path):
        print("Loading model weights...")
        model.load_weights(weights_checkpoint_path)
    elif os.path.exists(full_model_checkpoint_path):
        print("Loading full model...")
        model = load_model(full_model_checkpoint_path, custom_objects=custom_objects)
    else:
        print("Using freshly defined model...")

    # ::::::::::::::::::::::::Training Model Functions:::::::::::::::::::::::::::: #

    class CustomSchedule(LearningRateSchedule):
        def __init__(self, units, warmup_steps, custom_lr, lr_scale):
            super(CustomSchedule, self).__init__()
            self.lr_scale = lr_scale
            self.units = tf.cast(units, tf.float32)
            self.warmup_steps = warmup_steps
            self.custom_lr = custom_lr

        def __call__(self, step):
            if not self.custom_lr:
                return fixed_learning_rate
            
            step = tf.cast(step, tf.float32)
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.units) * tf.math.minimum(arg1, arg2) * self.lr_scale
        
        def get_config(self):
            return {
                'units': self.units.numpy(), 
                'warmup_steps': self.warmup_steps, 
                'custom_lr': self.custom_lr, 
                'lr_scale': self.lr_scale
            }

    class PrintLR(Callback):
        def __init__(self, print_enable=True):
            super().__init__()
            self.print_enable = print_enable

        def on_epoch_end(self, epoch, logs=None):
            if self.print_enable:
                step = self.model.optimizer.iterations
                current_lr = self.model.optimizer.learning_rate(step)
                if isinstance(current_lr, tf.Tensor):
                    current_lr = current_lr.numpy()
                print(f"Epoch {epoch + 1}, Current Learning Rate: {current_lr:.10f}")

    class PrintPredictionsAndLoss_0(Callback):
        def __init__(self, target_scalers, validation_data, source_tables, batch_size):
            super().__init__()
            self.target_scalers = target_scalers
            self.validation_data = validation_data
            self.source_tables = source_tables
            self.batch_size = batch_size

        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val_dict = self.validation_data
            predictions = self.model.predict(X_val, batch_size=self.batch_size)

            if not isinstance(predictions, list):
                predictions = [predictions]

            num_samples = X_val.shape[0]
            random_indices = np.random.choice(num_samples, 5, replace=False)

            for i, (key, y_val) in enumerate(y_val_dict.items()):
                pred = predictions[i]
                if pred.ndim > 2:
                    pred = pred[:, -1, :]  # Use only the last timestep
                elif pred.ndim > 1:
                    pred = np.mean(pred, axis=1)

                pred_sampled = pred[random_indices]
                y_val_sampled = y_val[random_indices]

                source_tables_sampled = self.source_tables[random_indices, -1].astype(str)  # Use last column and ensure it's string

                pred_unscaled = []
                real_unscaled = []

                for idx, source_table in enumerate(source_tables_sampled):
                    if source_table in self.target_scalers and key in self.target_scalers[source_table]:
                        scaler = self.target_scalers[source_table][key]
                        pred_val = pred_sampled[idx].reshape(1, -1)
                        real_val = y_val_sampled[idx].reshape(1, -1)

                        pred_unscaled.append(scaler.inverse_transform(pred_val).flatten()[0])
                        real_unscaled.append(scaler.inverse_transform(real_val).flatten()[0])
                    else:
                        print(f"Warning: Scaler for table '{source_table}' and output '{key}' not found.")

                print(f"Epoch {epoch + 1} - First 5 Validation Predictions for {key}: {pred_unscaled}")
                print(f"Epoch {epoch + 1} - First 5 Validation Real Values for {key}: {real_unscaled}")
                print(f"Corresponding source tables: {source_tables_sampled}")
                print()

    # Example code to validate shapes and intermediate values
    print(f"Validation data shapes - X_val: {X_val.shape}, y_val_dict: {[y.shape for y in y_val_dict.values()]}")
    print(f"Source tables shape: {source_tables_val.shape}")

    class PrintUnscaledLoss(Callback):
        def __init__(self, target_scalers, batch_size, validation_data, source_tables):
            super().__init__()
            self.target_scalers = target_scalers
            self.batch_size = batch_size
            self.validation_data = validation_data
            self.source_tables = source_tables

        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val_dict = self.validation_data
            predictions = self.model.predict(X_val, batch_size=self.batch_size)

            if not isinstance(predictions, list):
                predictions = [predictions]

            combined_unscaled = defaultdict(lambda: defaultdict(lambda: {"pred": [], "real": []}))
            all_source_tables = set(self.source_tables.flatten())

            # Flatten the source tables
            flat_source_tables = self.source_tables[:, -1]  # Use only the last column
            #print(f"Flat Source Tables (first 10): {flat_source_tables[:10]} ...")

            for i, (output_name, y_val) in enumerate(y_val_dict.items()):
                pred = predictions[i]

                if pred.ndim == 3:
                    pred = pred[:, -1, :]  # Use only the last timestep
                if y_val.ndim == 1:
                    y_val = y_val[:, np.newaxis]

                for idx in range(len(pred)):
                    source_table = flat_source_tables[idx]
                    if source_table in self.target_scalers and output_name in self.target_scalers[source_table]:
                        scaler = self.target_scalers[source_table][output_name]
                        pred_val = pred[idx].reshape(1, -1)
                        real_val = y_val[idx].reshape(1, -1)

                        try:
                            unscaled_pred = scaler.inverse_transform(pred_val)
                            unscaled_real = scaler.inverse_transform(real_val)
                            combined_unscaled[source_table][output_name]["pred"].append(unscaled_pred[0, 0])
                            combined_unscaled[source_table][output_name]["real"].append(unscaled_real[0, 0])
                        except Exception as e:
                            print(f"Error: {e} - Skipping index {idx} for table '{source_table}' and output '{output_name}'.")
                    else:
                        print(f"Warning: Scaler for table '{source_table}' and output '{output_name}' not found. Skipping this entry.")

            # Check for any missing source tables
            missing_source_tables = all_source_tables - set(combined_unscaled.keys())
            for source_table in missing_source_tables:
                print(f"Warning: Source table '{source_table}' found in source_tables but not processed.")

            for source_table, output_data in combined_unscaled.items():
                for output_name, data in output_data.items():
                    if data["pred"] and data["real"]:
                        pred_unscaled = np.array(data["pred"])
                        y_val_unscaled = np.array(data["real"])

                        mae_val_unscaled = mean_absolute_error(y_val_unscaled, pred_unscaled)
                        print(f'Epoch {epoch + 1} - Validation {output_name} Unscaled MAE for {source_table}: {mae_val_unscaled}')

            """ 
            # Debugging: Print all source tables and their counts
            print("Processed source tables and their counts (first 10):")
            for source_table, output_data in list(combined_unscaled.items())[:10]:
                for output_name, data in output_data.items():
                    print(f"{source_table} - {output_name}: {len(data['pred'])} predictions, {len(data['real'])} actuals") 
            """

            #print(f"Target scalers keys (first 10): {list(self.target_scalers.keys())[:10]} ...")

    class RandomizeDataCallback(Callback):
        def __init__(self, data_generator):
            super().__init__()
            self.data_generator = data_generator

        def on_epoch_begin(self, epoch, logs=None):
            self.data_generator.shuffle_data()
            print("Randomizing Next Epoch Data Complete.")

    class CustomDataGenerator(Sequence):
        def __init__(self, X, y, source_tables, batch_size):
            self.X = X
            self.y = y
            self.source_tables = source_tables
            self.batch_size = batch_size
            self.indices = np.arange(len(self.X))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))

        def __getitem__(self, index):
            batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            X_batch = self.X[batch_indices]
            y_batch = {key: value[batch_indices] for key, value in self.y.items()}
            source_tables_batch = self.source_tables[batch_indices]
            return X_batch, y_batch

        def on_epoch_end(self):
            self.shuffle_data()

        def shuffle_data(self):
            np.random.shuffle(self.indices)
            
    def loss_function(y_true, y_pred):
        # Ensuring y_true and y_pred are at least 2D
        if len(tf.shape(y_true)) == 1:
            y_true = tf.expand_dims(y_true, -1)
        if len(tf.shape(y_pred)) == 1:
            y_pred = tf.expand_dims(y_pred, -1)

        if len(tf.shape(y_pred)) == 3: 
            y_pred = tf.reduce_mean(y_pred, axis=1)
            y_true = tf.reduce_mean(y_true, axis=1)

        loss_per_timestep = tf.keras.losses.mean_absolute_error(y_true, y_pred)
        
        return tf.reduce_mean(loss_per_timestep)

    # ::::::::::::::::::::::::Training & Validation Loop:::::::::::::::::::::::::::: #

    # Initialize optimizer
    learning_rate = CustomSchedule(d_ffn, warmup_steps, custom_lr, lr_scale)
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)

    training_data = (X_train, y_train_dict)
    validation_data = (X_val, y_val_dict)

    # callbacks
    unscaled_loss_callback = PrintUnscaledLoss(
        target_scalers=target_scalers, 
        batch_size=batch_size,  
        validation_data=validation_data,
        source_tables=source_tables_val
    )
    print_predictions_and_loss_0 = PrintPredictionsAndLoss_0(
        target_scalers=target_scalers,
        validation_data=validation_data,
        source_tables=source_tables_val, 
        batch_size=batch_size
    )

    train_data_generator = CustomDataGenerator(X_train, y_train_dict, source_tables_train, batch_size)
    randomize_data_callback = RandomizeDataCallback(train_data_generator)

    dashboard = RealTimeDashboard()
    print_lr = PrintLR()

    other_callbacks = [dashboard, print_lr, unscaled_loss_callback, print_predictions_and_loss_0]
    main_callbacks = [weights_checkpoint, full_model_checkpoint] + other_callbacks + [randomize_data_callback]

    # loss dictionary using specific output names
    loss_dict = {name: loss_func for name in output_names} # loss_func or loss_function
    model.compile(optimizer=optimizer, loss=loss_dict) # metrics=['mae']

    # Evaluate model immediately after loading weights (for baseline performance)
    val_loss = model.evaluate(X_val, y_val_dict)
    print(f"Baseline Mean Absolute Error on Validation Set after Loading: {val_loss}")

    # Training the model
    history = model.fit(
        train_data_generator, 
        epochs=epochs, 
        batch_size=batch_size, 
        validation_data=(X_val, y_val_dict), 
        callbacks=main_callbacks
    )     
    print("Training complete.")

# ::::::::::::::::::Evaluation 2nd Model (Save Predictions):::::::::::::::::::::: #

print("Starting Evaluation with predictions(.csv).")

# Predict model outputs
y_pred_list = model.predict(X_val)

# Ensure y_pred_list is a list
if not isinstance(y_pred_list, list):
    y_pred_list = [y_pred_list]

# Dictionary to store predictions and actual values by source table
source_table_data = {str(source_table).strip(): {name: {"pred_unscaled": [], "real_unscaled": []} for name in output_names} for source_table in np.unique(source_tables_val)}

# Process predictions and actual values
for name, pred in zip(output_names, y_pred_list):
    valid_indices = range(len(source_tables_val))
    
    for i in valid_indices:
        source_table = source_tables_val[i, -1] if source_tables_val.ndim > 1 else source_tables_val[i]
        table_name = str(source_table).strip()
        
        if table_name in target_scalers and name in target_scalers[table_name]:
            scaler = target_scalers[table_name][name]
            try:
                pred_val = pred[i].reshape(1, -1)
                real_val = y_val_dict[name][i].reshape(1, -1)

                source_table_data[table_name][name]["pred_unscaled"].append(scaler.inverse_transform(pred_val).flatten())
                source_table_data[table_name][name]["real_unscaled"].append(scaler.inverse_transform(real_val).flatten())
            except IndexError as e:
                print(f"IndexError: {e} - Skipping index {i} for table '{table_name}' and output '{name}'.")
                continue
        else:
            print(f"Warning: Scaler for table '{table_name}' and output '{name}' not found. Skipping this entry.")

# Save predictions and actuals to separate tabs in the Excel file
with pd.ExcelWriter( "/home/jd/Desktop/trading/model_testing_text/training_models/training_progress/predictions_5m_multi_low.xlsx") as writer:
    for table_name, output_data in source_table_data.items():
        for name, data in output_data.items():
            if data["pred_unscaled"] and data["real_unscaled"]:
                pred_unscaled = np.concatenate(data["pred_unscaled"])
                real_unscaled = np.concatenate(data["real_unscaled"])

                # Ensure that the length of actual and predicted arrays match
                min_length = min(len(real_unscaled), len(pred_unscaled))
                real_unscaled = real_unscaled[:min_length]
                pred_unscaled = pred_unscaled[:min_length]

                # Create a DataFrame from the predictions and actuals
                result_df = pd.DataFrame({
                    'Actual': real_unscaled,
                    'Predicted': pred_unscaled
                })

                # Save the DataFrame to a separate sheet
                sheet_name = f"{table_name}_{name}"[:31]  # Excel sheet names are limited to 31 characters
                result_df.to_excel(writer, sheet_name=sheet_name, index=False)

print("Predictions saved to multiple tabs in the Excel file.")
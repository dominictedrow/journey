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

import joblib
import sqlite3
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
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

seq_len = 16                     # Currently: 15 Minute data: with positional encoding has 80 features per seq_len     
seq_step = 1                    # Number of steps per dequence
epochs = 1000                   # Training iterations
batch_size = 512                # Samples per batch (last was 32)

# :::: Adaptive Sphere Transformation ::::

asta_threshold_1 = 0.1          # Determines when to consider the model's performance stable.   
asta_threshold_2 = 0.1          # Defines the performance metric value above which the centers are locked.             
asta_preformance = 0.01         # Controls how quickly the model perceives stability. (0.01 default)

# :::: learning warmup & Adam Optimizer ::::

custom_lr = False               # Activate warmup_steps
warmup_steps = int(2000)         # Number of batches for lr warmup_steps  
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
#drop_targets = -2               # exclude the last target columns
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

# Set device configuration early in the script
def configure_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

# Set up the first group of GPUs (0, 1, 2, 3)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
configure_gpus()

# Use MirroredStrategy for multi-GPU training
strategy6 = tf.distribute.MirroredStrategy()


print('Number of devices: {}'.format(strategy6.num_replicas_in_sync))
print("Eager execution:", tf.executing_eagerly()) 
print()

# ::::::::::::::::::::::Data Processing Transformation::::::::::::::::::::::: #

def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_overlapping_sequences(data, seq_len, step=(seq_step)):
    sequences = []
    for i in range(0, len(data) - seq_len + 1, step):  # step > 1 to reduce overlap
        seq = data[i:i + seq_len]
        sequences.append(seq)
    return np.array(sequences)

def preprocess_data(df, seq_len, target_columns, output_names, feature_scaler=None, target_scaler=None, encoder=None):
    if not set(target_columns).issubset(df.columns):
        raise KeyError(f"DataFrame does not contain the required target columns: {target_columns}")

    print("Initial target columns:", target_columns)
    #used_targets = target_columns[:drop_targets]
    used_targets = [target_columns[1]]
    df_targets = df[used_targets].astype('float32')
    df.drop(columns=target_columns, inplace=True)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    print("Using targets:", used_targets, "\n")

    # Columns to encode and their value ranges
    columns_to_encode = {
        0: [0, 15, 30, 45],
        1: list(range(1, 32)),
        2: list(range(1, 13)),
        3: list(range(0, 24)),
        **{i: [0, 1] for i in range(80, 128)},
        128: list(range(0, 5000))
    }
    # Ordinal encode specified columns
    columns_indices = sorted(columns_to_encode.keys())

    if encoder is None:
        encoder = OrdinalEncoder(categories=[columns_to_encode[i] for i in columns_indices])
        encoded_features = encoder.fit_transform(df.iloc[:, columns_indices])
        print("Encoder fit and transformed data...")
    else:
        encoded_features = encoder.transform(df.iloc[:, columns_indices])
        print("Encoder transformed data using existing encoder...")

    encoded_df = pd.DataFrame(encoded_features, columns=[f'encoded_{i}' for i in columns_indices])

    # Remove original columns that were encoded
    df.drop(columns=[df.columns[i] for i in columns_indices], inplace=True)

    # Combine encoded features with other features
    df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

    columns_to_drop = df.columns[0:60].tolist()
    print("Columns to be dropped:", columns_to_drop)
    df.drop(columns=columns_to_drop, inplace=True)

    # Apply scaler separately to encoded columns
    encoded_columns = encoded_df.columns.tolist()
    remaining_columns = df.columns.difference(encoded_columns).tolist()

    if feature_scaler is None:
        feature_scaler = {'encoded': MinMaxScaler(), 'remaining': MinMaxScaler()}

        # Scale encoded columns
        if df[encoded_columns].isnull().any().any():
            df[encoded_columns].fillna(df[encoded_columns].mean(), inplace=True)
        scaled_encoded_features = feature_scaler['encoded'].fit_transform(df[encoded_columns].astype('float32'))

        # Scale remaining columns
        if df[remaining_columns].isnull().any().any():
            df[remaining_columns].fillna(df[remaining_columns].mean(), inplace=True)
        scaled_remaining_features = feature_scaler['remaining'].fit_transform(df[remaining_columns].astype('float32'))

        print("Scalers loaded and data scaled...")
    else:
        # Scale encoded columns using existing scaler
        if df[encoded_columns].isnull().any().any():
            mean_dict = {col: feature_scaler['encoded'].data_min_[i] for i, col in enumerate(encoded_columns)}
            df[encoded_columns].fillna(mean_dict, inplace=True)
        scaled_encoded_features = feature_scaler['encoded'].transform(df[encoded_columns].astype('float32'))

        # Scale remaining columns using existing scaler
        if df[remaining_columns].isnull().any().any():
            mean_dict = {col: feature_scaler['remaining'].data_min_[i] for i, col in enumerate(remaining_columns)}
            df[remaining_columns].fillna(mean_dict, inplace=True)
        scaled_remaining_features = feature_scaler['remaining'].transform(df[remaining_columns].astype('float32'))

        print("Data scaled using existing scalers...")

    # Combine scaled encoded and remaining columns
    scaled_df = pd.DataFrame(scaled_encoded_features, columns=encoded_columns).join(pd.DataFrame(scaled_remaining_features, columns=remaining_columns))

    # Scale targets separately
    if target_scaler is None:
        target_scaler = {name: MinMaxScaler() for name in output_names}
        target_scaled = np.zeros(df_targets.shape)
        for i, col in enumerate(df_targets.columns):
            df_targets[col].fillna(df_targets[col].mean(), inplace=True)
            target_scaled[:, i] = target_scaler[output_names[i]].fit_transform(df_targets[[col]].values).flatten()
        print("Targets scaled separately...")
    else:
        target_scaled = np.zeros(df_targets.shape)
        for i, col in enumerate(df_targets.columns):
            df_targets[col].fillna(df_targets[col].mean(), inplace=True)
            target_scaled[:, i] = target_scaler[output_names[i]].transform(df_targets[[col]].values).flatten()
        print("Targets scaled with existing scalers...")

    # Create overlapping sequences
    X = create_overlapping_sequences(scaled_df.values, seq_len, seq_step)

    target_dict = {output_names[idx]: target_scaled[seq_len - 1:, idx] for idx in range(len(output_names))}

    return X, target_dict, feature_scaler, target_scaler, encoder, columns_to_encode

# ::::::::::::::::::::::Randomize Data::::::::::::::::::::::: #

def randomize_data(X, y):
    """Shuffles X and y consistently."""
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = {key: value[perm] for key, value in y.items()}
    return X_shuffled, y_shuffled
    
# ::::::::::::::::::::::Function loading::::::::::::::::::::::: #

# Load and preprocess data
print("About to load and preprocess training data...")

df_train = load_data_from_sqlite("/opt/dlami/nvme/XAUUSD_15m_OHLC_Train_4_4_new.db", 'XAUUSD')
df_val = load_data_from_sqlite("/opt/dlami/nvme/XAUUSD_15m_OHLC_Val_4_4_new.db", 'XAUUSD')


print("Finished loading training data...\n")

print("Starting to preprocess training data...")
X_train, y_train_dict, train_scaler, target_scaler, encoder, columns_to_encode = preprocess_data(df_train, seq_len, target_columns, output_names)
print("Finished preprocessing training data...")
X_train, y_train_dict = randomize_data(X_train, y_train_dict)
print(f"Training data shapes - X: {X_train.shape}, y: {[y.shape for y in y_train_dict.values()]}\n")

# Save the scalers for future use
print("Saved the scaler for future use...\n")
joblib.dump({
    'feature_scaler': train_scaler,
    'target_scaler': target_scaler,
    'encoder': encoder,
    'columns_to_encode': columns_to_encode,
    'seq_len': seq_len,
    'seq_step': seq_step
}, 'scalers_15m_low_XAUUSD.pkl')

# Load the scalers and encoder for validation or live data
scalers_encoder = joblib.load('scalers_15m_low_XAUUSD.pkl')
train_scaler = scalers_encoder['feature_scaler']
target_scaler = scalers_encoder['target_scaler']
encoder = scalers_encoder['encoder']
columns_to_encode = scalers_encoder['columns_to_encode']
seq_len = scalers_encoder['seq_len']
seq_step = scalers_encoder['seq_step']

print("Starting to preprocess validation data...")
X_val, y_val_dict, _, _, _, _ = preprocess_data(df_val, seq_len, target_columns, output_names, train_scaler, target_scaler, encoder)
print("Finished preprocessing validation data...")
print(f"Validation data shapes - X: {X_val.shape}, y: {[y.shape for y in y_val_dict.values()]}\n")  

# :::::::::::::::::::::::::::Real-time Dashboard::::::::::::::::::::::::::::::: #

class RealTimeDashboard(Callback):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('15m XAUUSD Predictions Low')
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
        plt.savefig("/home/ubuntu/Desktop/trading/model_testing_text/training_models/training_progress/XAUUSD_high_train_loss_v1_15m_low.png") 
        self.root.update()

print("About to initialize the model...")

with strategy6.scope():

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
                Dense(512, activation='relu', use_bias=True, kernel_initializer=HeNormal()),
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
                Dense(512, activation='relu', use_bias=True, kernel_initializer=HeNormal()),
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
    num_enc_layers = 3                    # Encoding Attention layers
    num_dec_layers = 3                    # Decoding Attention layers
    dropout_rate = 0.00                   # Dropout rate

    conv_layers = 2                       # 1st convolutional layers
    initial_filters = int(256)            # 1nd unit hidden layer exspansion (MHA)
    d_ffn = int(initial_filters//2)       # 1nd unit exspansion/supression (LSTM)

    # ::::::::::::::Main Model Training (Multi Target Prediction):::::::::::::: #

    outputs = []
    for name in output_names: 
        q_ffn = initial_filters
        q_ffn_skip= q_ffn // conv_layers
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
    weights_checkpoint_path = "/home/ubuntu/Desktop/trading/model_testing_text/training_models/saved_weights/XAUUSD_high_weights_v1_15m_low.h5"
    weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', 
                                        verbose=1, save_best_only=True, mode='min', save_weights_only=True)

    # Define the model checkpoint for the full model
    full_model_checkpoint_path = "/home/ubuntu/Desktop/trading/model_testing_text/training_models/saved_weights/XAUUSD_high_model_v1_FULL_15m_low.h5"
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
        def __init__(self, target_scalers, validation_data, batch_size):
            super().__init__()
            self.target_scalers = target_scalers
            self.validation_data = validation_data
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
                if pred.ndim > 1:
                    pred = np.mean(pred, axis=1)

                pred_sampled = pred[random_indices]
                y_val_sampled = y_val[random_indices]

                scaler = self.target_scalers[key]

                pred_sampled = pred_sampled.reshape(-1, 1)
                y_val_sampled = y_val_sampled.reshape(-1, 1)

                pred_unscaled = scaler.inverse_transform(pred_sampled).flatten()
                real_unscaled = scaler.inverse_transform(y_val_sampled).flatten()

                print(f"Epoch {epoch + 1} - First 5 Validation Predictions for {key}: {pred_unscaled}")
                print(f"Epoch {epoch + 1} - First 5 Validation Real Values for {key}: {real_unscaled}")
                print()

    class PrintUnscaledLoss(Callback):
        def __init__(self, target_scalers, batch_size, validation_data):
            super().__init__()
            self.target_scalers = target_scalers
            self.batch_size = batch_size
            self.validation_data = validation_data

        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val_dict = self.validation_data
            predictions = self.model.predict(X_val, batch_size=self.batch_size)

            if not isinstance(predictions, list):
                predictions = [predictions]

            for i, (key, y_val) in enumerate(y_val_dict.items()):
                pred = predictions[i]

                if pred.ndim == 3:
                    pred = np.mean(pred, axis=1)
                if y_val.ndim == 1:
                    y_val = y_val[:, np.newaxis] 

                pred = pred.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)

                scaler = self.target_scalers[key]
                combined = np.hstack([pred, y_val])
                combined_unscaled = scaler.inverse_transform(combined)
                pred_unscaled = combined_unscaled[:, 0]
                y_val_unscaled = combined_unscaled[:, 1]

                mae_val_unscaled = np.mean(np.abs(pred_unscaled - y_val_unscaled))
                print(f'Epoch {epoch + 1} - Validation {key} Unscaled MAE: {mae_val_unscaled}')

    class CustomDataGenerator(Sequence):
        def __init__(self, X, y, batch_size):
            self.X = X
            self.y = y
            self.batch_size = batch_size
            self.indices = np.arange(len(self.X))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))

        def __getitem__(self, index):
            batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            X_batch = self.X[batch_indices]
            y_batch = {key: value[batch_indices] for key, value in self.y.items()}
            return X_batch, y_batch

        def on_epoch_end(self):
            self.shuffle_data()

        def shuffle_data(self):
            np.random.shuffle(self.indices)

    class RandomizeDataCallback(Callback):
        def __init__(self, data_generator):
            super().__init__()
            self.data_generator = data_generator

        def on_epoch_begin(self, epoch, logs=None):
            self.data_generator.shuffle_data()
            print("Randomizing Next Epoch Data Complete.")

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

    # callbacks
    unscaled_loss_callback = PrintUnscaledLoss(
        target_scalers=target_scaler, 
        batch_size=batch_size,  
        validation_data=(X_val, y_val_dict)
    )
    print_predictions_and_loss_0 = PrintPredictionsAndLoss_0(
        target_scalers=target_scaler, 
        validation_data=(X_val, y_val_dict), 
        batch_size=batch_size
    )

    train_data_generator = CustomDataGenerator(X_train, y_train_dict, batch_size)
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

# Flatten predictions and unscale using the appropriate scalers
if not isinstance(y_pred_list, list):
    y_pred_list = [y_pred_list]

y_pred_dict = {name: target_scaler[name].inverse_transform(pred.reshape(-1, 1)).flatten() for name, pred in zip(output_names, y_pred_list)}
y_real_dict = {name: target_scaler[name].inverse_transform(y_val_dict[name].reshape(-1, 1)).flatten() for name in output_names}

# Calculate metrics and store results
metrics = {}
for name in output_names:
    mse = mean_squared_error(y_real_dict[name], y_pred_dict[name])
    mae = mean_absolute_error(y_real_dict[name], y_pred_dict[name])
    metrics[name] = {"MSE": mse, "MAE": mae}
    print(f"{name} - Mean Squared Error: {mse}")
    print(f"{name} - Mean Absolute Error: {mae}")

# Combine all predictions and actuals into one DataFrame
data = []
for name in output_names:
    actual = y_real_dict[name]
    predicted = y_pred_dict[name]

    print(f"Shape of Actual {name}: {actual.shape}")
    print(f"Shape of Predicted {name}: {predicted.shape}")

    # Ensure that the length of actual and predicted arrays match
    min_length = min(len(actual), len(predicted))
    actual = actual[:min_length]
    predicted = predicted[:min_length]

    # Append actual and predicted data to the list
    data.append((actual, predicted))

# Create a DataFrame from the data collected
result_df = pd.DataFrame(
    np.column_stack([np.concatenate([pair[0] for pair in data]), np.concatenate([pair[1] for pair in data])]),
    columns=[f'Actual_{name}' for name in output_names] + [f'Predicted_{name}' for name in output_names]
)

# Save the combined results to a single CSV file
file_path = "/home/ubuntu/Desktop/trading/model_testing_text/training_models/training_progress/predictions_15m_low.csv"
result_df.to_csv(file_path, index=False)
print(f"Saving all predictions to: {file_path}")

print("Predictions complete.")







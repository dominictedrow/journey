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
from collections import defaultdict
from tensorflow.keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder, RobustScaler
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.initializers import HeNormal, RandomNormal
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.layers import MultiHeadAttention, Add, Conv2D, Concatenate, Conv1D
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, LayerNormalization, Flatten, LSTM
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

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
epochs = 25                     # Training iterations
batch_size = 512                # Samples per batch (last was 32)

target_group = False

# :::: learning warmup & Adam Optimizer ::::

custom_lr = False               # Activate warmup_steps
warmup_steps = int(4000)        # Number of batches for lr warmup_steps  
lr_scale = 1                    # 1 is default for no change (increases starting lr 4 warmup_steps)

fixed_learning_rate = 0.000001  # Used if custom_lr=False
clipnorm = 1.0
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-7
amsgrad = False

#:::: Training Target Predictions ::::
""" 
target_columns should be the name of any Target columns and output_names
should be the the targets you want to use, which requires adjusting drop_targets
to match the output_names you selected: ['High', 'Low', 'Close', 'Signal']. Curretly only
using 1 Target of 4 possible in dataset 
"""
drop_targets = -1            # exclude the last target columns
target_columns = ['Target1', 'Target2', 'Target3', 'Target4']
output_names =   ['Signal']       

# :::: Training Loss Function ::::

loss_func = 'mean_squared_error'
""" 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'
'mean_squared_logarithmic_error', 'huber_loss', 'log_cosh', 'cosine_similarity' """

# :::::::::::::::::::::::::::::TensorFlow setup:::::::::::::::::::::::::::::: #

# Suppresses TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configure GPUs if necessary
def configure_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
configure_gpus()

print("Eager execution:", tf.executing_eagerly()) 
print()

# ::::::::::::::::::::::Data Processing Transformation::::::::::::::::::::::: #

class AdaptiveMinMaxScaler:
    def __init__(self):
        self.scalers = {}

    def fit_transform(self, data, key):
        if key not in self.scalers:
            self.scalers[key] = MinMaxScaler()
        return self.scalers[key].fit_transform(data)

    def transform(self, data, key):
        if key not in self.scalers:
            raise ValueError(f"Scaler for key {key} not found.")
        return self.scalers[key].transform(data)

    def inverse_transform(self, values, key):
        if key not in self.scalers:
            raise ValueError(f"Scaler for key {key} not found.")
        scaler = self.scalers[key]
        return scaler.inverse_transform(values)


def load_data_from_sqlite(db_path):
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    print("Executing query to get table names...")
    tables = pd.read_sql_query(query, conn)
    all_data = []
    source_tables = []
    
    for table_name in tables['name']:
        print(f"Loading data from table: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Process Target4 column if it exists in the table
        if 'Target4' in df.columns:
            df['Target4'] = df['Target4'].replace({5000: 10, 1000: -10})
        
        all_data.append(df)
        source_tables.append(table_name)
    
    conn.close()
    print("Data loading complete.")
    return all_data, source_tables

def create_overlapping_sequences(data, seq_len, step):
    sequences = []
    for i in range(0, len(data) - seq_len + 1, step):
        seq = data[i:i + seq_len]
        sequences.append(seq)
    return np.array(sequences)

def preprocess_and_combine_data(data_list, source_tables, seq_len, seq_step, target_columns, output_names, is_training=True, existing_scalers=None):
    print("Preprocessing and combining data...")

    all_X_encoded = []
    all_X_remaining = []
    all_y = {name: [] for name in output_names}
    all_encoded_4 = []
    all_source_tables = []
    all_symbol_ids = []

    # Initialize AdaptiveMinMaxScaler or use existing ones
    if existing_scalers:
        feature_scaler = existing_scalers.get('feature_scaler')
        target_scaler = existing_scalers.get('target_scaler')
        encoder = existing_scalers.get('encoder')
        columns_to_encode = existing_scalers.get('columns_to_encode')
    else:
        feature_scaler = AdaptiveMinMaxScaler()
        target_scaler = AdaptiveMinMaxScaler()
        encoder = None
        columns_to_encode = None

    for idx, (df, table_name) in enumerate(zip(data_list, source_tables)):
        print(f"Processing data from table: {table_name}")
        used_targets = [target_columns[3]]  # Use 'Target4'
        df_targets = df[used_targets].astype('float32')
        df_targets.columns = output_names  # Rename target columns

        df = df.drop(columns=target_columns, errors='ignore')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        df['table_id'] = idx
        df['source_table'] = table_name

        # Handling 'symbol_id' column
        if 'symbol_id' in df.columns:
            symbol_ids = df['symbol_id'].values
        else:
            symbol_ids = np.full(len(df), idx)

        """ # Initialize columns_to_encode if not done already
        if columns_to_encode is None:
            columns_to_encode = {  # Define your encoding scheme here
                0: list(range(1, 32)),
                1: list(range(1, 13)),
                2: list(range(0, 24)),
                3: list(range(0, 56)),
                4: [5, 15, 30, 60],
                **{i: [0, 1, 2, 3, 4, 5, 6] for i in range(57, 76)},
                76: list(range(0, 160))
            }
        columns_indices = sorted(columns_to_encode.keys()) """

        if columns_to_encode is None:
            columns_to_encode = {
                0: list(range(1, 32)),
                1: list(range(1, 13)),
                2: list(range(0, 24)),
                3: list(range(0, 60)),
                4: [1, 5, 15, 30, 60],
                **{i: [0, 1, 2, 3, 4, 5, 6] for i in range(57, 76)},
                76: list(range(0, 160))
            }
        columns_indices = sorted(columns_to_encode.keys())

        if encoder is None:
            encoder = OrdinalEncoder(categories=[columns_to_encode[i] for i in columns_indices])
            encoded_features = encoder.fit_transform(df.iloc[:, columns_indices])
        else:
            encoded_features = encoder.transform(df.iloc[:, columns_indices])

        encoded_df = pd.DataFrame(encoded_features, columns=[f'encoded_{i}' for i in columns_indices])
        df = df.drop(columns=[df.columns[i] for i in columns_indices])
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        encoded_columns = encoded_df.columns.tolist()
        remaining_columns = df.columns.difference(encoded_columns + ['source_table', 'table_id']).tolist()

        # Scale remaining columns and targets using AdaptiveMinMaxScaler
        if is_training:
            df[remaining_columns] = feature_scaler.fit_transform(df[remaining_columns].astype('float32'), table_name)
            df_targets_scaled = target_scaler.fit_transform(df_targets, table_name)
        else:
            df[remaining_columns] = feature_scaler.transform(df[remaining_columns].astype('float32'), table_name)
            df_targets_scaled = target_scaler.transform(df_targets, table_name)

        # Sorting and balancing encoded_4 groups
        sorted_encoded_values = sorted(df['encoded_4'].unique()) if 'encoded_4' in df.columns else [0]
        for group in sorted_encoded_values:
            group_df = df[df['encoded_4'] == group]
            group_df_targets = df_targets[df['encoded_4'] == group] if 'encoded_4' in df.columns else df_targets

            print(f"Processing Group: {group}, Group Size Row Count: {len(group_df)}")
            group_sequences_encoded = create_overlapping_sequences(group_df[encoded_columns].values, seq_len, seq_step)
            group_sequences_remaining = create_overlapping_sequences(group_df[remaining_columns].values, seq_len, seq_step)

            group_indices = group_df.index
            group_targets = df_targets_scaled[group_indices][seq_len - 1:len(group_sequences_encoded) + seq_len - 1]

            # Balance the groups during training
            if is_training:
                target_values = group_targets[:, 0]  # Assuming one target column
                unique_targets, counts_per_target = np.unique(target_values, return_counts=True)
                min_count = np.min(counts_per_target)

                print(f"Group {group}: Balancing to {min_count} sequences per target...")

                balanced_group_sequences_encoded = []
                balanced_group_sequences_remaining = []
                balanced_group_targets = []

                for target_value in unique_targets:
                    target_indices = np.where(target_values == target_value)[0]
                    if len(target_indices) > min_count:
                        target_indices = np.random.choice(target_indices, size=min_count, replace=False)
                    balanced_group_sequences_encoded.append(group_sequences_encoded[target_indices])
                    balanced_group_sequences_remaining.append(group_sequences_remaining[target_indices])
                    balanced_group_targets.append(group_targets[target_indices])

                group_sequences_encoded = np.concatenate(balanced_group_sequences_encoded, axis=0)
                group_sequences_remaining = np.concatenate(balanced_group_sequences_remaining, axis=0)
                group_targets = np.concatenate(balanced_group_targets, axis=0)

            all_X_encoded.append(group_sequences_encoded)
            all_X_remaining.append(group_sequences_remaining)
            for i, name in enumerate(output_names):
                all_y[name].append(group_targets[:, i])

            all_encoded_4.append(np.full(group_sequences_encoded.shape[0], group))
            all_source_tables.append(np.full(group_sequences_encoded.shape[0], table_name))
            all_symbol_ids.append(np.full(group_sequences_encoded.shape[0], symbol_ids[group_indices][:len(group_sequences_encoded)]))

        # Clear variables to free memory
        del df, df_targets, df_targets_scaled, encoded_df, group_df, group_df_targets, group_sequences_encoded, group_sequences_remaining, group_targets

    return all_X_encoded, all_X_remaining, all_y, all_encoded_4, all_symbol_ids, feature_scaler, target_scaler, encoder, columns_to_encode, all_source_tables

def scale_combined_data(all_X_encoded, all_X_remaining, all_y, all_encoded_4, all_symbol_ids, all_source_tables, feature_scaler=None, is_training=True):
    print("Scaling encoded data from all tables")
    
    # Combine data
    combined_X_encoded = np.concatenate(all_X_encoded, axis=0)
    combined_X_remaining = np.concatenate(all_X_remaining, axis=0)
    combined_y = {name: np.concatenate(values, axis=0).astype(np.float32) for name, values in all_y.items() if len(values) > 0}
    combined_encoded_4 = np.concatenate(all_encoded_4, axis=0)
    combined_source_tables = np.concatenate(all_source_tables, axis=0)
    combined_symbol_ids = np.concatenate(all_symbol_ids, axis=0)

    # Clear memory
    del all_X_encoded, all_X_remaining, all_y, all_encoded_4, all_source_tables, all_symbol_ids

    # Reshape and scale encoded part
    num_encoded_columns = combined_X_encoded.shape[2]
    encoded_part_reshaped = combined_X_encoded.reshape(-1, num_encoded_columns)
    
    if is_training:
        scaled_encoded = feature_scaler.fit_transform(encoded_part_reshaped, 'combined')
    else:
        scaled_encoded = feature_scaler.transform(encoded_part_reshaped, 'combined')
    
    # Clear memory
    del combined_X_encoded, encoded_part_reshaped

    # Reshape back to original shape
    scaled_encoded = scaled_encoded.reshape(-1, combined_X_remaining.shape[1], num_encoded_columns)
    
    # Combine scaled encoded part with remaining part
    final_X = np.concatenate([scaled_encoded, combined_X_remaining], axis=2).astype(np.float32)

    # Clear memory
    del scaled_encoded, combined_X_remaining

    print(f"Final combined shapes - X: {final_X.shape}, y: {[y.shape for y in combined_y.values()]}")
    return final_X, combined_y, combined_encoded_4, combined_symbol_ids, combined_source_tables, feature_scaler


def randomize_data(X, y, source_tables, symbol_ids):
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = {key: value[perm] for key, value in y.items()}
    source_tables_shuffled = source_tables[perm]
    symbol_ids_shuffled = symbol_ids[perm]
    return X_shuffled, y_shuffled, source_tables_shuffled, symbol_ids_shuffled

print("About to load and preprocess training data...")
df_train, source_tables_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//Processed_Stacked_HLC_XAUUSD_Train_2.db")
df_val, source_tables_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//Processed_Stacked_HLC_XAUUSD_Val_2.db")

# Preprocess and combine training data
print("Starting to preprocess training data...")
all_X_train_encoded, all_X_train_remaining, all_y_train, all_encoded_4_train, all_symbol_ids_train, feature_scaler, target_scaler, encoder, columns_to_encode, source_tables_train = preprocess_and_combine_data(
    df_train, source_tables_train, seq_len, seq_step, target_columns, output_names, is_training=True
)

# Scale and combine all training data
print("Scaling and combining all training data...")
X_train, y_train_dict, encoded_4_train, symbol_ids_train, source_tables_train, feature_scaler = scale_combined_data(
    all_X_train_encoded, all_X_train_remaining, all_y_train, all_encoded_4_train, all_symbol_ids_train, source_tables_train, feature_scaler=feature_scaler, is_training=True
)

# num_symbols is now the number of unique symbol_ids
num_symbols = len(np.unique(symbol_ids_train))

# Randomize the training data
X_train, y_train_dict, source_tables_train, symbol_ids_train = randomize_data(X_train, y_train_dict, source_tables_train, symbol_ids_train)
print(f"Final training data shapes - X: {X_train.shape}, y: {[y.shape for y in y_train_dict.values()]}, source_tables: {len(source_tables_train)}, symbol_ids: {len(symbol_ids_train)}\n")

# Save the scalers and encoder for future use
print("Saving the scalers and encoder for future use...")
joblib.dump({
    'feature_scaler': feature_scaler,
    'target_scaler': target_scaler,
    'encoder': encoder,
    'columns_to_encode': columns_to_encode,
    'seq_len': seq_len,
    'seq_step': seq_step
}, 'scalers_stacked_multi.pkl')

# Load the scalers and encoder for validation data
scalers_encoder = joblib.load('scalers_stacked_multi.pkl')
feature_scaler = scalers_encoder['feature_scaler']
target_scaler = scalers_encoder['target_scaler']
encoder = scalers_encoder['encoder']
columns_to_encode = scalers_encoder['columns_to_encode']
seq_len = scalers_encoder['seq_len']
seq_step = scalers_encoder['seq_step']

# Preprocess and combine validation data
print("Starting to preprocess validation data...")
all_X_val_encoded, all_X_val_remaining, all_y_val, all_encoded_4_val, all_symbol_ids_val, _, _, _, _, source_tables_val = preprocess_and_combine_data(
    df_val, source_tables_val, seq_len, seq_step, target_columns, output_names, is_training=False,
    existing_scalers={'feature_scaler': feature_scaler, 'target_scaler': target_scaler, 'encoder': encoder, 'columns_to_encode': columns_to_encode}
)

# Scale and combine validation data
print("Scaling and combining validation data...")
X_val, y_val_dict, encoded_4_val, symbol_ids_val, source_tables_val, _ = scale_combined_data(
    all_X_val_encoded, all_X_val_remaining, all_y_val, all_encoded_4_val, all_symbol_ids_val, source_tables_val, feature_scaler=feature_scaler, is_training=False
)

# Display shapes of the validation data
print(f"Validation data shapes - X: {X_val.shape}, y: {[y.shape for y in y_val_dict.values()]}, source_tables: {len(source_tables_val)}, symbol_ids: {symbol_ids_val.shape}\n")


# :::::::::::::::::::::::::::Real-time Dashboard::::::::::::::::::::::::::::::: #

class RealTimeDashboard(Callback):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('5m multi Predictions')
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
        plt.savefig("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/multi_train_loss_v1_stacked_multi.png") 
        self.root.update()

print("About to initialize the model...")

# :::::::::::::::::::::::::::Custom Positional Encoding Layer::::::::::::::::::::::::::::::: #

class AdaptiveSphereTransformLayer(Layer):
    def __init__(self, **kwargs):
        super(AdaptiveSphereTransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        super(AdaptiveSphereTransformLayer, self).build(input_shape)

    def perform_transformation(self, inputs, seq_len, num_features):
        # Center the sphere transformation at the middle of the input
        center_seq = tf.cast(tf.round(0.5 * tf.cast(seq_len, tf.float32)), tf.int32)  # Cast to float32, then round and cast to int32
        center_feature = tf.cast(tf.round(0.5 * tf.cast(num_features, tf.float32)), tf.int32)

        # Generate a grid from -1 to 1 for the sequence and feature dimensions
        i = tf.linspace(-1.0, 1.0, seq_len)
        j = tf.linspace(-1.0, 1.0, num_features)
        ii, jj = tf.meshgrid(i, j, indexing='ij')

        # Center the grid to the middle of the input space
        ii_centered = ii - (2.0 * tf.cast(center_seq, tf.float32) / tf.cast(seq_len, tf.float32) - 1.0)
        jj_centered = jj - (2.0 * tf.cast(center_feature, tf.float32) / tf.cast(num_features, tf.float32) - 1.0)

        # Compute spherical coordinates
        r = tf.sqrt(ii_centered**2 + jj_centered**2 + 1e-9)  # Avoid divide by zero
        theta = tf.atan2(jj_centered, ii_centered)
        phi = r * np.pi

        # Convert spherical coordinates to Cartesian
        x = r * tf.sin(phi) * tf.cos(theta)
        y = r * tf.sin(phi) * tf.sin(theta)
        z = r * tf.cos(phi)

        # Combine the Cartesian coordinates
        sphere_sum = (x + y + z)

        # Expand and tile the spherical surface to match the batch size
        sphere_sum = tf.expand_dims(sphere_sum, axis=0)
        sphere_sum = tf.tile(sphere_sum, [tf.shape(inputs)[0], 1, 1])

        # Cast both inputs and sphere_sum to the same type (float32)
        inputs = tf.cast(inputs, tf.float32)  # Ensure inputs are float32
        sphere_sum = tf.cast(sphere_sum, tf.float32)  # Ensure sphere_sum is float32

        # Perform the addition
        transformed_inputs = tf.reshape(sphere_sum, tf.shape(inputs)) + inputs

        return transformed_inputs
    
    def call(self, inputs, training=None):
        # Extract dimensions of the input tensor
        batch_size, seq_len, num_features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        
        # Perform the transformation
        transformed_inputs = self.perform_transformation(inputs, seq_len, num_features)

        return transformed_inputs

    def get_config(self):
        # If you need to save/load the layer later
        config = super(AdaptiveSphereTransformLayer, self).get_config()
        return config
    
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
            pos=tf.range(self.seq_len, dtype=tf.float32)[:, tf.newaxis],
            i=tf.range(self.d_model, dtype=tf.float32)[tf.newaxis, :],
            d_model=self.d_model
        )

        # Apply sin to even indices and cos to odd indices
        sines = tf.math.sin(angle_rads[:, 0::2])
        cosines = tf.math.cos(angle_rads[:, 1::2])

        # Concatenate sines and cosines along the last axis
        pos_encoding = tf.concat([sines, cosines], axis=-1)
        pos_encoding = pos_encoding[tf.newaxis, ...]  # Add batch dimension
        return tf.cast(pos_encoding, tf.float32)

    def call(self, inputs):
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
    
class CustomLearnablePositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super(CustomLearnablePositionalEncoding, self).__init__(**kwargs)
        self.seq_len = seq_len
        self.d_model = d_model
        self.pos_encoding = self.add_weight(
            name='learnable_pos_encoding',
            shape=(1, seq_len, d_model),
            initializer=tf.keras.initializers.RandomNormal(),
            trainable=True
        )

    def call(self, inputs):
        seq_len = tf.shape(inputs)[1]
        return inputs + self.pos_encoding[:, :seq_len, :]
    
    def get_config(self):
        config = super(CustomLearnablePositionalEncoding, self).get_config()
        config.update({
            'seq_len': self.seq_len,
            'd_model': self.d_model
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(seq_len=config['seq_len'], d_model=config['d_model'])
    
# :::::::::::::::::::::::::::Custom Transformer::::::::::::::::::::::::::::::: #

# Symbol Embedding Layer to capture symbol-specific patterns
class SymbolEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_symbols, embed_dim, **kwargs):
        super(SymbolEmbedding, self).__init__(**kwargs)
        self.num_symbols = num_symbols
        self.embed_dim = embed_dim
        self.symbol_embeddings = tf.keras.layers.Embedding(num_symbols, embed_dim)
    
    def call(self, symbol_id):
        return self.symbol_embeddings(symbol_id)
    
    def get_config(self):
        config = super(SymbolEmbedding, self).get_config()
        config.update({
            'num_symbols': self.num_symbols,
            'embed_dim': self.embed_dim
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(num_symbols=config['num_symbols'], embed_dim=config['embed_dim'])

# Custom Multi-Head Attention Layer with symbol-aware modulation
class CustomMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attention_heads = tf.keras.layers.MultiHeadAttention(num_heads, embed_dim)
        self.layernorm = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, symbol_embedding):
        attention_output = self.attention_heads(inputs, inputs)
        # Modulate attention with the symbol embedding
        modulated_attention = attention_output + symbol_embedding
        output = self.layernorm(modulated_attention + inputs)
        return output
    
    def get_config(self):
        config = super(CustomMultiHeadAttention, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(embed_dim=config['embed_dim'], num_heads=config['num_heads'])


# Custom Transformer Block with symbol and time awareness
class CustomTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
        super(CustomTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = CustomMultiHeadAttention(embed_dim, num_heads)
        self.ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation="elu", use_bias=True, kernel_initializer='he_normal'),
            tf.keras.layers.Dense(embed_dim, use_bias=False, kernel_initializer='he_normal')
        ])
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(dropout_rate)
        self.dropout2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, inputs, symbol_embedding, training):
        # Attention modulated by symbol embedding
        attn_output = self.att(inputs, symbol_embedding)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feedforward + normalization
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    
    def get_config(self):
        config = super(CustomTransformerBlock, self).get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'dropout_rate': self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            embed_dim=config['embed_dim'],
            num_heads=config['num_heads'],
            ff_dim=config['ff_dim'],
            dropout_rate=config['dropout_rate']
        )


# Encoder block for transformer, adds original input back at each layer
class Encoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.transformer_blocks = [
            CustomTransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
    
    def call(self, x, symbol_embed, original_inputs, training):
        # Pass input through each transformer block and add original input back
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, symbol_embed, training)
            x = x + original_inputs 
        return x

# Decoder block for transformer, adds original input back at each layer
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, embed_dim, num_heads, ff_dim, dropout_rate):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.self_attention_blocks = [
            CustomTransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)
            for _ in range(num_layers)
        ]
        self.cross_attention = CustomMultiHeadAttention(embed_dim, num_heads)

    def call(self, x, enc_output, symbol_embed, original_inputs, training):
        # Self-attention layers with original input added back
        for transformer_block in self.self_attention_blocks:
            x = transformer_block(x, symbol_embed, training)
            x = x + original_inputs 

        # Cross-attention between decoder input and encoder output
        cross_attn_output = self.cross_attention(x, enc_output)
        cross_attn_output = cross_attn_output + original_inputs  
        return cross_attn_output

# Market-Aware Transformer Model with original inputs added back at each layer
class MarketAwareTransformer(tf.keras.Model):
    def __init__(self, num_symbols, seq_len, embed_dim, num_heads, ff_dim, num_enc_layers, num_dec_layers, dropout_rate, **kwargs):
        super(MarketAwareTransformer, self).__init__(**kwargs)
        self.num_symbols = num_symbols
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers
        self.dropout_rate = dropout_rate

        self.symbol_embedding = SymbolEmbedding(num_symbols, embed_dim)
        self.pos_encoding = CustomLearnablePositionalEncoding(seq_len, embed_dim)
        self.positional_encoding = PositionalEncoding(seq_len, embed_dim)

        # Encoder and Decoder
        self.encoder = Encoder(num_enc_layers, embed_dim, num_heads, ff_dim, dropout_rate)
        self.decoder = Decoder(num_dec_layers, embed_dim, num_heads, ff_dim, dropout_rate)

    def call(self, inputs, symbol_id, training=False):
        # Symbol-specific embedding
        symbol_embed = self.symbol_embedding(symbol_id)
        
        # Apply learnable positional encoding to inputs
        x = self.positional_encoding(inputs)
        #x = self.pos_encoding(x)

        # Save original inputs to add back at each block
        original_inputs = inputs

        # Encoder output, adding original inputs back at each layer
        enc_output = self.encoder(x, symbol_embed, original_inputs, training=training)

        # Decoder output, adding original inputs back at each layer
        dec_output = self.decoder(x, enc_output, symbol_embed, original_inputs, training=training)

        # Final output
        return dec_output
    
    def get_config(self):
        config = super(MarketAwareTransformer, self).get_config()
        config.update({
            'num_symbols': self.num_symbols,
            'seq_len': self.seq_len,
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'ff_dim': self.ff_dim,
            'num_enc_layers': self.num_enc_layers,
            'num_dec_layers': self.num_dec_layers,
            'dropout_rate': self.dropout_rate
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
# :::::::::::::::::::::::::::Custom 1D Convolutional::::::::::::::::::::::::::::::: #
    
def build_conv_layers(mha_model, initial_filters, seq_len, conv_layers, use_bias=False):
    # Dynamic generation of Conv1D layers
    for i in range(conv_layers):
        filters = initial_filters * (2 ** i)  # Double filters at each step
        kernel_size = seq_len // (4 // (i + 1))  # Dynamic kernel size based on sequence length
        mha_model = Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            activation='elu',
            kernel_initializer='he_normal',
            padding='same',
            use_bias=use_bias
        )(mha_model)
    return mha_model

# ::::::::::::::::::::::::Main Model Parameters::::::::::::::::::::::::::::: #

# Assuming your dataset has a shape of (num_samples, seq_len, d_model)
seq_len = X_train.shape[1]            # Time-series length
d_model = X_train.shape[2]            # Number of features (input dim per time step)

# Use the number of unique symbol IDs directly
num_symbols = len(np.unique(symbol_ids_train))
num_enc_layers = 8                    # Encoding Attention layers
num_dec_layers = 8                    # Decoding Attention layers
num_heads = 8                         # Number of attention heads
ff_dim = d_model*4                    # Feedforward network dimension
dropout_rate = 0.000                  # Dropout rate for regularization

conv_layers = 3                       # 1st convolutional layers
initial_filters = int(128)            # 1nd unit hidden layer exspansion (MHA)

# Input layer for time-series data (shape: (seq_len, d_model))
inputs = tf.keras.Input(shape=(seq_len, d_model), name='input_1')

# Input layer for symbol IDs (categorical)
symbol_id = tf.keras.Input(shape=(1,), dtype=tf.int32, name='input_2')

# ::::::::::::::Main Model Training (Multi Target Prediction):::::::::::::: #

outputs = []
for name in output_names: 
    
    market_aware_transformer = MarketAwareTransformer(
        num_symbols=num_symbols,
        seq_len=seq_len,
        embed_dim=d_model,             # Embedding dimension same as input feature dimension
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_enc_layers=num_enc_layers, 
        num_dec_layers=num_dec_layers,
        dropout_rate=dropout_rate
    )

    mha_model = market_aware_transformer(inputs, symbol_id) 
    #mha_model = AdaptiveSphereTransformLayer()(mha_model)

    """ mha_model = build_conv_layers(mha_model, initial_filters, seq_len, conv_layers, use_bias=True)
    mha_model = LayerNormalization(epsilon=1e-6)(mha_model)  """

    #mha_model = Flatten()(mha_model)
      
    mha_model = Dense(ff_dim, activation='elu', kernel_initializer=HeNormal(), use_bias=True)(mha_model)
    mha_model = Dense(d_model, kernel_initializer=HeNormal(), use_bias=False)(mha_model) 
    mha_model = Add()([mha_model, inputs])
    mha_model = LayerNormalization(epsilon=1e-6)(mha_model)

    # Output Layer
    mha_model = Dense(
        1,
        activation='linear',
        kernel_initializer=HeNormal(),
        use_bias=False,
        name=name,
        dtype='float32'
    )(mha_model) 
    
    outputs.append(mha_model)
model = Model([inputs, symbol_id], outputs=outputs)

model.summary()
print("Model initialized!")
print("Total number of parameters in the model:", model.count_params())

# :::::::::::::::::::::Model Checkpoint For Weights::::::::::::::::::::::: #

# Define the model checkpoint for weights only
weights_checkpoint_path =    "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/weights_v1_stacked_multi.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/model_v1_FULL_stacked_multi.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', 
                                        verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

custom_objects = {
    'CustomLearnablePositionalEncoding': CustomLearnablePositionalEncoding,
    'PositionalEncoding': PositionalEncoding,
    'SymbolEmbedding': SymbolEmbedding,
    'CustomMultiHeadAttention': CustomMultiHeadAttention,
    'CustomTransformerBlock': CustomTransformerBlock,
    'MarketAwareTransformer': MarketAwareTransformer,
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

# Class 1: PrintPredictionsAndLoss_0
class PrintPredictionsAndLoss_0(Callback):
    def __init__(self, target_scalers, validation_data, source_tables, batch_size, encoded_4_val):
        super().__init__()
        self.target_scalers = target_scalers  # Expecting a dictionary of scalers per table
        self.validation_data = validation_data
        self.source_tables = source_tables
        self.encoded_4_val = encoded_4_val
        self.batch_size = batch_size

        # Mapping source_table strings to unique integers
        self.source_table_mapping = {table: i for i, table in enumerate(np.unique(self.source_tables))}

        # Replace source_tables strings with their corresponding integer values
        self.mapped_source_tables = np.array([self.source_table_mapping[table] for table in self.source_tables])

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_dict = self.validation_data

        # Ensure that source_tables are integers and properly formatted for model prediction
        X_val_inputs = {
            'input_1': X_val,
            'input_2': self.mapped_source_tables
        }

        predictions = self.model.predict(X_val_inputs, batch_size=self.batch_size)

        if not isinstance(predictions, list):
            predictions = [predictions]  # Ensure consistent list handling

        num_samples = X_val.shape[0]
        random_indices = np.random.choice(num_samples, 5, replace=False)

        source_tables_sampled = self.source_tables[random_indices].astype(str)

        for i, output_name in enumerate(self.model.output_names):
            pred = predictions[i]
            y_val = y_val_dict[output_name]

            pred_sampled = pred[random_indices]
            y_val_sampled = y_val[random_indices]

            pred_unscaled = []
            real_unscaled = []

            for idx, source_table in enumerate(source_tables_sampled):
                # Ensure you have the correct scaler for the current source_table
                if source_table in self.target_scalers.scalers:
                    scaler = self.target_scalers.scalers[source_table]  # Use the specific scaler for the table
                    pred_val = pred_sampled[idx].reshape(1, -1)
                    real_val = y_val_sampled[idx].reshape(1, -1)

                    try:
                        unscaled_pred = scaler.inverse_transform(pred_val)
                        unscaled_real = scaler.inverse_transform(real_val)

                        pred_unscaled.append(unscaled_pred.flatten()[0])
                        real_unscaled.append(unscaled_real.flatten()[0])
                    except Exception as e:
                        print(f"Error in scaling for source_table '{source_table}': {e}")
                else:
                    print(f"Warning: Scaler for source_table '{source_table}' not found.")

            print(f"Epoch {epoch + 1} - First 5 Validation Predictions for {output_name}: {pred_unscaled}")
            print(f"Epoch {epoch + 1} - First 5 Validation Real Values for {output_name}: {real_unscaled}")
            print(f"Corresponding source tables: {source_tables_sampled}")
            print()


class PrintUnscaledLoss(Callback):
    def __init__(self, target_scalers, batch_size, validation_data, source_tables, encoded_4_val):
        super().__init__()
        self.target_scalers = target_scalers  # Dictionary of scalers
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.source_tables = source_tables
        self.encoded_4_val = encoded_4_val

        # Mapping source_table strings to unique integers
        self.source_table_mapping = {table: i for i, table in enumerate(np.unique(self.source_tables))}

        # Replace source_tables strings with their corresponding integer values
        self.mapped_source_tables = np.array([self.source_table_mapping[table] for table in self.source_tables])

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_dict = self.validation_data

        # Ensure that source_tables are integers for model input
        X_val_inputs = {
            'input_1': X_val,
            'input_2': self.mapped_source_tables
        }

        predictions = self.model.predict(X_val_inputs, batch_size=self.batch_size)

        if not isinstance(predictions, list):
            predictions = [predictions]

        combined_unscaled = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"pred": [], "real": []})))
        all_source_tables = set(self.source_tables)

        for i, output_name in enumerate(self.model.output_names):
            pred = predictions[i]
            y_val = y_val_dict[output_name]

            # Loop over each prediction and real value
            for idx in range(len(pred)):
                source_table = self.source_tables[idx]
                group = self.encoded_4_val[idx]

                pred_val = pred[idx].reshape(1, -1)
                real_val = y_val[idx].reshape(1, -1)

                try:
                    # Ensure that each source_table has its own scaler
                    if source_table in self.target_scalers.scalers:
                        scaler = self.target_scalers.scalers[source_table]

                        # Inverse scaling
                        unscaled_pred = scaler.inverse_transform(pred_val)
                        unscaled_real = scaler.inverse_transform(real_val)

                        combined_unscaled[source_table][output_name][group]["pred"].append(unscaled_pred[0, 0])
                        combined_unscaled[source_table][output_name][group]["real"].append(unscaled_real[0, 0])
                    else:
                        print(f"Warning: Scaler for source_table '{source_table}' not found.")
                except Exception as e:
                    print(f"Error: {e} - Skipping index {idx} for table '{source_table}', output '{output_name}', and group {group}.")

        # Check for missing source tables
        missing_source_tables = all_source_tables - set(combined_unscaled.keys())
        for source_table in missing_source_tables:
            print(f"Warning: Source table '{source_table}' found in source_tables but not processed.")

        # Calculate and print MAE for each combination of source table, output, and group
        for source_table, output_data in combined_unscaled.items():
            for output_name, group_data in output_data.items():
                sorted_groups = sorted(group_data.keys())
                for group in sorted_groups:
                    data = group_data[group]
                    if data["pred"] and data["real"]:
                        pred_unscaled = np.array(data["pred"])
                        y_val_unscaled = np.array(data["real"])

                        mae_val_unscaled = mean_absolute_error(y_val_unscaled, pred_unscaled)
                        print(f'Epoch {epoch + 1} - Validation {output_name} Unscaled MAE for {source_table}, Group {group}: {mae_val_unscaled}')

        # Calculate overall MAE for each output across all tables and groups
        for output_name in self.model.output_names:
            all_pred = []
            all_real = []
            for source_table, output_data in combined_unscaled.items():
                if output_name in output_data:
                    for group_data in output_data[output_name].values():
                        all_pred.extend(group_data["pred"])
                        all_real.extend(group_data["real"])

            if all_pred and all_real:
                overall_mae = mean_absolute_error(all_real, all_pred)
                print(f'Epoch {epoch + 1} - Overall Validation {output_name} Unscaled MAE: {overall_mae}')

def loss_function(y_true, y_pred):
    # Ensuring y_true and y_pred are at least 2D
    if len(tf.shape(y_true)) == 1:
        y_true = tf.expand_dims(y_true, -1)
    if len(tf.shape(y_pred)) == 1:
        y_pred = tf.expand_dims(y_pred, -1)

    if len(tf.shape(y_pred)) == 3: 
        y_pred = tf.reduce_mean(y_pred, axis=1)
        y_true = tf.reduce_mean(y_true, axis=1)

    loss_per_timestep = loss_func(y_true, y_pred)
    
    return tf.reduce_mean(loss_per_timestep)

class CustomDataGenerator(Sequence):
    def __init__(self, X, y, symbol_ids, batch_size):
        assert len(X) == len(symbol_ids), "X and symbol_ids must have the same length"
        assert len(X) == len(next(iter(y.values()))), "X and y must have the same length"
        
        self.X = X
        self.y = y
        self.symbol_ids = symbol_ids
        self.batch_size = batch_size
        self.indices = np.arange(len(self.X))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        X_batch = self.X[batch_indices]
        symbol_ids_batch = self.symbol_ids[batch_indices]
        y_batch = {key: value[batch_indices].astype(np.float32) for key, value in self.y.items()}
        
        # Return a tuple of inputs and outputs
        return ({'input_1': X_batch, 'input_2': symbol_ids_batch}, y_batch)

    def on_epoch_end(self):
        self.shuffle_data()

    def shuffle_data(self):
        np.random.shuffle(self.indices)

class CaptureLayerOutputs(tf.keras.callbacks.Callback):
    def __init__(self, model, layer_name, output_save_dir, sample_rate=0.1):
        super(CaptureLayerOutputs, self).__init__()
        self.model = model
        self.layer_name = layer_name
        self.output_save_dir = output_save_dir
        self.sample_rate = sample_rate  # Fraction of batches to sample
        os.makedirs(output_save_dir, exist_ok=True)

    def on_epoch_end(self, epoch, logs=None):
        # Get the layer by name
        layer = self.model.get_layer(self.layer_name)

        # Iterate over the batches randomly
        for batch_idx, (x_batch, _) in enumerate(self.model.train_data):
            if np.random.rand() < self.sample_rate:  # Randomly sample
                # Forward pass through the model until the chosen layer
                intermediate_model = tf.keras.Model(inputs=self.model.input, outputs=layer.output)
                output = intermediate_model(x_batch, training=False)  # Pass through the network
                
                # Save the output
                output_file = os.path.join(self.output_save_dir, f"output_epoch_{epoch}_batch_{batch_idx}.npy")
                np.save(output_file, output.numpy())

                print(f"Captured output for epoch {epoch}, batch {batch_idx}, saved to {output_file}")
                break 

# ::::::::::::::::::::::::Training & Validation Loop:::::::::::::::::::::::::::: #

# Initialize optimizer
learning_rate = CustomSchedule(d_model, warmup_steps, custom_lr, lr_scale)
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)

X_val = X_val.astype(np.float32)
y_val_dict = {key: value.astype(np.float32) for key, value in y_val_dict.items()}

X_val_inputs = {
    'input_1': X_val,
    'input_2': symbol_ids_val 
}

# Debugging prints before prediction
print("X_val shape:", X_val.shape)
#print("X_val inputs type:", {k: v.dtype for k, v in X_val_inputs.items()})
print("Processed Source Tables:", set(source_tables_val)) 

# Callbacks initialization
unscaled_loss_callback = PrintUnscaledLoss(
    target_scalers=target_scaler, 
    batch_size=batch_size,
    validation_data=(X_val, y_val_dict),
    source_tables=source_tables_val,
    encoded_4_val=encoded_4_val,
)

print_predictions_and_loss_0 = PrintPredictionsAndLoss_0(
    target_scalers=target_scaler,
    validation_data=(X_val, y_val_dict),
    source_tables=source_tables_val,
    batch_size=batch_size,
    encoded_4_val=encoded_4_val,
)

capture_callback = CaptureLayerOutputs(model=mha_model, 
                                       layer_name='adaptive_sphere_transform_layer',
                                       output_save_dir='./outputs/',
                                       sample_rate=0.1)

X_train = X_train.astype(np.float32)
symbol_ids_train = symbol_ids_train.astype(np.int32)
y_train_dict = {key: value.astype(np.float32) for key, value in y_train_dict.items()}

# Create the train data generator
train_data_generator = CustomDataGenerator(X_train, y_train_dict, symbol_ids_train, batch_size)

dashboard = RealTimeDashboard()
print_lr = PrintLR()

other_callbacks_2 = [dashboard, print_lr, unscaled_loss_callback, print_predictions_and_loss_0] 
all_callbacks = [weights_checkpoint] + other_callbacks_2

# Compile the model
loss_dict = {name: loss_function for name in output_names} 
model.compile(optimizer=optimizer, loss=loss_func)

print("Model output names:", model.output_names)
print("y_val_dict keys:", y_val_dict.keys())

for key, value in y_val_dict.items():
    print(f"{key} shape:", value.shape)
    print(f"{key} dtype:", value.dtype)

# Evaluate model immediately after loading weights (for baseline performance)
print("Evaluating baseline performance...")
val_loss = model.evaluate(X_val_inputs, y_val_dict, verbose=1)
# history = model.evaluate(X_val, y_val_dict, verbose=0)

print(f"Baseline loss on Validation Set after Loading: {val_loss}")

# Training the model using the custom data generator
print("Starting model training...")
history = model.fit(
    train_data_generator,
    epochs=epochs,
    validation_data=(X_val_inputs, y_val_dict),
    callbacks=all_callbacks,
    verbose=1
)
print("Training complete.")


# ::::::::::::::::::Evaluation 2nd Model (Save Predictions):::::::::::::::::::::: #

# Ensure source_tables_val is a NumPy array
source_tables_val = np.array(source_tables_val)

print("Starting Evaluation with predictions(.xlsx).")

# Predict model outputs
X_val_inputs = {
    'input_1': X_val,
    'input_2': symbol_ids_val
}
y_pred_list = model.predict(X_val_inputs, batch_size=batch_size)

# Ensure y_pred_list is a list
if not isinstance(y_pred_list, list):
    y_pred_list = [y_pred_list]

# Initialize source_table_data dictionary based on target_group
if target_group:
    # When targets are scaled together, we don't need to separate by source table for scaling
    source_table_data = {
        'All_Tables': {
            name: {"pred_unscaled": [], "real_unscaled": []}
            for name in output_names
        }
    }
else:
    # When targets are scaled per table, we need to separate by source table
    unique_tables = np.unique(source_tables_val.flatten() if source_tables_val.ndim > 1 else source_tables_val)
    source_table_data = {
        str(source_table).strip(): {
            name: {"pred_unscaled": [], "real_unscaled": []}
            for name in output_names
        }
        for source_table in unique_tables
    }

# Process predictions and actual values
for name, pred in zip(output_names, y_pred_list):
    valid_indices = range(len(source_tables_val))

    for i in valid_indices:
        # Handle both 1D and 2D source_tables
        source_table = source_tables_val[i] if source_tables_val.ndim == 1 else source_tables_val[i, -1]
        table_name = str(source_table).strip()

        if target_group:
            # Use the scaler from target_scalers directly
            if table_name in target_scaler.scalers:
                scaler = target_scaler.scalers[table_name]
            else:
                print(f"Warning: Scaler for table '{table_name}' not found. Skipping this entry.")
                continue
        else:
            # Use the scaler for the specific table and output
            if table_name in target_scaler.scalers:
                scaler = target_scaler.scalers[table_name]
            else:
                print(f"Warning: Scaler for table '{table_name}' not found. Skipping this entry.")
                continue

        try:
            pred_val = pred[i].reshape(1, -1)
            real_val = y_val_dict[name][i].reshape(1, -1)

            # Inverse transform predictions and actual values
            unscaled_pred = scaler.inverse_transform(pred_val).flatten()
            unscaled_real = scaler.inverse_transform(real_val).flatten()

            if target_group:
                # Store under 'All_Tables'
                source_table_data['All_Tables'][name]["pred_unscaled"].append(unscaled_pred)
                source_table_data['All_Tables'][name]["real_unscaled"].append(unscaled_real)
            else:
                # Store under the specific table
                source_table_data[table_name][name]["pred_unscaled"].append(unscaled_pred)
                source_table_data[table_name][name]["real_unscaled"].append(unscaled_real)
        except IndexError as e:
            print(f"IndexError: {e} - Skipping index {i} for table '{table_name}' and output '{name}'.")
            continue

# Save predictions and actuals to separate tabs in the Excel file
output_file = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/predictions_multi_stacked.xlsx"
with pd.ExcelWriter(output_file) as writer:
    sheet_written = False
    for table_name, output_data in source_table_data.items():
        # Initialize a DataFrame to hold all outputs' actuals and predictions
        all_output_data = pd.DataFrame()

        for name, data in output_data.items():
            if data["pred_unscaled"] and data["real_unscaled"]:
                pred_unscaled = np.array(data["pred_unscaled"]).flatten()
                real_unscaled = np.array(data["real_unscaled"]).flatten()

                # Ensure that the length of actual and predicted arrays match
                min_length = min(len(real_unscaled), len(pred_unscaled))
                real_unscaled = real_unscaled[:min_length]
                pred_unscaled = pred_unscaled[:min_length]

                # Add the actual and predicted data to the DataFrame
                all_output_data[f'Actual_{name}'] = real_unscaled
                all_output_data[f'Predicted_{name}'] = pred_unscaled

        if not all_output_data.empty:
            # Save the DataFrame to a separate sheet
            sheet_name = f"{table_name}"[:31]  # Excel sheet names are limited to 31 characters
            all_output_data.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_written = True

    # Ensure at least one sheet is written
    if not sheet_written:
        pd.DataFrame({"No Data": []}).to_excel(writer, sheet_name="No_Data", index=False)

print(f"Predictions saved to '{output_file}' with multiple tabs for each table.")











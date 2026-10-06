"""
TTTTT   TTTTT   CCCC  M   M  |  SSSS  EEEEE  QQQ
  T       T    C      MM MM  |  S     E     Q   Q
  T       T    C      M M M  |  SSSS  EEEEE Q   Q
  T       T    C      M   M  |     S  E     Q  QQ
  T       T     CCCC  M   M  |  SSSS  EEEEE  QQQ Q
_______________________________________________________ 
:Welcome to Transformer Time-series Convolutional Mixer (TTCM) | SEQ

Version 1: Designed for sequenced, time series data, which can be used for predicting continuous values or classes for stock/currency data.

Model:
- **Input Processing**: The model accepts multiple target columns, with both features and targets scaled using MinMax Scaler to normalize the data.
- **Architecture**: It utilizes a Transformer architecture featuring Multi-Head Attention layers for both encoding and decoding processes.
- **Feature Projection**: A dense layer is employed to project the feature dimensions linearly into a higher-dimensional space.
- **Temporal Pattern Capture**: Convolutional layers are integrated to effectively capture temporal patterns within the data.
- **Learning Enhancement**: Residual connections are used to enhance learning by facilitating better gradient flow.
- **Output Layer**: The final output is generated through a dense layer with softmax activation, suitable for classification tasks.

Info:
- This model is specifically designed to handle multi-target prediction tasks, with a focus on financial time-series data.
- It leverages advanced and custom deep learning techniques to accurately capture and predict complex temporal dependencies and patterns.

By: JD
"""  
import os 

"""
C++ minimum log level to filter out warnings for mixed precision 16bit 
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import json
import joblib
import sqlite3
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import multiply
from collections import defaultdict
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import RandomNormal
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, LayerNormalization 
from tensorflow.keras.layers import MultiHeadAttention, Embedding, Conv1D, Concatenate   

# Set the global mixed precision policy to 'mixed_float16'
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# Verify the current policy
print(f"Current mixed precision policy: {mixed_precision.global_policy()}")

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

# :::: Main Parameters: scroll to line 895 <Model Start> to change unit dimentions ::::

# Model and training parameters
seq_len = 4                      # Sequence length (15-minute data or any other sequence)
seq_step = 8                     # Number of steps per dequence
epochs = 10                      # Number of epochs
batch_size = 512                 # Batch size
custom_lr = True                 # Activate warmup_steps
lr_scale = 1                     # Scale for learning rate adjustment
fixed_learning_rate = 0.0001     # Fixed LR if custom_lr=False
total_samples = 8_655_360        # Actual number of data rows 1_489_408(XAUUSD)

# Calculate total steps and dynamic warmup steps
total_steps = (total_samples // batch_size) * epochs  # Total training steps
warmup_steps = int(0.05 * total_steps)

beta_1 = 0.9                      # Adam optimizer beta_1
beta_2 = 0.98                     # Adam optimizer beta_2
epsilon = 1e-9                    # Adam optimizer epsilon
amsgrad = False                   # AMSGrad variant of Adam

#:::: Training Target Predictions ::::
""" 
target_columns should be the name of any Target columns and output_names
should be the targets you want to use, which requires adjusting drop_targets
to match the output_names you selected: ['High', 'Low', 'Close', 'Signal']. Curretly only
using Signal Target of 4 possible in dataset, which is a binary classification.
"""
drop_targets = -1            # exclude the last target columns
target_columns = ['Target1', 'Target2', 'Target3', 'Target4']
output_names =   ['Signal']       

# :::: Training Loss Function ::::

loss_func = 'sparse_categorical_crossentropy' 

# :::::::::::::::::::::::::::::TensorFlow setup:::::::::::::::::::::::::::::: #

# Suppresses TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Configure GPUs
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

    def fit_transform(self, X, key):
        if key not in self.scalers:
            self.scalers[key] = MinMaxScaler()
        return self.scalers[key].fit_transform(X)

    def transform(self, X, key):
        if key not in self.scalers:
            raise ValueError(f"Scaler for key '{key}' not found.")
        return self.scalers[key].transform(X)

    def inverse_transform(self, X, key):
        if key not in self.scalers:
            raise ValueError(f"Scaler for key '{key}' not found.")
        return self.scalers[key].inverse_transform(X)

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
            # Correct the target class mapping for `Target4`
            df['Target4'] = df['Target4'].replace({5000: 1, 0: 0, 1000: 1})  # Class convertion

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
            # Correct the target class mapping for `Target4`
            df['Target4'] = df['Target4'].replace({5000: 1, 0: 0, 1000: 1})  # Class convertion

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
        encoder = existing_scalers.get('encoder')
        columns_to_encode = existing_scalers.get('columns_to_encode')
    else:
        feature_scaler = AdaptiveMinMaxScaler()
        encoder = None
        columns_to_encode = None

    # Load or initialize symbol_id mapping
    mapping_filename = 'symbol_id_mapping_2.json'
    if os.path.exists(mapping_filename):
        with open(mapping_filename, 'r') as f:
            symbol_id_mapping = json.load(f)
        symbol_id_updated = False
    else:
        symbol_id_mapping = {}
        symbol_id_updated = True  # Mapping will be updated since it doesn't exist

    for idx, (df, table_name) in enumerate(zip(data_list, source_tables)):
        print(f"Processing data from table: {table_name}")
        used_targets = [target_columns[3]]  # Use 'Target4'
        df_targets = df[used_targets].astype('float32')
        df_targets.columns = output_names  # Rename target columns

        df = df.drop(columns=target_columns, errors='ignore')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        df['source_table'] = table_name

        # Assign symbol_id based on the mapping
        if table_name in symbol_id_mapping:
            symbol_id = symbol_id_mapping[table_name]
        else:
            symbol_id = len(symbol_id_mapping)
            symbol_id_mapping[table_name] = symbol_id
            symbol_id_updated = True

        symbol_ids = np.full(len(df), symbol_id)
        df['symbol_id'] = symbol_ids

        if columns_to_encode is None:
            columns_to_encode = {
                'Day': list(range(1, 32)),
                'Month': list(range(1, 13)),
                'Hour': list(range(0, 24)),
                'Minute': list(range(0, 56)),
                'Group': [5, 15, 30, 60],
            }    
            # Add 'Pattern_1' to 'Pattern_23 and 'Pattern_Class'
            for i in range(1, 24):
                columns_to_encode[f'Pattern_{i}'] = [0, 1, 2, 3, 4, 5, 6]
            columns_to_encode['Pattern_Class'] = list(range(0, 360))

        """ # Initialize columns_to_encode if not done already
        if columns_to_encode is None:
            columns_to_encode = {
                'Day': list(range(1, 32)),
                'Month': list(range(1, 13)),
                'Hour': list(range(0, 24)),
                'Minute': list(range(0, 56)),
                'Group': [5, 15, 30, 60],
            }
            # Add 'Pattern_1' to 'Pattern_19' and 'Pattern_Class'
            for i in range(1, 20):
                columns_to_encode[f'Pattern_{i}'] = [0, 1, 2, 3, 4, 5, 6]
            columns_to_encode['Pattern_Class'] = list(range(0, 160)) """

        # Drop unnecessary columns
        """ columns_to_drop = ['New_SMA_5', 'New_SMA_20', 'New_BB_Upper', 'New_BB_Lower']
        df = df.drop(columns=columns_to_drop, errors='ignore') """

        # Drop columns 5 to 57 (inclusive) by names, but ensure 'Group' is not dropped
        columns_to_drop = df.columns[5:57].tolist()
        # Exclude 'Group' from being dropped
        columns_to_drop = [col for col in columns_to_drop if col != 'Group']
        df = df.drop(columns=columns_to_drop, errors='ignore')

        # Adjust columns_to_encode to only include columns still in df
        columns_to_encode = {col: cats for col, cats in columns_to_encode.items() if col in df.columns}

        # Encode columns using column names
        columns_to_encode_list = list(columns_to_encode.keys())
        categories = list(columns_to_encode.values())
        if encoder is None:
            encoder = OrdinalEncoder(categories=categories)
            encoded_features = encoder.fit_transform(df[columns_to_encode_list])
        else:
            encoded_features = encoder.transform(df[columns_to_encode_list])
        encoded_column_names = [f'encoded_{col}' for col in columns_to_encode_list]
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_column_names)

        # Drop the original columns that have been encoded
        df = df.drop(columns=columns_to_encode_list)

        # Drop specific encoded columns if needed
        encoded_columns_to_drop = [f'encoded_{col}' for col in ['Day', 'Month']]
        encoded_df = encoded_df.drop(columns=encoded_columns_to_drop, errors='ignore')

        # Concatenate the encoded features back to df
        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        # Update encoded_columns and remaining_columns
        encoded_columns = encoded_df.columns.tolist()
        remaining_columns = df.columns.difference(encoded_columns + ['source_table', 'symbol_id']).tolist()
        print(f"Table '{table_name}' remaining columns for scaling: {remaining_columns}")

        # Scale remaining columns using per-table scaling
        if is_training:
            df[remaining_columns] = feature_scaler.fit_transform(df[remaining_columns].astype('float32'), table_name)
        else:
            df[remaining_columns] = feature_scaler.transform(df[remaining_columns].astype('float32'), table_name)

        # Sort and balance encoded_Group groups
        if 'encoded_Group' in df.columns:
            sorted_encoded_values = sorted(df['encoded_Group'].unique())
        else:
            sorted_encoded_values = [0]

        for group in sorted_encoded_values:
            group_df = df[df['encoded_Group'] == group]
            group_df_targets = df_targets[df['encoded_Group'] == group] if 'encoded_Group' in df.columns else df_targets
            print(f"Processing Group: {group}, Group Size Row Count: {len(group_df)}")
            group_sequences_encoded = create_overlapping_sequences(group_df[encoded_columns].values, seq_len, seq_step)
            group_sequences_remaining = create_overlapping_sequences(group_df[remaining_columns].values, seq_len, seq_step)
            group_indices = group_df.index
            group_targets = group_df_targets.iloc[seq_len - 1 : len(group_sequences_encoded) + seq_len - 1].values

            # Balance the groups during training & validation
            if is_training or not is_training:
                unique_targets, counts_per_target = np.unique(group_targets, return_counts=True)
                min_count = np.min(counts_per_target)
                print(f"Before balancing: {dict(zip(unique_targets, counts_per_target))}")
                print(f"Group {group}: Balancing to {min_count} sequences per target...")
                balanced_group_sequences_encoded = []
                balanced_group_sequences_remaining = []
                balanced_group_targets = []

                for target_value in unique_targets:
                    target_indices = np.where(group_targets == target_value)[0]

                    if len(target_indices) > min_count:
                        target_indices = np.random.choice(target_indices, size=min_count, replace=False)

                    balanced_group_sequences_encoded.append(group_sequences_encoded[target_indices])
                    balanced_group_sequences_remaining.append(group_sequences_remaining[target_indices])
                    balanced_group_targets.append(group_targets[target_indices])

                group_sequences_encoded = np.concatenate(balanced_group_sequences_encoded, axis=0)
                group_sequences_remaining = np.concatenate(balanced_group_sequences_remaining, axis=0)
                group_targets = np.concatenate(balanced_group_targets, axis=0)
                print(f"After balancing: {np.unique(group_targets, return_counts=True)}")

            all_X_encoded.append(group_sequences_encoded)
            all_X_remaining.append(group_sequences_remaining)
            for i, name in enumerate(output_names):
                all_y[name].append(group_targets)

            all_encoded_4.append(np.full(group_sequences_encoded.shape[0], group))
            all_source_tables.append(np.full(group_sequences_encoded.shape[0], table_name))
            all_symbol_ids.append(np.full(group_sequences_encoded.shape[0], symbol_id))

            # Clear variables to free memory
            del group_df, group_df_targets, group_sequences_encoded, group_sequences_remaining, group_targets

        # Clear variables to free memory
        del df, df_targets, encoded_df

    # Save the updated symbol_id_mapping if in training mode and mapping has been updated
    if is_training and symbol_id_updated:
        with open(mapping_filename, 'w') as f:
            json.dump(symbol_id_mapping, f)
        print(f"Symbol ID mapping saved to {mapping_filename}")

    return all_X_encoded, all_X_remaining, all_y, all_encoded_4, all_symbol_ids, feature_scaler, encoder, columns_to_encode, all_source_tables

def scale_combined_data(all_X_encoded, all_X_remaining, all_y, all_encoded_4, all_symbol_ids, 
                        all_source_tables, feature_scaler=None, is_training=True):
    print("Scaling encoded data from all tables")
    
    # Combine data
    combined_X_encoded = np.concatenate(all_X_encoded, axis=0)
    combined_X_remaining = np.concatenate(all_X_remaining, axis=0)
    combined_y = {name: np.concatenate(values, axis=0).astype(np.float32) 
                  for name, values in all_y.items() if len(values) > 0}
    combined_encoded_4 = np.concatenate(all_encoded_4, axis=0)
    combined_source_tables = np.concatenate(all_source_tables, axis=0)
    combined_symbol_ids = np.concatenate(all_symbol_ids, axis=0)

    # Clear memory
    del all_X_encoded, all_X_remaining, all_y, all_encoded_4, all_source_tables, all_symbol_ids

    # Reshape and scale encoded part
    num_encoded_columns = combined_X_encoded.shape[2]
    encoded_part_reshaped = combined_X_encoded.reshape(-1, num_encoded_columns)
    
    if is_training:
        # Fit and transform the encoded features
        scaled_encoded = feature_scaler.fit_transform(encoded_part_reshaped, 'combined')
    else:
        # Transform using the existing scaler
        scaled_encoded = feature_scaler.transform(encoded_part_reshaped, 'combined')
    
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
df_train, source_tables_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//Processed_multi_stacked_HLC_Train_2.db")
df_val, source_tables_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//Processed_multi_stacked_HLC_Val_2.db")

# Preprocess and combine training data
print("Starting to preprocess training data...")
all_X_train_encoded, all_X_train_remaining, all_y_train, all_encoded_4_train, all_symbol_ids_train, feature_scaler, encoder, columns_to_encode, source_tables_train = preprocess_and_combine_data(
    df_train, source_tables_train, seq_len, seq_step, target_columns, output_names, is_training=True
)

# Print out the shape of the feature arrays
print("Encoded feature shape:", all_X_train_encoded[0].shape)
print("Remaining feature shape:", all_X_train_remaining[0].shape)

# Scale and combine all training data
print("Scaling and combining all training data...")
X_train, y_train_dict, encoded_4_train, symbol_ids_train, source_tables_train, feature_scaler = scale_combined_data(
    all_X_train_encoded, all_X_train_remaining, all_y_train, all_encoded_4_train, all_symbol_ids_train, source_tables_train, feature_scaler=feature_scaler, is_training=True
)

# Randomize the training data
X_train, y_train_dict, source_tables_train, symbol_ids_train = randomize_data(X_train, y_train_dict, source_tables_train, symbol_ids_train)
print(f"Final training data shapes - X: {X_train.shape}, y: {[y.shape for y in y_train_dict.values()]}, source_tables: {len(source_tables_train)}, symbol_ids: {len(symbol_ids_train)}\n")

# Save the scalers and encoder for future use
print("Saving the scalers and encoder for future use...")
if not os.path.exists('scalers'):
    os.makedirs('scalers')
joblib.dump({
    'feature_scaler': feature_scaler,
    'encoder': encoder,
    'columns_to_encode': columns_to_encode,
    'seq_len': seq_len,
    'seq_step': seq_step
}, 'scalers/scalers_stacked_multi_main_2.pkl')

# Load the scalers and encoder for validation data
scalers_encoder = joblib.load('scalers/scalers_stacked_multi_main_2.pkl')
feature_scaler = scalers_encoder['feature_scaler']
encoder = scalers_encoder['encoder']
columns_to_encode = scalers_encoder['columns_to_encode']
seq_len = scalers_encoder['seq_len']
seq_step = scalers_encoder['seq_step']

# Preprocess and combine validation data
print("Starting to preprocess validation data...")
all_X_val_encoded, all_X_val_remaining, all_y_val, all_encoded_4_val, all_symbol_ids_val, feature_scaler, encoder, columns_to_encode, source_tables_val = preprocess_and_combine_data(
    df_val, source_tables_val, seq_len, seq_step, target_columns, output_names, is_training=False,
    existing_scalers={'feature_scaler': feature_scaler, 'encoder': encoder, 'columns_to_encode': columns_to_encode}
)

# Scale and combine validation data
print("Scaling and combining validation data...")
X_val, y_val_dict, encoded_4_val, symbol_ids_val, source_tables_val, feature_scaler = scale_combined_data(
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
    
# :::::::::::::::::::::::::::Custom Transformer::::::::::::::::::::::::::::::: #

# Symbol Embedding Layer to capture symbol-specific patterns
class SymbolEmbedding(Layer):
    def __init__(self, num_symbols, embed_dim, **kwargs):
        super(SymbolEmbedding, self).__init__(**kwargs)
        self.num_symbols = num_symbols
        self.embed_dim = embed_dim
        self.symbol_embeddings = Embedding(num_symbols, embed_dim)
    
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
class CustomMultiHeadAttention(Layer):
    def __init__(self, embed_dim, num_heads, **kwargs):
        super(CustomMultiHeadAttention, self).__init__(**kwargs)
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.attention_heads = MultiHeadAttention(num_heads, embed_dim)
        self.layernorm = LayerNormalization(epsilon=1e-6)

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
class CustomTransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate, **kwargs):
        super(CustomTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = CustomMultiHeadAttention(embed_dim, num_heads)
        self.ffn = Sequential([
            Dense(ff_dim, activation="swish", use_bias=True, kernel_initializer='he_normal'),
            Dense(embed_dim, use_bias=False, kernel_initializer='he_normal')
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)

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
class Encoder(Layer):
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
            x = x 
        return x

# Decoder block for transformer, adds original input back at each layer
class Decoder(Layer):
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
            x = x

        # Cross-attention between decoder input and encoder output
        cross_attn_output = self.cross_attention(x, enc_output)
        cross_attn_output = cross_attn_output + original_inputs  
        return cross_attn_output

# Market-Aware Transformer Model with original inputs added back at each layer
class MarketAwareTransformer(Model):
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
        self.positional_encoding = PositionalEncoding(seq_len, embed_dim)

        # Encoder and Decoder
        self.encoder = Encoder(num_enc_layers, embed_dim, num_heads, ff_dim, dropout_rate)
        self.decoder = Decoder(num_dec_layers, embed_dim, num_heads, ff_dim, dropout_rate)

    def call(self, inputs, symbol_id, training=False):
        # Symbol-specific embedding
        symbol_embed = self.symbol_embedding(symbol_id)
        
        # positional encoding to inputs
        x = self.positional_encoding(inputs)
        #x = self.pos_encoding(x)

        # Save original inputs to add back at each block
        original_inputs = x

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
    
def build_conv_layers(mha_model, filters, kernel_sizes, seq_len, use_bias=False):

    num_layers = min(len(filters), len(kernel_sizes))  # Automatically determine number of layers

    # Build the convolutional layers dynamically
    for i in range(num_layers):
        mha_model = Conv1D(
            filters=filters[i],
            kernel_size=kernel_sizes[i],
            activation='swish',
            kernel_initializer='he_normal',
            padding='same',
            use_bias=use_bias
        )(mha_model)

        # Apply Layer Normalization after each Conv1D layer
        mha_model = LayerNormalization(epsilon=1e-6)(mha_model)
    
    return mha_model

# ::::::::::::::::::::::::Main Model Parameters::::::::::::::::::::::::::::: #

# Load symbol_id mapping
with open('symbol_id_mapping_2.json', 'r') as f:
    symbol_id_mapping = json.load(f)

# Assuming your dataset has a shape of (num_samples, seq_len, d_model)
seq_len = X_train.shape[1]            # Time-series length
d_model = X_train.shape[2]            # Number of features (input dim per time step)

# Use the number of unique symbol IDs directly
num_symbols = len(symbol_id_mapping)
num_enc_layers = 6                    # Encoding Attention layers
num_dec_layers = 6                    # Decoding Attention layers
num_heads = 3                         # Number of attention heads
dropout_rate = 0.01                   # Dropout rate for regularization
ff_dim = d_model*4                    # Feedforward network dimension

# filter and kernel size values
filters = [64, 128, 128] 
kernel_sizes = [2, 4, 4] 

d_ffd = filters[-1]                   # Input Dense layer dimentions

# Input layer for time-series data (shape: (seq_len, d_model))
inputs = Input(shape=(seq_len, d_model), name='input_1')

# Input layer for symbol IDs (categorical)
symbol_id_input = Input(shape=(1,), dtype=tf.int32, name='input_2')

# ::::::::::::::Main Model Training (Multi Target Prediction):::::::::::::: #

outputs = []
for name in output_names: 

    projected_inputs = Dense(d_model, kernel_initializer='he_normal', use_bias=False)(inputs)
    mha_model_inputs = Concatenate(axis=-1)([projected_inputs, inputs])
    d_model_updated = mha_model_inputs.shape[-1]   
    
    market_aware_transformer = MarketAwareTransformer(
        num_symbols=num_symbols,
        seq_len=seq_len,
        embed_dim=d_model_updated,             # Embedding dimension same as input feature dimension
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_enc_layers=num_enc_layers, 
        num_dec_layers=num_dec_layers,
        dropout_rate=dropout_rate,
    )

    mha_model = market_aware_transformer(mha_model_inputs, symbol_id_input) 

    #mha_model = AdaptiveSphereTransformLayer()(mha_model)  
    mha_model = build_conv_layers(mha_model, filters, kernel_sizes, 
                                  seq_len, use_bias=False)   

    #mha_model = AdaptiveSphereTransformLayer()(mha_model)
    pool_1 = GlobalAveragePooling1D()(mha_model)
    pool_2 = GlobalMaxPooling1D()(mha_model)

    mha_model = multiply(pool_1, pool_2)

    mha_model = Dense(d_ffd, activation='swish', kernel_initializer='he_normal', use_bias=True)(mha_model) 
    mha_model = Dense(d_model_updated, kernel_initializer='he_normal', use_bias=False)(mha_model)    
    mha_model = Dropout(0.3)(mha_model)
    mha_model = LayerNormalization(epsilon=1e-6)(mha_model)

    # Output Layer
    mha_model = Dense(
        2,  # 3 classes made into 2
        activation='softmax',  # Softmax activation 
        kernel_initializer='he_normal',
        use_bias=False,
        name=name,
        dtype='float32'
    )(mha_model)
    
    outputs.append(mha_model)
model = Model([inputs, symbol_id_input], outputs=outputs)

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
    'PositionalEncoding': PositionalEncoding,
    'SymbolEmbedding': SymbolEmbedding,
    'CustomMultiHeadAttention': CustomMultiHeadAttention,
    'CustomTransformerBlock': CustomTransformerBlock,
    'MarketAwareTransformer': MarketAwareTransformer,
    'AdaptiveSphereTransformLayer': AdaptiveSphereTransformLayer,
    'build_conv_layers': build_conv_layers,
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

# Custom learning rate schedule with warm-up
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
    def __init__(self, validation_data, source_tables, batch_size, encoded_4_val):
        super().__init__()
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
        random_indices = np.random.choice(num_samples, 10, replace=False)

        source_tables_sampled = self.source_tables[random_indices].astype(str)

        for i, output_name in enumerate(self.model.output_names):
            pred = predictions[i]
            y_val = y_val_dict[output_name]

            pred_sampled = pred[random_indices]
            y_val_sampled = y_val[random_indices]

            pred_classes = np.argmax(pred_sampled, axis=1)  # still need np.argmax for predictions
            true_classes = y_val_sampled.ravel()  # Flatten the real values to print them horizontally

            print(f"Epoch {epoch + 1} - First 5 Validation Predictions for {output_name}: {pred_classes[:10]}")
            print(f"Epoch {epoch + 1} - First 5 Validation Real Values for {output_name}: {true_classes[:10]}")
            print(f"Corresponding source tables: {source_tables_sampled[:5]}")
            print()

class PrintUnscaledLoss(Callback):
    def __init__(self, batch_size, validation_data, source_tables, encoded_4_val):
        super().__init__()
        self.batch_size = batch_size
        self.validation_data = validation_data
        self.source_tables = source_tables
        self.encoded_4_val = encoded_4_val
        self.source_table_mapping = {table: i for i, table in enumerate(np.unique(self.source_tables))}
        self.mapped_source_tables = np.array([self.source_table_mapping[table] for table in self.source_tables])

    def on_epoch_end(self, epoch, logs=None):
        X_val, y_val_dict = self.validation_data
        X_val_inputs = {'input_1': X_val, 'input_2': self.mapped_source_tables}
        predictions = self.model.predict(X_val_inputs, batch_size=self.batch_size)
        if not isinstance(predictions, list):
            predictions = [predictions]

        combined_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"pred": [], "real": []})))
        all_pred_classes = []
        all_real_classes = []
        group_size = defaultdict(int)

        for i, output_name in enumerate(self.model.output_names):
            pred = predictions[i]
            y_val = y_val_dict[output_name]

            for idx in range(len(pred)):
                source_table = self.source_tables[idx]
                group = self.encoded_4_val[idx]
                pred_class = np.argmax(pred[idx])
                real_class = y_val[idx][0]  # Assuming y_val is shape (n, 1) for sparse categorical

                combined_results[source_table][output_name][group]["pred"].append(pred_class)
                combined_results[source_table][output_name][group]["real"].append(real_class)
                all_pred_classes.append(pred_class)
                all_real_classes.append(real_class)
                group_size[(source_table, group)] += 1

        for source_table, output_data in combined_results.items():
            for output_name, group_data in output_data.items():
                for group in sorted(group_data.keys()):
                    data = group_data[group]
                    if data["pred"] and data["real"]:
                        pred_classes = np.array(data["pred"])
                        real_classes = np.array(data["real"])
                        accuracy = np.mean(pred_classes == real_classes)
                        print(f'Epoch {epoch + 1} - Validation {output_name} Accuracy for {source_table}, Group {group}: {accuracy:.4f}')

        all_pred_classes = np.array(all_pred_classes)
        all_real_classes = np.array(all_real_classes)
        overall_accuracy = np.mean(all_pred_classes == all_real_classes)
        print(f'Epoch {epoch + 1} - Overall Validation Accuracy: {overall_accuracy:.4f}')

        total_samples = sum(group_size.values())
        weighted_accuracy_sum = 0.0
        for (source_table, group), size in group_size.items():
            for output_name, group_data in combined_results[source_table].items():
                data = group_data.get(group)
                if data and data["pred"] and data["real"]:
                    pred_classes = np.array(data["pred"])
                    real_classes = np.array(data["real"])
                    group_accuracy = np.mean(pred_classes == real_classes)
                    weighted_accuracy_sum += group_accuracy * size

        if total_samples > 0:
            weighted_accuracy = weighted_accuracy_sum / total_samples
            print(f'Epoch {epoch + 1} - Weighted Validation Accuracy: {weighted_accuracy:.4f}')

# Custom Data Generator for Training
class CustomTrainDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_train_inputs, y_train_dict, batch_size, shuffle=True):
        self.X_train_inputs = X_train_inputs
        self.y_train_dict = y_train_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.X_train_inputs['input_1'].shape[0])
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.X_train_inputs['input_1']) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices for the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data for the batch
        X_batch = {
            'input_1': self.X_train_inputs['input_1'][batch_indices],
            'input_2': self.X_train_inputs['input_2'][batch_indices]
        }
        y_batch = {key: value[batch_indices] for key, value in self.y_train_dict.items()}

        return X_batch, y_batch

    def on_epoch_end(self):
        # Shuffle the indices after each epoch
        if self.shuffle:
            np.random.shuffle(self.indices)

# Custom Data Generator for Validation (no shuffle)
class CustomValDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X_val_inputs, y_val_dict, batch_size, shuffle=False):
        self.X_val_inputs = X_val_inputs
        self.y_val_dict = y_val_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(self.X_val_inputs['input_1'].shape[0])

    def __len__(self):
        return int(np.floor(len(self.X_val_inputs['input_1']) / self.batch_size))

    def __getitem__(self, index):
        # Generate indices for the batch
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data for the batch
        X_batch = {
            'input_1': self.X_val_inputs['input_1'][batch_indices],
            'input_2': self.X_val_inputs['input_2'][batch_indices]
        }
        y_batch = {key: value[batch_indices] for key, value in self.y_val_dict.items()}

        return X_batch, y_batch

# ::::::::::::::::::::::::Training & Validation Loop:::::::::::::::::::::::::::: #

# Initialize optimizer
learning_rate = CustomSchedule(d_model, warmup_steps, custom_lr, lr_scale)
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)

X_val = X_val.astype(np.float32)
y_val_dict = {key: value.astype(np.int32) for key, value in y_val_dict.items()}
symbol_ids_val = symbol_ids_val.astype(np.int32)

X_val_inputs = {
    'input_1': X_val,
    'input_2': symbol_ids_val
}

# Debugging prints before prediction
print("X_val shape:", X_val.shape)
print("Processed Source Tables Val:", set(source_tables_val))

# Callbacks initialization
unscaled_loss_callback = PrintUnscaledLoss( 
    batch_size=batch_size,
    validation_data=(X_val, y_val_dict),
    source_tables=source_tables_val,
    encoded_4_val=encoded_4_val,
)

print_predictions_and_loss_0 = PrintPredictionsAndLoss_0(
    validation_data=(X_val, y_val_dict),
    source_tables=source_tables_val,
    batch_size=batch_size,
    encoded_4_val=encoded_4_val,
)

X_train = X_train.astype(np.float32)
y_train_dict = {key: value.astype(np.int32) for key, value in y_train_dict.items()}
symbol_ids_train = symbol_ids_train.astype(np.int32)

X_train_inputs = {
    'input_1': X_train,
    'input_2': symbol_ids_train
}

# Debugging prints before prediction
print("X_train shape:", X_train.shape)
print("Processed Source Tables Train:", set(source_tables_train))

dashboard = RealTimeDashboard()
print_lr = PrintLR()

other_callbacks_2 = [dashboard, print_lr, unscaled_loss_callback, print_predictions_and_loss_0] 
all_callbacks = [weights_checkpoint] + other_callbacks_2

# Compile the model
model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])

print("Model output names:", model.output_names)
print("y_val_dict keys:", y_val_dict.keys())

for key, value in y_val_dict.items():
    print(f"{key} shape:", value.shape)
    print(f"{key} dtype:", value.dtype)

# Evaluate model immediately after loading weights (for baseline performance)
print("Evaluating baseline performance...")
val_loss = model.evaluate(X_val_inputs, y_val_dict, batch_size=batch_size, verbose=1)

print(f"Baseline loss on Validation Set after Loading: {val_loss}")

# Instantiate data generators
train_generator = CustomTrainDataGenerator(X_train_inputs, y_train_dict, batch_size=batch_size, shuffle=True)
val_generator = CustomValDataGenerator(X_val_inputs, y_val_dict, batch_size=batch_size, shuffle=False)

# Training the model using the custom data generator
print("Starting model training...")
history = model.fit(
    x=train_generator,  # Use the custom training generator
    epochs=epochs,
    validation_data=val_generator,  # Use the custom validation generator
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

# Initialize source_table_data dictionary
unique_tables = np.unique(source_tables_val.flatten() if source_tables_val.ndim > 1 else source_tables_val)
source_table_data = {
    str(source_table).strip(): {
        name: {"pred_classes": [], "real_classes": []}
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

        try:
            # Get predicted class by taking the argmax of the softmax predictions
            pred_class = np.argmax(pred[i], axis=-1)
            
            # Directly use the real class since the labels are sparse integers
            real_class = y_val_dict[name][i]  # No need for np.argmax

            # Store the predicted and real classes
            source_table_data[table_name][name]["pred_classes"].append(pred_class)
            source_table_data[table_name][name]["real_classes"].append(real_class)
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
            if data["pred_classes"] and data["real_classes"]:
                pred_classes = np.array(data["pred_classes"]).flatten()
                real_classes = np.array(data["real_classes"]).flatten()

                # Ensure that the length of actual and predicted arrays match
                min_length = min(len(real_classes), len(pred_classes))
                real_classes = real_classes[:min_length]
                pred_classes = pred_classes[:min_length]

                # Add the actual and predicted data to the DataFrame
                all_output_data[f'Actual_{name}'] = real_classes
                all_output_data[f'Predicted_{name}'] = pred_classes

        if not all_output_data.empty:
            # Save the DataFrame to a separate sheet
            sheet_name = f"{table_name}"[:31]  # Excel sheet names are limited to 31 characters
            all_output_data.to_excel(writer, sheet_name=sheet_name, index=False)
            sheet_written = True

    # Ensure at least one sheet is written
    if not sheet_written:
        pd.DataFrame({"No Data": []}).to_excel(writer, sheet_name="No_Data", index=False)

print(f"Predictions saved to '{output_file}' with multiple tabs for each table.")












"""


By: JD (ThinkBe)
"""    
import os 

"""
C++ minimum log level to filter out warnings for mixed precision 16bit 
"""
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import gc 
import re
import sys
import glob
import json
import joblib
import sqlite3
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
from collections import defaultdict
import tensorflow.keras.backend as K
from tensorflow.keras import mixed_precision
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC, Precision, Recall
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import MultiHeadAttention, Embedding
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, SimpleRNN
from tensorflow.keras.layers import LayerNormalization, Concatenate, Add, RNN
from tensorflow.keras.layers import Multiply

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

# :::: Main Parameters: scroll to line 1236 <Model Start> to change unit dimentions ::::

# Model and training parameters
seq_len = 8                        # Sequence length (15-minute data or any other sequence)
epochs = 16                         # When 0, skips training and goes to evaluation
batch_size = 2056                   # Batch size
custom_lr = True                    # Activate training warmup_steps 
lr_scale = 1                        # Scale for learning rate adjustment
fixed_learning_rate = 0.00001945325 # Fixed LR if custom_lr=False
elip_len = 1e-6                     # Epsilon for layer normalization (default=1e-6)

beta_1 = 0.9                        # Default value for Adam optimizer beta_1 in TensorFlow
beta_2 = 0.98                       # Default value for Adam optimizer beta_2 in TensorFlow
epsilon = 1e-9                      # Default value for Adam optimizer epsilon in TensorFlow
amsgrad = False                     # Default value for AMSGrad variant of Adam in TensorFlow

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

loss_func = 'binary_crossentropy' 
#loss_func = 'categorical_crossentropy' 

log_file_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/training_progress/console_output_testing_new.txt"
class Tee(object):
    def __init__(self, name, mode):
        self.file = open(name, mode)
        self.stdout = sys.stdout

    def __del__(self):
        self.close()  # Ensure file is closed upon deletion

    def close(self):
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)
        self.flush()

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def isatty(self):
        return self.stdout.isatty()
    
sys.stdout = sys.stderr = Tee(log_file_path, 'w')

class Tee2(object):
       def __init__(self, stdout, file):
           self.stdout = stdout
           self.file = file

       def write(self, data):
           self.stdout.write(data)
           self.file.write(data)

       def flush(self):
           self.stdout.flush()
           self.file.flush()

       def close(self):
           self.file.close()

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
        
        # Clean Target4 data
        if 'Target4' in df.columns:
            # Drop rows with NaN in Target4
            df = df.dropna(subset=['Target4'])
            
            # Handle string type Target4
            if df['Target4'].dtype == 'object':
                df = df[df['Target4'].str.strip() != '']
            
            # Reset index after filtering
            df = df.reset_index(drop=True)
            
            # Convert Target4 to numeric and map values
            df['Target4'] = pd.to_numeric(df['Target4'], errors='coerce')
            df['Target4'] = df['Target4'].replace({5000: 0, 1000: 1})
            
            # Final cleanup of any remaining NaN values
            df = df.dropna(subset=['Target4'])
            df = df.reset_index(drop=True)

        all_data.append(df)
        source_tables.append(table_name)
    
    conn.close() 
    print("Data loading complete.")
    return all_data, source_tables

def preprocess_and_combine_data(data_list, source_tables, target_columns, output_names, 
                                is_training=True, existing_scalers=None):
    print("Preprocessing and combining data...")

    all_X_encoded = []
    all_X_remaining = []
    all_y = {name: [] for name in output_names}
    all_encoded_4 = []
    all_symbol_ids = []
    all_source_tables = []

    ########################################################################
    # NEW: Instead of a single mapped_classes, build a global list
    ########################################################################
    all_pattern_classes = []

    # Initialize or retrieve scalers/encoder
    if existing_scalers:
        feature_scaler = existing_scalers.get('feature_scaler')
        encoded_scaler = existing_scalers.get('encoded_scaler')
        encoder = existing_scalers.get('encoder')
        columns_to_encode = existing_scalers.get('columns_to_encode')
    else:
        feature_scaler = AdaptiveMinMaxScaler()
        encoded_scaler = AdaptiveMinMaxScaler()
        encoder = None
        columns_to_encode = None

    # Load or initialize symbol_id and pattern_class mappings
    mapping_filename = 'symbol_id_mapping_small_main_11.json'
    if os.path.exists(mapping_filename):
        with open(mapping_filename, 'r') as f:
            symbol_id_mapping = json.load(f)
        symbol_id_updated = False
    else:
        symbol_id_mapping = {}
        symbol_id_updated = True

    pattern_class_mapping_filename = 'pattern_class_mapping_small_main_11.json'
    if os.path.exists(pattern_class_mapping_filename):
        with open(pattern_class_mapping_filename, 'r') as f:
            pattern_class_mapping = json.load(f)
        pattern_class_updated = False
    else:
        pattern_class_mapping = {}
        pattern_class_updated = True

    for idx, (df, table_name) in enumerate(zip(data_list, source_tables)):
        print(f"Processing data from table: {table_name}")
        used_targets = [target_columns[3]]  # e.g., 'Target4'
        df_targets = df[used_targets].astype('float32')
        df_targets.columns = output_names

        # Drop target columns
        df = df.drop(columns=target_columns, errors='ignore')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        df['source_table'] = table_name

        # Symbol ID
        if table_name in symbol_id_mapping:
            symbol_id = symbol_id_mapping[table_name]
        else:
            symbol_id = len(symbol_id_mapping)
            symbol_id_mapping[table_name] = symbol_id
            symbol_id_updated = True
        symbol_ids = np.full(len(df), symbol_id)
        df['symbol_id'] = symbol_ids

        # Pattern_Class mapping
        if 'Pattern_Class' in df.columns:
            df['Pattern_Class'] = df['Pattern_Class'].astype(int)
            mapped_classes = []
            for pc in df['Pattern_Class']:
                if pc not in pattern_class_mapping:
                    next_idx = len(pattern_class_mapping)
                    pattern_class_mapping[pc] = next_idx
                    pattern_class_updated = True
                mapped_classes.append(pattern_class_mapping[pc])
            df['pattern_class_id'] = mapped_classes
        else:
            df['pattern_class_id'] = 0

        # Initialize columns_to_encode if needed
        if columns_to_encode is None:
            columns_to_encode = {
                'Day': list(range(1, 32)),
                'Month': list(range(1, 13)),
                'Hour': list(range(0, 24)),
                'Minute': list(range(0, 56)),
                'Group': [5, 15, 30, 60],
                'New_Pattern_Sum': list(range(0, 234)),
            }
            for i in range(1, 26):
                columns_to_encode[f'Pattern_{i}'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
            columns_to_encode['Pattern_Class'] = list(range(360)) # large 9216 | small 360

        # Optionally remove the original 'Pattern_Class':
        """ if 'Pattern_Class' in df.columns:
            df.drop(columns=['Pattern_Class'], inplace=True) """

        # Drop columns 5 to 57 (pricing data) if desired, but keep 'Group'
        """ columns_to_drop = df.columns[5:57].tolist()
        columns_to_drop = [col for col in columns_to_drop if col != 'Group']
        df = df.drop(columns=columns_to_drop, errors='ignore') """

        # Actual encoding steps
        columns_to_encode = {col: cats for col, cats in columns_to_encode.items() if col in df.columns}
        columns_to_encode_list = list(columns_to_encode.keys())
        categories = list(columns_to_encode.values())

        if encoder is None:
            encoder = OrdinalEncoder(categories=categories)
            encoded_features = encoder.fit_transform(df[columns_to_encode_list])
        else:
            encoded_features = encoder.transform(df[columns_to_encode_list])

        encoded_column_names = [f'encoded_{col}' for col in columns_to_encode_list]
        encoded_df = pd.DataFrame(encoded_features, columns=encoded_column_names)
        df = df.drop(columns=columns_to_encode_list)

        # Drop specific encoded columns if needed
        """ encoded_columns_to_drop = [f'encoded_{col}' for col in ['Day', 'Month']]
        encoded_df = encoded_df.drop(columns=encoded_columns_to_drop, errors='ignore') """

        df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

        encoded_columns = encoded_df.columns.tolist()
        remaining_columns = df.columns.difference(encoded_columns + ['source_table', 'symbol_id']).tolist()
        
        # Scale remaining columns per table
        if is_training:
            df[remaining_columns] = feature_scaler.fit_transform(df[remaining_columns].astype('float32'), table_name)
        else:
            df[remaining_columns] = feature_scaler.transform(df[remaining_columns].astype('float32'), table_name)
        
        # Scale remaining columns globally
        remaining_key = 'global_remaining_columns'
        if is_training:
            df[remaining_columns] = feature_scaler.fit_transform(
                df[remaining_columns].astype('float32'),
                remaining_key
            )
        else:
            df[remaining_columns] = feature_scaler.transform(
                df[remaining_columns].astype('float32'),
                remaining_key
            )

        """ # Scale encoded columns per table (like remaining columns)
        if encoded_columns:
            # Optionally exclude 'encoded_Group'
            if 'encoded_Group' in encoded_columns:
                encoded_columns_no_grp = [col for col in encoded_columns if col != 'encoded_Group']
            else:
                encoded_columns_no_grp = encoded_columns

            if is_training:
                df[encoded_columns_no_grp] = encoded_scaler.fit_transform(
                    df[encoded_columns_no_grp].astype('float32'), table_name
                )
            else:
                df[encoded_columns_no_grp] = encoded_scaler.transform(
                    df[encoded_columns_no_grp].astype('float32'), table_name
                )
            if 'encoded_Group' in df.columns:
                df['encoded_Group'] = df['encoded_Group'].astype('int32') """

        # Scale encoded columns globally
        if encoded_columns:
            encoded_key = 'global_encoded_columns'
            # Optionally exclude 'encoded_Group'
            if 'encoded_Group' in encoded_columns:
                encoded_columns_no_grp = [col for col in encoded_columns if col != 'encoded_Group']
            else:
                encoded_columns_no_grp = encoded_columns

            if is_training:
                df[encoded_columns_no_grp] = encoded_scaler.fit_transform(
                    df[encoded_columns_no_grp].astype('float32'), encoded_key
                )
            else:
                df[encoded_columns_no_grp] = encoded_scaler.transform(
                    df[encoded_columns_no_grp].astype('float32'), encoded_key
                )
            if 'encoded_Group' in df.columns:
                df['encoded_Group'] = df['encoded_Group'].astype('int32')

        # Group by 'encoded_Group' if it exists
        grouped_values = df['encoded_Group'].unique() if 'encoded_Group' in df.columns else [0]
        for group in sorted(grouped_values):
            if 'encoded_Group' in df.columns:
                group_df = df[df['encoded_Group'] == group]
                group_df_targets = df_targets[df['encoded_Group'] == group]
            else:
                group_df = df
                group_df_targets = df_targets

            # Convert to numpy
            encoded_array = group_df[encoded_columns].values.reshape(-1, 1, len(encoded_columns))
            remaining_array = group_df[remaining_columns].values.reshape(-1, 1, len(remaining_columns))
            group_targets = group_df_targets.values
            # pattern_class array
            group_pattern = group_df['pattern_class_id'].values  # shape (N,)

            # Append to global lists
            all_X_encoded.append(encoded_array)
            all_X_remaining.append(remaining_array)

            for out_i, name in enumerate(output_names):
                if len(output_names) == 1:
                    all_y[name].append(group_targets.ravel())
                else:
                    all_y[name].append(group_targets[:, out_i].ravel())

            all_encoded_4.append(np.full(encoded_array.shape[0], group))
            all_source_tables.append(np.array([table_name]*encoded_array.shape[0], dtype=object))
            all_symbol_ids.append(group_df['symbol_id'].values)

            ########################################################################
            # APPEND pattern_class IDs to a global list, matching row counts
            ########################################################################
            all_pattern_classes.append(group_pattern)

        del df, df_targets, encoded_df

    # Save updates to mapping
    if is_training and symbol_id_updated:
        with open(mapping_filename, 'w') as f:
            json.dump(symbol_id_mapping, f)
        print(f"Symbol ID mapping saved to {mapping_filename}")

    if is_training and pattern_class_updated:
        with open(pattern_class_mapping_filename, 'w') as f:
            json.dump(pattern_class_mapping, f)
        print(f"Pattern Class mapping saved to {pattern_class_mapping_filename}")

    return (
        all_X_encoded,
        all_X_remaining,
        all_y,
        all_encoded_4,
        all_symbol_ids,
        feature_scaler,
        encoded_scaler,
        encoder,
        columns_to_encode,
        all_source_tables,
        ########################################################################
        # Instead of a single 'mapped_classes', return all_pattern_classes for all data
        ########################################################################
        all_pattern_classes
    )

def combined_data(all_X_encoded, all_X_remaining, all_y, all_encoded_4, all_symbol_ids, 
                        all_source_tables, feature_scaler=None, is_training=True):
    print("Combining scaled data from all tables")
    
    # Calculate total size needed
    total_rows = sum(x.shape[0] for x in all_X_encoded)
    feat_encoded = all_X_encoded[0].shape[2]
    feat_remaining = all_X_remaining[0].shape[2]
    
    # Pre-allocate final array
    final_X = np.empty((total_rows, all_X_encoded[0].shape[1], 
                       feat_encoded + feat_remaining), dtype=np.float32)
    
    # Copy data in chunks
    current_idx = 0
    for enc, rem in zip(all_X_encoded, all_X_remaining):
        rows = enc.shape[0]
        final_X[current_idx:current_idx + rows, :, :feat_encoded] = enc
        final_X[current_idx:current_idx + rows, :, feat_encoded:] = rem
        current_idx += rows
        
        # Clear references immediately
        del enc, rem
    
    # Clear original lists
    del all_X_encoded, all_X_remaining
    gc.collect()  # Force garbage collection

    # Handle smaller concatenations
    combined_y = {name: np.concatenate(values, axis=0).astype(np.float32) 
                 for name, values in all_y.items()}
    combined_encoded_4 = np.concatenate(all_encoded_4, axis=0)
    combined_source_tables = np.concatenate(all_source_tables, axis=0)
    combined_symbol_ids = np.concatenate(all_symbol_ids, axis=0)

    return final_X, combined_y, combined_encoded_4, combined_symbol_ids, combined_source_tables, feature_scaler

if epochs > 0:
    print("About to load and preprocess training data...")
    df_train, source_tables_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//new_multi_stacked_XAUUSD_Train_11_new.db")

    # Preprocess training data
    print("Starting to preprocess training data...")
    (all_X_train_encoded, all_X_train_remaining, all_y_train, all_encoded_4_train,
     all_symbol_ids_train, feature_scaler, encoded_scaler, encoder, columns_to_encode,
     source_tables_train, all_pattern_classes_train) = preprocess_and_combine_data(
         df_train, source_tables_train, target_columns, output_names, is_training=True
    )  # Make sure your function signature no longer returns 'all_unused_majority_*'.

    # Scale training data
    print("Scaling and combining all training data...")
    X_train, y_train_dict, encoded_4_train, symbol_ids_train, source_tables_train, feature_scaler = combined_data(
        all_X_train_encoded, all_X_train_remaining, all_y_train, 
        all_encoded_4_train, all_symbol_ids_train, source_tables_train,
        feature_scaler=feature_scaler, is_training=True
    )

    print(f"Training data shapes - X: {X_train.shape}, y: {[y.shape for y in y_train_dict.values()]}, source_tables: {len(source_tables_train)}, symbol_ids: {symbol_ids_train.shape}\n")

    # Because we need a single y_train array to pass to DynamicClassWeightCallback,
    # choose the first output as 'primary' for class weighting:
    primary_output = list(y_train_dict.keys())[0]  # e.g. 'output_1'
    y_train = y_train_dict[primary_output].ravel()

    print("Saving the scalers and encoder for future use...")
    if not os.path.exists('scalers'):
        os.makedirs('scalers')
    joblib.dump({
        'feature_scaler': feature_scaler,
        'encoded_scaler': encoded_scaler,
        'encoder': encoder,
        'columns_to_encode': columns_to_encode,
        'seq_len': seq_len
    }, 'scalers/scalers_stacked_multi_4_small_main_11.pkl')

print("Loading the scalers and encoder for validation data...")
scalers_encoder = joblib.load('scalers/scalers_stacked_multi_4_small_main_11.pkl')
feature_scaler = scalers_encoder['feature_scaler']
encoded_scaler = scalers_encoder['encoded_scaler']
encoder = scalers_encoder['encoder']
columns_to_encode = scalers_encoder['columns_to_encode']
seq_len = scalers_encoder['seq_len']

print("Starting to preprocess validation data...")
df_val, source_tables_val = load_data_from_sqlite(
    "C://Users//crgon//OneDrive//Desktop//trading//data//new_multi_stacked_XAUUSD_Val_11_new.db"
)
all_X_val_encoded, all_X_val_remaining, all_y_val, all_encoded_4_val, all_symbol_ids_val, \
    feature_scaler, encoded_scaler, encoder, columns_to_encode, source_tables_val, all_pattern_classes_val = preprocess_and_combine_data(
        df_val, source_tables_val, target_columns, output_names, is_training=False,
        existing_scalers={
            'feature_scaler': feature_scaler,
            'encoded_scaler': encoded_scaler,
            'encoder': encoder,
            'columns_to_encode': columns_to_encode
        }
    )

print("Scaling and combining validation data...")
X_val, y_val_dict, encoded_4_val, symbol_ids_val, source_tables_val, feature_scaler = combined_data(
    all_X_val_encoded, all_X_val_remaining, all_y_val, 
    all_encoded_4_val, all_symbol_ids_val, source_tables_val,
    feature_scaler=feature_scaler, is_training=False
)

# Display shapes of the validation data
print(f"Validation data shapes - X: {X_val.shape}, y: {[y.shape for y in y_val_dict.values()]}, source_tables: {len(source_tables_val)}, symbol_ids: {symbol_ids_val.shape}\n")

# :::::::::::::::::::::::::::Real-time Dashboard::::::::::::::::::::::::::::::: #

class RealTimeDashboard(Callback):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Multi Predictions')
        self.fig, self.ax = plt.subplots(1, 1)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        self.losses = []
        self.val_losses = []
        self.epoch_count = 1  # Start epoch count at 1

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs['loss'])
        self.val_losses.append(logs['val_loss'])

        self.ax.clear()
        self.ax.plot(range(1, self.epoch_count + 1), self.losses, label='loss')  # Adjust range to start from 1
        self.ax.plot(range(1, self.epoch_count + 1), self.val_losses, label='val_loss')  # Adjust range to start from 1
        self.ax.legend()
        self.ax.set_title('Epoch vs Loss')
        self.ax.set_xlabel('Epoch')
        self.ax.set_ylabel('Loss')
        
        self.canvas.draw()
        plt.savefig("C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/training_progress/multi_train_loss_v1_stacked_mult_testing_new.png") 
        self.root.update()

        self.epoch_count += 1  # Increment epoch count after plotting

print("About to initialize the model...")

# ::::::::::::::::::::::::New Model Start::::::::::::::::::::::::::::: #

class AdaptivePositionalEncoding(Layer):
    def __init__(self, seq_len, d_model, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.seq_len = seq_len
        
    def build(self, input_shape):
        # Initialize learnable position embeddings
        self.position_embeddings = self.add_weight(
            "position_embeddings",
            shape=(self.seq_len, self.d_model),
            initializer=tf.keras.initializers.TruncatedNormal(stddev=0.02),
            trainable=True
        )
        
        # Add learnable scaling factor
        self.scale = self.add_weight(
            "scale",
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )
        
        # Add learnable content-position mixing weights
        self.content_weight = self.add_weight(
            "content_weight",
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )
        
        self.position_weight = self.add_weight(
            "position_weight",
            shape=(),
            initializer=tf.keras.initializers.Constant(1.0),
            trainable=True
        )
        
    def call(self, inputs):
        # Scale the input content
        content = inputs * self.scale
        
        # Add weighted position embeddings
        position_encoding = (
            self.content_weight * content + 
            self.position_weight * self.position_embeddings
        )
        
        return position_encoding
    
# ::::::::::::::SDMLP function:::::::::::::: #
    
# Add this new layer class before the model definition
class SDMLP(Layer):
    """Snake Dynamic Matrix Learning Processor with Adaptive Grid Alignment"""
    def __init__(self, units, focus_blocks=4, activation=None, 
                 kernel_initializer='orthogonal', **kwargs):  # Add initializer param
        super().__init__(**kwargs)
        self.units = units
        self.focus_blocks = focus_blocks
        self.activation = tf.keras.activations.get(activation)
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.grid = None
        
    def build(self, input_shape):
        # Create weight grid with snake-optimized initialization
        self.kernel = self.add_weight(
            name="snake_grid",
            shape=(input_shape[-1], self.units),
            initializer=self.kernel_initializer,  # Use provided initializer
            trainable=True
        )
        self.bias = self.add_weight(
            name="bias", shape=(self.units,), 
            initializer=tf.keras.initializers.RandomUniform(-0.01, 0.01),
            trainable=True
        )
        self.grid = self.kernel
        
    def get_focus_blocks(self):
        """Returns dynamically padded focus blocks with snake path optimization"""
        rows, cols = tf.shape(self.grid)[0], tf.shape(self.grid)[1]
        
        # Calculate required padding dynamically
        row_pad = (self.focus_blocks - (rows % self.focus_blocks)) % self.focus_blocks
        col_pad = (self.focus_blocks - (cols % self.focus_blocks)) % self.focus_blocks
        
        padded_grid = tf.pad(self.grid, [[0, row_pad], [0, col_pad]])
        adjusted_rows = rows + row_pad
        adjusted_cols = cols + col_pad
        
        block_rows = adjusted_rows // self.focus_blocks
        block_cols = adjusted_cols // self.focus_blocks
        
        # Create snake-like scanning pattern
        blocks = []
        for i in tf.range(self.focus_blocks):
            for j in tf.range(self.focus_blocks):
                # Alternate scanning direction for snake pattern
                j_eff = tf.cond(i % 2 == 0, lambda: j, lambda: self.focus_blocks - j - 1)
                block = padded_grid[i*block_rows:(i+1)*block_rows,
                                    j_eff*block_cols:(j_eff+1)*block_cols]
                blocks.append(tf.reduce_sum(block))
        
        return tf.stack(blocks)

    def call(self, inputs):
        # Dynamic dimension alignment using tensorflow shape ops
        input_dim = tf.shape(inputs)[-1]
        output_dim = tf.shape(self.grid)[-1]
        
        # Adaptive scaling for grid mismatch
        if input_dim != tf.shape(self.grid)[0]:
            scale_factor = tf.cast(tf.shape(self.grid)[0], tf.float32) / tf.cast(input_dim, tf.float32)
            inputs = inputs * scale_factor
            
        output = tf.matmul(inputs, self.grid) + self.bias
        return self.activation(output) if self.activation else output

class SnakeMind(Layer):
    """Neural Snake Game Engine with Dynamic Reward Learning"""
    def __init__(self, num_segments=3, exploration_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_segments = num_segments
        self.exploration_rate = exploration_rate
        self.reward_matrix = None
        self.penalty_matrix = None
        self.snake_path = None
        
    def build(self, input_shape):
        # Initialize snake movement parameters
        self.reward_matrix = self.add_weight(
            name='reward_weights',
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.Orthogonal(gain=0.5),
            trainable=True
        )
        self.penalty_matrix = self.add_weight(
            name='penalty_weights',
            shape=(input_shape[-1], 1),
            initializer=tf.keras.initializers.Orthogonal(gain=0.5),
            trainable=True
        )
        # Snake memory with exponential decay
        self.snake_path = self.add_weight(
            name='snake_path',
            shape=(self.num_segments, input_shape[-1]),
            initializer=tf.keras.initializers.Zeros(),
            trainable=False
        )
        
    def snake_dynamics(self, x, training=False):
        """Implements snake game mechanics with exploration-exploitation"""
        # Ensemble predictions
        avg_pred = tf.reduce_mean(x, axis=1, keepdims=True)
        
        # Exploration noise
        if training:
            noise = tf.random.normal(tf.shape(x), stddev=self.exploration_rate)
            x += noise * (1 - avg_pred)  # Reduce noise as confidence increases
            
        # Dynamic reward calculation
        reward = tf.where(avg_pred > 0.5,
                         self.reward_matrix * tf.square(avg_pred),
                         self.penalty_matrix * tf.sqrt(1 - avg_pred))
        
        # Update snake path with momentum
        new_head = reward * x
        updated_path = tf.concat([new_head, self.snake_path[:-1]], axis=0)
        self.snake_path.assign(updated_path)
        
        # Combine path influence with residual connection
        path_effect = tf.reduce_sum(self.snake_path, axis=0) 
        return tf.keras.activations.swish(x + path_effect)
    
    def call(self, inputs, training=False):
        snake_output = self.snake_dynamics(inputs, training)
        return tf.reduce_mean(snake_output, axis=-1)

# Dimension-Aligned Attention Mechanism
def create_attention_gate(x, focus_blocks, d_model_updated):
    # Get dynamic dimensions
    batch_size = tf.shape(x)[0]
    seq_len = tf.shape(x)[1]
    features = tf.shape(x)[2]
    
    # Create SDMLP layer with dimension matching
    sdmlp = SDMLP(units=features, focus_blocks=focus_blocks)
    grid_output = sdmlp(x)
    
    # Generate attention scores with proper dimension alignment
    grid_weights = sdmlp.get_focus_blocks()
    attention_scores = tf.nn.softmax(grid_weights)
    
    # Calculate required repetitions dynamically
    total_elements = tf.reduce_prod(tf.shape(x))
    required_repeats = total_elements // tf.size(attention_scores) + 1
    attention_tiled = tf.tile(attention_scores, [required_repeats])
    
    # Reshape to match input dimensions
    attention_scores = tf.reshape(attention_tiled[:total_elements], 
                                tf.shape(x))
    
    # Apply gating with residual connection
    return x * attention_scores + x

# ::::::::::::::::::::::::Main Model Parameters::::::::::::::::::::::::::::: #

d_model = 117                          # Number of features (input dim per time step)
pattern_class_embedding_dim = 483      # Clustering class embedding dimension
d_ffn = 320                            # Projected input dimension or expanded dimension

dropout_rate = 0.01                    # Dropout rate for regularization

# SDMLP Parameters
focus_blocks = 4                   # Number of focus blocks for weight grid analysis
num_experts = 4                    # Number of experts for SnakeMind processing

# ::::::::::::::::::::::::Main Model Embeddings::::::::::::::::::::::::::::: #

# >>> NEW: Load pattern_class mapping <<<
pattern_class_mapping_filename = 'pattern_class_mapping_small_main_11.json'
with open(pattern_class_mapping_filename, 'r') as f:
    pattern_class_mapping = json.load(f)

num_pattern_classes = len(pattern_class_mapping)
print("Number of pattern classes loaded:", num_pattern_classes)

class SequencePatternEmbedding(Layer):
    def __init__(self, num_pattern_classes, embedding_dim, **kwargs):
        super(SequencePatternEmbedding, self).__init__(**kwargs)
        self.embedding = Embedding(
            input_dim=num_pattern_classes,
            output_dim=embedding_dim,
            name='pattern_embedding_inner'
        )
        self.layer_norm = LayerNormalization(epsilon=elip_len)  # Added normalization
    
    def call(self, inputs):
        embedded = self.embedding(inputs)
        return self.layer_norm(embedded) 

# Input layers
inputs = Input(shape=(seq_len, d_model), name='input_1', dtype='float32')
symbol_id_input = Input(shape=(1,), dtype=tf.int32, name='input_2')
pattern_class_input = Input(shape=(seq_len,), dtype=tf.int32, name='input_3')

pattern_embed_layer = SequencePatternEmbedding(
    num_pattern_classes=num_pattern_classes,
    embedding_dim=pattern_class_embedding_dim
)
pattern_embeddings = pattern_embed_layer(pattern_class_input)

concatenated_inputs = Concatenate(axis=-1)([inputs, pattern_embeddings])

# ::::::::::::::Main Model Training (Multi Target Prediction):::::::::::::: #

outputs = []
for name in output_names:
    model_inputs = concatenated_inputs

    d_model_updated = model_inputs.shape[-1]
    seq_len_updated = model_inputs.shape[-2]
    ff_dim_updated = d_model_updated*4

    x = AdaptivePositionalEncoding(seq_len_updated, d_model_updated)(model_inputs)   
    x = LayerNormalization(epsilon=elip_len)(x)

    # Create SDMLP grid system
    sdmlp_layer = SDMLP(
        units=d_model_updated,  # Match output dimension to input
        focus_blocks=focus_blocks,
        activation='gelu',
        name=f'{name}_sdmlp_grid'
    )
    grid_output = sdmlp_layer(x)  # Then call with input
    
    # Generate dynamic attention scores
    def create_snake_attention(input_tensor, sdmlp_layer):
        # Get focus blocks with snake scanning pattern
        grid_weights = sdmlp_layer.get_focus_blocks()
        
        # Calculate attention scores with dynamic dimension matching
        attention_scores = tf.nn.softmax(grid_weights)
        
        # Calculate required repetitions using tensorflow ops
        input_shape = tf.shape(input_tensor)
        total_elements = tf.reduce_prod(input_shape)
        num_blocks = tf.size(attention_scores)
        
        # Dynamically calculate required repetitions
        required_repeats = (total_elements // num_blocks) + 1
        attention_tiled = tf.tile(attention_scores, [required_repeats])
        
        # Slice to exact size and reshape
        attention_scores = tf.reshape(
            attention_tiled[:total_elements],
            input_shape
        )
        
        # Apply residual connection for dimension stability
        return input_tensor * attention_scores + input_tensor

    # Apply snake attention mechanism
    gated_output = create_snake_attention(x, sdmlp_layer)

    last_row = tf.expand_dims(gated_output[:, -1, :], axis=1)
    mha_model = tf.reshape(last_row, [-1, last_row.shape[-1]])
    
    # Connect to final SDMLP output
    mha_model = SDMLP(
        units=3,  # Three outputs for ensemble voting
        focus_blocks=focus_blocks,
        activation='sigmoid',
        kernel_initializer='he_normal',  # Now properly accepted
        name=name
    )(mha_model)
    outputs.append(mha_model)

model = Model([inputs, symbol_id_input, pattern_class_input], outputs=outputs)

model.summary()
print("Model initialized!")
print("Total number of parameters in the model:", model.count_params())

# :::::::::::::::::::::Model Checkpoint For Weights::::::::::::::::::::::: #

# Modify the weights_checkpoint to save weights every epoch
weights_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/saved_weights/weights_v1_stacked_multi_small_main_11_epoch_{epoch:02d}.h5"
weights_checkpoint = ModelCheckpoint(
    filepath=weights_checkpoint_path,
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/saved_weights/model_v1_FULL_stacked_multi.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', 
                                        verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

custom_objects = {
    
}

if epochs > 0:
    # Try loading weights or use fresh model
    weights_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/saved_weights/"
    weight_files = glob.glob(os.path.join(weights_dir, "weights_v1_stacked_multi_small_main_11_epoch_*.h5"))

    if weight_files:
        # Extract epoch numbers from filenames
        available_epochs = []
        for filepath in weight_files:
            match = re.search(r"weights_v1_stacked_multi_small_main_11_epoch_(\d+).h5", os.path.basename(filepath))
            if match:
                epoch_num = int(match.group(1))
                available_epochs.append(epoch_num)
        
        available_epochs.sort()
        print(f"\nAvailable epoch weights: {available_epochs}")
        print("Enter epoch number to load weights from, or 'no' for fresh model:")
        user_input = input().strip().lower()
        
        if user_input != 'no':
            try:
                epoch_to_load = int(user_input)
                if epoch_to_load in available_epochs:
                    weights_path = os.path.join(weights_dir, f"weights_v1_stacked_multi_small_main_11_epoch_{epoch_to_load:02d}.h5")
                    print(f"Loading weights from epoch {epoch_to_load}...")
                    model.load_weights(weights_path)
                    print("Weights loaded successfully!")
                else:
                    print(f"Epoch {epoch_to_load} not found. Using fresh model.")
            except ValueError:
                print("Invalid input. Using fresh model.")
        else:
            print("Using fresh model.")
    else:
        print("No saved weights found. Using fresh model.")

# ::::::::::::::::::::::::Training Model Functions:::::::::::::::::::::::::::: #

# Custom learning rate schedule with warm-up
class CustomSchedule(LearningRateSchedule):
    def __init__(self, d_model, warmup_steps, custom_lr, lr_scale):
        super(CustomSchedule, self).__init__()
        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = warmup_steps
        self.custom_lr = custom_lr
        self.lr_scale = lr_scale

    def __call__(self, step):
        # Early return for fixed learning rate
        if not self.custom_lr:
            return fixed_learning_rate
        
        step = tf.cast(step, tf.float32)
        
        # Formula from "Attention Is All You Need" paper
        arg1 = tf.math.rsqrt(self.d_model)
        arg2 = tf.math.minimum(
            tf.math.rsqrt(step),
            step * (tf.math.rsqrt(tf.cast(self.warmup_steps, tf.float32)) ** 3)
        )
        
        return arg1 * arg2 * self.lr_scale

    def get_config(self):
        return {
            'd_model': self.d_model.numpy(),
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
    def __init__(self, val_generator, num_to_sample_per_class=10):
        """
        val_generator: your ShiftingSequenceGenerator (or similar) for validation
        num_to_sample_per_class: how many samples of each class to print per group
        """
        super().__init__()
        self.val_generator = val_generator
        self.num_to_sample_per_class = num_to_sample_per_class

    def relaxed_tp_fp(self, true_labels, pred_labels):
        """
        For class=1, give 'credit' if the model predicts a '1' at the same index
        or one step before/after. This method returns (tp, fp) based on that rule.

        If true[i] == 1, then if pred[i], pred[i-1], or pred[i+1] == 1
        (and wasn't already counted for some other true[j]), we count +1 to TP,
        and mark that prediction index as 'used' so we don't double-count it.
        """
        n = len(true_labels)
        used_pred = np.zeros(n, dtype=bool)  # track which predicted 1's we've assigned
        tp = 0

        for i in range(n):
            if true_labels[i] == 1:
                idx_candidates = [i]
                if i > 0:
                    idx_candidates.append(i - 1)
                if i < n - 1:
                    idx_candidates.append(i + 1)

                found_match = False
                for j in idx_candidates:
                    if pred_labels[j] == 1 and not used_pred[j]:
                        used_pred[j] = True
                        found_match = True
                        tp += 1
                        break

        total_pred_ones = np.sum(pred_labels)
        fp = total_pred_ones - tp
        return tp, fp

    def on_epoch_end(self, epoch, logs=None):
        # Use the model to predict on the entire val_generator
        predictions = self.model.predict(self.val_generator, verbose=0)

        # If multiple outputs => predictions is a list of arrays. If single => just one array
        if not isinstance(predictions, list):
            predictions = [predictions]

        # group_list, table_list, y_true_dict are aligned with val_generator order
        group_list, table_list, y_true_dict = self.val_generator.get_full_order()

        # Ensure table_list is a NumPy array for proper indexing
        table_list = np.array(table_list)

        unique_groups = np.unique(group_list)

        for i, output_name in enumerate(self.model.output_names):
            all_preds = predictions[i].ravel()  # shape: (total_sequences,)
            all_labels = np.array(y_true_dict[output_name]).ravel()
            # Convert probabilities to {0,1}
            pred_classes = (all_preds > 0.5).astype(int)

            print(f"\n[Epoch {epoch+1}] === Processing output: {output_name} ===")

            # For each group, compute relaxed precision
            for grp in unique_groups:
                grp_indices = np.where(group_list == grp)[0]
                if grp_indices.size == 0:
                    continue

                grp_pred = pred_classes[grp_indices]
                grp_true = all_labels[grp_indices]

                # --- Compute "relaxed tp / fp" ---
                grp_tp, grp_fp = self.relaxed_tp_fp(grp_true, grp_pred)
                precision = grp_tp / (grp_tp + grp_fp) if (grp_tp + grp_fp) > 0 else 0.0

                print(f"[Epoch {epoch+1}] Group {grp} => {output_name} (Relaxed) Precision: {precision:.4f}")

                # ------------------------------------------------------------------
                # Sample predictions for a quick human-inspectable readout
                # ------------------------------------------------------------------
                idx_class_0 = np.where(grp_true == 0)[0]
                idx_class_1 = np.where(grp_true == 1)[0]
                n0 = min(len(idx_class_0), self.num_to_sample_per_class)
                n1 = min(len(idx_class_1), self.num_to_sample_per_class)

                if n0 == 0 or n1 == 0:
                    print(f"[Epoch {epoch+1}] Group {grp} => Not enough samples for both classes.")
                    continue

                # Sample each class
                s0 = np.random.choice(idx_class_0, n0, replace=False)
                s1 = np.random.choice(idx_class_1, n1, replace=False)
                chosen = np.random.permutation(np.concatenate([s0, s1]))

                pred_sampled = grp_pred[chosen]
                true_sampled = grp_true[chosen]
                tables_sampled = table_list[grp_indices][chosen]

                print(f"[Epoch {epoch+1}] Group {grp} => {output_name} Pred: {pred_sampled}")
                print(f"[Epoch {epoch+1}] Group {grp} => {output_name} True: {true_sampled}")
                print(f"Source tables: {tables_sampled}\n")

class PrintUnscaledLoss(Callback):
    def __init__(self, val_generator):
        """
        val_generator: ShiftingSequenceGenerator for validation.
        """
        super().__init__()
        self.val_generator = val_generator

    def on_epoch_end(self, epoch, logs=None):
        # Predict
        predictions = self.model.predict(self.val_generator, verbose=0)
        if not isinstance(predictions, list):
            predictions = [predictions]
        group_list, table_list, y_true_dict = self.val_generator.get_full_order()
        
        # We combine results for summary stats (accuracy, etc.)
        combined_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"pred": [], "real": []})))
        group_sizes = defaultdict(int)
        all_pred_classes = []
        all_real_classes = []

        for i, output_name in enumerate(self.model.output_names):
            preds = predictions[i].ravel()
            reals = np.array(y_true_dict[output_name], dtype=np.float32).ravel()
            pred_classes = (preds > 0.5).astype(int)
            real_classes = reals.astype(int)

            for idx in range(len(pred_classes)):
                src_table = table_list[idx]
                grp = group_list[idx]
                p = pred_classes[idx]
                r = real_classes[idx]
                combined_results[src_table][output_name][grp]["pred"].append(p)
                combined_results[src_table][output_name][grp]["real"].append(r)
                all_pred_classes.append(p)
                all_real_classes.append(r)
                group_sizes[(src_table, grp)] += 1

        # Print group-level accuracies
        for src_table, out_dict in combined_results.items():
            for output_name, group_dict in out_dict.items():
                for grp, data_map in sorted(group_dict.items()):
                    preds_arr = np.array(data_map["pred"])
                    reals_arr = np.array(data_map["real"])
                    accuracy = (preds_arr == reals_arr).mean() if len(preds_arr) > 0 else 0
                    print(f"[Epoch {epoch+1}] {output_name} Acc for {src_table}, Group {grp}: {accuracy:.4f}")

        # Overall accuracy
        all_pred_classes = np.array(all_pred_classes)
        all_real_classes = np.array(all_real_classes)
        overall_acc = (all_pred_classes == all_real_classes).mean() if len(all_pred_classes) > 0 else 0
        print(f"[Epoch {epoch+1}] Overall Validation Accuracy: {overall_acc:.4f}")

        # Weighted accuracy
        total_samples = sum(group_sizes.values())
        weighted_acc_sum = 0.0
        for (table, grp), size in group_sizes.items():
            for out_name, group_dict in combined_results[table].items():
                data = group_dict.get(grp)
                if data and data["pred"] and data["real"]:
                    p_arr = np.array(data["pred"])
                    r_arr = np.array(data["real"])
                    grp_acc = (p_arr == r_arr).mean()
                    weighted_acc_sum += grp_acc * size

        if total_samples > 0:
            weighted_accuracy = weighted_acc_sum / total_samples
            print(f"[Epoch {epoch+1}] Weighted Validation Accuracy: {weighted_accuracy:.4f}")

# ::::::::::::::::::::::::Training Model Functions:::::::::::::::::::::::::::: #

class ShiftingSequenceGenerator(tf.keras.utils.Sequence):
        def __init__(
            self,
            X,             # shape (N, 1, features)
            y_dict,        # dict of {output_name: np.array(shape=(N,))}
            symbol_ids,    # shape (N,)
            pattern_classes, # <--- Add this
            source_tables, # shape (N,) 
            group_ids,     # shape (N,)
            seq_len,
            batch_size
        ):
            super().__init__()
            # Ensure X is 3D => (N, 1, features)
            if X.ndim == 2:
                X = np.expand_dims(X, axis=1)
            self.X = X  # shape (N, 1, features)
            self.y_dict = y_dict
            self.symbol_ids = symbol_ids
            self.pattern_classes = pattern_classes  # <--- Store pattern_class array
            self.source_tables = source_tables
            self.group_ids = group_ids
            self.seq_len = seq_len
            self.batch_size = batch_size

            # We'll keep all output_names (keys of y_dict)
            self.output_names = list(y_dict.keys())

            # 1) Separate by group in chronological order
            self._grouped_data = {}
            for idx in range(len(self.X)):
                grp = self.group_ids[idx]
                if grp not in self._grouped_data:
                    self._grouped_data[grp] = {
                        "X": [],
                        "symbol": [],
                        "pattern": [],  # <--- Add pattern storage
                        "table": [],
                        "targets": {k: [] for k in self.output_names}
                    }
                self._grouped_data[grp]["X"].append(self.X[idx])
                self._grouped_data[grp]["symbol"].append(self.symbol_ids[idx])
                self._grouped_data[grp]["pattern"].append(self.pattern_classes[idx])  # <--- Append pattern
                self._grouped_data[grp]["table"].append(self.source_tables[idx])
                for k in self.output_names:
                    self._grouped_data[grp]["targets"][k].append(self.y_dict[k][idx])

            # Convert to arrays
            for grp in self._grouped_data:
                self._grouped_data[grp]["X"] = np.array(self._grouped_data[grp]["X"])  # shape (G, 1, features)
                self._grouped_data[grp]["symbol"] = np.array(self._grouped_data[grp]["symbol"])
                self._grouped_data[grp]["pattern"] = np.array(self._grouped_data[grp]["pattern"])  # <--- Convert to array
                self._grouped_data[grp]["table"] = np.array(self._grouped_data[grp]["table"], dtype=object)
                for k in self.output_names:
                    self._grouped_data[grp]["targets"][k] = np.array(self._grouped_data[grp]["targets"][k])

            # 2) Build list of (group_id, start_index) for valid sliding windows in each group
            self._all_sequences = []
            for grp in sorted(self._grouped_data.keys()):
                data_len = len(self._grouped_data[grp]["X"])
                n_valid_starts = data_len - self.seq_len + 1
                if n_valid_starts < 1:
                    continue  # skip groups too small for the window
                for start_i in range(n_valid_starts):
                    self._all_sequences.append((grp, start_i))

            # 3) Batch building: each batch is from one group only (no mixing)
            self._batches = []
            current_grp = None
            current_batch = []
            for (grp, start_i) in self._all_sequences:
                if current_grp is None:
                    current_grp = grp
                if grp != current_grp and current_batch:
                    # flush the current batch
                    for i in range(0, len(current_batch), self.batch_size):
                        self._batches.append((current_grp, current_batch[i:i+self.batch_size]))
                    current_batch = []
                    current_grp = grp
                current_batch.append(start_i)
            # flush the last group
            if current_grp is not None and current_batch:
                for i in range(0, len(current_batch), self.batch_size):
                    self._batches.append((current_grp, current_batch[i:i+self.batch_size]))

        def __len__(self):
            return len(self._batches)

        def __getitem__(self, index):
            grp, start_positions = self._batches[index]
            group_data = self._grouped_data[grp]

            batch_X_list = []
            batch_symbol_ids = []
            batch_pattern_ids = []  # we want shape (batch_size, seq_len)
            batch_y_dict = {k: [] for k in self.output_names}

            for st_i in start_positions:
                seq_X = group_data["X"][st_i : st_i + self.seq_len]
                if seq_X.ndim == 3:
                    seq_X = seq_X[:, 0, :]

                last_idx = st_i + self.seq_len - 1
                # Symbol ID once per sequence
                seq_symbol = group_data["symbol"][last_idx]

                # Gather the FULL pattern-class sequence => shape (seq_len,)
                seq_pattern_seq = group_data["pattern"][st_i : st_i + self.seq_len]
                batch_pattern_ids.append(seq_pattern_seq)

                # Target is from the last row
                for k in self.output_names:
                    seq_y = group_data["targets"][k][last_idx]
                    batch_y_dict[k].append(seq_y)

                batch_X_list.append(seq_X)
                batch_symbol_ids.append(seq_symbol)

            batch_X = np.array(batch_X_list, dtype=np.float32)  # => (batch_size, seq_len, feats)
            batch_symbol_ids = np.array(batch_symbol_ids, dtype=np.int32)  # => (batch_size,)
            batch_pattern_ids = np.array(batch_pattern_ids, dtype=np.int32)  # => (batch_size, seq_len)

            final_y = []
            for k in self.output_names:
                final_y.append(np.array(batch_y_dict[k], dtype=np.float32))
            if len(self.output_names) == 1:
                final_y = final_y[0]

            return {
                'input_1': batch_X,                            # shape => (batch_size, seq_len, feats)
                'input_2': batch_symbol_ids.reshape(-1, 1),    # shape => (batch_size, 1)
                'input_3': batch_pattern_ids                   # shape => (batch_size, seq_len)
            }, final_y

        def get_full_order(self):
            """
            Returns a tuple:
            group_list: [grp_1, grp_2, ...]
            table_list: [table_1, table_2, ...]
            y_true_dict: { 'Signal': [val_1, val_2, ...],
                            ... (for each output name) }
            in the exact order that model.predict() produces predictions.

            Each element corresponds to one sequence (the last row's label).
            """
            group_list = []
            table_list = []
            y_true_dict = {k: [] for k in self.output_names}

            for (grp, start_positions) in self._batches:
                for st_i in start_positions:
                    last_i = st_i + self.seq_len - 1
                    group_list.append(grp)
                    last_table = self._grouped_data[grp]['table'][last_i]
                    table_list.append(last_table)
                    for k in self.output_names:
                        real_val = self._grouped_data[grp]["targets"][k][last_i]
                        y_true_dict[k].append(real_val)

            return group_list, table_list, y_true_dict
        
class ChronologicalBalancedSequenceGenerator(ShiftingSequenceGenerator):
    """
    ChronologicalBalancedSequenceGenerator preserves chronological order but oversamples
    (duplicates) the minority class in-place.
    """

    def __init__(
        self,
        X,
        y_dict,
        symbol_ids,
        pattern_classes,
        source_tables,
        group_ids,
        seq_len,
        batch_size,
        balance_on_output,
        replicate_variation,
        augmentation_noise_std,      # <-- Add a small random noise parameter
        augmentation_scale_range     # <-- Add an optional scale parameter
    ):
        """
        :param augmentation_noise_std: Standard deviation of Gaussian noise to add
                                       to each feature when replicating a sample.
        :param augmentation_scale_range: Range of uniform scaling factors around 1.0.
                                         e.g. 0.02 => random scale in [0.98, 1.02].
        """
        # Call parent constructor (ShiftingSequenceGenerator).
        super().__init__(X, y_dict, symbol_ids, pattern_classes, source_tables, group_ids, seq_len, batch_size)

        self.balance_on_output = (
            balance_on_output
            if balance_on_output in self.output_names
            else self.output_names[0]
        )
        self.replicate_variation = replicate_variation

        # Store new augmentation parameters
        self.augmentation_noise_std = augmentation_noise_std
        self.augmentation_scale_range = augmentation_scale_range

        self._build_oversampled_batches()

    def _apply_augmentation(self, seq_X):
        """
        Apply mild random noise and/or random scaling to seq_X for data augmentation.
        seq_X shape => (seq_len, features) if the generator has shaped it that way.
        """
        # Make a copy so we don't mutate self._grouped_data directly.
        seq_X = np.copy(seq_X)

        # If user specified a scale range, pick a random scale factor around 1.0
        if self.augmentation_scale_range > 0.0:
            scale_factor = 1.0 + np.random.uniform(
                -self.augmentation_scale_range, self.augmentation_scale_range
            )
            seq_X = seq_X * scale_factor

        # If user specified a noise std, add Gaussian noise
        if self.augmentation_noise_std > 0.0:
            noise = np.random.normal(
                loc=0.0, 
                scale=self.augmentation_noise_std, 
                size=seq_X.shape
            )
            seq_X = seq_X + noise

        return seq_X

    def _build_oversampled_batches(self):
        """
        Build self._batches chronologically. This is unchanged except we store
        how often we replicate each window.
        """
        self._compressed_windows = {}
        self._prefix_sums = {}
        all_batches = []

        for grp in sorted(self._grouped_data.keys()):
            data_len = len(self._grouped_data[grp]["X"])
            n_valid_starts = data_len - self.seq_len + 1
            if n_valid_starts < 1:
                continue

            windows = []
            for start_i in range(n_valid_starts):
                last_i = start_i + self.seq_len - 1
                label = self._grouped_data[grp]["targets"][self.balance_on_output][last_i]
                windows.append((start_i, label))

            # Count minority vs. majority
            count0 = sum(1 for (_, lbl) in windows if lbl == 0)
            count1 = len(windows) - count0

            # If one class is zero, skip balancing
            if count0 == 0 or count1 == 0:
                compressed = [(w[0], 1) for w in windows]
            else:
                if count0 < count1:
                    minority_label = 0
                    majority_count = count1
                    minority_count = count0
                else:
                    minority_label = 1
                    majority_count = count0
                    minority_count = count1

                base_factor = max(1, int(round(majority_count / float(minority_count))))

                compressed = []
                for (start_i, lbl) in windows:
                    replicate_count = 1
                    if lbl == minority_label:  # oversample the minority
                        if self.replicate_variation and self.replicate_variation > 0.0:
                            low = int(round(base_factor * (1 - self.replicate_variation)))
                            high = int(round(base_factor * (1 + self.replicate_variation)))
                            low = max(low, 1)
                            high = max(high, 1)
                            replicate_count = np.random.randint(low, high + 1)
                        else:
                            replicate_count = base_factor
                    compressed.append((start_i, replicate_count))

            self._compressed_windows[grp] = compressed

            prefix_sums = [0]
            for (_, rc) in compressed:
                prefix_sums.append(prefix_sums[-1] + rc)
            self._prefix_sums[grp] = prefix_sums

            start_idx = 0
            total_expanded = prefix_sums[-1]
            while start_idx < total_expanded:
                end_idx = min(start_idx + self.batch_size, total_expanded)
                all_batches.append((grp, start_idx, end_idx))
                start_idx = end_idx

        self._batches = all_batches

    def __getitem__(self, index):
        grp, start_expanded, end_expanded = self._batches[index]
        compressed = self._compressed_windows[grp]
        prefix_sums = self._prefix_sums[grp]

        batch_size = end_expanded - start_expanded
        batch_X_list = []
        batch_symbol_ids = []
        batch_pattern_ids = []
        batch_y_dict = {k: [] for k in self.output_names}

        needed = batch_size
        c_idx = self._binary_search(prefix_sums, start_expanded)
        offset_within_entry = start_expanded - prefix_sums[c_idx]

        while needed > 0 and c_idx < len(compressed):
            (start_i, replicate_count) = compressed[c_idx]
            expansions_left_in_entry = replicate_count - offset_within_entry
            use_here = min(needed, expansions_left_in_entry)

            for _ in range(use_here):
                # Pull original sequence from group_data
                seq_X = self._grouped_data[grp]["X"][start_i:start_i + self.seq_len]
                # If needed, remove the extra dimension: shape => (seq_len, features)
                if seq_X.ndim == 3:
                    seq_X = seq_X[:, 0, :]

                # Apply augmentation only if we are replicating (or always if desired)
                # You can refine logic if you want the first copy to remain "original".
                seq_X = self._apply_augmentation(seq_X)

                last_idx = start_i + self.seq_len - 1
                seq_symbol = self._grouped_data[grp]["symbol"][last_idx]
                seq_pattern_seq = self._grouped_data[grp]["pattern"][start_i:start_i + self.seq_len]

                # Targets come from the last row
                for k in self.output_names:
                    seq_y = self._grouped_data[grp]["targets"][k][last_idx]
                    batch_y_dict[k].append(seq_y)

                batch_X_list.append(seq_X)
                batch_symbol_ids.append(seq_symbol)
                batch_pattern_ids.append(seq_pattern_seq)

            needed -= use_here
            offset_within_entry += use_here
            if offset_within_entry >= replicate_count:
                c_idx += 1
                offset_within_entry = 0

        batch_X = np.array(batch_X_list, dtype=np.float32)             # (batch_size, seq_len, feats)
        batch_symbol_ids = np.array(batch_symbol_ids, dtype=np.int32)   # (batch_size,)
        batch_pattern_ids = np.array(batch_pattern_ids, dtype=np.int32) # (batch_size, seq_len)

        final_y = []
        for k in self.output_names:
            final_y.append(np.array(batch_y_dict[k], dtype=np.float32))
        if len(self.output_names) == 1:
            final_y = final_y[0]

        return {
            'input_1': batch_X,
            'input_2': batch_symbol_ids.reshape(-1, 1),
            'input_3': batch_pattern_ids
        }, final_y
    
    @staticmethod
    def _binary_search(prefix_sums, x):
        """Find i so that prefix_sums[i] <= x < prefix_sums[i+1]."""
        lo, hi = 0, len(prefix_sums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if prefix_sums[mid] <= x < prefix_sums[mid + 1]:
                return mid
            elif x < prefix_sums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        return lo

# ::::::::::::::::::::::::Training & Validation Loop:::::::::::::::::::::::::::: #

if epochs > 0:
    # -------------------------------------------------------------------------
    # 2. Prepare Validation Data
    # -------------------------------------------------------------------------
    X_val = X_val.astype(np.float32)
    y_val_dict = {key: value.astype(np.int32) for key, value in y_val_dict.items()}
    symbol_ids_val = symbol_ids_val.astype(np.int32)
    pattern_class_val = np.concatenate(all_pattern_classes_val, axis=0).astype(np.int32)

    #X_val_inputs = {'input_1': X_val, 'input_2': symbol_ids_val, 'input_3': pattern_class_val}

    print("X_val shape:", X_val.shape)
    print("Processed Source Tables Val:", set(source_tables_val))    

    # -------------------------------------------------------------------------
    # 3. Prepare Training Data
    # -------------------------------------------------------------------------
    X_train = X_train.astype(np.float32)
    y_train_dict = {key: value.astype(np.int32) for key, value in y_train_dict.items()}
    symbol_ids_train = symbol_ids_train.astype(np.int32)
    pattern_class_train = np.concatenate(all_pattern_classes_train, axis=0).astype(np.int32)

    # Same note about dictionary vs. arrays
    #X_train_inputs = {'input_1': X_train, 'input_2': symbol_ids_train, 'input_3': pattern_class_train}

    print("X_train shape:", X_train.shape)
    print("Processed Source Tables Train:", set(source_tables_train))

    dashboard = RealTimeDashboard()
    print_lr = PrintLR()

    # -------------------------------------------------------------------------
    # 4. Pick a Primary Output so that we can flatten y_train for class weighting
    # -------------------------------------------------------------------------
    primary_output_name = list(y_train_dict.keys())[0]  # e.g. first key
    y_train = y_train_dict[primary_output_name].ravel()

    print("Model output names:", model.output_names)
    print("y_val_dict keys:", y_val_dict.keys())
    for key, value in y_val_dict.items():
        print(f"{key} shape:", value.shape)
        print(f"{key} dtype:", value.dtype)

    # -------------------------------------------------------------------------
    # 7. Evaluate Baseline Using a Generator
    #    (Because direct evaluation with X_val_inputs => shape mismatch)
    # -------------------------------------------------------------------------
    """ val_generator_for_eval = ShiftingSequenceGenerator(
        X_val,       # pass array not dict
        y_val_dict,
        symbol_ids_val,
        pattern_class_val,
        source_tables_val,
        encoded_4_val,
        seq_len,
        batch_size
    ) """

    # -------------------------------------------------------------------------
    # 8. Create Shifting Generators for Training & Val
    # -------------------------------------------------------------------------

    train_generator = ChronologicalBalancedSequenceGenerator(
        X_train,
        y_train_dict,
        symbol_ids_train,
        pattern_class_train,
        source_tables_train,
        encoded_4_train,
        seq_len=seq_len,
        batch_size=batch_size,
        balance_on_output="Signal",
        replicate_variation=0.2,          # 0.2 is the default
        # Augmentation params:
        augmentation_noise_std=0.01,      # e.g. add small Gaussian noise (std=0.01)
        augmentation_scale_range=0.02     # e.g. random scale in [0.98, 1.02]
    )

    # Calculate warmup_steps based on training data
    num_batches_per_epoch = len(train_generator)
    total_training_steps = num_batches_per_epoch * epochs
    
    # Set warmup steps to be ~10% of total training steps
    # You can adjust this percentage by changing the 0.1
    warmup_steps = int(total_training_steps * 0.1)
    print(f"Automatically calculated warmup steps: {warmup_steps}")
    print(f"Based on: {num_batches_per_epoch} batches/epoch * {epochs} epochs = {total_training_steps} total steps")

    learning_rate = CustomSchedule(d_model, warmup_steps, custom_lr, lr_scale)
    optimizer = Adam(
        learning_rate=learning_rate,
        beta_1=beta_1,
        beta_2=beta_2,
        epsilon=epsilon,
        amsgrad=amsgrad
    )

    # Add this custom loss function before model compilation
    def ensemble_loss(y_true, y_pred):
        # y_pred shape: (batch_size, 3)
        # Average the three predictions
        avg_pred = tf.reduce_mean(y_pred, axis=-1)  # shape: (batch_size,)
        
        # Calculate binary crossentropy using averaged predictions
        bce = tf.keras.losses.binary_crossentropy(y_true, avg_pred)
        
        # Reward confidence bonus - encourages individual predictors to agree
        confidence_bonus = tf.reduce_mean(tf.abs(0.5 - y_pred), axis=[0,1])
        
        return bce - confidence_bonus*0.1  # Adjust bonus weight as needed

    # Update the metric compilation section to:
    def wrapped_metric(metric_fn, name):
        def wrapper(y_true, y_pred):
            avg_pred = tf.reduce_mean(y_pred, axis=-1)
            return metric_fn(y_true, avg_pred)
        wrapper.__name__ = name
        return wrapper

    # Update model compilation with explicit names:
    # Create metric instances outside the lambda scope
    auc_metric = AUC(curve='PR', name='ensemble_auc')
    precision_metric = Precision(name='ensemble_precision')
    recall_metric = Recall(name='ensemble_recall')

    model.compile(
        optimizer=optimizer,
        loss=ensemble_loss,
        metrics=[
            wrapped_metric(
                lambda yt, yp: tf.keras.metrics.binary_accuracy(yt, yp),
                name='ensemble_acc'
            ),
            wrapped_metric(
                lambda yt, yp: auc_metric(yt, yp),
                name='ensemble_auc'
            ),
            wrapped_metric(
                lambda yt, yp: precision_metric(yt, yp),
                name='ensemble_precision'
            ),
            wrapped_metric(
                lambda yt, yp: recall_metric(yt, yp),
                name='ensemble_recall'
            )
        ]
    )

    # Modified loss function for ensemble learning
    def snake_ensemble_loss(y_true, y_pred):
        # y_pred shape: [batch_size, 3]
        avg_pred = tf.reduce_mean(y_pred, axis=-1)  # Average of 3 predictions
        
        # Base binary crossentropy
        bce_loss = tf.keras.losses.binary_crossentropy(y_true, avg_pred)
        
        # Confidence reward for clear predictions
        confidence_reward = tf.reduce_mean(
            tf.abs(0.5 - y_pred) * 2  # Scales 0-0.5 to 0-1, 0.5-1 to 0-1
        )
        
        # Snake movement penalty (encourage strategic weight adjustments)
        snake_penalty = tf.reduce_mean(
            tf.abs(y_pred[:, 0] - y_pred[:, 1]) + 
            tf.abs(y_pred[:, 1] - y_pred[:, 2])
        )
        
        return bce_loss - confidence_reward + 0.1 * snake_penalty

    # Modified model compilation
    model.compile(
        optimizer=optimizer,
        loss=snake_ensemble_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name='acc', threshold=0.5),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')
        ]
    )


    val_generator = ShiftingSequenceGenerator(
        X_val,    # again, array instead of dict
        y_val_dict,
        symbol_ids_val,
        pattern_class_val,
        source_tables_val,
        encoded_4_val,
        seq_len,
        batch_size,
    )

    unscaled_loss_callback = PrintUnscaledLoss(val_generator=val_generator)
    print_predictions_and_loss_0 = PrintPredictionsAndLoss_0(val_generator=val_generator)

    # other_callbacks_2 = [dashboard, print_lr, unscaled_loss_callback, print_predictions_and_loss_0] # dynamic_class_weight_callback
    other_callbacks_2 = [dashboard, print_lr, unscaled_loss_callback, print_predictions_and_loss_0] # dynamic_class_weight_callback
    # Instantiate the custom stop callback
    all_callbacks = [weights_checkpoint] + other_callbacks_2 

    """ print("Evaluating baseline performance via generator...")
    val_loss = model.evaluate(val_generator_for_eval, verbose=1)
    print(f"Baseline loss on Validation Set after Loading: {val_loss}") """

    # -------------------------------------------------------------------------
    # 9. Run Training Loop
    # -------------------------------------------------------------------------
    print("Starting model training with sliding windows (group-isolated, no shuffle)...")
    print("Starting model training...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train for one epoch
        history = model.fit(
            x=train_generator,
            epochs=epoch + 1,
            initial_epoch=epoch,
            validation_data=val_generator,
            callbacks=all_callbacks,
            verbose=1
        )
        
        # Optional: Add explicit checks here
        if hasattr(history, 'history'):
            print(f"Epoch {epoch + 1} metrics:", history.history)
        else:
            print(f"Warning: No history for epoch {epoch + 1}")
            
        # Force garbage collection to free memory
        gc.collect()
    print("\nTraining complete.")

    # ::::::::::::::::::::::::::Post-Training Cleanup and User Interaction:::::::::::::::::::::::::::: #

# Close the Tee object properly
if isinstance(sys.stdout, Tee):
    original_stdout = sys.stdout.stdout  # Save reference to original stdout
    sys.stdout.close()  # Close the Tee object
    sys.stdout = original_stdout  # Restore original stdout

# After training ends (whether completed or interrupted), prompt the user
print("\nWould you like to keep all saved epochs? (yes/no)")
user_input = input().strip().lower()
if user_input == 'no':
    print("Cleaning up weight files based on your selection...")
    weights_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/saved_weights/"
    weight_files = glob.glob(os.path.join(weights_dir, "weights_v1_stacked_multi_small_main_11_epoch_*.h5"))
    
    # Extract epoch numbers from the filenames
    saved_epochs = []
    for filepath in weight_files:
        match = re.search(r"weights_v1_stacked_multi_small_main_11_epoch_(\d+).h5", os.path.basename(filepath))
        if match:
            epoch_num = int(match.group(1))
            saved_epochs.append(epoch_num)
    
    if not saved_epochs:
        print("No saved weight files found.")
    else:
        saved_epochs.sort()
        print(f"Saved epochs: {saved_epochs}")
        print("Please enter the epoch numbers to keep, separated by commas:")
        epochs_to_keep_input = input().strip()
        epochs_to_keep = [int(e.strip()) for e in epochs_to_keep_input.split(',') if e.strip().isdigit()]
        epochs_to_keep = set(epochs_to_keep)
        
        # Delete weight files for epochs not in epochs_to_keep
        for filepath in weight_files:
            match = re.search(r"weights_v1_stacked_multi_small_main_11_epoch_(\d+).h5", os.path.basename(filepath))
            if match:
                epoch_num = int(match.group(1))
                if epoch_num not in epochs_to_keep:
                    try:
                        os.remove(filepath)
                        print(f"Deleted weights from epoch {epoch_num}.")
                    except Exception as e:
                        print(f"Error deleting file {filepath}: {e}")
                else:
                    print(f"Kept weights from epoch {epoch_num}.")
else:
    print("All saved epochs are kept.")

# ::::::::::::::::::Evaluation 2nd Model (Save Predictions):::::::::::::::::::::: #

# Close the Tee object properly
if isinstance(sys.stdout, Tee):
    original_stdout = sys.stdout.stdout  # Save reference to original stdout
    sys.stdout.close()  # Close the Tee object
    sys.stdout = original_stdout  # Restore original stdout
        
print("Skipping training and run evaluation models")

# Ensure source_tables_val is a NumPy array
source_tables_val = np.array(source_tables_val)
print("Starting Evaluation with predictions.")

# Directories for saving outputs
predictions_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/training_progress/predictions/"
output_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/training_progress/output/"

# Check for existing prediction files
weights_dir = "C:/Users/crgon/OneDrive/Desktop/trading/model_training/training_models/saved_weights/"
weight_files = glob.glob(os.path.join(weights_dir, "weights_v1_stacked_multi_small_main_11_epoch_*.h5"))

if weight_files:
    # Extract epoch numbers from filenames
    available_epochs = []
    for filepath in weight_files:
        match = re.search(
            r"weights_v1_stacked_multi_small_main_11_epoch_(\d+).h5", 
            os.path.basename(filepath)
        )
        if match:
            epoch_num = int(match.group(1))
            available_epochs.append(epoch_num)
    
    available_epochs.sort()
    print(f"\nAvailable epoch weights for prediction: {available_epochs}")
    print("Would you like to select a saved weight? (yes/no)")
    user_input = input().strip().lower()
    
    if user_input == 'yes':
        print("Enter epoch number to generate predictions from, or 'no' to skip prediction generation:")
        user_epoch_input = input().strip().lower()
        if user_epoch_input != 'no':
            try:
                epoch_to_predict = int(user_epoch_input)
                if epoch_to_predict in available_epochs:
                    weight_epochs = [epoch_to_predict]
                else:
                    print(f"Epoch {epoch_to_predict} not found. Skipping prediction generation.")
                    weight_epochs = []
            except ValueError:
                print("Invalid input. Skipping prediction generation.")
                weight_epochs = []
        else:
            print("Skipping prediction generation.")
            weight_epochs = []
    else:
        # If user says 'no', we'll run through all available epochs automatically
        weight_epochs = available_epochs

    # Build a validation generator matching our training loop approach
    # So we can evaluate identically to PrintUnscaledLoss + PrintPredictionsAndLoss_0
    val_generator_for_eval = ShiftingSequenceGenerator(
        X_val,
        y_val_dict,
        symbol_ids_val,
        np.concatenate(all_pattern_classes_val, axis=0).astype(np.int32),
        source_tables_val,
        encoded_4_val,
        seq_len=seq_len,
        batch_size=batch_size,
    )

    # Functions replicating PrintUnscaledLoss and PrintPredictionsAndLoss_0 logic for post-training evaluation
    def compute_and_print_unscaled_loss_2(
        model, val_generator, epoch
    ):
        """
        Replicates the PrintUnscaledLoss callback logic: 
        computes unscaled accuracy and group-level stats based on the entire val_generator.
        """
        # Predict on the entire val_generator
        predictions = model.predict(val_generator, verbose=0)
        if not isinstance(predictions, list):
            predictions = [predictions]

        group_list, table_list, y_true_dict = val_generator.get_full_order()

        combined_results = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"pred": [], "real": []})))
        group_sizes = defaultdict(int)
        all_pred_classes = []
        all_real_classes = []

        for i, output_name in enumerate(model.output_names):
            preds = predictions[i].ravel()  # shape: (total_sequences,)
            reals = np.array(y_true_dict[output_name], dtype=np.float32).ravel()

            # Convert probabilities to {0,1}
            pred_classes = (preds > 0.5).astype(int)
            real_classes = reals.astype(int)

            for idx in range(len(pred_classes)):
                src_table = table_list[idx]
                grp = group_list[idx]
                p = pred_classes[idx]
                r = real_classes[idx]
                combined_results[src_table][output_name][grp]["pred"].append(p)
                combined_results[src_table][output_name][grp]["real"].append(r)
                all_pred_classes.append(p)
                all_real_classes.append(r)
                group_sizes[(src_table, grp)] += 1

        # Print group-level accuracies
        for src_table, out_dict in combined_results.items():
            for output_name, group_dict in out_dict.items():
                for grp, data_map in sorted(group_dict.items()):
                    preds_arr = np.array(data_map["pred"])
                    reals_arr = np.array(data_map["real"])
                    accuracy = (preds_arr == reals_arr).mean() if len(preds_arr) > 0 else 0
                    print(f"[Epoch {epoch}] {output_name} Acc for {src_table}, Group {grp}: {accuracy:.4f}")

        # Overall accuracy
        all_pred_classes = np.array(all_pred_classes)
        all_real_classes = np.array(all_real_classes)
        if len(all_pred_classes) > 0:
            overall_acc = (all_pred_classes == all_real_classes).mean()
        else:
            overall_acc = 0
        print(f"[Epoch {epoch}] Overall Validation Accuracy: {overall_acc:.4f}")

        # Weighted accuracy
        total_samples = sum(group_sizes.values())
        weighted_acc_sum = 0.0
        for (table, grp), size in group_sizes.items():
            for out_name, group_dict in combined_results[table].items():
                data = group_dict.get(grp)
                if data and data["pred"] and data["real"]:
                    p_arr = np.array(data["pred"])
                    r_arr = np.array(data["real"])
                    grp_acc = (p_arr == r_arr).mean()
                    weighted_acc_sum += grp_acc * size

        if total_samples > 0:
            weighted_accuracy = weighted_acc_sum / total_samples
            print(f"[Epoch {epoch}] Weighted Validation Accuracy: {weighted_accuracy:.4f}")

    class PrintPredictionsAndLossHelper:
        @staticmethod
        def relaxed_tp_fp(true_labels, pred_labels):
            """
            For class=1, 'credit' if model predicts a '1' at the same index or one step
            before/after. Returns (tp, fp) using that rule.
            """
            n = len(true_labels)
            used_pred = np.zeros(n, dtype=bool)
            tp = 0

            for i in range(n):
                if true_labels[i] == 1:
                    idx_candidates = [i]
                    if i > 0:
                        idx_candidates.append(i - 1)
                    if i < n - 1:
                        idx_candidates.append(i + 1)

                    found_match = False
                    for j in idx_candidates:
                        if pred_labels[j] == 1 and not used_pred[j]:
                            used_pred[j] = True
                            found_match = True
                            tp += 1
                            break

            total_pred_ones = np.sum(pred_labels)
            fp = total_pred_ones - tp
            return tp, fp

        @staticmethod
        def compute_and_print_predictions_and_loss_2(model, val_generator, epoch):
            """
            Replicates the PrintPredictionsAndLoss_0 callback logic:
            prints sample predictions for each group in a short, human-inspectable format.
            """
            predictions = model.predict(val_generator, verbose=0)
            if not isinstance(predictions, list):
                predictions = [predictions]

            group_list, table_list, y_true_dict = val_generator.get_full_order()
            table_list = np.array(table_list)
            unique_groups = np.unique(group_list)
            num_to_sample_per_class = 10

            for i, output_name in enumerate(model.output_names):
                all_preds = predictions[i].ravel()  # shape: (total_sequences,)
                all_labels = np.array(y_true_dict[output_name]).ravel()
                pred_classes = (all_preds > 0.5).astype(int)

                print(f"\n[Epoch {epoch}] === Processing output: {output_name} ===")

                for grp in unique_groups:
                    grp_indices = np.where(group_list == grp)[0]
                    if grp_indices.size == 0:
                        continue

                    grp_pred = pred_classes[grp_indices]
                    grp_true = all_labels[grp_indices]

                    # Compute relaxed precision
                    grp_tp, grp_fp = PrintPredictionsAndLossHelper.relaxed_tp_fp(grp_true, grp_pred)
                    precision = grp_tp / (grp_tp + grp_fp) if (grp_tp + grp_fp) > 0 else 0.0
                    print(f"[Epoch {epoch}] Group {grp} => {output_name} (Relaxed) Precision: {precision:.4f}")

                    # Sample predictions for a quick readout
                    idx_class_0 = np.where(grp_true == 0)[0]
                    idx_class_1 = np.where(grp_true == 1)[0]
                    n0 = min(len(idx_class_0), num_to_sample_per_class)
                    n1 = min(len(idx_class_1), num_to_sample_per_class)

                    if n0 == 0 or n1 == 0:
                        print(f"[Epoch {epoch}] Group {grp} => Not enough samples for both classes.")
                        continue

                    s0 = np.random.choice(idx_class_0, n0, replace=False)
                    s1 = np.random.choice(idx_class_1, n1, replace=False)
                    chosen = np.random.permutation(np.concatenate([s0, s1]))

                    pred_sampled = grp_pred[chosen]
                    true_sampled = grp_true[chosen]
                    tables_sampled = table_list[grp_indices][chosen]

                    print(f"[Epoch {epoch}] Group {grp} => {output_name} Pred: {pred_sampled}")
                    print(f"[Epoch {epoch}] Group {grp} => {output_name} True: {true_sampled}")
                    print(f"Source tables: {tables_sampled}\n")

    # Now generate predictions and xlsx output
    # referencing the new approach (three-input generator) for each chosen epoch
    for epoch_to_predict in weight_epochs:
        # Load the weights
        weights_path = os.path.join(
            weights_dir,
            f"weights_v1_stacked_multi_small_main_11_epoch_{epoch_to_predict:02d}.h5"
        )
        print(f"Loading weights from epoch {epoch_to_predict} for prediction...")
        model.load_weights(weights_path)
        print("Weights loaded successfully!")

        # Prepare a text file to capture logs
        txt_filename = os.path.join(
            output_dir,
            f"evaluation_output_testing_new_epoch_{epoch_to_predict:02d}.txt"
        )

        with open(txt_filename, 'w') as f_out:
            original_stdout = sys.stdout
            sys.stdout = Tee2(sys.stdout, f_out)

            # 1) Compute & print unscaled loss across the validation generator
            compute_and_print_unscaled_loss_2(
                model=model,
                val_generator=val_generator_for_eval,
                epoch=epoch_to_predict
            )

            # 2) Print sample predictions, relaxed precision, etc.
            PrintPredictionsAndLossHelper.compute_and_print_predictions_and_loss_2(
                model=model,
                val_generator=val_generator_for_eval,
                epoch=epoch_to_predict
            )

            # 3) Save predictions to XLSX
            # We'll reuse the same approach: gather predictions in a structured dict.
            # 3) Save predictions to XLSX
            print("Generating predictions for XLSX output...")
            # Get predictions using the generator's three inputs
            predictions = model.predict(
                val_generator_for_eval,
                verbose=1
            )
            if not isinstance(predictions, list):
                predictions = [predictions]

            # Get the ordered data from the generator
            group_list, tbl_list, y_true_dict = val_generator_for_eval.get_full_order()
            tbl_list = np.array(tbl_list)
            group_list = np.array(group_list)

            # Build structured dict for predictions
            source_table_data = defaultdict(
                lambda: defaultdict(
                    lambda: defaultdict(
                        lambda: {'pred_classes': [], 'real_classes': []}
                    )
                )
            )

            # Process predictions for each output
            for out_name, pred_array in zip(model.output_names, predictions):
                pred_array = np.asarray(pred_array).flatten()
                y_array = np.array(y_true_dict[out_name]).flatten()
                
                min_length = min(len(pred_array), len(y_array), len(tbl_list))
                for i_idx in range(min_length):
                    table_val = tbl_list[i_idx]
                    group_val = group_list[i_idx]
                    table_name = str(table_val).strip()
                    
                    pred_class = (pred_array[i_idx] > 0.5).astype(int)
                    real_class = y_array[i_idx].astype(int)
                    
                    source_table_data[table_name][group_val][out_name]["pred_classes"].append(pred_class)
                    source_table_data[table_name][group_val][out_name]["real_classes"].append(real_class)

            # Write predictions to Excel
            print(f"Writing predictions to Excel...")
            output_file = os.path.join(
                predictions_dir,
                f"predictions_multi_stacked_testing_new_epoch_{epoch_to_predict:02d}.xlsx"
            )

            with pd.ExcelWriter(output_file) as writer:
                sheet_written = False
                
                for table_name, group_dict in source_table_data.items():
                    print(f"Processing table: {table_name}")
                    all_output_data = pd.DataFrame()
                    max_length = 0
                    
                    # Find max length among all groups + outputs
                    for grp_val in group_dict:
                        for o_name in group_dict[grp_val]:
                            length = len(group_dict[grp_val][o_name]["pred_classes"])
                            if length > max_length:
                                max_length = length
                    
                    # Build columns
                    for grp_val in sorted(group_dict.keys()):
                        for o_name in group_dict[grp_val]:
                            data_obj = group_dict[grp_val][o_name]
                            pred_classes = data_obj["pred_classes"]
                            real_classes = data_obj["real_classes"]
                            
                            min_len = min(len(pred_classes), len(real_classes))
                            pred_classes = pred_classes[:min_len]
                            real_classes = real_classes[:min_len]
                            
                            # Pad with NaN if needed
                            if min_len < max_length:
                                pred_classes += [np.nan] * (max_length - min_len)
                                real_classes += [np.nan] * (max_length - min_len)
                            
                            real_col_name = f'Actual_{grp_val}_{o_name}'
                            pred_col_name = f'Pred_{grp_val}_{o_name}'
                            all_output_data[real_col_name] = real_classes
                            all_output_data[pred_col_name] = pred_classes
                    
                    if not all_output_data.empty:
                        sheet_name = f"{table_name}"[:31]  # Excel sheet names limited to 31 chars
                        all_output_data.to_excel(writer, sheet_name=sheet_name, index=False)
                        sheet_written = True
                        print(f"Written sheet: {sheet_name}")
                
                if not sheet_written:
                    pd.DataFrame({"No Data": []}).to_excel(writer, sheet_name="No_Data", index=False)

            print(f"Predictions from epoch {epoch_to_predict} saved to '{output_file}'")

            # Restore stdout
            sys.stdout.close()
            sys.stdout = original_stdout

else:
    print("No saved weights found. Cannot generate predictions.")




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
import datetime
import threading
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import multiply
from tensorflow.keras import mixed_precision
from sklearn.preprocessing import MinMaxScaler, OrdinalEncoder
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, Sequential
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

# :::: Main Parameters: scroll to line 391 <Model Start> to change unit dimentions ::::

# Retrieve the active symbols from the environment variable
active_symbols = os.environ.get('ACTIVE_SYMBOLS', '')
active_symbols_list = [sym for sym in active_symbols.split(',') if sym]  # Ensure no empty strings

# Determine if load all symbols
load_all_symbols = not active_symbols_list  # True if active_symbols_list is empty

if load_all_symbols:
    print("No active symbols provided. Loading all tables as normal.")
else:
    print(f"Active symbols received: {active_symbols_list}")


def get_dynamic_group_label_load():
    current_time = datetime.datetime.now()
    minute = current_time.minute

    if minute % 5 != 0:
        return None  # Not a 5-minute interval, skip loading data

    if minute == 0:
        return [5, 15, 30, 60]
    elif minute == 15:
        return [5, 15]
    elif minute == 30:
        return [5, 15, 30]
    elif minute == 45:
        return [5, 15]
    else:
        return [5]

# Replace the static group_label_load with this function call
group_label_load = get_dynamic_group_label_load()

# check before loading and processing data
if group_label_load is None:
    print("Not a 5-minute interval. Skipping data loading and processing.")
    # Exit the script or return from the function
    import sys
    sys.exit()

#group_label_load = [5, 15, 30, 60]

print(f"Loading data for groups: {group_label_load}")

# :::::::::::::::::::::::::::::TensorFlow setup:::::::::::::::::::::::::::::: #

# Suppresses TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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
    print(f"Connecting to live database:")
    conn = sqlite3.connect(db_path)
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    print("Executing query to get table names...")
    tables = pd.read_sql_query(query, conn)
    all_data = []
    source_tables = []
    
    for table_name in tables['name']:
        # If active_symbols_list is not empty, filter tables
        if not load_all_symbols:
            if table_name not in active_symbols_list:
                print(f"Skipping table '{table_name}' as it's not an active symbol.")
                continue

        print(f"Loading data from table: {table_name}")
        df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
        
        # Filter rows based on group_label_load
        if 'Group' in df.columns:
            df = df[df['Group'].isin(group_label_load)]
            print(f"Filtered rows based on group_label_load: {group_label_load}")
        
        all_data.append(df)
        source_tables.append(table_name)
        
    conn.close()
    
    if not all_data:
        print("No data was loaded from the database.")
        sys.exit()

    print("Live data loading complete.")
    return all_data, source_tables

def create_overlapping_sequences(data, seq_len, step):
    sequences = []
    for i in range(0, len(data) - seq_len + 1, step):
        seq = data[i:i + seq_len]
        sequences.append(seq)
    return np.array(sequences)

def preprocess_and_combine_data(data_list, source_tables, seq_len, seq_step, is_training=True, existing_scalers=None):
    print("Preprocessing and combining data...")

    all_X_encoded = []
    all_X_remaining = []
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
    mapping_filename = 'C:\\Users\\crgon\\OneDrive\\Desktop\\trading\\metatrader\\dwxconnect-main\\python\\models\\symbol_id_mapping.json'
    if os.path.exists(mapping_filename):
        with open(mapping_filename, 'r') as f:
            symbol_id_mapping = json.load(f)
        symbol_id_updated = False
    else:
        symbol_id_mapping = {}
        symbol_id_updated = True  # Mapping will be updated since it doesn't exist

    for idx, (df, table_name) in enumerate(zip(data_list, source_tables)):
        print(f"Processing data from table: {table_name}")

        # Drop target columns if they exist
        target_columns = ['Target1', 'Target2', 'Target3', 'Target4']
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

        """ if columns_to_encode is None:
            columns_to_encode = {
                'Day': list(range(1, 32)),
                'Month: list(range(1, 13)),
                'Hour': list(range(0, 24)),
                'Minute': list(range(0, 60)),
                'Group': [1, 5, 15, 30, 60],                
            }    
            # Add 'Pattern_1' to 'Pattern_19' and 'Pattern_Class'
            for i in range(1, 20):
                columns_to_encode[f'Pattern_{i}'] = [0, 1, 2, 3, 4, 5, 6]
            columns_to_encode['Pattern_Class'] = list(range(0, 160)) """

        # Initialize columns_to_encode if not done already
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
            columns_to_encode['Pattern_Class'] = list(range(0, 160))

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
        #print(f"Table '{table_name}' remaining columns for scaling: {remaining_columns}")

        # Scale remaining columns using per-table scaling
        if is_training:
            df[remaining_columns] = feature_scaler.fit_transform(df[remaining_columns].astype('float32'), table_name)
        else:
            df[remaining_columns] = feature_scaler.transform(df[remaining_columns].astype('float32'), table_name)

        # Sort and process encoded_Group groups
        if 'encoded_Group' in df.columns:
            sorted_encoded_values = sorted(df['encoded_Group'].unique())
        else:
            sorted_encoded_values = [0]

        for group in sorted_encoded_values:
            group_df = df[df['encoded_Group'] == group]
            print(f"Processing Group: {group}, Group Size Row Count: {len(group_df)}")

            if not is_training:
                # Only create a sequence ending at the last row of the group
                if len(group_df) >= seq_len:
                    seq_start = len(group_df) - seq_len
                    seq_end = len(group_df)
                    seq_encoded = group_df[encoded_columns].iloc[seq_start:seq_end].values
                    seq_remaining = group_df[remaining_columns].iloc[seq_start:seq_end].values
                    group_sequences_encoded = np.array([seq_encoded])
                    group_sequences_remaining = np.array([seq_remaining])
                else:
                    print(f"Group {group} in table {table_name} does not have enough data to create a sequence of length {seq_len}. Skipping...")
                    continue  # Skip groups that don't have enough data
            else:
                # Create overlapping sequences for training
                group_sequences_encoded = create_overlapping_sequences(group_df[encoded_columns].values, seq_len, seq_step)
                group_sequences_remaining = create_overlapping_sequences(group_df[remaining_columns].values, seq_len, seq_step)

            all_X_encoded.append(group_sequences_encoded)
            all_X_remaining.append(group_sequences_remaining)
            all_encoded_4.append(np.full(group_sequences_encoded.shape[0], group))
            all_source_tables.append(np.full(group_sequences_encoded.shape[0], table_name))
            all_symbol_ids.append(np.full(group_sequences_encoded.shape[0], symbol_id))

            # Clear variables to free memory
            del group_df, group_sequences_encoded, group_sequences_remaining

        # Clear variables to free memory
        del df, encoded_df

    # Save the updated symbol_id_mapping if in training mode and mapping has been updated
    if is_training and symbol_id_updated:
        with open(mapping_filename, 'w') as f:
            json.dump(symbol_id_mapping, f)
        print(f"Symbol ID mapping saved to {mapping_filename}")

    return all_X_encoded, all_X_remaining, all_encoded_4, all_symbol_ids, feature_scaler, encoder, columns_to_encode, all_source_tables

def scale_combined_data(all_X_encoded, all_X_remaining, all_encoded_4, all_symbol_ids, 
                        all_source_tables, feature_scaler=None, is_training=True):
    print("Scaling encoded data from all tables")
    
    # Combine data
    combined_X_encoded = np.concatenate(all_X_encoded, axis=0)
    combined_X_remaining = np.concatenate(all_X_remaining, axis=0)
    combined_encoded_4 = np.concatenate(all_encoded_4, axis=0)
    combined_source_tables = np.concatenate(all_source_tables, axis=0)
    combined_symbol_ids = np.concatenate(all_symbol_ids, axis=0)

    # Clear memory
    del all_X_encoded, all_X_remaining, all_encoded_4, all_source_tables, all_symbol_ids

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

    print(f"Final combined shapes - X: {final_X.shape}")
    return final_X, combined_encoded_4, combined_symbol_ids, combined_source_tables, feature_scaler


print("Initializing and loading scalers/encoders...")

# Load the scalers and encoder for validation data
scalers_encoder = joblib.load("C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/models/scalers/scalers_stacked_multi_main_1.pkl")
feature_scaler = scalers_encoder['feature_scaler']
encoder = scalers_encoder['encoder']
columns_to_encode = scalers_encoder['columns_to_encode']
seq_len = scalers_encoder['seq_len']
seq_step = scalers_encoder['seq_step']

# Load symbol_id mapping
with open('C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/models/symbol_id_mapping.json', 'r') as f:
    symbol_id_mapping = json.load(f)
num_symbols = len(symbol_id_mapping)

# Model and training parameters
seq_len = 36                    # Sequence length (15-minute data or any other sequence)
seq_step = 1                    # Number of steps per dequence
batch_size = 1                  # Adjust as needed
output_names = ['Signal']       # Adjust as needed

# Assuming your dataset has a shape of (num_samples, seq_len, d_model)
d_model = 31
ff_dim = d_model*4 

# Use the number of unique symbol IDs directly
num_enc_layers = 9                    # Encoding Attention layers
num_dec_layers = 9                    # Decoding Attention layers
num_heads = 9                         # Number of attention heads
dropout_rate = 0.00                    # Dropout rate for regularization                   # Feedforward network dimension

# filter and kernel size values
filters = [64, 128, 256, 256]         
# last saved model used [64, 128, 256, 256] = weights_v1_stacked_multi_1_#0.71.h5

kernel_sizes = [6, 9, 18, 36]
# last saved model used [6, 9, 18, 36] = weights_v1_stacked_multi_1_#0.71.h5
# last saved model used [36, 18, 9, 36] = weights_v1_stacked_multi_2_#0.70.h5

d_ffd = filters[-1]                   # Input Dense layer dimentions

# Prepare to load model weights
weights_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/models/model_weights/weights_v1_stacked_multi_main_1.h5"

# Define functions
def load_and_preprocess_data():
    global X_val, symbol_ids_val, encoded_4_val, source_tables_val, feature_scaler, encoder, columns_to_encode
    print("About to load and preprocess training data...")
    df_val, source_tables_val = load_data_from_sqlite("C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/data/databases/Processed_multi_stacked_1.db")

    # Preprocess and combine validation data
    print("Starting to preprocess Live data...")
    all_X_val_encoded, all_X_val_remaining, all_encoded_4_val, all_symbol_ids_val, feature_scaler, encoder, columns_to_encode, source_tables_val = preprocess_and_combine_data(
        df_val, source_tables_val, seq_len, seq_step, is_training=False,
        existing_scalers={'feature_scaler': feature_scaler, 'encoder': encoder, 'columns_to_encode': columns_to_encode}
    )

    # Scale and combine validation data
    print("Scaling and combining Live data...")
    X_val, encoded_4_val, symbol_ids_val, source_tables_val, feature_scaler = scale_combined_data(
        all_X_val_encoded, all_X_val_remaining, all_encoded_4_val, all_symbol_ids_val, source_tables_val, feature_scaler=feature_scaler, is_training=False
    )

    # Display shapes of the validation data
    print(f"Live data shapes - X: {X_val.shape}, source_tables: {len(source_tables_val)}, symbol_ids: {symbol_ids_val.shape}\n")            # Number of features (input dim per time step)


def initialize_model():
    global model, weights
    print("Initializing the model...")

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

    # Load weights immediately after model definition
    if os.path.exists(weights_checkpoint_path):
        print("Loading model weights...")
        model.load_weights(weights_checkpoint_path)
        print("Weights loaded successfully.")
    else:
        print("No saved weights found. Using random initialization.")

    print("Model initialized!")
    print("Total number of parameters in the model:", model.count_params())

# ::::::::::::::::::Production & Predictions:::::::::::::::::::::: #

def load_data_and_initialize_model():
    # Create threads for loading data and initializing model
    data_thread = threading.Thread(target=load_and_preprocess_data)
    model_thread = threading.Thread(target=initialize_model)

    # Start both threads
    data_thread.start()
    model_thread.start()

    # Wait for both threads to complete
    data_thread.join()
    model_thread.join()

    print("Data loading and model initialization completed.")

# Load data and initialize model
load_data_and_initialize_model()

# Ensure source_tables_val is a NumPy array
source_tables_val = np.array(source_tables_val).flatten()

print("Starting evaluation and saving predictions to text file.")

X_val = X_val.astype(np.float32)
symbol_ids_val = symbol_ids_val.astype(np.int32)
encoded_4_val = np.array(encoded_4_val).astype(int).flatten()

X_val_inputs = {
    'input_1': X_val,
    'input_2': symbol_ids_val
}

# Get predictions from the model
y_pred = model.predict(X_val_inputs, batch_size=batch_size)

# Get predicted class labels
pred_classes = np.argmax(y_pred, axis=1)

# Initialize predictions dictionary
predictions_dict = {}

for i in range(len(pred_classes)):
    source_table = source_tables_val[i].strip()
    group = encoded_4_val[i]
    pred_class = pred_classes[i]
    
    if source_table not in predictions_dict:
        predictions_dict[source_table] = {}
    # Store the prediction for the group
    # If multiple predictions exist for the same group, keep the latest one
    predictions_dict[source_table][group] = pred_class

# Now, for each source_table, generate the output string
output_lines = []

for source_table in predictions_dict:
    group_predictions = predictions_dict[source_table]
    # Sort the groups from smallest to largest
    sorted_groups = sorted(group_predictions.keys())
    group_pred_strings = [f"{group}:{group_predictions[group]}" for group in sorted_groups]
    # Combine into the desired format
    output_line = f"{source_table}[{', '.join(group_pred_strings)}]"
    output_lines.append(output_line)

# Define the output file path
output_file = "C:/Users/crgon/OneDrive/Desktop/trading/metatrader/dwxconnect-main/python/models/predictions/predictions_multi_stacked_1.txt"  # Replace with your desired path

# Ensure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

# Write to the file, overwriting any existing content
with open(output_file, 'w') as f:
    f.write('\n'.join(output_lines))

print(f"Predictions saved to predictions_multi_stacked_1.txt.")












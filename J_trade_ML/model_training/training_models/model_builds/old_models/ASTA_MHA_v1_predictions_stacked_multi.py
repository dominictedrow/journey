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
#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

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
from tensorflow.keras.layers import MultiHeadAttention, Add, Conv2D, Concatenate, Conv1D
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, LayerNormalization, Flatten
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

seq_len = 16                     # Currently: 15 Minute data: with positional encoding has 80 features per seq_len     
seq_step = 1                    # Number of steps per dequence
epochs = 100                    # Training iterations
batch_size = 256                # Samples per batch (last was 32)

target_group = True
remaining_group = False

# :::: Adaptive Sphere Transformation ::::

asta_threshold_1 = 0.1          # Determines when to consider the model's performance stable.   
asta_threshold_2 = 0.1          # Defines the performance metric value above which the centers are locked.             
asta_preformance = 0.01         # Controls how quickly the model perceives stability. (0.01 default)

# :::: learning warmup & Adam Optimizer ::::

custom_lr = False               # Activate warmup_steps
warmup_steps = int(4000)        # Number of batches for lr warmup_steps  
lr_scale = 1                    # 1 is default for no change (increases starting lr 4 warmup_steps)

fixed_learning_rate = 0.000005 # Used if custom_lr=False
clipnorm = 1.0
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-9
amsgrad = False

# :::: Training Target Predictions ::::
""" 
target_columns should be the name of any Target columns and output_names
should be the the targets you want to use, which requires adjusting drop_targets
to match the output_names you selected: ['High', 'Low', 'Close']. Curretly only
using 1 Target of 3 possible in dataset 
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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['NCCL_DEBUG_SUBSYS'] = 'ALL'
os.environ['NCCL_P2P_DISABLE'] = '1'  # Adjust this based on your setup

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

# Initialize MultiWorkerMirroredStrategy
strategy = tf.distribute.MultiWorkerMirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
print("Eager execution:", tf.executing_eagerly())

# ::::::::::::::::::::::Data Processing Transformation::::::::::::::::::::::: #

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

def preprocess_and_combine_data(data_list, source_tables, seq_len, seq_step, target_columns, output_names, is_training=True, existing_scalers=None, target_group=False, remaining_group=False):
    print("Preprocessing and combining data...")

    all_X_encoded = []
    all_X_remaining = []
    all_y = {name: [] for name in output_names}
    all_encoded_4 = []
    all_source_tables = []

    # Initialize feature_scalers and target_scalers
    if existing_scalers:
        feature_scalers = existing_scalers.get('feature_scalers', {})
        target_scalers = existing_scalers.get('target_scalers', {})
        encoder = existing_scalers.get('encoder')
        columns_to_encode = existing_scalers.get('columns_to_encode')
    else:
        feature_scalers = {}
        target_scalers = {}
        encoder = None
        columns_to_encode = None

    # For target_group=True, collect global min and max per target column
    if target_group and is_training:
        target_mins = {name: None for name in output_names}
        target_maxs = {name: None for name in output_names}

    # For remaining_group=True, collect global min and max per remaining column
    if remaining_group and is_training:
        remaining_column_mins = {}
        remaining_column_maxs = {}

    # Initialize sets to keep track of numeric and non-numeric columns
    numeric_columns_set = set()
    non_numeric_columns_set = set()
    all_dfs = []

    for idx, (df, table_name) in enumerate(zip(data_list, source_tables)):
        print(f"Processing data from table: {table_name}")

        # Check if the required target columns exist in the DataFrame
        missing_columns = set(target_columns) - set(df.columns)
        if missing_columns:
            print(f"Warning: DataFrame for {table_name} is missing these required target columns: {missing_columns}")
            print(f"Available columns: {df.columns.tolist()}")
            print(f"Skipping table: {table_name}")
            continue

        print("Initial target columns:", target_columns)
        #used_targets = target_columns
        #used_targets = target_columns[:drop_targets]
        used_targets = [target_columns[3]]
        df_targets = df[used_targets].astype('float32')

        # Rename the target columns to match output_names
        df_targets.columns = output_names  # Now df_targets has column 'Signal'

        df = df.drop(columns=target_columns, errors='ignore')
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        df['table_id'] = idx
        df['source_table'] = table_name

        if columns_to_encode is None:
            columns_to_encode = {
                0: list(range(1, 32)),
                1: list(range(1, 13)),
                2: list(range(0, 24)),
                3: list(range(0, 56)),
                4: [5, 15, 30, 60],
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

        # Collect numeric and non-numeric columns
        numeric_columns_df = df.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_columns_df = [col for col in df.columns if col not in numeric_columns_df and col != 'source_table']
        numeric_columns_set.update(numeric_columns_df)
        non_numeric_columns_set.update(non_numeric_columns_df)
        # Drop non-numeric columns from df
        df.drop(columns=non_numeric_columns_df, inplace=True)

        # Collect min and max for remaining_columns or scale per table
        if is_training:
            if remaining_group:
                # Compute global min and max per column
                for col in remaining_columns:
                    col_min = df[col].min()
                    col_max = df[col].max()
                    if col in remaining_column_mins:
                        remaining_column_mins[col] = min(remaining_column_mins[col], col_min)
                        remaining_column_maxs[col] = max(remaining_column_maxs[col], col_max)
                    else:
                        remaining_column_mins[col] = col_min
                        remaining_column_maxs[col] = col_max
            else:
                # Scale remaining columns for each table separately
                if table_name not in feature_scalers:
                    feature_scalers[table_name] = MinMaxScaler()
                df[remaining_columns] = feature_scalers[table_name].fit_transform(df[remaining_columns].astype('float32'))
        else:
            if remaining_group:
                if 'remaining_scaler' not in feature_scalers:
                    print("Error: Remaining columns scaler not provided for remaining_group=True.")
                    raise KeyError("Remaining columns scaler not provided for remaining_group=True.")
                scaler = feature_scalers['remaining_scaler']
                df[remaining_columns] = scaler.transform(df[remaining_columns])
            else:
                scaler = feature_scalers.get(table_name)
                if scaler:
                    df[remaining_columns] = scaler.transform(df[remaining_columns].astype('float32'))
                else:
                    print(f"Error: Feature scaler for table '{table_name}' not found.")
                    raise KeyError(f"Feature scaler for table '{table_name}' not found.")

        # Collect targets if target_group is True
        if is_training and target_group:
            for i, name in enumerate(output_names):
                col_values = df_targets[name].values
                col_min = col_values.min()
                col_max = col_values.max()
                if target_mins[name] is None or col_min < target_mins[name]:
                    target_mins[name] = col_min
                if target_maxs[name] is None or col_max > target_maxs[name]:
                    target_maxs[name] = col_max
        elif is_training and not target_group:
            # Scale targets for each table separately
            if table_name not in target_scalers:
                target_scalers[table_name] = {name: MinMaxScaler() for name in output_names}
            df_targets_scaled_list = []
            for i, name in enumerate(output_names):
                target_data = df_targets[[name]]  # Use the renamed column
                scaler = target_scalers[table_name][name]
                df_targets_scaled_list.append(scaler.fit_transform(target_data))
            df_targets_scaled = np.column_stack(df_targets_scaled_list)
        elif not is_training:
            if target_group:
                if not target_scalers:
                    print("Error: Target scalers not provided for target_group=True.")
                    raise KeyError("Target scalers not provided for target_group=True.")
                df_targets_scaled_list = []
                for i, name in enumerate(output_names):
                    target_data = df_targets[[name]]  # Use the renamed column
                    scaler = target_scalers[name]
                    df_targets_scaled_list.append(scaler.transform(target_data))
                df_targets_scaled = np.column_stack(df_targets_scaled_list)
            else:
                if table_name in target_scalers:
                    df_targets_scaled_list = []
                    for i, name in enumerate(output_names):
                        target_data = df_targets[[name]]  # Use the renamed column
                        scaler = target_scalers[table_name][name]
                        df_targets_scaled_list.append(scaler.transform(target_data))
                    df_targets_scaled = np.column_stack(df_targets_scaled_list)
                else:
                    print(f"Error: Target scaler for table '{table_name}' not found.")
                    raise KeyError(f"Target scaler for table '{table_name}' not found.")

        sorted_encoded_values = sorted(df['encoded_4'].unique())
        for group in sorted_encoded_values:
            group_df = df[df['encoded_4'] == group]
            group_df_targets = df_targets[df['encoded_4'] == group]

            print(f"Processing Group: {group}, Group Size Row Count: {len(group_df)}")

            # Create sequences separately for encoded_columns and remaining_columns
            group_sequences_encoded = create_overlapping_sequences(group_df[encoded_columns].values, seq_len, seq_step)
            group_sequences_remaining = create_overlapping_sequences(group_df[remaining_columns].values, seq_len, seq_step)

            # Get group targets
            group_indices = group_df.index
            if is_training and target_group:
                group_targets = group_df_targets.values[seq_len - 1:len(group_sequences_encoded) + seq_len - 1]
            else:
                group_targets = df_targets_scaled[group_indices][seq_len - 1:len(group_sequences_encoded) + seq_len - 1]

            if is_training:
                # Balancing step for training data only
                target_values = group_targets[:, 0]  # Assuming we have one target column
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
                    else:
                        target_indices = np.array(target_indices)

                    balanced_group_sequences_encoded.append(group_sequences_encoded[target_indices])
                    balanced_group_sequences_remaining.append(group_sequences_remaining[target_indices])
                    balanced_group_targets.append(group_targets[target_indices])

                group_sequences_encoded = np.concatenate(balanced_group_sequences_encoded, axis=0)
                group_sequences_remaining = np.concatenate(balanced_group_sequences_remaining, axis=0)
                group_targets = np.concatenate(balanced_group_targets, axis=0)

            # Store the sequences separately
            all_X_encoded.append(group_sequences_encoded)
            all_X_remaining.append(group_sequences_remaining)

            for i, name in enumerate(output_names):
                all_y[name].append(group_targets[:, i])

            all_encoded_4.append(np.full(group_sequences_encoded.shape[0], group))
            all_source_tables.append(np.full(group_sequences_encoded.shape[0], table_name))

            all_dfs.append(df)

    if is_training and target_group:
        # Fit scalers using global min and max per target column
        for name in output_names:
            data_min_ = target_mins[name]
            data_max_ = target_maxs[name]

            # Create a DataFrame with min and max values
            min_max_df = pd.DataFrame({name: [data_min_, data_max_]})

            # Create and fit the scaler
            target_scalers[name] = MinMaxScaler()
            target_scalers[name].fit(min_max_df)

        # Now scale the collected targets
        for name in output_names:
            for i in range(len(all_y[name])):
                target_data = pd.DataFrame(all_y[name][i], columns=[name])
                all_y[name][i] = target_scalers[name].transform(target_data).flatten()

    if is_training and remaining_group:
        # Compute global min and max per remaining column
        data_min_ = np.array([remaining_column_mins[col] for col in remaining_columns])
        data_max_ = np.array([remaining_column_maxs[col] for col in remaining_columns])

        # Create a DataFrame with min and max values
        min_max_df = pd.DataFrame([data_min_, data_max_], columns=remaining_columns)

        # Create and fit the scaler
        remaining_scaler = MinMaxScaler()
        remaining_scaler.fit(min_max_df)

        # Store the scaler
        feature_scalers['remaining_scaler'] = remaining_scaler

        # Now, transform the remaining_columns in each DataFrame
        for df in all_dfs:
            df[remaining_columns] = remaining_scaler.transform(df[remaining_columns])

    if not all_dfs:
        raise ValueError("No valid data to process. All tables were skipped.")

    numeric_columns = list(numeric_columns_set)
    non_numeric_columns = list(non_numeric_columns_set)
    print(f"Dropping non-numeric columns: {non_numeric_columns}")

    # Ensure remaining_columns only contains numeric columns
    remaining_columns = [col for col in remaining_columns if col in numeric_columns]

    return (all_X_encoded, all_X_remaining, all_y, all_encoded_4, feature_scalers, target_scalers, encoder, columns_to_encode,
            all_source_tables, encoded_columns, remaining_columns)

def scale_combined_data(all_X_encoded, all_X_remaining, all_y, all_encoded_4, all_source_tables, feature_scaler=None, is_training=True):
    print("Scaling encoded data from all tables")
    
    # Concatenate encoded and remaining sequences separately
    combined_X_encoded = np.concatenate(all_X_encoded, axis=0)
    combined_X_remaining = np.concatenate(all_X_remaining, axis=0)
    combined_y = {name: np.concatenate(values, axis=0).astype(np.float32) for name, values in all_y.items() if len(values) > 0}
    combined_encoded_4 = np.concatenate(all_encoded_4, axis=0)
    combined_source_tables = np.concatenate(all_source_tables, axis=0)
    
    # Reshape encoded part for scaling
    num_encoded_columns = combined_X_encoded.shape[2]
    encoded_part_reshaped = combined_X_encoded.reshape(-1, num_encoded_columns)
    
    # Scale only the encoded columns
    if is_training:
        if feature_scaler is None:
            feature_scaler = MinMaxScaler()
            scaled_encoded = feature_scaler.fit_transform(encoded_part_reshaped)
        else:
            scaled_encoded = feature_scaler.fit_transform(encoded_part_reshaped)
    else:
        if feature_scaler is None:
            raise ValueError("Feature scaler must be provided for validation data.")
        scaled_encoded = feature_scaler.transform(encoded_part_reshaped)
    
    # Reshape scaled_encoded back to original 3D shape
    scaled_encoded = scaled_encoded.reshape(combined_X_encoded.shape[0], combined_X_encoded.shape[1], num_encoded_columns)
    
    # Concatenate scaled encoded columns with the remaining columns
    final_X = np.concatenate([scaled_encoded, combined_X_remaining], axis=2).astype(np.float32)
    
    print(f"Final combined shapes - X: {final_X.shape}, y: {[y.shape for y in combined_y.values()]}")
    
    return final_X, combined_y, combined_encoded_4, combined_source_tables, feature_scaler

def randomize_data(X, y, source_tables):
    print("Randomizing data...")
    perm = np.random.permutation(len(X))
    X_shuffled = X[perm]
    y_shuffled = {key: value[perm] for key, value in y.items()}
    source_tables_shuffled = source_tables[perm]
    print("Randomization complete.")
    return X_shuffled, y_shuffled, source_tables_shuffled

print("About to load and preprocess training data...")
df_train, source_tables_train = load_data_from_sqlite("/home/jd/Desktop/trading/data/Processed_multi_stacked_HLC_Train.db")
df_val, source_tables_val = load_data_from_sqlite("/home/jd/Desktop/trading/data/Processed_multi_stacked_HLC_Val.db")

# Preprocess and combine training data
print("Starting to preprocess training data...")
all_X_train_encoded, all_X_train_remaining, all_y_train, all_encoded_4_train, feature_scalers, target_scalers, encoder, columns_to_encode, source_tables_train, encoded_columns, remaining_columns = preprocess_and_combine_data(
    df_train, source_tables_train, seq_len, seq_step, target_columns, output_names, is_training=True, target_group=target_group, remaining_group=remaining_group
)

# Scale and combine all training data
print("Scaling and combining all training data...")
X_train, y_train_dict, encoded_4_train, source_tables_train, feature_scaler = scale_combined_data(
    all_X_train_encoded, all_X_train_remaining, all_y_train, all_encoded_4_train, source_tables_train, is_training=True
)

# Randomize the training data
X_train, y_train_dict, source_tables_train = randomize_data(X_train, y_train_dict, source_tables_train)
print(f"Final training data shapes - X: {X_train.shape}, y: {[y.shape for y in y_train_dict.values()]}, source_tables: {len(source_tables_train)}\n")

# Save the scalers for future use
print("Saving the scalers for future use...")
joblib.dump({
    'feature_scalers': feature_scalers,
    'target_scalers': target_scalers,
    'encoder': encoder,
    'columns_to_encode': columns_to_encode,
    'seq_len': seq_len,
    'seq_step': seq_step,
    'remaining_group': remaining_group
}, 'scalers_stacked_multi.pkl')

# Load the scalers and encoder for validation data
scalers_encoder = joblib.load('scalers_stacked_multi.pkl')
feature_scalers = scalers_encoder['feature_scalers']
target_scalers = scalers_encoder['target_scalers']
encoder = scalers_encoder['encoder']
columns_to_encode = scalers_encoder['columns_to_encode']
seq_len = scalers_encoder['seq_len']
seq_step = scalers_encoder['seq_step']
remaining_group = scalers_encoder.get('remaining_group', False)


# Preprocess and combine validation data
all_X_val_encoded, all_X_val_remaining, all_y_val, all_encoded_4_val, _, _, _, _, source_tables_val, encoded_columns, remaining_columns = preprocess_and_combine_data(
    df_val, source_tables_val, seq_len, seq_step, target_columns, output_names, is_training=False,
    existing_scalers={'feature_scalers': feature_scalers, 'target_scalers': target_scalers, 'encoder': encoder, 'columns_to_encode': columns_to_encode},
    target_group=target_group, remaining_group=remaining_group
)

if feature_scaler is None:
    raise ValueError("Feature scaler must be provided for validation data.")

# Scale and combine validation data
X_val, y_val_dict, encoded_4_val, source_tables_val, _ = scale_combined_data(
    all_X_val_encoded, all_X_val_remaining, all_y_val, all_encoded_4_val, source_tables_val, feature_scaler=feature_scaler, is_training=False
)

print(f"Validation data shapes - X: {X_val.shape}, y: {[y.shape for y in y_val_dict.values()]}, source_tables: {len(source_tables_val)}\n")

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
        plt.savefig("/home/jd/Desktop/trading/model_testing_text/training_models/training_progress/multi_train_loss_v1_stacked_multi.png") 
        self.root.update()

print("About to initialize the model...")

with strategy.scope():
    
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
    num_enc_layers = 8                    # Encoding Attention layers
    num_dec_layers = 8                    # Decoding Attention layers
    dropout_rate = 0.00                   # Dropout rate

    initial_filters = int(128)            # 1nd unit hidden layer exspansion (MHA)
    d_ffn = int(512)       # 1nd unit exspansion/supression (LSTM)

    # ::::::::::::::Main Model Training (Multi Target Prediction):::::::::::::: #

    outputs = []
    for name in output_names: 
        q_ffn = initial_filters
        pos = PositionalEncoding(seq_len, d_model)(inputs)
        
        # Custom Multi-Head Attention Model
        mha_model = CustomMultiHeadAttentionModel(
            seq_len, d_model, num_heads_1, num_enc_layers, num_dec_layers, dropout_rate
        )(pos)
        
        mha_model = Add()([mha_model, inputs])
        
        """ # Convolutional Positional Encoding (if applicable)
        conv_positional_encoding_layer = ConvolutionalPositionalEncoding()
        mha_model = conv_positional_encoding_layer(mha_model) """
        
        # Global Temporal Convolution
        mha_model = Conv1D(
            filters=q_ffn,
            kernel_size=min(seq_len // 2, 64),
            activation='relu',
            kernel_initializer='he_normal',
            padding='same',
            use_bias=True
        )(mha_model)

        # Determine the input width dynamically
        input_shape = tf.shape(mha_model)
        input_width = input_shape[2]

        # Global Feature Convolution
        mha_model = Conv1D(
            filters=q_ffn * 2,
            kernel_size=min(d_model // 2, 32),
            activation='relu',
            kernel_initializer='he_normal',
            padding='same',
            use_bias=True
        )(mha_model)

        # 1x1 Convolution to transform features
        mha_model = Conv1D(
            filters=q_ffn * 4,
            kernel_size=1,
            activation='relu',
            kernel_initializer='he_normal',
            padding='same',
            use_bias=True
        )(mha_model)
        
        # Flatten for Dense Layers
        mha_model = Flatten()(mha_model)
        mha_model = Dense(d_ffn, activation='relu', kernel_initializer=HeNormal(), use_bias=True)(mha_model)

        # Output Layer
        output = Dense(
            1,
            activation='linear',
            kernel_initializer=HeNormal(),
            use_bias=True,
            name=name,
            dtype='float32'
        )(mha_model)
        
        outputs.append(output)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    print("Model initialized!")
    print("Total number of parameters in the model:", model.count_params())

    # :::::::::::::::::::::Model Checkpoint For Weights::::::::::::::::::::::: #

    # Define the model checkpoint for weights only
    weights_checkpoint_path =    "/home/jd/Desktop/trading/model_testing_text/training_models/saved_weights/weights_v1_stacked_multi.h5"
    weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', 
                                        verbose=1, save_best_only=True, mode='min', save_weights_only=True)

    # Define the model checkpoint for the full model
    full_model_checkpoint_path = "/home/jd/Desktop/trading/model_testing_text/training_models/saved_weights/model_v1_FULL_stacked_multi.h5"
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
        def __init__(self, target_scalers, validation_data, source_tables, batch_size, encoded_4_val, target_group=False):
            super().__init__()
            self.target_scalers = target_scalers
            self.validation_data = validation_data
            self.source_tables = source_tables
            self.encoded_4_val = encoded_4_val
            self.batch_size = batch_size
            self.target_group = target_group  # Added to handle target scaling mode

        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val_dict = self.validation_data
            predictions = self.model.predict(X_val, batch_size=self.batch_size)

            if not isinstance(predictions, list):
                predictions = [predictions]

            num_samples = X_val.shape[0]
            random_indices = np.random.choice(num_samples, 5, replace=False)

            # Handle both 1D and 2D source_tables
            if self.source_tables.ndim == 1:
                source_tables_sampled = self.source_tables[random_indices].astype(str)
            else:
                source_tables_sampled = self.source_tables[random_indices, -1].astype(str)

            for i, (output_name, y_val) in enumerate(y_val_dict.items()):
                pred = predictions[i]

                # Handle 3D predictions (e.g., [samples, timesteps, features])
                if pred.ndim > 2:
                    pred = pred[:, -1, :]  # Use only the last timestep
                elif pred.ndim > 1:
                    pred = np.mean(pred, axis=1)  # Average over features if needed

                pred_sampled = pred[random_indices]
                y_val_sampled = y_val[random_indices]

                pred_unscaled = []
                real_unscaled = []

                for idx, source_table in enumerate(source_tables_sampled):
                    # Determine the correct scaler based on target_group
                    if self.target_group:
                        # When targets are scaled together
                        if output_name in self.target_scalers:
                            scaler = self.target_scalers[output_name]
                        else:
                            print(f"Warning: Scaler for output '{output_name}' not found.")
                            continue
                    else:
                        # When targets are scaled per table
                        if source_table in self.target_scalers and output_name in self.target_scalers[source_table]:
                            scaler = self.target_scalers[source_table][output_name]
                        else:
                            print(f"Warning: Scaler for table '{source_table}' and output '{output_name}' not found.")
                            continue

                    pred_val = pred_sampled[idx].reshape(1, -1)
                    real_val = y_val_sampled[idx].reshape(1, -1)

                    # Apply inverse scaling using the correct scaler
                    unscaled_pred = scaler.inverse_transform(pred_val)
                    unscaled_real = scaler.inverse_transform(real_val)

                    pred_unscaled.append(unscaled_pred.flatten()[0])
                    real_unscaled.append(unscaled_real.flatten()[0])

                print(f"Epoch {epoch + 1} - First 5 Validation Predictions for {output_name}: {pred_unscaled}")
                print(f"Epoch {epoch + 1} - First 5 Validation Real Values for {output_name}: {real_unscaled}")
                print(f"Corresponding source tables: {source_tables_sampled}")
                print()

    class PrintUnscaledLoss(Callback):
        def __init__(self, target_scalers, batch_size, validation_data, source_tables, encoded_4_val, target_group=False):
            super().__init__()
            self.target_scalers = target_scalers
            self.batch_size = batch_size
            self.validation_data = validation_data
            self.source_tables = source_tables
            self.encoded_4_val = encoded_4_val
            self.target_group = target_group  # Added to handle target scaling mode

        def on_epoch_end(self, epoch, logs=None):
            X_val, y_val_dict = self.validation_data
            predictions = self.model.predict(X_val, batch_size=self.batch_size)

            if not isinstance(predictions, list):
                predictions = [predictions]

            combined_unscaled = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"pred": [], "real": []})))
            all_source_tables = set(self.source_tables.flatten())

            # Handle both 1D and 2D source_tables
            if self.source_tables.ndim == 1:
                flat_source_tables = self.source_tables.astype(str)
            else:
                flat_source_tables = self.source_tables[:, -1].astype(str)

            for i, (output_name, y_val) in enumerate(y_val_dict.items()):
                pred = predictions[i]

                if pred.ndim == 3:
                    pred = pred[:, -1, :]  # Use only the last timestep
                if y_val.ndim == 1:
                    y_val = y_val[:, np.newaxis]

                pred = pred.reshape(-1, 1)
                y_val = y_val.reshape(-1, 1)

                for idx in range(len(pred)):
                    source_table = flat_source_tables[idx]
                    group = self.encoded_4_val[idx]

                    # Determine the correct scaler based on target_group
                    if self.target_group:
                        # When targets are scaled together
                        if output_name in self.target_scalers:
                            scaler = self.target_scalers[output_name]
                        else:
                            print(f"Warning: Scaler for output '{output_name}' not found. Skipping this entry.")
                            continue
                    else:
                        # When targets are scaled per table
                        if source_table in self.target_scalers and output_name in self.target_scalers[source_table]:
                            scaler = self.target_scalers[source_table][output_name]
                        else:
                            print(f"Warning: Scaler for table '{source_table}' and output '{output_name}' not found. Skipping this entry.")
                            continue

                    pred_val = pred[idx].reshape(1, -1)
                    real_val = y_val[idx].reshape(1, -1)

                    try:
                        # Inverse scaling
                        unscaled_pred = scaler.inverse_transform(pred_val)
                        unscaled_real = scaler.inverse_transform(real_val)
                        combined_unscaled[source_table][output_name][group]["pred"].append(unscaled_pred[0, 0])
                        combined_unscaled[source_table][output_name][group]["real"].append(unscaled_real[0, 0])
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
            for output_name in y_val_dict.keys():
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
        def __init__(self, X, y, source_tables, batch_size):
            assert len(X) == len(source_tables), "X and source_tables must have the same length"
            assert len(X) == len(next(iter(y.values()))), "X and y must have the same length"
            
            self.X = X
            self.y = y
            self.source_tables = np.array(source_tables)
            self.batch_size = batch_size
            self.indices = np.arange(len(self.X))
            self.on_epoch_end()

        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))

        def __getitem__(self, index):
            batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
            X_batch = self.X[batch_indices]
            y_batch = {key: value[batch_indices].astype(np.float32) for key, value in self.y.items()}
            return X_batch, y_batch  # we don't return source_tables_batch as it's not used in training

        def on_epoch_end(self):
            self.shuffle_data()

        def shuffle_data(self):
            np.random.shuffle(self.indices)

    # ::::::::::::::::::::::::Training & Validation Loop:::::::::::::::::::::::::::: #

    # Initialize optimizer
    learning_rate = CustomSchedule(d_model, warmup_steps, custom_lr, lr_scale)
    optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)

    # callbacks
    unscaled_loss_callback = PrintUnscaledLoss(
        target_scalers=target_scalers,
        batch_size=batch_size,
        validation_data=(X_val, y_val_dict),
        source_tables=source_tables_val,
        encoded_4_val=encoded_4_val,
        target_group=target_group
    )
    # Initialize callbacks with target_group parameter
    print_predictions_and_loss_0 = PrintPredictionsAndLoss_0(
        target_scalers=target_scalers,
        validation_data=(X_val, y_val_dict),
        source_tables=source_tables_val,
        batch_size=batch_size,
        encoded_4_val=encoded_4_val,
        target_group=target_group
    )

    train_data_generator = CustomDataGenerator(X_train, y_train_dict, source_tables_train, batch_size)

    dashboard = RealTimeDashboard()
    print_lr = PrintLR()

    other_callbacks_2 = [dashboard, print_lr, unscaled_loss_callback, print_predictions_and_loss_0] 
    all_callbacks = [weights_checkpoint, full_model_checkpoint] + other_callbacks_2

    # Ensure X_val and y_val_dict are in the correct format
    X_val = X_val.astype(np.float32)
    y_val_dict = {key: value.astype(np.float32) for key, value in y_val_dict.items()}

    # Create the train data generator
    train_data_generator = CustomDataGenerator(X_train, y_train_dict, source_tables_train, batch_size)

    # Compile the model
    loss_dict = {name: loss_function for name in output_names} 
    model.compile(optimizer=optimizer, loss=loss_func)

    for key, value in y_val_dict.items():
        print(f"{key} shape:", value.shape)
        print(f"{key} dtype:", value.dtype)

    # Evaluate model immediately after loading weights (for baseline performance)
    print("Evaluating baseline performance...")
    val_loss = model.evaluate(X_val, y_val_dict, verbose=1)
    print(f"Baseline loss on Validation Set after Loading: {val_loss}")

    # Training the model using the custom data generator
    print("Starting model training...")
    history = model.fit(
        train_data_generator,
        epochs=epochs,
        validation_data=(X_val, y_val_dict),
        callbacks=all_callbacks,
        verbose=1
    )
    print("Training complete.")

# Ensure source_tables_val is a NumPy array
source_tables_val = np.array(source_tables_val)

# ::::::::::::::::::Evaluation 2nd Model (Save Predictions):::::::::::::::::::::: #

print("Starting Evaluation with predictions(.xlsx).")

# Predict model outputs
y_pred_list = model.predict(X_val)

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
            if name in target_scalers:
                scaler = target_scalers[name]
            else:
                print(f"Warning: Scaler for output '{name}' not found. Skipping this entry.")
                continue
        else:
            # Use the scaler for the specific table and output
            if table_name in target_scalers and name in target_scalers[table_name]:
                scaler = target_scalers[table_name][name]
            else:
                print(f"Warning: Scaler for table '{table_name}' and output '{name}' not found. Skipping this entry.")
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
output_file = "/home/jd/Desktop/trading/model_testing_text/training_models/training_progress/predictions_multi_stacked.xlsx"
with pd.ExcelWriter(output_file) as writer:
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

print(f"Predictions saved to '{output_file}' with multiple tabs for each table.")











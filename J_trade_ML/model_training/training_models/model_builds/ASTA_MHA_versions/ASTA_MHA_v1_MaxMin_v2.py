"""
   AA     SSSS  TTTTT    AAA      |  SSS   EEEEE   QQQ
  A  A    S       T     A   A     |  S     E      Q   Q
  AAAA    SSSS    T     AAAAA     |  SSS   EEEEE  Q   Q
 A    A      S    T    A     A    |    S   E      Q  QQ
A      A  SSSS    T   A       A   |  SSS   EEEEE   QQQ Q
_______________________________________________________ 
:Welcome to Custom ASTA (Adaptive Sphere Transformation) Sequencer LSTM Multi Target Prediction | (float32): (Version 2)

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
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras.initializers import GlorotUniform, LecunUniform
from tensorflow.keras.optimizers.schedules import LearningRateSchedule
from tensorflow.keras.losses import mean_squared_error, mean_absolute_error
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OrdinalEncoder
from tensorflow.keras.layers import GlobalAveragePooling2D, Concatenate, Flatten, Conv2D
from tensorflow.keras.layers import Input, Dense, Layer, Dropout, LayerNormalization, Reshape

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
epochs = 1000                   # Training iterations
batch_size = 32                 # Samples per batch (last was 32)

# :::: Adaptive Sphere Transformation ::::

asta_threshold_1 = 0.1          # Determines when to consider the model's performance stable.   
asta_threshold_2 = 0.1          # Defines the performance metric value above which the centers are locked.             
asta_preformance = 0.01          # Controls how quickly the model perceives stability. (0.01 default)

# :::: Advanced Positional Encoding ::::
"""
A smaller max_relative_position will limit the relative positional encodings to a more localized context, 
which might be beneficial for tasks where local context is more important than distant relationships.

A larger max_relative_position will allow the model to consider a wider range of relative positions, 
capturing more global context, but at the cost of increased computational complexity and potential overfitting.
"""
max_relative_position = 4

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
drop_targets = -2               # exclude the last target columns
target_columns = ['Target1', 'Target2', 'Target3']
output_names =   ['High']       

# :::: Training Target Predictions ::::

Scaler_features = MinMaxScaler()
Scaler_targets = MinMaxScaler()   

# :::: Training Loss Function ::::

loss_func = 'mean_absolute_error'
""" 'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error'
'mean_squared_logarithmic_error', 'huber_loss', 'log_cosh', 'cosine_similarity' """

# :::::::::::::::::::::::::::::TensorFlow setup:::::::::::::::::::::::::::::: #

# Suppresses TensorFlow warnings
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

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

def load_data_from_sqlite(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.drop(df.columns[0], axis=1, inplace=True)
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
    used_targets = target_columns[:drop_targets]
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
        **{i: [0, 1] for i in range(80, 144)},
        144: list(range(-1, 255))
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

    columns_to_drop = df.columns[0:72].tolist()
    #print("Columns to be dropped:", columns_to_drop)
    df.drop(columns=columns_to_drop, inplace=True)

    cols_to_scale = df.columns.tolist()
    #print("Used columns:", cols_to_scale)
    
    if feature_scaler is None:
        feature_scaler = MinMaxScaler()
        if df.isnull().any().any():
            df.fillna(df.mean(), inplace=True)
        scaled_features = feature_scaler.fit_transform(df[cols_to_scale].astype('float32'))
        print("Scalers loaded and data scaled...")
    else:
        if df.isnull().any().any():
            mean_dict = {col: feature_scaler.data_min_[i] for i, col in enumerate(cols_to_scale)}
            df.fillna(mean_dict, inplace=True)
        scaled_features = feature_scaler.transform(df[cols_to_scale].astype('float32'))
        print("Data scaled using existing scalers...")

    scaled_df = pd.DataFrame(scaled_features, columns=cols_to_scale)

    # Scale targets separately
    if target_scaler is None:
        target_scaler = {name: Scaler_targets for name in output_names}
        target_scaled = np.zeros(df_targets.shape)
        for i, col in enumerate(df_targets.columns):
            df_targets[col].fillna(df_targets[col].mean(), inplace=True)
            target_scaled[:, i] = target_scaler[output_names[i]].fit_transform(df_targets[[col]]).flatten()
        print("Targets scaled separately...")
    else:
        target_scaled = np.zeros(df_targets.shape)
        for i, col in enumerate(df_targets.columns):
            df_targets[col].fillna(df_targets[col].mean(), inplace=True)
            target_scaled[:, i] = target_scaler[output_names[i]].transform(df_targets[[col]]).flatten()
        print("Targets scaled with existing scalers...")

    # Create overlapping sequences
    X = create_overlapping_sequences(scaled_df.values, seq_len)

    # Create overlapping sequences for targets
    Y = create_overlapping_sequences(target_scaled, seq_len)
    target_dict = {output_names[idx]: Y[:, -1, idx] for idx in range(len(output_names))}

    return X, target_dict, feature_scaler, target_scaler, encoder

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

df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_OHLC_Train_3_3_new.db", 'XAUUSD')
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//XAUUSD_15m_OHLC_Val_3_3_new.db", 'XAUUSD')


print("Finished loading training data...\n")

print("Starting to preprocess training data...")
X_train, y_train_dict, train_scaler, target_scaler, encoder = preprocess_data(df_train, seq_len, target_columns, output_names)
print("Finished preprocessing training data...")
X_train, y_train_dict = randomize_data(X_train, y_train_dict)
print(f"Training data shapes - X: {X_train.shape}, y: {[y.shape for y in y_train_dict.values()]}\n")

# Save the scalers for future use
print("Saved the scaler for future use...\n")
joblib.dump({'feature_scaler': train_scaler, 'target_scaler': target_scaler, 'encoder': encoder}, 'scalers_MaxMin_v2.pkl')

""" 
# Load the scalers and encoder for validation or live data
scalers_encoder = joblib.load('scalers_encoder.pkl')
train_scaler = scalers_encoder['feature_scaler']
target_scaler = scalers_encoder['target_scaler']
encoder = scalers_encoder['encoder'] 
"""

print("Starting to preprocess validation data...")
X_val, y_val_dict, _, _, _ = preprocess_data(df_val, seq_len, target_columns, output_names, train_scaler, target_scaler, encoder)
print("Finished preprocessing validation data...")
print(f"Validation data shapes - X: {X_val.shape}, y: {[y.shape for y in y_val_dict.values()]}\n")  

# :::::::::::::::::::::::::::Real-time Dashboard::::::::::::::::::::::::::::::: #

class RealTimeDashboard(Callback):
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('Custom Spartial Gating Multi Output')
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
        plt.savefig("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/XAUUSD_high_train_loss_v1_MaxMin_v2.png") 
        self.root.update()

print("About to initialize the model...")

# :::::::::::::::::::::::::::Adaptive Sphere Transform Layer #Version 3::::::::::::::::::::::::::::::: #

""" 
make sure to disable UpdatePerformanceMetricCallback in the call back if this function is
not used. This version incorporates a custom callback that learns from the predicted error
"""

class AdaptiveSphereTransformLayer(Layer):
    def __init__(self, stability_threshold=asta_threshold_1, **kwargs):
        super(AdaptiveSphereTransformLayer, self).__init__(**kwargs)
        self.stability_threshold = stability_threshold
        

    def build(self, input_shape):
        self.performance_metric = self.add_weight(name='performance_metric',
                                                  shape=(),
                                                  initializer='zeros',
                                                  trainable=False,
                                                  dtype=tf.float32)
        self.center_seq_idx = self.add_weight(name='center_seq_idx',
                                              shape=(),
                                              initializer=tf.keras.initializers.Constant(0.5),
                                              trainable=True,
                                              dtype=tf.float32)
        self.center_feature_idx = self.add_weight(name='center_feature_idx',
                                                  shape=(),
                                                  initializer=tf.keras.initializers.Constant(0.5),
                                                  trainable=True,
                                                  dtype=tf.float32)
        self.performance_improvement = self.add_weight(name='performance_improvement',
                                                       shape=(),
                                                       initializer='zeros',
                                                       trainable=False,
                                                       dtype=tf.float32)
        super(AdaptiveSphereTransformLayer, self).build(input_shape)

    def perform_transformation(self, inputs, center_seq, center_feature, seq_len, num_features, improvement_factor):
        i = tf.linspace(-1.0, 1.0, seq_len)
        j = tf.linspace(-1.0, 1.0, num_features)
        ii, jj = tf.meshgrid(i, j, indexing='ij')

        ii_centered = ii - (2.0 * tf.cast(center_seq, tf.float32) / tf.cast(seq_len, tf.float32) - 1.0)
        jj_centered = jj - (2.0 * tf.cast(center_feature, tf.float32) / tf.cast(num_features, tf.float32) - 1.0)

        r = tf.sqrt(ii_centered**2 + jj_centered**2 + 1e-9)
        theta = tf.atan2(jj_centered, ii_centered)
        phi = r * np.pi

        x = r * tf.sin(phi) * tf.cos(theta)
        y = r * tf.sin(phi) * tf.sin(theta)
        z = r * tf.cos(phi)

        sphere_sum = (x + y + z) * improvement_factor

        sphere_sum = tf.expand_dims(sphere_sum, axis=0)
        sphere_sum = tf.tile(sphere_sum, [tf.shape(inputs)[0], 1, 1])

        transformed_inputs = tf.reshape(sphere_sum, tf.shape(inputs)) + inputs
        return transformed_inputs
    
    def call(self, inputs, training=None):
        batch_size, seq_len, num_features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]

        center_seq = tf.cast(tf.round(self.center_seq_idx * tf.cast(seq_len, tf.float32)), tf.int32)
        center_feature = tf.cast(tf.round(self.center_feature_idx * tf.cast(num_features, tf.float32)), tf.int32)

        if training:
            if self.performance_metric < self.stability_threshold:
                new_center_seq_idx = tf.random.uniform(shape=(), minval=0, maxval=1)
                new_center_feature_idx = tf.random.uniform(shape=(), minval=0, maxval=1)
                self.center_seq_idx.assign(new_center_seq_idx)
                self.center_feature_idx.assign(new_center_feature_idx)
                self.performance_improvement.assign(0.0)
            else:
                self.center_seq_idx.assign(tf.stop_gradient(self.center_seq_idx))
                self.center_feature_idx.assign(tf.stop_gradient(self.center_feature_idx))
                self.performance_improvement.assign(self.performance_improvement + asta_preformance)

            if self.performance_improvement > 1.0:
                self.performance_improvement.assign(1.0)

        transformed_inputs = self.perform_transformation(inputs, center_seq, center_feature, seq_len, num_features, self.performance_metric)
        return transformed_inputs

    def update_performance_metric(self, new_metric_value):
        self.performance_metric.assign(new_metric_value)

    def get_config(self):
        config = super(AdaptiveSphereTransformLayer, self).get_config()
        config.update({
            "stability_threshold": self.stability_threshold
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(
            stability_threshold=config['stability_threshold']
        )
    
# :::::::::::::::::::::::Custom Layer (undefined)::::::::::::::::::::::::::: #

class QuantumFractalTransform(Layer):
    def __init__(self, **kwargs):
        super(QuantumFractalTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        self.seq_len = input_shape[1]
        self.d_model = input_shape[2]

    def call(self, inputs):
        # Quantum-inspired transformation: Apply complex-valued operations
        complex_transform = tf.math.angle(tf.cast(inputs, tf.complex64))
        
        # Fractal geometry transformation: Iteratively downsample and upsample
        fractal_output = complex_transform
        for _ in range(3):
            fractal_output = tf.image.resize(fractal_output, [self.seq_len // 2, self.d_model // 2])
        
        # Restore original dimensions
        fractal_output = tf.image.resize(fractal_output, [self.seq_len, self.d_model])
        
        return tf.cast(tf.reshape(fractal_output, (-1, self.seq_len, self.d_model)), tf.float32)

class HypercomplexAttention(Layer):
    def __init__(self, **kwargs):
        super(HypercomplexAttention, self).__init__(**kwargs)
        self.units = None

    def build(self, input_shape):
        print(f"HypercomplexAttention build input_shape: {input_shape}")
        self.units = input_shape[-1]
        print(f"HypercomplexAttention units set to: {self.units}")
        self.dense_real = Dense(self.units)
        self.dense_imag = Dense(self.units)
        self.dense_j = Dense(self.units)
        self.dense_k = Dense(self.units)

    def call(self, inputs):
        print(f"HypercomplexAttention call inputs shape: {tf.shape(inputs)}")
        # Split inputs into hypercomplex components
        real_part = self.dense_real(inputs)
        imag_part = self.dense_imag(inputs)
        j_part = self.dense_j(inputs)
        k_part = self.dense_k(inputs)

        # Combine parts to form hypercomplex attention
        attention_output = real_part + imag_part + j_part + k_part
        return attention_output

class HolographicEncoding(Layer):
    def __init__(self, **kwargs):
        super(HolographicEncoding, self).__init__(**kwargs)

    def build(self, input_shape):
        print(f"HolographicEncoding build input_shape: {input_shape}")
        self.units = 2400
        self.dense = Dense(self.units)

    def call(self, inputs):
        print(f"HolographicEncoding call inputs shape: {tf.shape(inputs)}")
        # Holographic encoding
        encoded_output = self.dense(inputs)
        return encoded_output

class SOMLayer(Layer):
    def __init__(self, map_size, **kwargs):
        super(SOMLayer, self).__init__(**kwargs)
        self.map_size = map_size

    def build(self, input_shape):
        print(f"SOMLayer build input_shape: {input_shape}")
        self.map_dim = input_shape[-1]
        self.som_weights = self.add_weight(shape=(self.map_size * self.map_size, self.map_dim),
                                           initializer='random_normal', trainable=True)
    
    def call(self, inputs):
        print(f"SOMLayer call inputs shape: {tf.shape(inputs)}")
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        # Flatten inputs correctly
        inputs_flattened = tf.reshape(inputs, [batch_size * seq_len, d_model])

        # Expand dims for SOM operation
        expanded_inputs = tf.expand_dims(inputs_flattened, axis=1)  # shape: [batch_size * seq_len, 1, d_model]
        expanded_weights = tf.expand_dims(self.som_weights, axis=0)  # shape: [1, map_size * map_size, map_dim]
        
        # Calculate the distances between input vectors and weight vectors
        distances = tf.reduce_sum(tf.square(expanded_inputs - expanded_weights), axis=-1)
        
        # Find the Best Matching Units (BMUs)
        bmu_indices = tf.argmin(distances, axis=1)
        
        # Update the weight vectors to move closer to the input vectors
        bmu_updates = tf.gather(self.som_weights, bmu_indices)
        new_weights = bmu_updates + 0.1 * (inputs_flattened - bmu_updates)
        
        # Update the SOM weights
        self.som_weights.assign(tf.tensor_scatter_nd_update(self.som_weights, tf.expand_dims(bmu_indices, axis=1), new_weights))
        
        # Reshape the output to the required shape
        output = tf.reshape(self.som_weights, [self.map_size, self.map_size, self.map_dim])
        output = tf.expand_dims(output, axis=0)  # Add batch dimension
        output = tf.tile(output, [batch_size, 1, 1, 1])  # Match batch size
        
        return output
    
# ::::::::::::::::::::::::Main Model Parameters::::::::::::::::::::::::::::: #

# ::Data Model Transformation::
seq_len = X_train.shape[1]               
d_model = X_train.shape[2]                # Input shape in units
inputs = Input(shape=(seq_len, d_model), name='main_input_features')                      # Dropout rate

d_ffn = int(4)                    # 1nd unit exspansion/supression
q_ffn = int(d_model*4) 

# ::::::::::::::Main Model Training (Multi Target Prediction):::::::::::::: #

#apose = AdvancedPositionalEncoding(seq_len, d_model)(inputs)
#apose = PositionalEncoding(seq_len, d_model)(inputs)
x_loop = inputs

outputs = []
for name in output_names: 
    
    x_loop = AdaptiveSphereTransformLayer()(x_loop)
    # Apply QuantumFractalTransform
    quantum_fractal_transform = QuantumFractalTransform()(x_loop)
    
    # Apply HypercomplexAttention
    hypercomplex_attention = HypercomplexAttention()(quantum_fractal_transform)
    
    # Apply HolographicEncoding
    holographic_encoding = HolographicEncoding()(hypercomplex_attention)
    
    # Reinvented Dense Layer: SOMLayer
    som_layer = SOMLayer(map_size=8)(holographic_encoding)

    # Flatten and reshape to match the input shape
    som_layer_shape = tf.shape(som_layer)
    som_layer_flattened = tf.reshape(som_layer, [-1, tf.reduce_prod(som_layer.shape[1:])])
    
    reshaped_output = Dense(seq_len * d_model)(som_layer_flattened)
    reshaped_output = Reshape((seq_len, d_model))(reshaped_output)
    
    # Ensure output shape is correct
    reshaped_output = Reshape((seq_len, d_model))(reshaped_output)

    output = Dense(1, activation='linear', use_bias=True, name=name, dtype='float32')(reshaped_output)
    outputs.append(output)

model = Model(inputs=inputs, outputs=outputs)

model.summary()
print("Model initialized!")
print("Total number of parameters in the model:", model.count_params())

# :::::::::::::::::::::Model Checkpoint For Weights::::::::::::::::::::::: #

# Define the model checkpoint for weights only
weights_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/XAUUSD_high_weights_v1_MaxMin_v2.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/XAUUSD_high_model_v1_FULL_MaxMin_v2.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', 
                                        verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

custom_objects={
    'AdaptiveSphereTransformLayer': AdaptiveSphereTransformLayer,
    'QuantumFractalTransform': QuantumFractalTransform,
    'HypercomplexAttention': HypercomplexAttention,
    'HolographicEncoding': HolographicEncoding,
    'SOMLayer': SOMLayer,
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

            if scaler.n_features_in_ > 1:
                dummy_columns = np.zeros((5, scaler.n_features_in_ - 1))
                pred_sampled = np.hstack([pred_sampled.reshape(5, 1), dummy_columns])
                y_val_sampled = np.hstack([y_val_sampled.reshape(5, 1), dummy_columns])
            else:
                pred_sampled = pred_sampled.reshape(5, 1)
                y_val_sampled = y_val_sampled.reshape(5, 1)

            pred_unscaled = scaler.inverse_transform(pred_sampled)[:, 0]
            real_unscaled = scaler.inverse_transform(y_val_sampled)[:, 0]

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

            if isinstance(pred, list):
                pred = np.array(pred)

            if pred.ndim == 3:
                pred = np.mean(pred, axis=1)
            if y_val.ndim == 1:
                y_val = y_val[:, np.newaxis] 

            # Use the appropriate scaler for each target
            scaler = self.target_scalers[key]
            combined = np.hstack([pred, y_val])
            combined_unscaled = scaler.inverse_transform(combined)
            pred_unscaled = combined_unscaled[:, 0]
            y_val_unscaled = combined_unscaled[:, 1]

            mae_val_unscaled = np.mean(np.abs(pred_unscaled - y_val_unscaled))
            print(f'Epoch {epoch + 1} - Validation {key} Unscaled MAE: {mae_val_unscaled}')

class RandomizeDataCallback(Callback):
    def __init__(self, X, y):
        super().__init__()
        self.X = X
        self.y = y

    def on_epoch_begin(self, epoch, logs=None):
        perm = np.random.permutation(len(self.X))
        perm = tf.constant(perm, dtype=tf.int32) 
        self.X = tf.gather(self.X, perm)
        self.y = {key: tf.gather(value, perm) for key, value in self.y.items()}
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

    loss_per_timestep = mean_absolute_error(y_true, y_pred)
    
    return tf.reduce_mean(loss_per_timestep)

class UpdatePerformanceMetricCallback(Callback):
    def __init__(self, layer, threshold=asta_threshold_2):
        super().__init__()
        self.layer = layer
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None):
        # Retrieve loss from the logs
        current_loss = logs.get('loss') 
        if current_loss is not None:
            # Here we use a simple inverse scale as an example
            normalized_metric = 1 / (1 + current_loss)
            self.layer.performance_metric.assign(normalized_metric)
            # Check if the performance metric is above the threshold
            if normalized_metric > self.threshold:
                # Lock the center coordinates by making them non-trainable
                self.layer.center_seq_idx.assign(tf.stop_gradient(self.layer.center_seq_idx))
                self.layer.center_feature_idx.assign(tf.stop_gradient(self.layer.center_feature_idx))
            else:
                # Allow the center coordinates to be trainable
                self.layer.center_seq_idx._trainable = True
                self.layer.center_feature_idx._trainable = True

# ::::::::::::::::::::::::Training & Validation Loop:::::::::::::::::::::::::::: #

# Initialize optimizer
learning_rate = CustomSchedule(d_ffn, warmup_steps, custom_lr, lr_scale)
optimizer = Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon, amsgrad=amsgrad)

# loss dictionary using specific output names
loss_dict = {name: loss_func for name in output_names} # loss_func or loss_function
model.compile(optimizer=optimizer, loss=loss_dict) # metrics=['mae']

# Evaluate model immediately after loading weights (for baseline performance)
val_loss = model.evaluate(X_val, y_val_dict)
print(f"Baseline Mean Absolute Error on Validation Set after Loading: {val_loss}")

training_data = (X_train, y_train_dict)
validation_data = (X_val, y_val_dict)

# callbacks
unscaled_loss_callback = PrintUnscaledLoss(
    target_scalers=target_scaler, 
    batch_size=batch_size,  
    validation_data=validation_data
)
print_predictions_and_loss_0 = PrintPredictionsAndLoss_0(target_scaler, validation_data=(X_val, y_val_dict), batch_size=batch_size)

randomize_data_callback = RandomizeDataCallback(X_train, y_train_dict)
dashboard = RealTimeDashboard()
print_lr = PrintLR()

adaptive_layer = next((layer for layer in model.layers if isinstance(layer, AdaptiveSphereTransformLayer)), None)

if adaptive_layer is not None:
    # Creating the custom callback if the adaptive layer is found
    performance_callback = UpdatePerformanceMetricCallback(adaptive_layer)
    other_callbacks_1 = [dashboard, print_lr, unscaled_loss_callback, randomize_data_callback, print_predictions_and_loss_0, performance_callback]
    all_callbacks = [weights_checkpoint] + other_callbacks_1  # full_model_checkpoint
else:
    print("AdaptiveSphereTransformLayer not found in the model.")
    other_callbacks_2 = [dashboard, print_lr, unscaled_loss_callback, randomize_data_callback, print_predictions_and_loss_0]
    all_callbacks = [weights_checkpoint] + other_callbacks_2  # full_model_checkpoint

# Training the model
history = model.fit(
    X_train, y_train_dict, 
    epochs=epochs, 
    batch_size=batch_size, 
    validation_data=(X_val, y_val_dict), 
    callbacks=all_callbacks
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
file_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/predictions_MaxMin_v2.csv"
result_df.to_csv(file_path, index=False)
print(f"Saving all predictions to: {file_path}")

print("Predictions complete.")







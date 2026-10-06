"""
  QQQ    GGG     M    M    L      PPPP .........
 Q   Q  G        MM  MM    L      P   P ..........
Q     Q G  GG    M M  M    L      PPPP .............
 Q   Q  G   G   M      M   L      P ...................   
  QQQQ    GGG  M        M  LLLLL  P .....................    
       Q       
               
:Welcome to Quantum Gated Multi-Layer Perceptron Regressive (float32): version 1

QgMLP Transformation (based on principals double-slit experiment):
The QgMLP equation transforms the incoming data from each layer using the principles of quantum field theory, 
information entropy, and Markov chains. This transformation enables the QgMLP to find and build upon tiny 
variations in inputs, allowing for more effective pattern recognition and learning.

The QgMLP equation can be broken down into three components:
1.  Quantum Field-inspired Layer: This layer represents the interference patterns observed in the double slit problem. 
    It simulates the wave function ψ(x) using the Schrödinger equation and calculates the probability density ρ(x) using the Born rule.

    Layer 1 (Quantum Field-inspired Layer)
    x → h1 = ∑(φ(x)|α + φ(x)|β)

2.  Information Entropy-inspired Layer: This layer models the measurement process as an interaction between particles and 
    spacetime continuum. The Shannon entropy formula is used to calculate the information entropy of the system.

    Layer 2 (Information Entropy-inspired Layer)
    h1 → h2 = - ∑(p log p)

3.  Markov Chain-inspired Layer: This layer represents the Markov chain, where the probability of each outcome is calculated 
    using the von Neumann equation.

    Layer 3 (Markov Chain-inspired Layer)
    h2 → y = ∏(h2|φ(x)|α + h2|φ(x)|β)   

4.  The QgMLP Equation combines all the layers and transforms the output of layer 3.

    QNN Equation
    y = ∏(h1 × h2) × ∑(φ(x)|α + φ(x)|β)

The α and β coefficients are learnable parameters that represent the strength of the connections between these layers. They are 
separate from the weights between layers, which are represented by the QgMLP equation itself.

Output of Each Unit:
Each unit in a QgMLP has both h1 and h2 values. The output of each unit is a combination of these values, representing the weighted
connection between layers. 

Next Steps: Quantum Self Attention, being that attention is still the best case for building contexr of inputs for LLMs, the focus with this
model is to reduce the need of multi-head attentino, to a single head Quantum Self Attention. Convolutional NNs will replace convolutional 
layers with CQNN (Convolutional Quantum Neural Netwrok). There is no computational benifit to QgMLP, but rather focused on using a much
smaller amounts of units and layers, to make smaller models more powerful.

By: JD (ThinkBe)
"""   
import os
import sqlite3
import numpy as np
import pandas as pd
import tkinter as tk
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.mixed_precision import experimental
from tensorflow.keras.layers import Dense, LayerNormalization
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from sklearn.metrics import mean_squared_error, mean_absolute_error 

"""
TTTTT  H   H  IIIII  N   N  K   K  B B B  EEEE ...
  T    H   H    I    NN  N  K  K   B   B  E    .......
  T    HHHHH    I    N N N  KKK    BBBB   EEE  .............
  T    H   H    I    N  NN  K  K   B   B  E    ......................
  T    H   H  IIIII  N   N  K   K  B B B  EEEE ...............................

:parameters: Below are the model settings.

"""
seq_len = 1
d_ffn = 500
block_layers = 3
learning_rate = 0.0001
epochs = 1000
batch_size = 32

# tf.keras.activations.linear, tf.nn.relu, tf.nn.sigmoid, tf.tanh,
activation = tf.nn.sigmoid

# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

tf.config.experimental.set_visible_devices
print("Eager execution:", tf.executing_eagerly())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e) 

# Data Loading and Preprocessing Functions
def load_data_from_sqlite(db_path, table_name, columns=None):
    conn = sqlite3.connect(db_path)
    if columns:
        query = f"SELECT {','.join(columns)} FROM {table_name}"
    else:
        query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def create_overlapping_sequences(data, seq_len):
    sequences = []
    for i in range(seq_len - 1, len(data)):
        seq = data[i - (seq_len - 1):i + 1] 
        sequences.append(np.array(seq))
    return np.array(sequences)

def preprocess_data(df, scaler=None, seq_len=seq_len):
    if 'Target' not in df.columns:
        raise KeyError("The DataFrame does not contain a 'Target' column.")

    target = df.pop('Target').astype('float32')
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(df.mean(numeric_only=True), inplace=True)

    if scaler is None:
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df.values.astype('float32'))
    else:
        scaled = scaler.transform(df.values.astype('float32'))

    X = create_overlapping_sequences(scaled, seq_len)
    y = target.values[seq_len - 1:]  

    return X, y, scaler

# Before loading data
print("About to load training data...")

# Load data
columns_to_load = None
df_train = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//xauusd_train.db", 'XAUUSD', columns=columns_to_load)
df_val = load_data_from_sqlite("C://Users//crgon//OneDrive//Desktop//trading//data//xauusd_val.db", 'XAUUSD', columns=columns_to_load)

def flatten_and_reshape(data):
    return np.array([np.array(x).flatten() for x in data])

# After loading data
print("Finished loading training data...")

print("Starting to preprocess training data...")
X_train, y_train, scaler = preprocess_data(df_train, seq_len=seq_len)
print("Finished preprocessing training data...")
X_train = flatten_and_reshape(X_train)
print("starting X_train shape:", X_train.shape)

print("Starting to preprocess validation data...")
X_val, y_val, _ = preprocess_data(df_val, scaler, seq_len=seq_len)
X_val = flatten_and_reshape(X_val)
print("Finished preprocessing validation data...")
print("starting X_val shape:", X_val.shape)

# Real-time Dashboard
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
        plt.savefig("C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/QgMLP_training_loss.png") 
        self.root.update()

# Create RealTimeDashboard callback instance
dashboard = RealTimeDashboard()

print("About to initialize the model...")

# Define the model
class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, d_ffn, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.d_ffn = d_ffn
        self.layer_norm = LayerNormalization()
        self.spatial_projection = Dense(d_ffn, activation='sigmoid', use_bias=True, kernel_initializer='glorot_normal')
        
         # Forget and input gates
        self.forget_gate = Dense(d_ffn, activation='sigmoid', kernel_initializer='glorot_normal') 
        self.input_gate = Dense(d_ffn, activation='sigmoid', kernel_initializer='glorot_normal')  

    def build(self, input_shape):
        self.spatial_gating = self.add_weight(
            shape=(self.d_ffn,),
            initializer='he_normal',
            trainable=True
        )

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        f_gate = self.forget_gate(normalized_inputs)
        i_gate = self.input_gate(normalized_inputs)
        gated_temporal = f_gate * normalized_inputs + i_gate * self.spatial_projection(normalized_inputs)
        return gated_temporal * tf.sigmoid(self.spatial_gating) + inputs

    def get_config(self):
        config = super(SpatialGatingUnit, self).get_config()
        config.update({"d_ffn": self.d_ffn})
        return config

class QgMLPBlock(tf.keras.layers.Layer):
    def __init__(self, d_ffn): 
        super().__init__() 
        self.d_ffn = d_ffn
        self.minval_thresholds = None
        self.maxval_thresholds = None
        self.alpha = None
        self.beta = None

        self.a_norm = LayerNormalization()
        self.b_norm = LayerNormalization()
        self.x1_norm = LayerNormalization()
        self.x2_norm = LayerNormalization()
        self.h1_0_norm = LayerNormalization() 
        self.h1_1_norm = LayerNormalization() 
        self.h2_0_norm = LayerNormalization() 
        self.h2_1_norm = LayerNormalization() 
        self.output_norm = LayerNormalization()         

    def build(self, input_shape):
        input_dim = input_shape[-1]
        self.alpha = self.add_weight(shape=(self.d_ffn,), initializer="ones", trainable=True, name='alpha')
        self.beta = self.add_weight(shape=(self.d_ffn,), initializer="zeros", trainable=True, name='beta')        

        self.alpha_weights = self.add_weight(shape=(input_dim, self.d_ffn), initializer="glorot_normal", trainable=True, name='alpha_weights')
        self.bias_1 = self.add_weight(name='bias_1', shape=(self.d_ffn,), initializer='zeros', trainable=True)
        self.beta_weights = self.add_weight(shape=(input_dim, self.d_ffn), initializer="glorot_normal", trainable=True, name='beta_weights')
        self.bias_2 = self.add_weight(name='bias_2', shape=(self.d_ffn,), initializer='zeros', trainable=True)

        super().build(input_shape)

    def phi_x_alpha(self, x):
        result = tf.pow(tf.abs(tf.cos(x)), 2)
        return result

    def phi_x_beta(self, x):
        result = tf.pow(tf.abs(tf.sin(x)), 2)
        return result

    def Information_Entropy(self, x1, x2):
        p = tf.exp(x1)
        entropy_0 = -tf.math.log(p)

        p = tf.exp(x2)
        entropy_1 = -tf.math.log(p)
        return entropy_0, entropy_1
    
    def Markov_Chain(self, h1_0, h1_1, x1, x2):
        product_x1 = tf.abs(x1)   
        Chain_0 = h1_0**2 * tf.matmul(product_x1, self.alpha_weights)  
        Chain_0 += self.bias_1
        
        product_x2 = tf.abs(x2) 
        Chain_1 = h1_1**2 * tf.matmul(product_x2, self.beta_weights) 
        Chain_1 += self.bias_2
        
        return Chain_0, Chain_1
    
    def QuantumLayer(self, inputs):
        a = self.phi_x_alpha(inputs)
        a = activation(a)
        a_norm_output = self.a_norm(a)        
        act_x = tf.multiply(a_norm_output, self.alpha)        
        act_x = activation(act_x)

        b = self.phi_x_beta(inputs) 
        b = activation(b)
        b_norm_output = self.b_norm(b)
        act_y = tf.multiply(b_norm_output, self.beta)        
        act_y = activation(act_y)

        x1 = tf.matmul(act_x, self.alpha_weights)
        x1 += self.bias_1
        x1_norm_output = self.x1_norm(x1)
        x1 = activation(x1_norm_output)

        x2 = tf.matmul(act_y, self.beta_weights)
        x2 += self.bias_2
        x2_norm_output = self.x1_norm(x2)
        x2 = activation(x2_norm_output)

        h1_0, h1_1 = self.Information_Entropy(x1, x2) 
        h1_0_norm_output = self.h1_0_norm(h1_0)
        h1_0 = activation(h1_0_norm_output)
        h1_1_norm_output = self.h1_1_norm(h1_1)
        h1_1 = activation(h1_1_norm_output)

        
        h2_0, h2_1 = self.Markov_Chain(h1_0, h1_1, x1, x2)
        h2_0_norm_output = self.h2_0_norm(h2_0)
        h2_0_norm_output = activation(h2_0_norm_output)
        h2_1_norm_output = self.h2_1_norm(h2_1)
        h2_1_norm_output = activation(h2_1_norm_output)

        output = tf.multiply(h2_0_norm_output, h2_1_norm_output)
        output = self.output_norm(output)
        output = activation(output)
        return output

    def call(self, inputs):
        return self.QuantumLayer(inputs)

    def get_config(self):
        config = super(QgMLPBlock, self).get_config()
        config.update({"d_ffn": self.d_ffn})
        return config

print("Model initialized!")

inputs = tf.keras.layers.Input(shape=(X_train.shape[1],))

# Project the input embeddings to match d_ffn
norm = LayerNormalization()(inputs)
projected_inputs = Dense(d_ffn, activation= activation, kernel_initializer='glorot_normal')(norm)
x = projected_inputs

for i in range(block_layers):
    qnn_block = QgMLPBlock(d_ffn)
    x = qnn_block(x) 
    x = SpatialGatingUnit(d_ffn)(x) # not required

outputs = Dense(1, activation='linear', kernel_initializer="glorot_normal", dtype='float32')(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

# model summary
model.summary()
print("Total number of parameters in the model:", model.count_params())

# Define the model checkpoint for weights only
weights_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/QgMLP_weights_version_1_1.h5"
weights_checkpoint = ModelCheckpoint(filepath=weights_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)

# Define the model checkpoint for the full model
full_model_checkpoint_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/saved_weights/QgMLP_model_version_1.h5"
full_model_checkpoint = ModelCheckpoint(filepath=full_model_checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=False, save_format='h5')

# Try loading the full model, then the weights, or use the freshly defined model if neither exists
if os.path.exists(full_model_checkpoint_path):
    print("Loading full model...")
    model = tf.keras.models.load_model(full_model_checkpoint_path, custom_objects={'SpatialGatingUnit': SpatialGatingUnit, 'QgMLPBlock': QgMLPBlock})
elif os.path.exists(weights_checkpoint_path):
    print("Loading model weights...")
    model.load_weights(weights_checkpoint_path)
else:
    print("Using freshly defined model...")

# Initialize optimizer
optimizer = Adam(learning_rate=learning_rate)

# Compile the model
# mean_squared_error, mean_absolute_error
model.compile(optimizer=optimizer, loss='mean_absolute_error')

# Evaluate model immediately after loading weights (for baseline performance)
val_loss = model.evaluate(X_val, y_val)
print(f"Baseline Mean Absolute Error on Validation Set after Loading: {val_loss}")

# Fit the model
print(f"Training on {X_train.shape[0]} examples, validating on {X_val.shape[0]} examples.")
history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[weights_checkpoint, dashboard])

epochs = 0 
batch_size = 1
if epochs > 0:
    print(f"Training on {X_train.shape[0]} examples, validating on {X_val.shape[0]} examples.")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), callbacks=[weights_checkpoint, dashboard])

    # Close the Tkinter window
    dashboard.root.destroy()
    print("Training complete.") 

# Model evaluation for both MSE and MAE
y_pred = model.predict(X_val).flatten()
mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)

print(f"Model Mean Squared Error on Validation Set: {mse}")
print(f"Model Mean Absolute Error on Validation Set: {mae}")

# Obtain predictions on validation data
predictions = model.predict(X_val)

# Calculate MSE and MAE
mse = np.mean((y_val - predictions.flatten())**2)
mae = np.mean(np.abs(y_val - predictions.flatten()))

print(f"Mean Squared Error on validation data: {mse}")
print(f"Mean Absolute Error on validation data: {mae}")

# Save the actual and predicted values to a DataFrame
result_df = pd.DataFrame({"Actual": y_val, "Predicted": predictions.flatten()})

# Save result DataFrame to a CSV file
file_path = "C:/Users/crgon/OneDrive/Desktop/trading/model_testing_text/training_models/training_progress/qgmlp_version_1_1_predictions.csv"
print("Saving file to:", file_path)
result_df.to_csv(file_path, index=False)
print("File saved successfully!")



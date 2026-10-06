import os
import sqlite3
import pandas as pd
import numpy as np
import dask.dataframe as dd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.mixed_precision import experimental as mixed_precision

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.linear_model import LinearRegression
from collections import deque

# random seed for numpy and tensorflow
# np.random.seed(10)
# tf.random.set_seed(10)

# TensorFlow policy to mixed precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

print("Connecting to SQLite database...")

# path to your SQLite database
database_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\stock_data_1.db'
table_name = 'stock_data'

# Connect to the SQLite database
conn = sqlite3.connect(database_path)
print(f"Connected to SQLite database at {database_path}...")

print(f"Loading data from table: {table_name}")
query = f"SELECT * FROM {table_name}"
chunksize = 200000  # adjust based on system's memory
df = pd.DataFrame()  # initialize an empty dataframe

for chunk in pd.read_sql_query(query, conn, chunksize=chunksize):
    df = pd.concat([df, chunk])

# Convert to Dask DataFrame
ddf = dd.from_pandas(df, npartitions=10)

# Preprocessing
print("Starting preprocessing...")
# Separate features and target, ignore first column (1) or include first column (0)
features = df.iloc[:, 1:-1] # Exclude/include the first (index) and last (target) columns
target = df['Target']  # Original string targets

# Initialize and fit label encoder on the original targets
le = LabelEncoder()
encoded_target = le.fit_transform(target)
onehot_encoded = to_categorical(encoded_target)  # One-hot encoding

print(np.unique(encoded_target))
print(target.unique())

# Convert features to float32
values = features.values.astype('float32')

# Define sequence length
seq_len = 200 # Adjusted sequence length

# Ensure the total number of elements in 'values' is divisible by seq_len
total_rows = values.shape[0]
num_rows_drop = total_rows % seq_len
if num_rows_drop > 0:
    values = values[:-num_rows_drop, :]

# Scale features
scaler = StandardScaler()
scaled = scaler.fit_transform(values)

# Adjust target to match the number of sequences
if num_rows_drop > 0:
    onehot_encoded = onehot_encoded[:-num_rows_drop]

# Prepare sequences
X = np.reshape(scaled, (scaled.shape[0] // seq_len, seq_len, scaled.shape[1]))
y = onehot_encoded.reshape(-1, seq_len, onehot_encoded.shape[1])[:, -1, :] 


print("Preprocessing completed")

print("Building the model...")
# Define the model
class ZerosInitializer(tf.keras.initializers.Initializer):
    def __call__(self, shape, dtype=None):
        return tf.zeros(shape, dtype=dtype)

class SpatialGatingUnit(tf.keras.layers.Layer):
    def __init__(self, d_ffn, **kwargs):
        super(SpatialGatingUnit, self).__init__(**kwargs)
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.spatial_projection = Dense(d_ffn, use_bias=False, kernel_initializer='he_normal')

    def build(self, input_shape):
        self.spatial_gating = self.add_weight(
            shape=(input_shape[-1],),
            initializer=ZerosInitializer(),  # Using the custom initializer
            trainable=True
        )

    def call(self, inputs):
        normalized_inputs = self.layer_norm(inputs)
        return self.spatial_projection(normalized_inputs) * tf.sigmoid(self.spatial_gating)


class gMLPBlock(tf.keras.layers.Layer):
    def __init__(self, d_model, d_ffn):
        super(gMLPBlock, self).__init__()
        self.channel_projection_i = Dense(d_ffn, activation='relu', kernel_initializer='he_normal')
        self.sgu = SpatialGatingUnit(d_ffn)
        self.channel_projection_ii = Dense(d_model, kernel_initializer='he_normal')

    def call(self, inputs):
        x = self.channel_projection_i(inputs)
        x = self.sgu(x)
        x = self.channel_projection_ii(x)
        return x + inputs  # input to the output (residual connection)


d_model = 6  # model dimension related to # of features
d_ffn = 1000  # Feed Forward Network dimension

# Define your learning rate
learning_rate = 0.0001  # adjust as needed

# Create an instance of the Adam optimizer with your desired learning rate
optimizer = Adam(learning_rate=learning_rate)

# Define the input shape
inputs = tf.keras.layers.Input(shape=(seq_len, scaled.shape[1]))

x = inputs
for _ in range(20):  # Create GRU layers
    x = tf.keras.layers.GRU(d_ffn, return_sequences=True)(x)
    # x = tf.keras.layers.Dropout(0.5)(x)  # Dropout layer with dropout rate of 0.5

x = tf.keras.layers.GlobalAveragePooling1D()(x)
outputs = Dense(onehot_encoded.shape[1], activation='softmax', kernel_initializer='he_normal')(x)
model = tf.keras.Model(inputs, outputs)

# total number of model parameters
print("Total number of parameters in the model: ", model.count_params())

print("Compiling the model...")
model.compile(optimizer=optimizer, loss='categorical_crossentropy')  # Adjusted for categorical prediction

# Calculate the index at which to split the data
train_split = int(X.shape[0] * 0.4)

# Split the data
X_train = X[:train_split]
y_train = y[:train_split]
X_val = X[train_split:]
y_val = y[train_split:]

# Train the model using the split data
print(f"Training on {X_train.shape[0]} examples, validating on {X_val.shape[0]} examples.")
model.fit(X_train, y_train, epochs=500, batch_size=100, validation_data=(X_val, y_val))


print("Training complete.")

# Obtain predictions
predictions = model.predict(X)

# Convert softmax outputs to class labels
predicted_classes = np.argmax(predictions, axis=-1)

# Convert target back to 1D
true_classes = np.argmax(y, axis=-1)

accuracy = np.mean(predicted_classes == true_classes)
print(f"Accuracy: {accuracy}")

# Convert predicted classes to original labels
predicted_labels = le.inverse_transform(predicted_classes.flatten())

# Ensure position-based indexing with .iloc
target = target.iloc[:predicted_labels.shape[0]]

# Print the result DataFrame
print("Predictions and Actual Labels:")
result_df = pd.DataFrame({"Actual": target, "Predicted": predicted_labels})  # Adjusted target
print(result_df)

# Save result DataFrame to a CSV file
file_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\predictions.csv"
print("Saving file to:", file_path)
result_df.to_csv(file_path, index=False)
print("File saved successfully!")

# Print all unique classes in target
print("Unique classes in target:", np.unique(target))

# Select first 100,000 samples
target = target.iloc[:100000]
predicted_labels = predicted_labels[:100000]

# Plot the confusion matrix
conf_mat = confusion_matrix(target, predicted_labels, labels=["buy", "hold", "sell"])
plt.figure(figsize=(10,7))
sns.heatmap(conf_mat, annot=True, fmt="d", xticklabels=["buy", "hold", "sell"], yticklabels=["buy", "hold", "sell"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


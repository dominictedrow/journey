import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import AdditiveAttention, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import mean_squared_error
import datetime

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def data_generator(path_input, batch_size, img_size_input, context_window):
    all_input_images = sorted(glob.glob(path_input))
    total_images = len(all_input_images) - context_window

    while True:
        for i in range(0, total_images, batch_size):
            batch_input_images = all_input_images[i: i + batch_size + context_window]
            
            images_input = []
            images_output = []
            for j in range(len(batch_input_images) - context_window):
                img_sequence = []
                for k in range(context_window):
                    index = min(j + k, len(batch_input_images) - 1)  # don't go out of bounds
                    img_full = Image.open(batch_input_images[index]).convert('L')  # Open as grayscale
                    img_full = img_full.resize((img_size_input[0]*2, img_size_input[1]))
                    img_full = np.array(img_full, dtype=np.float32) / 255.0  # Normalize and convert to float32

                    img = img_full[:, img_size_input[0]:]  # Take the right half as input

                    img_sequence.append(img[..., np.newaxis])  # Add an extra dimension
                
                img_output_full = Image.open(batch_input_images[j + context_window]).convert('L')  # Open as grayscale
                img_output_full = img_output_full.resize((img_size_input[0]*2, img_size_input[1]))
                img_output_full = np.array(img_output_full, dtype=np.float32) / 255.0  # Normalize and convert to float32

                img_output = img_output_full[:, :img_size_input[0]]  # Take the left half as output

                images_input.append(img_sequence)
                images_output.append(img_output[..., np.newaxis])  # Add an extra dimension

            if len(images_input) == batch_size:
                images_input = np.array(images_input).reshape(-1, context_window, img_size_input[0], img_size_input[1], 1)  # 1 channel
                images_output = np.array(images_output).reshape(-1, img_size_input[0], img_size_input[1], 1)  # 1 channel
                yield (images_input, images_output)

def load_images(path_input, img_size):
    print("Loading images from: " + path_input)
    images_input = []
    images_output = []
    for img_path_input in sorted(glob.glob(path_input)):
        img_full = Image.open(img_path_input).convert('L')  # Open as grayscale
        img_full = img_full.resize((img_size[0]*2, img_size[1]))
        img_full = np.array(img_full, dtype=np.float32) / 255.0  # Normalize and convert to float32

        img_input = img_full[:, img_size[0]:]  # Take the right half as input
        img_output = img_full[:, :img_size[0]]  # Take the left half as output

        images_input.append(img_input[..., np.newaxis])  # Add an extra dimension
        images_output.append(img_output[..., np.newaxis])  # Add an extra dimension

    return np.stack(images_input), np.stack(images_output)

def create_sequences(images_input, images_output, context_window):
    print("Creating sequences...")
    X = []
    y = []
    for i in range(len(images_input) - context_window):
        sequence = images_input[i : i + context_window]
        X.append(sequence)
        y.append(images_output[i + context_window])
    return np.array(X), np.array(y)

class CustomAdditiveAttention(AdditiveAttention):
    def get_config(self):
        base_config = super().get_config()
        return {**base_config}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
class CustomConvLSTM2D(layers.ConvLSTM2D):
    def get_config(self):
        config = super().get_config()
        return config

def create_model(input_shape, output_shape, num_layers_conv, learning_rate):
    print("Creating model...")
    inputs = tf.keras.Input(input_shape)
    x = inputs
    for i in range(num_layers_conv):
        x_shortcut = x
        x = layers.TimeDistributed(layers.Conv2D(256, (3, 3), strides=2, dilation_rate=1, activation='relu', padding='same', kernel_initializer='he_normal'))(x)
        x = layers.BatchNormalization()(x)
                
        x_shortcut = layers.TimeDistributed(layers.Conv2D(256, (1, 1), strides=2, kernel_initializer='he_normal'))(x_shortcut)
        x_shortcut = layers.BatchNormalization()(x_shortcut)
        
        x = layers.Add()([x, x_shortcut])

    x = CustomConvLSTM2D(256, (3, 3), strides=2, activation='relu', padding='same', return_sequences=False, kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)

    x = CustomAdditiveAttention()([x, x])
    x = layers.Reshape((-1, x.shape[-2]*x.shape[-1]))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(np.prod(output_shape), activation='sigmoid')(x)
    outputs = layers.Reshape(output_shape)(x)

    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])     
    print("Model created")
    return model


img_size_input = (126, 126)
path_input = 'C:/Users/crgon/OneDrive/Desktop/thinkbe_app/ai_projects/trading/model_testing/val_predictions/cnn_lstm_(model_1)_main_4/07_25_2023_2/*.png'

images_input, images_output = load_images(path_input, img_size_input)

# split the data into training and test sets in a chronological order
split_index = int(len(images_input) * 0.6)

images_input_train = images_input[:split_index]
images_input_test = images_input[split_index:]

images_output_train = images_output[:split_index]
images_output_test = images_output[split_index:]

context_window = 1
X_train, y_train = create_sequences(images_input_train, images_output_train, context_window)
X_test, y_test = create_sequences(images_input_test, images_output_test, context_window)

num_layers_conv = 4
learning_rate = 0.0001
epochs = 0
batch_size = 24

csv_logger = CSVLogger('training.log')

model = create_model(input_shape=X_train.shape[1:], output_shape=y_train.shape[1:], num_layers_conv=num_layers_conv, learning_rate=learning_rate)

# Print a summary of the model
model.summary()

generator = data_generator(path_input, batch_size, img_size_input, context_window)

steps_per_epoch = len(sorted(glob.glob(path_input))) // batch_size

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath='my_checkpoint',  # Save models to a file that includes the epoch number
    save_weights_only=True,  # Save just weights, not whole model
    monitor='val_loss',
    mode='auto',
    save_best_only=True,  # Do not only save when validation loss improves
    save_freq='epoch',  # Save every epoch
    verbose=1)

# Load weights and continue training
model.load_weights('my_checkpoint')

history = model.fit(generator, 
            steps_per_epoch=steps_per_epoch, 
            epochs=epochs, 
            verbose=1, 
            validation_data=(X_test, y_test), 
            callbacks=[csv_logger, model_checkpoint_callback])  # Add the model checkpoint callback here

print("Making validation set predictions...")
predictions = model.predict(X_test)
print("Validation set predictions made")

# Define the path where you want to save your plot
plot_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_projects\\trading\\model_testing'
os.makedirs(plot_path, exist_ok=True)

# assuming 'epochs' variable exists
if epochs > 0:
    # Plot the training and validation loss
    plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    # Save the figure
    plt.savefig(f"{plot_path}\\loss_over_epochs.png")
    plt.show()

pred_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_projects\\trading\\model_testing\\val_predictions\\cnn_lstm_(model_2)_main\\07_27_2023'

os.makedirs(pred_path, exist_ok=True)

print("Saving side-by-side validation set images...")
for i in range(len(predictions) - 1):  # Change here
    # Convert arrays to images and resize to 126x126
    original_img = Image.fromarray((images_output_test[i + 1].squeeze() * 255).astype(np.uint8), 'L').resize((126, 126))  # Change here
    input_img = Image.fromarray((images_input_test[i + 1].squeeze() * 255).astype(np.uint8), 'L').resize((126, 126))  # Change here
    prediction_img = Image.fromarray((predictions[i].squeeze() * 255).astype(np.uint8), 'L').resize((126, 126))

    # Create a new image with appropriate dimensions and paste each image
    combined_img = Image.new('L', (378, 126))
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(input_img, (126, 0))
    combined_img.paste(prediction_img, (252, 0))

    # Save the combined image
    combined_filename = f"{pred_path}\\combined_{i}.png"
    combined_img.save(combined_filename)



""" # Path to the separate test set
test_set_input_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_projects\\trading\\model_testing\\test_set\\test_set_1\\*.png"
test_set_output_path = "C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_projects\\trading\\model_testing\\test_set\\target_test_set_1\\*.png"

# Load separate test set images
print("Loading separate test set images...")
images_input_test_new, images_output_test_new = load_images(test_set_input_path, test_set_output_path, img_size_input)

# Create sequences for the separate test set
print("Creating sequences for the separate test set...")
X_test_new, y_test_new = create_sequences(images_input_test_new, images_output_test_new, context_window)

# Make predictions on the separate test set
print("Making separate test set predictions...")
test_predictions_new = model.predict(X_test_new)
print("Separate test set predictions made")

# Calculate and print test loss for the separate test set
test_loss_new = mean_squared_error(y_test_new.flatten(), test_predictions_new.flatten(), squared=False)
print(f"Separate test set loss: {test_loss_new}")

# Save combined images of the separate test set predictions
test_predictions_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_projects\\trading\\model_testing\\test_predictions\\cnn_lstm_(model_1)_main_4\\07_25_2023'
os.makedirs(test_predictions_path, exist_ok=True)

print("Saving side-by-side separate test set images...")
for i, (original, prediction) in enumerate(zip(y_test_new, test_predictions_new)):
    # Add squeeze() to remove singleton dimensions.
    original_img = Image.fromarray((original.squeeze() * 255).astype(np.uint8), 'L')
    prediction_img = Image.fromarray((prediction.squeeze() * 255).astype(np.uint8), 'L')

    combined_filename = f"{test_predictions_path}\\combined_{i}.png"
    combined_img = Image.new('RGB', (original_img.width + prediction_img.width, original_img.height))
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(prediction_img, (original_img.width, 0))
    combined_img.save(combined_filename)

print("Separate test set prediction images saved.") """
   
model.save('model.h5')


    
   


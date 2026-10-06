import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.layers import AdditiveAttention, Reshape
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.data import Dataset

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.keras.mixed_precision.set_global_policy('mixed_float16')

@tf.function
def process_path(file_path, img_size_input):
    img = tf.io.read_file(file_path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, img_size_input)
    img = tf.cast(img, tf.float32) / 255.0
    return img

def load_images(path, img_size):
    print("Loading images from: " + path)
    image_paths = sorted(glob.glob(path))
    return tf.data.Dataset.from_tensor_slices(image_paths).map(lambda x: process_path(x, img_size))


img_size_input = (125, 125)
path_input = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\test_pics_2\\*.png'

def create_sequences(images_dataset, context_window):
    print("Creating sequences...")
    data = images_dataset.window(size=context_window+1, shift=1, drop_remainder=True)
    data = data.flat_map(lambda window: window.batch(context_window + 1))
    data = data.map(lambda window: (window[:-1], window[-1:]))
    return data

def create_model(input_shape, output_shape, num_layers_conv, learning_rate):
    x_input = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x_input)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
    x = layers.Conv2D(128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', kernel_initializer='he_normal')(x)

    # Reshape output from Conv2D layer(s) into 3D shape expected by GRU layer
    # -1 for the first dimension will ensure we maintain the same batch size
    # We multiply the last three dimensions together to make them a single dimension
    x = layers.Reshape((-1, np.prod(x.shape[1:])))(x)

    x = layers.GRU(256, activation='relu', return_sequences=False, kernel_initializer='he_normal')(x)


images_input = load_images(path_input, img_size_input)
split_index = int(len(images_input) * 0.8)

images_input_train = images_input.take(split_index)
images_input_test = images_input.skip(split_index)

context_window = 1
train_data = create_sequences(images_input_train, context_window)
test_data = create_sequences(images_input_test, context_window)

num_layers_conv = 6
learning_rate = .001
epochs = 3
batch_size = 12

csv_logger = CSVLogger('training.log')

model = create_model(input_shape=(context_window, img_size_input[0], img_size_input[1], 1), 
                     output_shape=(img_size_input[0], img_size_input[1], 1), 
                     num_layers_conv=num_layers_conv, 
                     learning_rate=learning_rate)
print("Model input shape:", model.input_shape)

train_data = train_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
test_data = test_data.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)

model.fit(train_data.batch(batch_size), epochs=epochs, verbose=1, validation_data=test_data.batch(batch_size), callbacks=[csv_logger])

print("Making predictions...")
predictions = model.predict(test_data)
print("Predictions made")

pred_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\predictions'

os.makedirs(pred_path, exist_ok=True)

print("Saving side-by-side comparison images...")
for i, (original, prediction) in enumerate(zip(y_test, predictions)):
    original_img = Image.fromarray((original * 255).astype(np.uint8))
    prediction_img = Image.fromarray((prediction * 255).astype(np.uint8))

    combined_filename = f"{pred_path}\\combined_{i}.png"
    combined_img = Image.new('RGB', (original_img.width + prediction_img.width, original_img.height))
    combined_img.paste(original_img, (0, 0))
    combined_img.paste(prediction_img, (original_img.width, 0))
    combined_img.save(combined_filename)

print("Comparison images saved")
import tensorflow as tf
import os
import glob

# Data Preparation
def preprocess_sequence(input_files, target_file, img_size_input, img_size_output, context_window):
    input_files = tf.constant(input_files, dtype=tf.string)
    img_sequence = tf.map_fn(lambda x: tf.image.decode_image(tf.io.read_file(x), channels=3), input_files, dtype=tf.uint8)
    img_sequence = [tf.image.resize(tf.cast(img, tf.float32), img_size_input) / 255.0 for img in img_sequence]

    img_output = tf.image.decode_image(tf.io.read_file(target_file), channels=3)
    img_output = tf.image.resize(img_output, img_size_output)
    img_output = tf.cast(img_output, tf.float32) / 255.0

    return tf.stack(img_sequence, axis=0), img_output

def data_generator(base_folder_path, batch_size, img_size_input, img_size_output, context_window, mode):
    if mode == "training":
        training_folders = sorted(glob.glob(os.path.join(base_folder_path, "*_training_set")))
        target_folders = sorted(glob.glob(os.path.join(base_folder_path, "*_target_set")))
    elif mode == "validation":
        training_folders = sorted(glob.glob(os.path.join(base_folder_path, "*_validation_set")))
        target_folders = sorted(glob.glob(os.path.join(base_folder_path, "*_target_set")))
    else:
        raise ValueError("Invalid mode")

    all_sequences = []
    for train_folder, target_folder in zip(training_folders, target_folders):
        input_image_files = sorted(glob.glob(os.path.join(train_folder, "*.png")))
        target_image_files = sorted(glob.glob(os.path.join(target_folder, "*.png")))
        for idx in range(0, len(input_image_files) - context_window):
            sequence = (input_image_files[idx:idx + context_window], target_image_files[idx + context_window])
            all_sequences.append(sequence)

    def generator():
        for input_files, target_file in all_sequences:
            yield preprocess_sequence(input_files, target_file, img_size_input, img_size_output, context_window)

    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(context_window, *img_size_input, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(*img_size_output, 3), dtype=tf.float32)
        )
    )

    dataset = dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    return dataset

# Model Architecture
input_shape = (4, 64, 64, 3)  # 5 time steps, 64x64 image, 3 channels
output_shape = (25, 25, 3)  # 25x25 target image, 3 channels

encoder_inputs = tf.keras.layers.Input(shape=input_shape, name='encoder_inputs')
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(encoder_inputs)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D((2, 2), padding='same'))(x)
x = tf.keras.layers.Flatten()(x)
encoded = tf.keras.layers.Dense(128, activation='relu')(x)

# Clustering layer
clustering_layer = tf.keras.layers.Dense(128, name='clustering')(encoded)

# Decoder
# Before Reshape
x = tf.keras.layers.Dense(51200, activation='relu')(clustering_layer)
x = tf.keras.layers.Reshape((5, 16, 16, 64))(x)

x = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((2, 2)))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'))(x)
x = tf.keras.layers.TimeDistributed(tf.keras.layers.UpSampling2D((2, 2)))(x)
decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same'))(x)

# Model
autoencoder = tf.keras.Model(inputs=encoder_inputs, outputs=[decoded, clustering_layer])

autoencoder.compile(optimizer='adam', loss=['mse', 'mse'], metrics=['accuracy'])

# Parameters
batch_size = 32
img_size_input = (64, 64)
img_size_output = (25, 25)
context_window = 4
epochs = 50

# Prepare data
training_data = data_generator('/home/thinkbe/Desktop/trading/model_testing/training_set/training_set_3/', batch_size, img_size_input, img_size_output, context_window, 'training')
validation_data = data_generator('/home/thinkbe/Desktop/trading/model_testing/training_set/training_set_3/', batch_size, img_size_input, img_size_output, context_window, 'validation')

# Train the model
history = autoencoder.fit(
    training_data,
    validation_data=validation_data,
    epochs=epochs,
    steps_per_epoch=200,  # Adjust based on your dataset size
    validation_steps=50  # Adjust based on your dataset size
)

# After training, you can use clustering_layer to identify similar groups. Cluster analysis can be done with K-means or other techniques.

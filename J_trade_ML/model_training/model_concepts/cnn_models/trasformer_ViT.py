import os
import glob
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Reshape, Input
from tensorflow.keras.callbacks import CSVLogger
from transformers import TFViTModel, ViTConfig

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
tf.keras.mixed_precision.set_global_policy('mixed_float16')

def load_images(path, img_size=(224, 224)):
    print("Loading images from: " + path)
    images = []
    for img_path in sorted(glob.glob(path)):
        img = Image.open(img_path).convert('RGBA')
        img = img.resize(img_size)
        img = np.array(img, dtype=np.float32) / 255.0  
        if img.shape != (img_size[0], img_size[1], 4):  
            print(f"Skipping image {img_path} due to incorrect shape {img.shape}")
            continue
        img = img[:, :, :3]
        images.append(img)
    return np.stack(images)

def create_sequences(images, context_window):
    print("Creating sequences...")
    X = []
    y = []
    for i in range(len(images) - context_window):
        X.append(images[i])
        y.append(images[i + context_window])
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes, learning_rate):
    print("Creating model...")

    config = ViTConfig.from_pretrained('google/vit-base-patch16-224')
    vit_model = TFViTModel(config)

    inputs = Input(shape=input_shape)
    x = vit_model(inputs)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(np.prod(input_shape), activation='sigmoid')(x)
    outputs = Reshape(input_shape)(outputs)

    model = Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate), loss='mse')

    print("Model created")
    return model

# Set paths and image size
img_size = (224, 224) 
input_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\test_pics_main\\*.png'

# Load image data
images = load_images(input_path, img_size)

print(images.shape)

context_window = 1
X, y = create_sequences(images, context_window)

print("Splitting data into train and validation sets...")
validation_split = 0.40
num_train_samples = int((1 - validation_split) * len(X))
X_train, y_train = X[:num_train_samples], y[:num_train_samples]
X_val, y_val = X[num_train_samples:], y[num_train_samples:]

input_shape = (224, 224, 3) 
num_classes = 1000
learning_rate = 0.001

model = create_model(input_shape, num_classes, learning_rate)

log_csv_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\training_logs.csv'
csv_logger = CSVLogger(log_csv_path, append=True)

print("Training model...")
batch_size = 1
epochs = 100
model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_val, y_val), callbacks=[csv_logger])
print("Model trained")

print("Making predictions...")
predictions = model.predict(X_val)
print("Predictions made")

orig_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\original_images'
pred_path = 'C:\\Users\\crgon\\OneDrive\\Desktop\\thinkbe_app\\ai_side_projects\\trading\\predicted_images'

os.makedirs(orig_path, exist_ok=True)
os.makedirs(pred_path, exist_ok=True)

print("Saving images...")
for i, (original, prediction) in enumerate(zip(y_val, predictions)):
    Image.fromarray((original * 255).astype(np.uint8)).save(f"{orig_path}\\original_{i}.png")
    Image.fromarray((prediction * 255).astype(np.uint8)).save(f"{pred_path}\\prediction_{i}.png")
print("Images saved")

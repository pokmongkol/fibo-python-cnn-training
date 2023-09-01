import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

# Assuming you have two folders: 'good' and 'bad', each containing their respective images

# Load and preprocess data
image_size = (1920, 1808)
num_samples_per_class = 50
num_classes = 2

# Placeholder arrays for images and labels
images = np.zeros((num_samples_per_class * num_classes, image_size[0], image_size[1], 3))
labels = np.zeros((num_samples_per_class * num_classes,))

# Load 'good' images
for i in range(num_samples_per_class):
    img = tf.keras.preprocessing.image.load_img(f'good/good_{i}.jpg', target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    images[i] = img_array
    labels[i] = 1  # Good label

# Load 'bad' images
for i in range(num_samples_per_class):
    img = tf.keras.preprocessing.image.load_img(f'bad/bad_{i}.jpg', target_size=image_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    images[i + num_samples_per_class] = img_array
    labels[i + num_samples_per_class] = 0  # Bad label

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Build the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='sigmoid')  # Use sigmoid for binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=16, epochs=10, validation_data=(X_test, y_test))

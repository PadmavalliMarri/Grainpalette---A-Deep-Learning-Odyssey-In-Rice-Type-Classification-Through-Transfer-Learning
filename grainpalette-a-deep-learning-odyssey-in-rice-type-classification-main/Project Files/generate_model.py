import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Settings
num_classes = 5
image_shape = (64, 64, 3)
samples_per_class = 100

# Generate dummy image data and labels
X = np.random.rand(num_classes * samples_per_class, *image_shape).astype('float32')
y = np.repeat(np.arange(num_classes), samples_per_class)
y = to_categorical(y, num_classes)

# Define CNN model
model = models.Sequential([
    layers.Input(shape=image_shape),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train with dummy data
model.fit(X, y, epochs=3, batch_size=32, verbose=1)

# Save the model
model.save("rice_classification_model.h5")
print("✅ Model saved as rice_classification_model.h5")

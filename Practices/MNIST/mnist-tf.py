import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# load datasets (mnist)
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

# normalize datasets
train_images = train_images / 255.0
test_images = test_images / 255.0

plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.show()

# build model
model = models.Sequential([
    layers.Rescaling(1./255, input_shape=(28, 28, 1)),  # normalisze datasets
    layers.Conv2D(32, (3, 3), activation='relu'),       # Conv2D
    layers.MaxPooling2D((2, 2)),                        # Pooling
    layers.Conv2D(64, (3, 3), activation='relu'),       # Conv2D
    layers.MaxPooling2D((2, 2)),                        # Pooling
    layers.Conv2D(64, (3, 3), activation='relu'),       # Conv2D
    layers.Flatten(),                                   # Flatten
    layers.Dense(64, activation='relu'),                # Fully-connected
    layers.Dense(10, activation='softmax')              # Output layer（10 neuron for 0 - 9）
])

# compile model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# train model
model.fit(train_images, train_labels, epochs=5, batch_size=64)

# evaluate model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# prediction by the trained model
predictions = model.predict(test_images[:5])
for i, prediction in enumerate(predictions):
    print(f"Prediction: {prediction.argmax()}, True Label: {test_labels[i]}")

plt.imshow(test_images[0], cmap=plt.cm.binary)
plt.title(f"Prediction: {predictions[0].argmax()}, True Label: {test_labels[0]}")
plt.show()


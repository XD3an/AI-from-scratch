import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt


# load datasets (CIFAR-10)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# normalize the dataset to [0, 255] to [0, 1]
x_train, x_test = x_train / 255.0, x_test / 255.0

# show datasets shape
print(f"Training data: {x_train.shape}, {y_train.shape}")
print(f"Test data: {x_test.shape}, {y_test.shape}")

# define CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(10)
])

model.summary()

# compile model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# train model
model.fit(x_train, y_train, batch_size=128, epochs=10, validation_data=(x_test, y_test))

# evaluate model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

# predict
predictions = model.predict(x_test)

for i in range(5):
    plt.imshow(x_test[i])
    plt.title(f"Prediction: {predictions[i].argmax()}, True Label: {y_test[i]}")
    plt.show()

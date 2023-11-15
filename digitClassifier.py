# Import required libraries
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Preprocess the data: normalize the pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0

# Explore the data: visualize some training images
plt.figure(figsize=(10, 10))
for i in range(25):
	plt.subplot(5, 5, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(train_images[i], cmap=plt.cm.binary)
	plt.xlabel(train_labels[i])
plt.show()

# Build the neural network model
model = models.Sequential([
	layers.Flatten(input_shape=(28, 28)),  # Flatten the 28x28 images
	layers.Dense(128, activation='relu'),  # Fully connected layer with 128 units
	layers.Dropout(0.2),              	# Dropout layer to reduce overfitting
	layers.Dense(10, activation='softmax') # Output layer for 10 classes
])

# Compile the model
model.compile(optimizer='adam',
          	loss='sparse_categorical_crossentropy',
          	metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=5)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# Make predictions on test set
predictions = model.predict(test_images)
predicted_labels = [np.argmax(pred) for pred in predictions]

# Visualize some test images and their predicted labels
plt.figure(figsize=(10, 10))
for i in range(25):
	plt.subplot(5, 5, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel(f"True: {test_labels[i]}, Pred: {predicted_labels[i]}")
plt.show()

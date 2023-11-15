import tensorflow as tf #For building neural networks
from tensorflow.keras import layers, models #For model layers and architecture
import matplotlib.pyplot as plt #For plotting and visualization
import numpy as np

#Load the MNIST dataset
mnist = tf.keras.datasets.mnist #Accessing MNIST dataset from TensorFlow's dataset library
(train_images, train_labels), (test_images, test_labels) = mnist.load_data() #Loading training and test data

# Preprocess the data by normalizing the pixel values
train_images, test_images = train_images / 255.0, test_images / 255.0 #Normalize pixel values to be between 0 and 1

#Explore the data: visualize some training images
plt.figure(figsize=(10, 10))  #Setting the figure size for the plot
for i in range(25):
	plt.subplot(5, 5, i + 1)  #Creating a subplot for each image
	plt.xticks([])  #Removing x-axis ticks
	plt.yticks([])  #Removing y-axis ticks
	plt.imshow(train_images[i], cmap=plt.cm.binary)  #Displaying the image in grayscale
	plt.xlabel(train_labels[i])  #Showing the label of the image
plt.show()  #Display the plot

#Build the neural network model
model = models.Sequential([
	layers.Flatten(input_shape=(28, 28)),  #Flatten the 28x28 images to a 1D array
	layers.Dense(128, activation='relu'),  #Fully connected layer with 128 neurons, ReLU activation
	layers.Dropout(0.2),                    #Dropout layer to prevent overfitting, dropping 20% of the neurons
	layers.Dense(10, activation='softmax')  #Output layer with 10 neurons (one for each class), Softmax activation
])

#Compile the model
model.compile(optimizer='adam',  #Using Adam optimizer
          	loss='sparse_categorical_crossentropy',  #Loss function for multi-class classification
          	metrics=['accuracy'])  #Metric to monitor is accuracy

#Train the model
model.fit(train_images, train_labels, epochs=5)  #Training the model for 5 epochs

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels)  #Evaluating the model on test data
print(f"Test accuracy: {test_acc}")  #Printing the test accuracy

#Make predictions on test set
predictions = model.predict(test_images)  #Making predictions on the test set
predicted_labels = [np.argmax(pred) for pred in predictions]  #Converting predictions to label indices

#Visualize some test images and their predicted labels
plt.figure(figsize=(10, 10))  #Setting the figure size for the plot
for i in range(25):
	plt.subplot(5, 5, i + 1)  #Creating a subplot for each image
	plt.xticks([])  #Removing x-axis ticks
	plt.yticks([])  #Removing y-axis ticks
	plt.imshow(test_images[i], cmap=plt.cm.binary)  #Displaying the image in grayscale
	plt.xlabel(f"True: {test_labels[i]}, Pred: {predicted_labels[i]}")  #Showing the true and predicted labels
plt.show()  #Display the plot
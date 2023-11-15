import tensorflow as tf 					#For building neural networks
from tensorflow.keras import layers, models	#For model layers and architecture
import matplotlib.pyplot as plt 			#For plotting and visualization
import numpy as np

#Load the MNIST dataset
#Each image is a 28x28 pixel grayscale image of a handwritten digit
#Training data (train_images, train_labels): the larger part of the dataset used to train the machine learning model
#train_images contains the image data, while train_labels contains the corresponding numerical labels (the actual digit each image represents)
#Testing data (test_images, test_labels): This is a smaller part of the dataset used to test or evaluate the model after it's been trained
#Similar to the training data, test_images contains the image data, and test_labels contains the corresponding labels

mnist = tf.keras.datasets.mnist 												#Accessing MNIST dataset from TensorFlow's dataset library
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()	#Loading training and test data

#Preprocess the data by normalizing the pixel values
#This process can lead to a more stable and faster convergence during training, as it helps to standardize the range of input features

train_images, test_images = train_images / 255.0, test_images / 255.0 			#Normalize pixel values to be between 0 and 1

#Visualize some training images

plt.figure(figsize=(10, 10))  													#Setting the figure size for the plot
for i in range(25): 															#Plot 25 images
	plt.subplot(5, 5, i + 1)  													#Creating a subplot for each image
	plt.xticks([])
	plt.yticks([])
	plt.imshow(train_images[i], cmap=plt.cm.binary)								#Displaying the image in grayscale
	plt.xlabel(train_labels[i])  												#Showing the label of the image
plt.show()

#Build the neural network model
#A Sequential model is a linear stack of layers. It's the simplest kind of model that you can build in Keras, suitable for most problems.
#The first layer is a Flatten layer, which converts the 2D image data into a 1D array. 
#This is necessary because the following dense layers require 1D input.
#Next is a fully connected (dense) layer with 128 neurons.
#The activation='relu' argument specifies that the layer uses the ReLU (Rectified Linear Unit) activation function.
#The ReLU function is mathematically defined as f(x)=max⁡(0,x).This means that for any input x, the output is the maximum of 0 and x.
#ReLU is a common activation function in neural networks, particularly useful for solving non-linear problems.
#The next (dropout) layer sets a fraction (0.2 in this case) of input units to 0 at each update during training time.
#This helps prevent overfitting.
#Overfitting occurs when a model learns the training data too well, including the noise and outliers,
#which then negatively affects its performance on new, unseen data.
#The final layer is another dense layer with 10 neurons, one for each class (digit) in the MNIST dataset.
#The softmax activation function is used for multi-class classification tasks.
#It converts the outputs to probability-like values and allows one to choose the class
#with the highest probability as the model's output prediction.

model = models.Sequential([
	layers.Flatten(input_shape=(28, 28)),										#Flatten the 28x28 images to a 1D array
	layers.Dense(128, activation='relu'),  										#Fully connected layer with 128 neurons, ReLU activation
	layers.Dropout(0.2),                    									#Dropout layer to prevent overfitting, dropping 20% of the neurons
	layers.Dense(10, activation='softmax')  									#Output layer with 10 neurons (one for each class), Softmax activation
])

#Compile the model
#The optimizer is an algorithm or method used to change the attributes
#of the neural network such as weights and learning rate in order to reduce the losses.
#Adam is a popular optimization algorithm in deep learning because it combines
#the best properties of the AdaGrad and RMSProp algorithms to provide an optimization algorithm that can handle sparse gradients on noisy problems.
#It's known for being robust and effective in a wide range of problems.
#The loss function is a measure of how good a prediction model does in terms of being able to predict the expected outcome. 
#In this case, sparse_categorical_crossentropy is used as the loss function for the model.
#This is a common choice for classification problems where the labels are integers.
#It differs from categorical_crossentropy in the way it handles labels:
#sparse_categorical_crossentropy takes integer labels, while categorical_crossentropy expects one-hot encoded labels.
#Metrics are used to evaluate the performance of the model. In this context, accuracy is used as a metric.
#Accuracy measures the proportion of correct predictions among the total number of cases examined.
#It's a common metric for classification tasks and provides a general idea of how often the model is correct across all classes.

model.compile(optimizer='adam',  												#Using Adam optimizer
          	loss='sparse_categorical_crossentropy',  							#Loss function for multi-class classification
          	metrics=['accuracy'])  												#Metric to monitor is accuracy

#Train the model
#fit() is a method of the Sequential class in Keras that is used for training the model. It fits the model to the training data.
#The epochs parameter specifies the number of times the learning algorithm will work through the entire training dataset. #FIXME
#One epoch means that each sample in the training dataset has had an opportunity to update the internal model parameters.
#An epoch is comprised of one or more batches, depending on the size of the training dataset and the defined batch size.
#In this case, setting epochs=5 means the entire dataset is passed forward and backward through the neural network five times.

model.fit(train_images, train_labels, epochs=5)  								#Training the model for 5 epochs

#Evaluate the model on the test set
#.evaluate() computes the loss and any additional metrics (in this case, accuracy) on the provided test dataset
#It's important to use a separate dataset (not used during training) for evaluation to ensure that the assessment of the model's performance is accurate and unbiased.

test_loss, test_acc = model.evaluate(test_images, test_labels)					#Evaluating the model on test data
print(f"Test accuracy: {test_acc}")												#Printing the test accuracy

#Make predictions on test set
#The .predict() method applies the trained neural network model to test_images
#For each image in the test set, the model outputs a set of values representing the probability that the image belongs to each of the possible classes
#The np.argmax(pred) function is used to find the index of the maximum value in pred, which is one of the arrays inside predictions. 
#Since pred contains probabilities for each class, np.argmax(pred) finds the class with the highest probability,
#which is the model's prediction for the corresponding test image.
#[np.argmax(pred) for pred in predictions] iterates over all the prediction arrays (one for each test image) and applies np.argmax to each one.
#The result is a list of predicted class labels (the most likely digit in each image, according to the model).

predictions = model.predict(test_images)  										#Making predictions on the test set
predicted_labels = [np.argmax(pred) for pred in predictions]  					#Converting predictions to label indices

#Visualize some test images

plt.figure(figsize=(10, 10))
for i in range(25):
	plt.subplot(5, 5, i + 1)
	plt.xticks([])
	plt.yticks([])
	plt.imshow(test_images[i], cmap=plt.cm.binary)
	plt.xlabel(f"True: {test_labels[i]}, Pred: {predicted_labels[i]}")
plt.show()
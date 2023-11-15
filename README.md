# MNIST Handwritten Digit Classification using TensorFlow

## Overview
This project involves building, training, and evaluating a neural network model using TensorFlow and Keras. The model is trained on the MNIST dataset, which consists of 28x28 pixel grayscale images of handwritten digits.

## Dependencies
- TensorFlow
- Keras
- Matplotlib
- NumPy

## Dataset
The MNIST dataset is used, which includes:
- Training data (`train_images`, `train_labels`): Images and corresponding labels for training the model.
- Testing data (`test_images`, `test_labels`): Data used to evaluate the model.

## Preprocessing
- Normalizing pixel values of images to be between 0 and 1 for stability and efficiency during training.

## Model Architecture
- A Sequential model comprising:
  - A Flatten layer to convert 2D images into a 1D array.
  - A Dense layer with 128 neurons and ReLU activation.
  - A Dropout layer to reduce overfitting.
  - A Dense output layer with 10 neurons (one for each digit) with Softmax activation.

## Compilation
- Optimizer: Adam
- Loss Function: Sparse Categorical Crossentropy
- Metrics: Accuracy

## Training
- The model is trained for 5 epochs.

## Evaluation
- The model's performance is evaluated on the test dataset.

## Predictions
- The trained model is used to predict digit classes for test images.

## Visualization
- Plots of training and test images with predictions are provided.

## Usage
1. Install dependencies.
2. Run the script to train and evaluate the model.
3. View the plots of digits with their predicted and actual labels.

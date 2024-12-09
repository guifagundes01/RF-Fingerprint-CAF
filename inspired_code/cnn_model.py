import tensorflow as tf
from keras import layers, models
from sklearn.metrics import accuracy_score
from keras import models, layers

def create_model(input_shape, num_classes):
    model = models.Sequential()

    # Convolutional layers with increased L2 regularization
    model.add(layers.Conv2D(8, (3, 3), padding='same', input_shape=input_shape))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
    
    model.add(layers.Conv2D(16, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))
   
    model.add(layers.Conv2D(32, (3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPooling2D((2, 2)))

    # Flatten layer
    model.add(layers.Flatten())

    # Fully connected (Dense) layers with Dropou

    # Output layer
    model.add(layers.Dense(num_classes, activation='softmax'))

    return model
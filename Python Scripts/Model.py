import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import PreProcessData
import sklearn.model_selection as skm

def train(X_train, y_train, X_val, y_val, epochs, img_width, img_height, tensorboard):
    # Reshape for Convolution
    X_train_reshape = np.reshape(X_train, (-1, img_width, img_height, 1))
    X_val_reshape = np.reshape(X_val, (-1, img_width, img_height, 1))

    # Create Neural Network Model
    CNN_model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (5, 5), activation='relu', padding='same', input_shape=(img_width, img_height, 1)),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(64, (5, 5), activation='relu', padding='same', input_shape=(img_width, img_height, 1)),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Conv2D(128, (5, 5), activation='relu', padding='same', input_shape=(img_width, img_height, 1)),
        tf.keras.layers.MaxPool2D((2, 2)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(50, activation='relu'),
        tf.keras.layers.Dense(9, activation='softmax')
    ])

    # Compile Model
    CNN_model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=['accuracy'])

    # Train Model
    CNN_model.fit(X_train_reshape, y_train, epochs=epochs, validation_data=(X_val_reshape, y_val), callbacks=[tensorboard])

    CNN_model.summary()
    return CNN_model


def test(model, X_test, y_test, img_width, img_height, tensorboard):
    X_test_reshape = np.reshape(X_test, (-1, img_width, img_height, 1))
    test_loss, test_acc = model.evaluate(X_test_reshape, y_test, verbose=2, callbacks=[tensorboard])
    print('\nTest accuracy:', test_acc)
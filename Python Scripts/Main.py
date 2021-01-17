import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import time
import PreProcessData
import sklearn.model_selection as skm
import Model
#import tf2onnx
#import keras2onnx
from collections import Counter


IMG_WIDTH = 80
IMG_HEIGHT = 80

# Load data
image_data, image_labels = PreProcessData.getData(IMG_WIDTH, IMG_HEIGHT)
image_data = image_data/255.0       # Normalize Training Images

X_train, X_test, y_train, y_test = skm.train_test_split(image_data, image_labels, test_size=0.2)
X_train, X_val, y_train, y_val = skm.train_test_split(X_train, y_train, test_size=0.1)

print("Total amount of Images: ", len(image_data))
print("Training Data: ", X_train.shape)
print("Training Labels: ", y_train.shape)
print("Testing Data: ", X_test.shape)
print("Testing Labels: ", y_test.shape)
print("Validation Data: ", X_val.shape)
print("Validation Labels: ", y_val.shape)

# Display Images
# plt.imshow(X_train[0])
# plt.show()

# Initialize Tensorboard
PATH = os.path.join('logs', time.strftime("%Y-%m-%d%H-%M-%S", time.gmtime()))
tensorboard = tf.keras.callbacks.TensorBoard(log_dir=PATH, histogram_freq=1)

# Call Train Function
CNN_Model = Model.train(X_train, y_train, X_val, y_val, 30, IMG_WIDTH, IMG_HEIGHT, tensorboard)

# Call Test Function
Model.test(CNN_Model, X_test, y_test, IMG_WIDTH, IMG_HEIGHT, tensorboard)

# Test Predictions
X_test_reshape = np.reshape(X_test, (-1, IMG_WIDTH, IMG_HEIGHT, 1))
predictions = CNN_Model.predict(X_test_reshape)
print("Prediction: ", np.argmax(predictions[9]))
print("True Label: ", y_test[9])

# Get most misclassified pose
predictions = [np.argmax(prediction) for prediction in predictions]
b = predictions != y_test
misclassifications = y_test[b]
print("Most misclassified: ", Counter(misclassifications).most_common(1)[0][0])

#builder = pb_builder.SavedModelBuilder('graphs')
#builder.save()

#onnx_model = keras2onnx.convert_keras(CNN_Model,         # keras model
#                         name="Model",           # the converted ONNX model internal name
#                         target_opset=9,           # the ONNX version to export the model to
#                         channel_first_inputs=None # which inputs to transpose from NHWC to NCHW
#                         )

#tf2onnx.onnx.save_model(onnx_model, "Model.onnx")

plt.imshow(X_test[9])
plt.show()
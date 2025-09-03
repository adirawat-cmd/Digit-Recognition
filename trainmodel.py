import numpy as np
import pandas as pd 
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,datasets,models
import matplotlib.pyplot as plt

(x_train,y_train),(x_test,y_test) = datasets.mnist.load_data()

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)

# print(y_train[0])
# plt.imshow(x_train[0])
# plt.show()

x_train = x_train/255
x_test = x_test/255
x_train = x_train.reshape(-1, 28, 28, 1) 
x_test = x_test.reshape(-1, 28, 28, 1) 

cnn = models.Sequential([
    layers.Conv2D(filters=32,activation = "relu",kernel_size=(3,3),input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(filters=32,activation = "relu",kernel_size=(3,3),input_shape=(28,28,1)),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(64,activation = "relu"),
    layers.Dense(10,activation  = "softmax")
])

cnn.compile(optimizer = "adam",
            loss = "sparse_categorical_crossentropy",
            metrics = ["accuracy"])


cnn.fit(x_train,y_train,epochs = 10)
cnn.save("firstmodel.h5")

# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 12:04:35 2021

@author: jcarini2
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import SGD

import os

os.chdir("C:/Users/jcarini2")
print (os.getcwd())

image_size = (150, 150)
batch_size = 32

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "output/train",
    labels="inferred",
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "output/val",
    image_size=image_size,
    batch_size=batch_size,
)
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    "output/test",
    image_size=image_size,
    batch_size=batch_size,
)

from tensorflow.keras import layers

data_augmentation = keras.Sequential(
    [
        layers.experimental.preprocessing.RandomFlip("horizontal"),
        layers.experimental.preprocessing.RandomRotation(0.1),
    ]
)
# from keras.applications.resnet50 import ResNet50
# base_model = ResNet50(weights = 'imagenet',    input_shape=(150, 150, 3),
#     include_top=False)

base_model = keras.applications.Xception(
    weights="imagenet",  # Load weights pre-trained on ImageNet.
    input_shape=(150, 150, 3),
    include_top=False,
)  # Do not include the ImageNet classifier at the top.

# Freeze the base_model
base_model.trainable = False

# Create new model on top
inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)  # Apply random data augmentation

# Pre-trained Xception weights requires that input be normalized
# from (0, 255) to a range (-1., +1.), the normalization layer
# does the following, outputs = (inputs - mean) / sqrt(var)
norm_layer = keras.layers.experimental.preprocessing.Normalization()
mean = np.array([127.5] * 3)
var = mean ** 2
# Scale inputs to [-1, +1]
x = norm_layer(x)
norm_layer.set_weights([mean, var])

# The base model contains batchnorm layers. We want to keep them in inference mode
# when we unfreeze the base model for fine-tuning, so we make sure that the
# base_model is running in inference mode here.
x = base_model(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = keras.layers.Dense(3,activation="softmax")(x)
model = keras.Model(inputs, outputs)

callbacks = [
    keras.callbacks.ModelCheckpoint("xception_at_{epoch}.h5"),
]
opt = SGD(lr=1e-2) 

model.compile(optimizer=opt,
              loss="sparse_categorical_crossentropy",
               metrics=["accuracy"])
model.fit(train_ds, epochs=10, callbacks=callbacks, validation_data=val_ds)


base_model.trainable = True
model.summary()

model.compile(
    optimizer=opt,  # Low learning rate
              loss="sparse_categorical_crossentropy",
               metrics=["accuracy"]
)

#model=keras.models.load_model("xception_at_10.h5")
opt = SGD(lr=1e-5) 

callbacks = [
    keras.callbacks.ModelCheckpoint("fine_tuning_{epoch}.h5"),
]
epochs = 10
model.fit(train_ds, epochs=epochs, validation_data=val_ds, callbacks=callbacks)

test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print('\nTest accuracy:', test_acc)



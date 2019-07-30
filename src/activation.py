#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import random
import numpy as np 
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

'''REGRESSION PROBLEM WITH ACTIVATION FUNCTION
'''

def f(x):
    return float(float(x)*2)

x = random.sample(range(1, 100), 10)
y = [f(i) for i in x]

# Necessary Variables
epochs = 10000
learning_rate = 0.01
loss = 'mean_squared_error'
optimizer = tf.keras.optimizers.Adam(lr=learning_rate) # Stochastic gradient descent
activation = tf.nn.relu

# Model
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(64, input_shape=(1,), activation=activation))
model.add(tf.keras.layers.Dense(64, activation=activation))
model.add(tf.keras.layers.Dense(1))
model.compile(loss=loss, optimizer=optimizer)

# Training
model.fit(x, y, epochs=epochs)

layers = model.layers
w = layers[0].get_weights()[0].flatten()
b = layers[0].get_weights()[1].flatten()

print("WEIGHT: {weight} ".format(weight=float(w[0])))
print("BIAS: {bias}".format(bias=float(b[0])))

PREDICT = [2141, 100, 1, 0, 500]

predictions = model.predict(PREDICT)

for prediction in predictions.flatten():
    print(round(prediction, 1))


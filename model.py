import csv
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
import csv

## Read in training data

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)
        
## Define a generator to feed online training data to the network 

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import sklearn
from random import shuffle

def generator(samples, batch_size=30):
    batch_size = int(batch_size / 6)
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples) # Shuffle once      
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []

            for batch_sample in batch_samples:
                for i in range(3): # Loop for left, right and center images
                    for j in range(2): # Loop for flipping images
                        name = './data/IMG/'+ batch_sample[i].split('/')[-1]
                        image = mpimg.imread(name)
                        angle = float(batch_sample[3])
                        if(j==0): # No flip
                            if(i == 1):
                                angle = angle + 0.15
                            if(i == 2):
                                angle = angle - 0.15
                        if(j==1): # flipped images
                            image = np.fliplr(image)
                            angle = -angle
                            if(i == 1):
                                angle = angle - 0.15 # Left camera becomes right in flipped image
                            if(i == 2):
                                angle = angle + 0.15 # Right camera becomes left in flipped image
                        images.append(image)
                        angles.append(angle)
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

# Set our batch size
batch_size= 180 # Should be multiple of 6

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

## Model Architecture

from keras.models import Sequential
from keras.layers import *
from math import ceil

model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,(5,10),strides=(2,2),activation='relu'))
model.add(Conv2D(36,(5,10),strides=(2,2),activation='relu'))
model.add(Conv2D(48,(5,10),strides=(2,2),activation='relu'))
model.add(Conv2D(64,(3,6),strides=(2,2),activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dropout(0.75))
model.add(Dense(50))
model.add(Dropout(0.75))
model.add(Dense(10))
model.add(Dense(1))

## Compile and train the network

model.compile(loss='mse', optimizer='adam')

model.fit_generator(train_generator,
            steps_per_epoch=ceil(len(train_samples)*6/batch_size),
            validation_data=validation_generator,
            validation_steps=ceil(len(validation_samples)*6/batch_size),
            epochs=3, verbose=1)

## Save the trained network

model.save('model.h5')


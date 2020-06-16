# **Behavioral Cloning** 
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"

### File Summary

|  File            |                Description                  |
|:----------------:|:-------------------------------------------:|
|model.py          | Python script for training and saving CovNet|
|drive.py          | Autonomous driving script					 |
|model.h5          | Trained CovNet								 |
|writeup_report.md | Project Report 					         |
|video.py          | Final recording of the car driving autonomously|

### Model Architecture and Training Strategy

#### 1. Architecture

The network implemented is based on NVIDIA's self-driving convolutional network architechture. The images saved by the simulator are slightly smaller than the input layer of NVIDIA's network. Hence, instead of 5 convolutional layers, only 4 have been used.
Following code block explanes the network architecture.

```python
model = Sequential()
model.add(Lambda(lambda x: x/127.5 - 1., input_shape=(160, 320, 3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,strides=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(36,5,strides=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(48,5,strides=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Conv2D(64,3,strides=(2,2),activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
```
#### 2. Image Pre-processing

* The image is mean normalized and scaled so that all pixels have intensities in range -1 to 1.
* In addition to this, the images have been internally cropped. (70 pixels from top and 25 from bottom). This prevents the model from being distracted from learning the steering predicitions.

#### 3. Training Data Augmentation

The training data used a mix of the one provided with the project and one lap of data collected on track two. This will help the model generalize and prevent overfitting.

The data has been augmented as follows.
1. The training data is highly biased towards taking left turns. In order to help the model generalize for both right and left turns, left-right flipped images are fed to the model during training.
2. Left and right camera images have been included in the training set. The steering measurements for these images have been altered by a small amount. A small positive correction has been added to the left image and an equal negative correction has been added to the right image. This would help the model steer back the the center of the lane, in case the car veers off the desired position.

*Note: The left and right camera images will also be flipped horizontally to increase the training data. In the mirrored image, the left camera would provide an image similar to the right camera and vice versa. Hence the correction factors also need to be flipped.*

#### 4. Training Strategy

In order to gauge how well the model was working, the training data is split in training and validation set. 20% of the entire data is used as a validation set after shuffling. Dropout layers have been added after every convolutional layer to prevent overfitting. The dropout percentage and number of epochs are chosen carefully to prevent overfitting.

The final trained model drives well around the track and all wheels stay within the driving area throughout the run.

The total number of data points after augmentation would be 48,216 images. Adam optimizer has been used to train the network.

The actual amount of images, just 24,108. The remaining half of the training data (flipped images) is generated online using a python generator to save memory. Also, the conversion of images to NumPy arrays has also been done within the generator itself to use memory more efficiently.

### Conclusion

The trained model is run using `drive.py` script and the data (recorded images) is collected in folder. These images are converted into a movie using `video.py` script.

The model successfully predicts the steering angle very accurately in autonomous mode. The model can be improved (generalized), but collected data on the challenge track and training the model using this new data. 

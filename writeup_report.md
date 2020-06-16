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

[image1]: ./examples/stats.png "Training Statistics"


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
The images have a very skewed aspect ratio (due to input image size and cropping layer), and accordingly, non-square convolutional filters have been implemented to capture the features more accurately.
Following code block explanes the network architecture.

```python
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
```
#### 2. Image Pre-processing

* The image is mean normalized and scaled so that all pixels have intensities in range -1 to 1.
* In addition to this, the images have been internally cropped. (70 pixels from top and 25 from bottom). This prevents the model from being distracted from learning the steering predicitions.

#### 3. Training Data Collection and Augmentation

The training data provided with the project was proving to be insufficient to train the model to handle all situations, especially sharp turns. Hence, one lap focussing on turns was recorded, with multiple forward and backward runs for sharp corners. In addition to this, one run was recorded only to teach the model how to recover if the vehicle veers of to the side of the road. This helped the model learn the required manuevers to drive successfully around the track.

Although the data was sufficiently large, the network that was being implemented was equally complex. To prevent overfitting and generalize the learning process, the training data has been augmented as follows.
1. The training data is highly biased towards taking left turns. In order to help the model generalize for both right and left turns, left-right flipped images are fed to the model during training.
2. Left and right camera images have been included in the training set. The steering measurements for these images have been altered by a small amount. A small positive correction has been added to the left image and an equal negative correction has been added to the right image. This would help the model steer back the the center of the lane, in case the car veers off the desired position.

*Note: The left and right camera images will also be flipped horizontally to increase the training data. In the mirrored image, the left camera would provide an image similar to the right camera and vice versa. Hence the correction factors also need to be flipped.*

#### 4. Training Strategy

In order to gauge how well the model was working, the training data is split in training and validation set. 20% of the entire data is used as a validation set after shuffling. 

Initial training was performed without any additional data collection (just on the provided dataset.) However, the resulting model was unable to make turns, despite several hyperparameter tuning iterations. Then, additional data was collected, which resulted in underfitting of the CovNet.

Initially, 4 dropout layers were implemented with `keep_ratio = 0.5`. This resulted in a highly underfit model, with both the training and validation error being very high. Then the dropout layers were shifted from convolutional layers to after the fully connected layers. This would help the convolutional layers capture relevant features from the images more easily, while preventing overfitting on the fully connected layers. The new architecture was found the be fitting the data set really well. The final architecture implements two dropout layers with `keep_ratio=0.75` which are added after two fully connected layers. The model is trained for `epcohs = 3`.

The final trained model drives well around the track and all wheels stay within the driving area throughout the run.

The total number of data points after augmentation would be 161,166 images. Adam optimizer has been used to train the network, and hence the learning rate is automatically chosen.

The actual amount of images is 80,583. The remaining half of the training data (flipped images) is generated online using a python generator to save memory. Also, the conversion of images to NumPy arrays has also been done within the generator itself to use memory more efficiently. The generated has been implemented in such a way that a mix of flipped and original images from center, left and right cameras are included in any batch of data. This makes sure that the stochastic gradient is a sufficiently accurate representation of the actual gradient, and helps the model train faster.

Following is the gist of the generator implementation.

```python
def generator(samples, batch_size=30):
	...
            for batch_sample in batch_samples:
                for i in range(3): # Loop for left, right and center images
                    for j in range(2): # Loop for flipping images
                        ...
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
                        ...
            yield sklearn.utils.shuffle(X_train, y_train)
```

### Conclusion

The training and validation loss keep decreasing throughout all three epochs. This is a strong evidence that the model doesn't overfit the training data. The validation loss is less than the training loss throughout all epochs, which is a consequence of implementing dropout layers.

![alt text][image1]

The trained model is run using `drive.py` script and the data (recorded images) is collected in folder. These images are converted into a movie using `video.py` script.

The model successfully predicts the steering angle very accurately in autonomous mode. The model can be improved (generalized), but collected data on the challenge track and training the model using this new data. 

#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/figure_1-2.png "Data histogram"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/left_2017_04_30_00_30_33_799.jpg "Recovery Image"
[image4]: ./examples/left_2017_04_30_00_31_41_830.jpg "Recovery Image"
[image5]: ./examples/right_2017_04_30_00_30_36_372.jpg "Recovery Image"
[image6]: ./examples/origcenter_2017_04_30_00_30_34_057.jpg "Original Image"
[image7]: ./examples/center_2017_04_30_00_30_34_148.jpg "Zoomed grayscale Image"
[image8]: ./examples/binary.png "Binary Image"
## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline and various strategy I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed
The model employs the nvidia model without dropouts. There is a preprocessing layer for cropping line 437. A lambda layer to achieve 0 mean line 438.
3 2x2 convolution layers line 439-441 followed by 2 1x1 convolution layers line 442-443. Followed by 5 fully connected layers line 444-448.

The five convolution layers mentioned above are followed by RELU activation to introduce non-linearity.

####2. Attempts to reduce overfitting in the model

The model did not it self contain drop out layer. However only a portion of the total data is used to train the model under 5 epochs. There is also a increase in drop out of lower steering data throughout epochs.
It is found that using 130k+ data and having 1/5-1/4 data utilized per epoch with steering degradation works best.
####3. Model parameter tuning

The model used an Nadam optimizer and learning rate is set to 0.0001 which seems to minimize loss while still provide reasonable speed at reducing loss.

####4. Appropriate training data

Training data contains the original Udacity driving data plus the manually recorded recovery data as well as data generated through the generator using an assortment of techniques such as flipping, left camera, right camera, rotation, zooming and brightness adjustment.

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to have a model that's sufficient at deriving correct behaviour.
This model have to be reasonably easy to train as well. I started with a few fully connected layers just to get started.
Clearly this model was not powerful enough to predict correct steering and I therefore switched to using the Nvidia model architecture.
I am running all of this with an 8GB memory CPU and the sheer amount of data was starting to bog down my computer and causing python to crash.
At this point I started using a generator. I started with generated middle, left and right images and steering per yield within the model.
I then added zooming in from 1 to 5 times into the generator with adjusted steering of 0.1. 
I've attempted to use pretrained VGG but the memory strain was too much for my computer. To backtrack I probably could have reduced the 
memory consumption by 8 folds but I've abandoned VGG in favor of the Nvidia architecture following some advice from past student submissions.
The Nvidia architeture contain 2 5x5 convolutional layers with strides of 2 followed by relu activation. This makes for fast processing and still 
meets my need for the project while keeping training time reasonable. It is then followed by 3x3 convolutional layers that is followed by relu activation.
This helps the NN look in granular details. There are 10 layers aside from preprocessing layers. I believe this suffices based on my previous experience with project 2.
Where I had used fewer activations. The model was improving but it seem to have overfitted in that it expects a set number of middle, left, right and zooms in sequence.
This was when I changed strategy to mark the data for their transformations and using the generator only to apply the transformation and steering adjustments.
I also performed a shuffle such that these marked data is now randomized. Although this helped it wasn't until I started generating histograms of the data
that I was able to pinpoint what I need to adjust for the data. I added constraints that the steering should stay between 1 and -1. Then with some finish touch
 on the steering adjustment for zooming and rotation along with adding preprocessing to grayscale. The model finally performed a full lap on course 1.

 ####2. Final Model Architecture

Some of suggestions were from https://medium.com/@mohankarthik/cloning-a-car-to-mimic-human-driving-5c2f7e8d8aff
I have followed suggestion for data augmentation but used zoom and rotation instead of shift. I also incorporated a formula of measurement = (1 + math.sqrt(float(abs(measurement))) / 25) * measurement per zoom.
I also pre mark my data for augmentation so that know exactly what data I am having more of. IE I want a lot more turning data.

Below is a visualization from model summary. Fancy ploty stuff refused to install them selves properly.
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d_1 (Cropping2D)    (None, 90, 320, 1)        0
_________________________________________________________________
lambda_1 (Lambda)            (None, 90, 320, 1)        0
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 43, 158, 24)       624
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 20, 77, 36)        21636
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 8, 37, 48)         43248
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 6, 35, 64)         27712
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 4, 33, 64)         36928
_________________________________________________________________
flatten_1 (Flatten)          (None, 8448)              0
_________________________________________________________________
dense_1 (Dense)              (None, 100)               844900
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11
=================================================================
Total params: 980,619.0
Trainable params: 980,619.0
Non-trainable params: 0.0
_________________________________________________________________

####3. Creation of the Training Set & Training Process

I use the Udacity data for typical driving

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to stay off the deep end.

![alt text][image3]
![alt text][image4]
![alt text][image5]

I then recorded every failed turns for recovery data.

To augment the data as the original data set was lacking in steep turning angles. I applied transforms on steep angles such as rotation, zooming, flip and finally grayscaling to make it easier to train with. Below are examples of original image and after zooming, flipping and grayscaling.

![alt text][image6]
![alt text][image7]

After the collection process, I had 130k+ data points with data augmentation. This results in data histogram below.

![alt text][image1]

I follow the training regiment for training using the retrain input keyword as described in architecture section. Of course I also incorporated a bit of transfer learning to help troubleshoot code and steering adjustments as the training is relatively fast and continue keyword to train additional epochs to see if there are any additional performance again.
I set the model to save in different files iteratively so I can analyse the progression in driving behaviours without having to manually interrupt or supervise the training cycles.

I have not tried track 2. Though using only grayscales of track 1 data I don't believe the model will drive too far. However my hypothesis is that using a binary hash of color and saturation will get much closer to completing track 2 even without track 2 data.
As exemplified below. It's much more intuitive for the CNN to learn where to drive. Instead of trying to make sense the where to drive based on a complexed image.
It only needs to know 3 things. Fuzzy is ok, Borders are bad and darkness is your friend( You might need to give it some cookies for this one. )
![alt text][image8]

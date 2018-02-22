import csv
import gc
import cv2
import math
import numpy as np
import sklearn
import numbers
import sys
import random
import scipy.stats as stats
import pylab as pl
import matplotlib.pyplot as plt
import plotly.plotly as py

lines = []
i = 0 
#Reading flags for actions
training = input('test_gen, retrain, continue, histogram or transfer: ')

#Loading Udacity data and marking them for augmentation
with open('./data/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	next(reader)
	for line in reader:
		if training == "retrain" or training == "continue" or training == "histogram" or training == "investigate":
			#I've set the data to take 1 out of ever 10 straight driving data as there are far too many straight driving
			if (line[3] == 0 and i%10 == 0):
				lines.append(line)
			
			if line[3] != 0:
				#We take every non straight driving be sure to flip it
				#The lines are tagged with their transformation data at index 0 is how many transform it performs positive for zoom and negative for left rotation
				#At index 1 is the line data it self
				#At index 2 is which camera is taken
				#At index 3 is if the image is going to be flipped
				#There is still plenty of small steering data so we don't apply many transforms
				lines.append([0,line, 'center', False])
				lines.append([0,line, 'center', True])
			if abs(float(line[3])) >0.125 and abs(float(line[3])) < 0.9:
				#We give 5 transformation each here 5 zoom and 5 rotations as lines greater than 0.125 is fewer
				#I know rotations are adjusted for 0.02 per transformation so the upper bound for this is 0.9 I could of used an inclusive upper bound <=
				#Yes I could of used classes and objects and have put this in a loops or have refactored this as an array of camera angles over 5 iteration and true/false but I was on a tight schedule
				lines.append([1,line, 'center', False])
				lines.append([2,line, 'center', False])
				lines.append([3,line, 'center', False])
				lines.append([4,line, 'center', False])
				lines.append([5,line, 'center', False])
				lines.append([1,line, 'left', False])
				lines.append([2,line, 'left', False])
				lines.append([3,line, 'left', False])
				lines.append([4,line, 'left', False])
				lines.append([5,line, 'left', False])
				lines.append([1,line, 'right', False])
				lines.append([2,line, 'right', False])
				lines.append([3,line, 'right', False])
				lines.append([4,line, 'right', False])
				lines.append([5,line, 'right', False])
				lines.append([1,line, 'center', True])
				lines.append([2,line, 'center', True])
				lines.append([3,line, 'center', True])
				lines.append([4,line, 'center', True])
				lines.append([5,line, 'center', True])
				lines.append([1,line, 'left', True])
				lines.append([2,line, 'left', True])
				lines.append([3,line, 'left', True])
				lines.append([4,line, 'left', True])
				lines.append([5,line, 'left', True])
				lines.append([1,line, 'right', True])
				lines.append([2,line, 'right', True])
				lines.append([3,line, 'right', True])
				lines.append([4,line, 'right', True])
				lines.append([5,line, 'right', True])
				
				lines.append([-1,line, 'center', False])
				lines.append([-2,line, 'center', False])
				lines.append([-3,line, 'center', False])
				lines.append([-4,line, 'center', False])
				lines.append([-5,line, 'center', False])
				lines.append([-1,line, 'left', False])
				lines.append([-2,line, 'left', False])
				lines.append([-3,line, 'left', False])
				lines.append([-4,line, 'left', False])
				lines.append([-5,line, 'left', False])
				lines.append([-1,line, 'right', False])
				lines.append([-2,line, 'right', False])
				lines.append([-3,line, 'right', False])
				lines.append([-4,line, 'right', False])
				lines.append([-5,line, 'right', False])
				lines.append([-1,line, 'center', True])
				lines.append([-2,line, 'center', True])
				lines.append([-3,line, 'center', True])
				lines.append([-4,line, 'center', True])
				lines.append([-5,line, 'center', True])
				lines.append([-1,line, 'left', True])
				lines.append([-2,line, 'left', True])
				lines.append([-3,line, 'left', True])
				lines.append([-4,line, 'left', True])
				lines.append([-5,line, 'left', True])
				lines.append([-1,line, 'right', True])
				lines.append([-2,line, 'right', True])
				lines.append([-3,line, 'right', True])
				lines.append([-4,line, 'right', True])
				lines.append([-5,line, 'right', True])
				
			if abs(float(line[3])) >0.5 and abs(float(line[3])) < 0.8:
				#We give 5 transformation each here 5 more zoom and rotations lines greater than 0.5 is even fewerer
				#I know rotations are adjusted for 0.02 per transformation so the upper bound for this is 0.8 as we added 5 transformations already I could of used an inclusive upper bound <=
				lines.append([6,line, 'center', False])
				lines.append([7,line, 'center', False])
				lines.append([8,line, 'center', False])
				lines.append([9,line, 'center', False])
				lines.append([10,line, 'center', False])
				lines.append([6,line, 'left', False])
				lines.append([7,line, 'left', False])
				lines.append([8,line, 'left', False])
				lines.append([9,line, 'left', False])
				lines.append([10,line, 'left', False])
				lines.append([6,line, 'right', False])
				lines.append([7,line, 'right', False])
				lines.append([8,line, 'right', False])
				lines.append([9,line, 'right', False])
				lines.append([10,line, 'right', False])
				lines.append([6,line, 'center', True])
				lines.append([7,line, 'center', True])
				lines.append([8,line, 'center', True])
				lines.append([9,line, 'center', True])
				lines.append([10,line, 'center', True])
				lines.append([6,line, 'left', True])
				lines.append([7,line, 'left', True])
				lines.append([8,line, 'left', True])
				lines.append([9,line, 'left', True])
				lines.append([10,line, 'left', True])
				lines.append([6,line, 'right', True])
				lines.append([7,line, 'right', True])
				lines.append([8,line, 'right', True])
				lines.append([9,line, 'right', True])
				lines.append([10,line, 'right', True])

				lines.append([-6,line, 'center', False])
				lines.append([-7,line, 'center', False])
				lines.append([-8,line, 'center', False])
				lines.append([-9,line, 'center', False])
				lines.append([-10,line, 'center', False])
				lines.append([-6,line, 'left', False])
				lines.append([-7,line, 'left', False])
				lines.append([-8,line, 'left', False])
				lines.append([-9,line, 'left', False])
				lines.append([-10,line, 'left', False])
				lines.append([-6,line, 'right', False])
				lines.append([-7,line, 'right', False])
				lines.append([-8,line, 'right', False])
				lines.append([-9,line, 'right', False])
				lines.append([-10,line, 'right', False])
				lines.append([-6,line, 'center', True])
				lines.append([-7,line, 'center', True])
				lines.append([-8,line, 'center', True])
				lines.append([-9,line, 'center', True])
				lines.append([-10,line, 'center', True])
				lines.append([-6,line, 'left', True])
				lines.append([-7,line, 'left', True])
				lines.append([-8,line, 'left', True])
				lines.append([-9,line, 'left', True])
				lines.append([-10,line, 'left', True])
				lines.append([-6,line, 'right', True])
				lines.append([-7,line, 'right', True])
				lines.append([-8,line, 'right', True])
				lines.append([-9,line, 'right', True])
				lines.append([-10,line, 'right', True])
				i+=1
#Loading Recovery data and marking them for augmentation
if training == "retrain" or training == "test_gen" or training == "continue"  or training == "histogram":
	with open('../RecoveryData/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			if float(line[3]) != 0:		
				lines.append(line)
			if abs(float(line[3])) >0.125 and abs(float(line[3])) < 0.9:
				lines.append([1,line, 'center', False])
				lines.append([2,line, 'center', False])
				lines.append([3,line, 'center', False])
				lines.append([4,line, 'center', False])
				lines.append([5,line, 'center', False])
				lines.append([1,line, 'left', False])
				lines.append([2,line, 'left', False])
				lines.append([3,line, 'left', False])
				lines.append([4,line, 'left', False])
				lines.append([5,line, 'left', False])
				lines.append([1,line, 'right', False])
				lines.append([2,line, 'right', False])
				lines.append([3,line, 'right', False])
				lines.append([4,line, 'right', False])
				lines.append([5,line, 'right', False])
				lines.append([1,line, 'center', True])
				lines.append([2,line, 'center', True])
				lines.append([3,line, 'center', True])
				lines.append([4,line, 'center', True])
				lines.append([5,line, 'center', True])
				lines.append([1,line, 'left', True])
				lines.append([2,line, 'left', True])
				lines.append([3,line, 'left', True])
				lines.append([4,line, 'left', True])
				lines.append([5,line, 'left', True])
				lines.append([1,line, 'right', True])
				lines.append([2,line, 'right', True])
				lines.append([3,line, 'right', True])
				lines.append([4,line, 'right', True])
				lines.append([5,line, 'right', True])
				
				lines.append([-1,line, 'center', False])
				lines.append([-2,line, 'center', False])
				lines.append([-3,line, 'center', False])
				lines.append([-4,line, 'center', False])
				lines.append([-5,line, 'center', False])
				lines.append([-1,line, 'left', False])
				lines.append([-2,line, 'left', False])
				lines.append([-3,line, 'left', False])
				lines.append([-4,line, 'left', False])
				lines.append([-5,line, 'left', False])
				lines.append([-1,line, 'right', False])
				lines.append([-2,line, 'right', False])
				lines.append([-3,line, 'right', False])
				lines.append([-4,line, 'right', False])
				lines.append([-5,line, 'right', False])
				lines.append([-1,line, 'center', True])
				lines.append([-2,line, 'center', True])
				lines.append([-3,line, 'center', True])
				lines.append([-4,line, 'center', True])
				lines.append([-5,line, 'center', True])
				lines.append([-1,line, 'left', True])
				lines.append([-2,line, 'left', True])
				lines.append([-3,line, 'left', True])
				lines.append([-4,line, 'left', True])
				lines.append([-5,line, 'left', True])
				lines.append([-1,line, 'right', True])
				lines.append([-2,line, 'right', True])
				lines.append([-3,line, 'right', True])
				lines.append([-4,line, 'right', True])
				lines.append([-5,line, 'right', True])
			if abs(float(line[3])) >0.5 and abs(float(line[3])) < 0.8:
				lines.append([6,line, 'center', False])
				lines.append([7,line, 'center', False])
				lines.append([8,line, 'center', False])
				lines.append([9,line, 'center', False])
				lines.append([10,line, 'center', False])
				lines.append([6,line, 'left', False])
				lines.append([7,line, 'left', False])
				lines.append([8,line, 'left', False])
				lines.append([9,line, 'left', False])
				lines.append([10,line, 'left', False])
				lines.append([6,line, 'right', False])
				lines.append([7,line, 'right', False])
				lines.append([8,line, 'right', False])
				lines.append([9,line, 'right', False])
				lines.append([10,line, 'right', False])
				lines.append([6,line, 'center', True])
				lines.append([7,line, 'center', True])
				lines.append([8,line, 'center', True])
				lines.append([9,line, 'center', True])
				lines.append([10,line, 'center', True])
				lines.append([6,line, 'left', True])
				lines.append([7,line, 'left', True])
				lines.append([8,line, 'left', True])
				lines.append([9,line, 'left', True])
				lines.append([10,line, 'left', True])
				lines.append([6,line, 'right', True])
				lines.append([7,line, 'right', True])
				lines.append([8,line, 'right', True])
				lines.append([9,line, 'right', True])
				lines.append([10,line, 'right', True])

				lines.append([-6,line, 'center', False])
				lines.append([-7,line, 'center', False])
				lines.append([-8,line, 'center', False])
				lines.append([-9,line, 'center', False])
				lines.append([-10,line, 'center', False])
				lines.append([-6,line, 'left', False])
				lines.append([-7,line, 'left', False])
				lines.append([-8,line, 'left', False])
				lines.append([-9,line, 'left', False])
				lines.append([-10,line, 'left', False])
				lines.append([-6,line, 'right', False])
				lines.append([-7,line, 'right', False])
				lines.append([-8,line, 'right', False])
				lines.append([-9,line, 'right', False])
				lines.append([-10,line, 'right', False])
				lines.append([-6,line, 'center', True])
				lines.append([-7,line, 'center', True])
				lines.append([-8,line, 'center', True])
				lines.append([-9,line, 'center', True])
				lines.append([-10,line, 'center', True])
				lines.append([-6,line, 'left', True])
				lines.append([-7,line, 'left', True])
				lines.append([-8,line, 'left', True])
				lines.append([-9,line, 'left', True])
				lines.append([-10,line, 'left', True])
				lines.append([-6,line, 'right', True])
				lines.append([-7,line, 'right', True])
				lines.append([-8,line, 'right', True])
				lines.append([-9,line, 'right', True])
				lines.append([-10,line, 'right', True])
#Transfer learning making use of only recovery data for quick analysis
if training == "transfer":		
	with open('../RecoveryData/driving_log.csv') as csvfile:
		reader = csv.reader(csvfile)
		next(reader)
		for line in reader:
			if float(line[3]) != 0:		
				lines.append(line)

#Generator returning batch size per yield				
def generator(samples, batch_size=32):
	num_samples = len(samples)
	while 1: # Loop forever so the generator never terminates
		sklearn.utils.shuffle(samples)
		for offset in range(0, num_samples, batch_size):
			batch_samples = samples[offset:offset+batch_size]
			images = []
			measurements = []
			correction = 0.1
			direction = 'center'
			flip = False
			for line in batch_samples:
				transformNum = 0
				shiftNum = 0
				#If data is marked data unravel it 
				if isinstance(line[0], numbers.Integral):
					transformNum = int(line[0])
					shiftNum = -1*int(line[0])
					direction = line[2]
					flip = line[3]
					line = line[1]
				source_path = line[0]
				filename = source_path.split('/')[-1].split('\\')[-1]
				current_path = './data/IMG/' + filename
				fileleftpath = './data/IMG/' + line[1].split('/')[-1].split('\\')[-1]
				filerightpath = './data/IMG/' + line[2].split('/')[-1].split('\\')[-1]
				#Set image and appropriate image loaded based on direction
				if direction == 'center':
					image = cv2.imread(current_path)
				if direction == 'left':
					image = cv2.imread(fileleftpath)
				if direction == 'right':
					image = cv2.imread(filerightpath)
				
				count = 0
				#Add steering correction for left and right camera
				measurement = float(line[3])
				if direction == "left":
					measurement = measurement+correction
				if direction == "right":
					measurement = measurement-correction
					
				#Zoom and adjust steering for each zoom
				while count < transformNum:
					image = resizeDown(image)
					measurement = (1 + math.sqrt(float(abs(measurement))) / 25) * measurement
					count += 1
					

				#Rotate and adjust steering for each rotation
				count = 0
				while count < shiftNum:
					image = shiftleft(image)
					measurement = measurement +0.02
					count += 1
				#Apply flipping and adjust for flipping
				if flip:
					image = np.fliplr(image)
					measurement = -1*measurement
				#Change brightness and convert to grayscale
				image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
				random_brightness = .1 + np.random.uniform()
				image[:,:,2] = image[:,:,2] * random_brightness
				image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
				#I have originally trained them to be BGR2GRAY
				image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
				images.append(np.reshape(image, (160, 320, 1)))
				measurements.append(measurement)
				#Check flag for if Testing generator. We only want to check for large transforms.
				if transformNum >= 5 and training == "test_gen":
					cv2.imwrite('./generatedimg/orig'+filename, cv2.imread(current_path))
					cv2.imwrite('./generatedimg/'+filename, image)
					print(filename)
					print("angle orig"+str(float(line[3])))
					print("angle: "+str(measurement))
			yield sklearn.utils.shuffle(np.array(images), np.array(measurements))

#Get the adjusted measurements for graphing. I should of reused this code
def getAdjustedMeasure(line):
	correction = 0.1
	direction = 'center'
	flip = False
	transformNum = 0
	shiftNum = 0
	if isinstance(line[0], numbers.Integral):
		transformNum = int(line[0])
		shiftNum = -1*int(line[0])
		direction = line[2]
		flip = line[3]
		line = line[1]
	count = 0
	measurement = float(line[3])
	if direction == "left":
		measurement = measurement+correction
	if direction == "right":
		measurement = measurement-correction
		

	while count < transformNum:
		measurement = (1 + math.sqrt(float(abs(measurement))) / 25) * measurement
		count += 1
		

	
	count = 0
	while count < shiftNum:
		measurement = measurement +0.05
		count += 1
	
	if flip:
		measurement = -1*measurement
	return measurement

#Zooming operation	
def resizeDown(img):
	resized = cv2.resize(img,(324,162))
	return resized[1:161, 2:322]

#Left rotation
def shiftleft(img):
	rows, cols = img.shape[:2]
	M = cv2.getRotationMatrix2D((cols/2,rows/2),5,1)
	return cv2.warpAffine(img,M,(cols,rows))

#Chop data into num pieces
def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg
  return out
#Use for progressive degradation of data based on filterlevel
def filterDown(data, filterlevel):
	return sklearn.utils.shuffle([d for d in data if ((not isinstance(d[0], numbers.Integral)) and abs(float(d[3])) >= (filterlevel*random.uniform(0, 1.0))) or (isinstance(d[0], numbers.Integral)) and abs(float(d[1][3])) >= (filterlevel*random.uniform(0, 1.0))])

import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input			
from keras.models import Sequential, Model, load_model
from keras.layers import Input, Flatten, Dense, Lambda, Cropping2D, ZeroPadding2D, MaxPooling2D, Dropout, Convolution2D
from sklearn.model_selection import train_test_split


model = None
if training == "test_gen":
	x = next(generator(lines, batch_size=200))
	sys.exit()
#Set up model for retraining
if training == "retrain":
	model = Sequential()
	model.add(Cropping2D(cropping=((50,20), (0,0)), input_shape=(160,320,1)))
	model.add(Lambda(lambda x: x/255.0 - 0.75))
	model.add(Convolution2D(24,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(36,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(48,5,5,subsample=(2,2),activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Convolution2D(64,3,3,activation="relu"))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Dense(50))
	model.add(Dense(10))
	model.add(Dense(1))
numOfEpoch = 5
#Investigate if any steering seem off in the data by outputing the steering and images if steering is greater than 0.3. It's found that most steering was good.
if training == "investigate":
	[print(x) for x in list(set([d[1][0] for d in lines if (((not isinstance(d[0], numbers.Integral)) and float(d[3]) >= 0.3) or ((isinstance(d[0], numbers.Integral)) and float(d[1][3]) >= 0.3))]))]
	sys.exit()
#Reload the last epoch's model and continue training it	
if training == "continue":
	model = load_model('model'+str(numOfEpoch-1)+'.h5')

#Load the 5th model and train with recovery data	
if training == "transfer":
	model = load_model('model4.h5')
	i = 0
	for l in model.layers:
		if(i<8):
			l.trainable = False
		i += 1
#Plot current data histogram
if training == "histogram":
	h = sorted([getAdjustedMeasure(l) for l in lines])
	plt.hist(h)
	plt.title("Gaussian Histogram")
	plt.xlabel("Value")
	plt.ylabel("Frequency")
	plt.show()
	# fig = plt.gcf()
	# py.plot_mpl(fig, filename='mpl-basic-histogram')
	sys.exit()
#Modle compilation with optimizer
model.compile(optimizer= keras.optimizers.Nadam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, schedule_decay=0.004),
              loss='mse')
epoch = 0
filterlevel = 0

#Model training
while epoch < numOfEpoch:
	#Data shuffled and split
	epochSplit = chunkIt(sklearn.utils.shuffle(lines), 4)
	train_samples, validation_samples = train_test_split(epochSplit[epoch%4], test_size=0.2)
	
	#Setup samples with filtered data
	train_samples = filterDown(train_samples, filterlevel)
	validation_samples = filterDown(validation_samples, filterlevel)
	train_generator = generator(train_samples, batch_size=24)
	validation_generator = generator(validation_samples, batch_size=24)
	#training with 24 batch size
	model.fit_generator(train_generator, steps_per_epoch =math.ceil(len(train_samples)/24), validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples)/24), nb_epoch = 1 )
	#Save as different name depending on flag
	if training == "retrain":
		model.save('model'+str(epoch)+'.h5')
	if training == "transfer":
		model.save('modelt'+str(epoch)+'.h5')
	if training == "continue":
		model.save('modelc'+str(epoch)+'.h5')
	#Gradual degradation per epoch 
	filterlevel += float(1)/(numOfEpoch-1)
	epoch += 1

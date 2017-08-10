import csv
import numpy as np
import cv2

lines = []
with open('drive_data/center/driving_log.csv') as csvfile:
	reader = csv.reader(csvfile)
	for line in reader:
		lines.append(line)

images = []
measurements = []

for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current_path = 'drive_data/center/IMG/' + filename
	image = cv2.imread(current_path, 1)
	images.append(image)
	measurements.append(float(line[3]))
	# print(current_path)

print(images[0].shape)

X_train = np.array(images)
y_train = np.array(measurements)

# Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Convolution2D(32, 3, 3, input_shape=(160, 320, 3)))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Convolution2D(32, 3, 3))
model.add(Dropout(0.5))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=10)

model.save('model.h5')

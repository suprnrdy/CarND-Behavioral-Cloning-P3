import csv
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse


# Initial Setup for Keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.backend import tf as ktf
from keras.models import load_model

def main():
  samples = []
  # If we do different types of training data in different folders, grab all of it and put
  # into one list
  dirs = [dI for dI in os.listdir('./drive_data/') if os.path.isdir(os.path.join('./drive_data',dI))]
  print("Number of Training Directories: ", len(dirs))
  for l in dirs:
    path = 'drive_data/' + l + '/driving_log.csv' 
    print(path)
    with open(path) as csvfile:
      reader = csv.reader(csvfile)
      l = 0
      for line in reader:
        # samples.append(line)
        if line != ' ':
          if abs(float(line[3])) < 0.85:
            l += 1
            if abs(float(line[3])) < 0.0005:
              if l == 55:
                samples.append(line)
                l = 0
            else:
              if l%2 == 0:
                samples.append(line)


  print("Total Samples: ", len(samples))

  samplearray = np.array(samples)[:,3].astype(np.float)
  print(samplearray)
  unique, counts = np.unique(samplearray, return_counts=True)
  n_classes = len(unique)
  print("Number of classes: ", n_classes)

  fig,ax = plt.subplots()
  n, bins, patches = ax.hist(samplearray,n_classes,normed=1,edgecolor='black')
  ax.set_xlabel('Angles')
  ax.set_ylabel('Density')
  ax.set_title('Histogram of sample data')
  fig.tight_layout()
  plt.show()

  # Split the data into training and validation samples
  samples = shuffle(samples)
  train_samples, validation_samples = train_test_split(samples, test_size=0.2)
  correction = 0.2

  def getImage(sample, item=0):
    source_path = sample[item]
    filename = source_path.split('/')[-1]
    if source_path.split('/')[0].replace(' ', '') == 'IMG':
      path = 'udacitydata'
    else:
      path = source_path.split('/')[-3]
    filepath = 'drive_data/' + path + '/IMG/' + filename
    img = cv2.imread(filepath, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # imgplot = plt.imshow(rgb_img)
    # plt.show()
    return rgb_img

  # Our generator that grabs the images and steering angles and batching the set
  def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
      samples = shuffle(samples)
      for offset in range(0, num_samples, batch_size):
        batch_samples = samples[offset:offset+batch_size]
        images = []
        angles = []
        for sample in batch_samples:
          # create adjusted steering measurements for the side camera images
          correction = 0.1 # this is a parameter to tune
          steering_center = float(sample[3])
          steering_left = steering_center + correction
          steering_right = steering_center - correction

          # read in images from center, left and right cameras
          # Grab the filenames for each camera
          
          img_center = getImage(sample,0)
          img_left = getImage(sample,1)
          img_right = getImage(sample,2)

          # add images and angles to data set
          images.extend([img_center, img_left, img_right])
          angles.extend([steering_center, steering_left, steering_right])

          # add flipped image
          images.append(cv2.flip(img_center, 1))
          angles.append(steering_center*-1.0)

        X_train = np.array(images)
        y_train = np.array(angles)
        yield shuffle(X_train, y_train)

  # 
  train_generator = generator(train_samples, batch_size=32)
  validation_generator = generator(validation_samples, batch_size=32)

  def resize_function(image):
    from keras.backend import tf as ktf
    # resized = ktf.image.resize_images(image, (35, 160))
    resized = image/255.0 - 0.5
    return resized

  # If we load an existing model, reload it and use existing weights to continue training.
  print("file: ", args.weights)
  if args.weights != ' ':
    print("Continuing training from model.h5...")
    model = load_model(args.weights)
    adam = Adam(lr=0.00001)
    model.compile(loss='mse', optimizer=adam)
  else:
    print("Training from scratch...")
    model = Sequential()
    # Preprocess incoming data, centered around zero with small standard deviation 
    model.add(Cropping2D(cropping=((60, 20), (5,5)), input_shape=(160,320,3)))
    model.add(Lambda(resize_function))
    # Network based on the NVIDIA Model
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.5))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')


  print(model.summary())  
  history_object = model.fit_generator(train_generator, samples_per_epoch=len(train_samples*4), validation_data=validation_generator, nb_val_samples=len(validation_samples), nb_epoch=args.epochs, verbose=1)
  model.save('model.h5')

  ### print the keys contained in the history object
  plt.ioff()
  print(history_object.history.keys())
  plt.plot(history_object.history['loss'])
  plt.plot(history_object.history['val_loss'])
  plt.title('model mean squared error loss')
  plt.ylabel('mean squared error loss')
  plt.xlabel('epoch')
  plt.legend(['training set', 'validation set'], loc='upper right')
  plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autonomous Driving Model Training')
    parser.add_argument(
            '-td',
            '--training_data',
            type=str,
            help='Relative path to training data e.g.".\Folder\"'
    )
    parser.add_argument(
            '-e',
            '--epochs',
            type=int,
            default=5,
            help='# of epochs to train the model'
    )
    parser.add_argument(
            '-w',
            '--weights',
            type=str,
            default=' ',
            help='Trained weights file (*.h5) to initialize parameters'
    )
    parser.add_argument(
            '-f',
            '--file',
            type=str,
            help='filename to save weights and model'
    )
    
    
    args = parser.parse_args()
    
    #Call main
    main()


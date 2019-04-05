# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers  import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# IMPORTING THE MNIST DATASET BUNDLED WITH KERAS
from keras.datasets import mnist

# SPLITTING THE DATASET INTO TRAINING AND TESTING DATASETS
(X_train, y_train), (X_test, y_test) = mnist.load_data()

'''
SINCE WE ARE USING TENSORFLOW AS BACKEND, WE NEED TO RESHAPE THE FEATURE MATRIX AS BEING:
1) NUMBER OF SAMPLES, 2)NUMBER OF ROWS, 3)NUMBER OF COLUMNS, 4) NUMBER OF CHANNELS
'''
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# NORMALIZING INPUTS FROM 0-255 TO 0.0-1.0
X_train = X_train.astype('float32')/255
X_test = X_test.astype('float32')/255

# CONVERTING LABELS TO KERAS USABLE CATEGORICAL FORMAT
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

# INITIATING THE SEQUENTIAL MODEL
classifier = Sequential()

'''
ADDING THE CONVOLUTIONAL LAYER WITH:
1)NUMBER OF FILTERS TO BE USED FOR CONVOLUTION - 32
2)FILTER SIZE - (3,3)
3)INPUT SHAPE OF THE IMAGES(NUMBER OF ROWS, NUMBER OF COLUMNS, NUMBER OF CHANNELS)
4)AN ACTIVATION FUNCTION
'''
classifier.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
# ADDING A POOLING LAYER MATRIX OF SIZE 2X2
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# DROPPING OUT RANDOM NODES TO AVOID OVERFITTING
classifier.add(Dropout(0.25))

'''
ADDING ONE MORE CONVOLUTIONAL LAYER, BUT THIS TIME THE INPUT OF THIS LAYER
WILL BE THE OUTPUT OF THE PREVIOUS LAYER
'''
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
# ADDING A POOLING LAYER MATRIX OF SIZE 2X2
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# DROPPING SOME MORE RANDOM NODES
classifier.add(Dropout(0.25))

# ADDING ONE MORE CONVOLUTIONAL LAYER WITH 128 FILTERS
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
# ADDING ANOTHER POOLING LAYER
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# DROPPING SOME MORE RANDOM NODES
classifier.add(Dropout(0.25))

# FLATTENING THE OUTPUT OF THE PREVIOUS LAYER TO 1D
classifier.add(Flatten())
# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH INPUT BEING THE FLATTEND ARRAY
classifier.add(Dense(units = 128, activation = 'relu'))
# DROPPING SOME MORE RANDOM NODES
classifier.add(Dropout(0.25))

'''
ADDING A FULLY CONNECTED DENSE NEURAL NETWORK, BUT THIS TIME THE NETWORK WILL ONLY
CONSIST OF NODES EQUIVALENT TO NUMBER OF CATEGORIES(LABELS). NOTICE THE ACTIVATION
FUNCTION IS ALSO BEEN CHANGED TO A SIGMOID INSTEAD OF A "Rectified Linear Unit"(ReLU)
'''
classifier.add(Dense(units = 10, activation = 'softmax'))

# COMPILING THE FINAL CREATED MODEL WITH AN OPTIMIZER, LOSS FUNCTION AND EVALUATION METRICS
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAINS THE MODEL ON TRAINING DATA BATCH-BY-BATCH
classifier.fit(X_train, y_train, batch_size=32, epochs=10, verbose=1, validation_data = (X_test, y_test))

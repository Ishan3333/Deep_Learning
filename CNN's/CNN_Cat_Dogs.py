# IMPORTING ALL THE NECESSARY LIBRARIES
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping

import matplotlib.pyplot as plt

# DEFINING A  VALIDATION 
VALIDATION_PATIENCE = 20

# INITIATING THE SEQUENTIAL MODEL
classifier = Sequential()

'''
ADDING THE CONVOLUTIONAL LAYER WITH:
1)NUMBER OF FILTERS TO BE USED FOR CONVOLUTION
2)FILTER SIZE
3)INPUT SHAPE OF THE IMAGES(NUMBER OF ROWS, NUMBER OF COLUMNS, NUMBER OF CHANNELS)
4)AN ACTIVATION FUNCTION
'''
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
# ADDING A POOLING LAYER MATRIX OF SIZE 2X2
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# DROPPING OUT RANDOM NODES TO AVOID OVERFITTING
classifier.add(Dropout(0.25))

'''
ADDING ONE MORE CONVOLUTIONAL LAYER, BUT THIS TIME THE INPUT OF THIS LAYER
WILL BE THE OUTPUT OF THE PREVIOUS LAYER
'''
classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
# ADDING ANOTHER POOLING LAYER
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# DROPPING SOME MORE RANDOM NODES
classifier.add(Dropout(0.25))

# ADDING ONE MORE CONVOLUTIONAL LAYER WITH 64 FILTERS
classifier.add(Conv2D(64, (3, 3), activation = 'relu'))
# ADDING ANOTHER POOLING LAYER
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# # DROPPING SOME MORE RANDOM NODES
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
# DROPPING OUT RANDOM NODES TO AVOID OVERFITTING
classifier.add(Dropout(0.25))

'''
ADDING A FULLY CONNECTED DENSE NEURAL NETWORK, BUT THIS TIME THE NETWORK WILL ONLY
CONSIST OF NODES EQUIVALENT TO NUMBER OF CATEGORIES(LABELS). NOTICE THE ACTIVATION
FUNCTION IS ALSO BEEN CHANGED TO A SIGMOID INSTEAD OF A "Rectified Linear Unit"(ReLU)
'''
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# COMPILING THE FINAL CREATED MODEL WITH AN OPTIMIZER, LOSS FUNCTION AND EVALUATION METRICS
classifier.compile(
	optimizer = 'adam',
	loss = 'binary_crossentropy',
	metrics = ['accuracy']
	)

'''
IMPLEMENTING A CONVERGENCE MONITOR SO THE THE MODEL AUTOMATICALLY STOPS THE ITERATION
AS SOON AS THE CONVERGENCE IS ACHIEVED
'''
stopper = EarlyStopping(monitor='val_loss', patience=VALIDATION_PATIENCE)

'''
CREATING AN "ImageDataGenerator" INSTANCE THAT WILL RANDOMLY-

1)ZOOM
2)RESCALE
3)FLIP(HORIZONTALLY & VERTICALLY)

-THE DATA
'''
train_datagen = ImageDataGenerator(
	rescale=1./255,
	shear_range=0.2,
	zoom_range=0.2,
	horizontal_flip=True,
	vertical_flip = True
	)

# CREATING AN "ImageDataGenerator" INSTANCE THAT WILL RESCALE THE DATA
test_datagen = ImageDataGenerator(rescale=1./255)

# CREATING A TRAINING DATASET
training_set = train_datagen.flow_from_directory(
	'../dataset/training_set',
	target_size=(64, 64),
	batch_size=32,
	class_mode='binary'
	)

# CREATING A TESTING DATASET
test_set = test_datagen.flow_from_directory(
	'../dataset/test_set',
	target_size=(64, 64),
	batch_size=32,
	class_mode='binary'
	)

# TRAINS THE MODEL ON DATA GENERATED BATCH-BY-BATCH
classifier.fit_generator(
	training_set,
	steps_per_epoch=8000,
	callbacks = [stopper],
	epochs=25,
	validation_data=test_set,
	validation_steps=2000,
	use_multiprocessing = True
	)

# PLOTTING OUT THE GRAPH OF VARIOUS LOSSES AND MODEL ACCURACY TO THE NUMBER OF EPOCHS
plt.figure()
plt.plot(classifier.history['loss'], label="train_loss")
plt.plot(classifier.history['val_loss'], label="val_loss")
plt.plot(classifier.history["acc"], label="train_acc")
plt.plot(classifier.history["val_acc"], label="val_acc")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()
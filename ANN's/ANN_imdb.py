-# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer

from keras.datasets import imdb

# DEFINING THE NUMBER OF FEATURES WE WANT
num_features = 10000

# LOADING THE DATA AND SPLITTING IT INTO TRAINING AND TESTING SET
(train_x, train_y), (test_x, test_y) = imdb.load_data(num_words = num_features)

# INITIALIZING A  WORD TOKENIZER
tokenizer = Tokenizer(num_words = num_features)

# CONVERTING THE TRAINING FEATURE SEQUENCE INTO A NUMPY MATRIX ARRAY
train_x = tokenizer.sequences_to_matrix(train_x, mode = 'binary')

# CONVERTING THE TESTING FEATURE SEQUENCE INTO A NUMPY MATRIX ARRAY
test_x = tokenizer.sequences_to_matrix(test_x, mode = 'binary')

# INITIALIZING THE MODEL
model = Sequential()

# ADDING A FULLY CONNECTED DENSE NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu', input_shape = (num_features,)))
# DROPPING OUT RANDOM NODES TO AVOID OVERFITTING
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NETWORK WITH NUMBER OF NODES EQUALS NUMBER OF OUTPUT
model.add(Dense(units = 1, activation = 'sigmoid'))

# COMPILING THE FINAL CREATED MODEL WITH AN OPTIMIZER, LOSS FUNCTION AND EVALUATION METRICS
model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# TRAINS THE MODEL ON TRAINING DATA BATCH-BY-BATCH
model.fit(train_x, train_y,
	batch_size = 100,
	epochs = 50,
	validation_data = (test_x, test_y),
	verbose = 1)
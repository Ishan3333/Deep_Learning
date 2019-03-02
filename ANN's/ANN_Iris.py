# IMPORT ALL THE NECESSARY LIBRARIES
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers  import Dense, Dropout
from sklearn.utils import shuffle

# READING IN THE DATASET
df = pd.read_csv('iris.csv')

# CONVERTING CATEGORICAL DATA TO NUMERICAL DATA
df['species'] = df['species'].astype('category').cat.codes

# RANDOM SHUFFLING
df = shuffle(df)

# SPLITTING THE DATA MANUALLY INTO TRAINING AND TESTING FEATURES & LABELS
x_train = df.iloc[0:101,0:4].values
y_train = df.iloc[0:101,-1:].values

x_test = df.iloc[101:,0:4].values
y_test = df.iloc[101:,-1:].values

# CONVERTING THE LABEL COLUMN TO KERAS USABLE CATEGORICAL FORMATS
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# INITIALIZING THE CLASSIFICATION MODEL
classifier = Sequential()

# ADDING THE FIRST FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES 
classifier.add(Dense(units = 32, activation = 'relu', input_dim = 4))

# ADDING ANOTHER FULLY CONNECTED DENSE NETWORK
classifier.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES FROM THE PREVIOUS NETWORK IN ORDER TO AVOID OVERFITTING
classifier.add(Dropout(0.25))

# ADDING ANOTHER FULLY CONNECTED DENSE NETWORK
classifier.add(Dense(units = 64, activation = 'relu'))
# DROPPING OUT RANDOM NODES TO AVOID OVERFITTING
classifier.add(Dropout(0.25))

# ADDING ANOTHER FULLY CONNECTED DENSE NETWORK
classifier.add(Dense(units = 64, activation = 'relu'))
# DROPPING OUT RANDOM NODES TO AVOID OVERFITTING
classifier.add(Dropout(0.25))

# ADDING A FINAL DENSE NETWORK WITH NUMBER OF NODES EQUAL TO NUMBER OF CLASSIFICATION
classifier.add(Dense(units = 3, activation = 'sigmoid'))

# COMPILING THE FINAL CREATED MODEL WITH AN OPTIMIZER, LOSS FUNCTION AND EVALUATION METRICS
classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAINS THE MODEL ON TRAINING DATA BATCH-BY-BATCH 
classifier.fit(x_train, y_train, batch_size=10, epochs=150)

# EVALUATION OF THE CREATED MODEL ON TESTING DATASET
score = classifier.evaluate(x_test, y_test, verbose=0)
print ('test_loss:', score[0])
print ('test_acc:', score[1])
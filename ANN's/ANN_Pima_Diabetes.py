# IMPORTING ALL THE NECESSARY LIBRARIES
import pandas as pd

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers  import Dense, Dropout

# READING IN THE DASET
df = pd.read_csv('pima-indians-diabetes.csv')

# SPLITTING THE DATASET INTO INPUT FEATURES AND OUTPUT LABELS
X = df.iloc[:,0:8].values
Y = df.iloc[:,8].values 

# INITIALIZING THE CLASSIFICATION MODEL
classifier = Sequential()

# ADDING A FULLY CONNECTED DENSE NETWORK OF 32 NODES
classifier.add(Dense(units = 32, activation = 'relu', input_dim = 8))
# DROPPING OUT RANDOM NODES FROM THE NETWORK TO AVOID OVERFITTING
classifier.add(Dropout(0.10))

# ADDING A FULLY CONNECTED DENSE NETWORK OF 64 NODES
classifier.add(Dense(units = 64, activation = 'relu'))
# DROPPING OUT RANDOM NODES FROM THE NETWORK TO AVOID OVERFITTING
classifier.add(Dropout(0.20))

# ADDING A FULLY CONNECTED DENSE NETWORK OF 64 NODES
classifier.add(Dense(units = 64, activation = 'relu'))
# DROPPING OUT RANDOM NODES FROM THE NETWORK TO AVOID OVERFITTING
classifier.add(Dropout(0.10))

# ADDING A FINAL DENSE NETWORK WITH NUMBER OF NODES EQUAL TO NUMBER OF CLASSIFICATION 
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# COMPILING THE FINAL CREATED MODEL WITH AN OPTIMIZER, LOSS FUNCTION AND EVALUATION METRICS
classifier.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# TRAINS THE MODEL ON TRAINING DATA BATCH-BY-BATCH 
classifier.fit(X, Y, batch_size=30, epochs=200, verbose=1)

# EVALUATION OF THE CREATED MODEL ON TESTING DATASET
score = classifier.evaluate(X, Y, verbose=0)
print ('test_loss:', score[0])
print ('test_acc:', score[1])

# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# SETTING UP A RANDOM SEED FOR THE SAKE OF PREDICTABILITY
np.random.seed(0)

# CREATING A DATASET SPECIFICALLY FOR REGRESSION ANALYSIS
x, y = make_regression(n_samples = 10000,
	n_features = 3,
	noise = 0.2,
	random_state = 1)

# SPLITTING THE DATASET INTO TRAINING AND TESTING DATASET
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.25, random_state = 0)

# INITIALIZING THE SCALER FROM SKLEARN
train_x_sc = StandardScaler()
train_y_sc = StandardScaler()
test_x_sc = StandardScaler()
test_y_sc = StandardScaler()

# FITTING AND TRANSFORMING THE RESPECTIVE DATA FOR SCALING
train_x_sc.fit(train_x)
train_y_sc.fit(train_y.reshape(7500, 1))
test_x_sc.fit(test_x)
test_y_sc.fit(test_y.reshape(2500, 1))

train_x = train_x_sc.transform(train_x)
train_y = train_y_sc.transform(train_y.reshape(7500, 1))
test_x = test_x_sc.transform(test_x)
test_y = test_y_sc.transform(test_y.reshape(2500, 1))

# INITIATING THE MODEL
model = Sequential()

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, input_dim = 3, activation = 'relu'))
# DROPPING RANDOM NODES TO AVOID OVERFITTING OF THE MODEL
model.add(Dropout(0.20))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING RANDOM NODES
model.add(Dropout(0.20))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 64 NODES
model.add(Dense(units = 64, activation = 'relu'))
# DROPPING RANDOM NODES
model.add(Dropout(0.20))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 64 NODES
model.add(Dense(units = 64, activation = 'relu'))
# DROPPING RANDOM NODES
model.add(Dropout(0.20))

# ADDING A FULLY CONNECTED DENSE NETWORK WITH NUMBER OF NODES EQUALS NUMBER OF OUTPUT
model.add(Dense(units = 1, activation = 'linear'))

# COMPILING THE FINAL CREATED MODEL WITH AN OPTIMIZER, LOSS FUNCTION AND EVALUATION METRICS
model.compile(loss='mse', optimizer='rmsprop', 	metrics = ['mse'])

# TRAINS THE MODEL ON TRAINING DATA BATCH-BY-BATCH
model.fit(train_x, train_y, epochs = 200, batch_size = 100, validation_data = (test_x, test_y), verbose = 1)
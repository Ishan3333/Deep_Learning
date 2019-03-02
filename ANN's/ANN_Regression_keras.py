# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression

# CREATING A DUMMY DATASET FOR PERFORMING REGRESSION
X, Y = make_regression(n_samples=100, n_features=4, noise=0.1, random_state=1)

# SCALING THE DATASET
scaled_X = StandardScaler()
scaled_Y = StandardScaler()

scaled_X.fit(X)
scaled_Y.fit(Y.reshape(100, 1))

X = scaled_X.transform(X)
Y = scaled_Y.transform(Y.reshape(100, 1))

# INITIALIZING THE MODEL
model = Sequential()

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, input_dim = 4, activation = 'relu'))
# DROPPING OUT RANDOM NODES TOAVOID OVERFITTING
model.add(Dropout(0.10))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES TOAVOID OVERFITTING
model.add(Dropout(0.20))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES TO AVOID OVERFITTING
model.add(Dropout(0.20))

# ADDING A FINAL NETWORK WITH NUMBER OF NODES EQUAL TO NUMBER OF OUTPUT
model.add(Dense(units = 1, activation = 'linear'))

# COMPILING THE FINAL CREATED MODEL WITH AN OPTIMIZER, LOSS FUNCTION AND EVALUATION METRICS
model.compile(loss='mean_squared_error', optimizer='adam', 	metrics = ['mae'])

# TRAINS THE MODEL ON TRAINING DATA BATCH-BY-BATCH 
model.fit(X, Y, epochs = 100, verbose = 0)

# GENERATING NEW DATA FOR TESTING PURPOSE
Xnew, a = make_regression(n_samples=3, n_features=4, noise=0.1, random_state=1)
# SCALING THE TEST DATA
Xnew = scaled_X.transform(Xnew)

# MAKE A PREDICTION
ynew = model.predict(Xnew)
# SHOW THE INPUTS AND PREDICTED OUTPUTS
for i in range(len(Xnew)):
	print("X =",Xnew[i], ",", "Predicted =",ynew[i])
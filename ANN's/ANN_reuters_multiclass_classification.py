# IMPORTING ALL THE NECESSARY LIBRARIES
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils.np_utils import to_categorical

from keras.datasets import reuters

# DEFINING THE NUMBER OF FEATURES WE WANT
num_features = 5000

# LOADING IN THE DATASET AND SPLITTING IT INTO TRAINING AND TESTING FEATURES & LABELS
(train_x, train_y), (test_x, test_y) = reuters.load_data(num_words = num_features)

# INITIALIZING A  WORD TOKENIZER
tokenizer = Tokenizer(num_words = num_features)

# CONVERTING THE TRAINING FEATURE SEQUENCE INTO A NUMPY MATRIX ARRAY
train_x = tokenizer.sequences_to_matrix(train_x, mode = 'binary')

# CONVERTING THE TESTING FEATURE SEQUENCE INTO A NUMPY MATRIX ARRAY
test_x = tokenizer.sequences_to_matrix(test_x, mode = 'binary')

# CONVERTING THE LABELS TO KERAS USABLE CATEGORICAL FORMAT
train_y = to_categorical(train_y)
test_y = to_categorical(test_y)

# INITIALIZING THE MODEL
model = Sequential()

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, input_shape = (num_features, ),activation = 'relu'))
# DROPPING OUT RANDOM NODES TO AVOID OVERFITTING
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH 32 NODES
model.add(Dense(units = 32, activation = 'relu'))
# DROPPING OUT RANDOM NODES
model.add(Dropout(0.15))

# ADDING A FULLY CONNECTED DENSE NEURAL NETWORK WITH NUMBER OF NODES EQUALS TO NUBER OF OUTPUT
model.add(Dense(units = 46, activation = 'softmax'))

# COMPILING THE FINAL CREATED MODEL WITH AN OPTIMIZER, LOSS FUNCTION AND EVALUATION METRICS
model.compile(loss = 'categorical_crossentropy', optimizer = 'rmsprop', metrics = ['accuracy'])

# TRAINS THE MODEL ON TRAINING DATA BATCH-BY-BATCH 
model.fit(train_x, train_y, batch_size = 100, epochs = 50, validation_data = (test_x, test_y), verbose = 1)
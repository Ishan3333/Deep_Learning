import numpy as np
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

number_of_features = 1000

(train_data, train_target), (test_data, test_target) = imdb.load_data(num_words=number_of_features)

train_features = sequence.pad_sequences(train_data, maxlen=400)
test_features = sequence.pad_sequences(test_data, maxlen=400)

print(len(train_data[0]))
print(len(train_features[0]))

model = Sequential()

model.add(Embedding(input_dim = number_of_features, output_dim = 128))

model.add(LSTM(units = 128))

model.add(Dense(units = 1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
                optimizer='Adam',
                metrics=['accuracy'])

history = model.fit(train_features,
                      train_target,
                      epochs=3,
                      verbose=1,
                      batch_size=1000,
                      validation_data=(test_features, test_target))
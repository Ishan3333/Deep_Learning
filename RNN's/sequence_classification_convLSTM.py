import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.embeddings import Embedding
from keras.layers import Conv1D, MaxPooling1D
from keras.preprocessing import sequence

np.random.seed(7)

top_words = 5000

(X_train, y_train),(X_test, y_test) = imdb.load_data(num_words = top_words)

max_review_len = 500

X_train = sequence.pad_sequences(X_train, maxlen = max_review_len)
X_test = sequence.pad_sequences(X_test, maxlen = max_review_len)

model = Sequential()

model.add(Embedding(top_words, 12, input_length = max_review_len))

model.add(LSTM(100, dropout = 0.2, recurrent_dropout = 0.2))

model.add(Conv1D(64, (3, 3), activation = 'relu', input_shape = (32, 32, 3)))

model.add(MaxPooling1D(pool_size = (2, 2)))

model.add(Dense(1, activation = 'sigmoid'))


model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=5, batch_size=64)

scores = model.evaluate(X_test, y_test, verbose=1)
print("Accuracy: %.2f%%" % (scores[1]*100))
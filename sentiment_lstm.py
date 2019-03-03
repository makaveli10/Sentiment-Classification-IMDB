from __future__ import print_function

from keras.preprocessing import sequence
from keras.layers import Dense, Embedding, LSTM
from keras.datasets import imdb
from keras.models import Sequential

max_feat = 20000
max_length = 80
batch_size = 32

print("Loading Dataset ...")

(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_feat)

print(len(x_train), 'train sequences')
print(len(x_test),'test sequences')

x_train = sequence.pad_sequences(x_train, maxlen=max_length)
x_test = sequence.pad_sequences(x_test, maxlen=max_length)

print('x_train shape', x_train.shape)
print('x_test shape', x_test.shape)

print("Bulding model...")

model = Sequential()
model.add(Embedding(max_feat, 128))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Training in progress...')

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=15,
                    validation_data=(x_test, y_test))

score, acc = model.evaluate(x_test, y_test, batch_size=batch_size)
print('Score =', score)
print('Accuracy =', acc)
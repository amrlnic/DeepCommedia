import sys
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import tensorflow as tf

import numpy as np
import os
import time
import pandas as pd
import pickle 

load = pd.read_csv("tentativo.csv", sep = ',')

YY = load.iloc[0:100,0:1].values

XX = load.iloc[0:100,1:].values



a = XX.tolist()
flatten_X = [item for sublist in a for item in sublist]


b = YY.tolist()
flatten_Y = [item for sublist in b for item in sublist]


# create mapping of unique chars to integers
chars_X = sorted(list(set(flatten_X)))
char_to_int_X = dict((c, i) for i, c in enumerate(chars_X))

chars_Y = sorted(list(set(flatten_Y)))
char_to_int_Y = dict((c, i) for i, c in enumerate(chars_Y))

# summarize the loaded data
n_chars = len(XX)
n_vocab = len(chars_X)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)


flatten_X_numbers =np.array([ char_to_int_X[el] for el in flatten_X])
X = numpy.reshape(flatten_X_numbers, (100, 100, 1))


flatten_Y_numbers =np.array([ char_to_int_Y[el] for el in flatten_Y])
Y = numpy.reshape(flatten_Y_numbers, (100, 1))


#conversion str --> int








# one hot encode the output variable
y = np_utils.to_categorical(Y)



# define the LSTM model
model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(256))
model.add(Dropout(0.2))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam')



# define the checkpoint
filepath="weights-improvement.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]



# fit the model
model.fit(X, y, epochs=5, batch_size=128, callbacks=callbacks_list)



# load the network weights
filename = "weights-improvement.hdf5"
model.load_weights(filename)
model.compile(loss='categorical_crossentropy', optimizer='adam')


int_to_char_X = dict((i, c) for i, c in enumerate(chars_X))
int_to_char_Y = dict((i, c) for i, c in enumerate(chars_Y))


# pick a random seed
start = numpy.random.randint(0, len(X)-1)
pattern = X[start].tolist()
pattern = [el[0] for el in pattern]

print ("Seed:")
print ("\"", ''.join([int_to_char_X[value] for value in pattern]), "\"")



# generate characters
for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	#x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char_Y[index]
	seq_in = [int_to_char_X[value] for value in pattern]
	if result==' ':
		sys.stdout.write('a')
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
print ("\nDone.")






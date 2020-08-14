# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 16:38:49 2020

@author: utente
"""
import sys
import keras
import numpy

# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
from matplotlib import pyplot

#####################################################################
                            # INPUT #
#####################################################################


# load ascii text and covert to lowercase
filename = "Commedia_cleaned.txt"
raw_text = open(filename, 'r', encoding='utf-8').read()
raw_text = raw_text.lower()


# create mapping of unique chars to integers, and a reverse mapping
chars = sorted(list(set(raw_text)))
char_to_int = dict((c, i) for i, c in enumerate(chars))
int_to_char = dict((i, c) for i, c in enumerate(chars))


# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(chars)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)


# prepare the dataset of input to output pairs encoded as integers
seq_length = 100
dataX = []
dataY = []
for i in range(0, n_chars - seq_length, 1):
	seq_in = raw_text[i:i + seq_length]
	seq_out = raw_text[i + seq_length]
	dataX.append([char_to_int[char] for char in seq_in])
	dataY.append(char_to_int[seq_out])
n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)


# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))


# normalize
X = X / float(n_vocab)


# one hot encode the output variable
y = np_utils.to_categorical(dataY)

#####################################################################
                            # MODEL #
#####################################################################
#################################################################################
#################################################################################
#################################################################################

# define the standalone discriminator model
def define_discriminator(in_shape=(1000)):
	model = Sequential()
	model.add(Dense(512, activation='relu'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Dense(512, activation='relu'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model
def define_generator():
    model = Sequential()
    model.add(LSTM(512, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax'))
    
    filename = "weights-improvement-10-1.8971.hdf5"
    model.load_weights(filename)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

#################################################################################
#################################################################################
#################################################################################

# select real samples
def generate_real_samples(raw_text, n_samples):
    N = []
    for i in range(0, n_samples):
        el= numpy.random.randint(0, len(raw_text))
        numpy.append(N,raw_text[el:el+1000])
        # generate 'real' class labels (1)
        y = ones((n_samples, 1))
        return X, y

#################################################################################
#################################################################################
#################################################################################

def gen_RNN_output(dataX, model ):
    start = numpy.random.randint(0, len(dataX)-1)
    pattern = dataX[start]
    R = ''
    # generate characters
    for i in range(1000):

    	x = numpy.reshape(pattern, (1, len(pattern), 1))
    	x = x / float(n_vocab)
    	prediction = model.predict(x, verbose=0)
    	index = numpy.argmax(prediction)
    	result = int_to_char[index]
    	R = R + result
    	sys.stdout.write(result)
    	pattern.append(index)
    	pattern = pattern[1:len(pattern)]
    return R

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(dataX, g_model,  n_samples):
    D=[]
    for i in range(0, n_samples):
        D.append(gen_RNN_output(dataX, g_model))

    # create 'fake' class labels (0)
    y = zeros((n_samples, 1))
    return D, y

#################################################################################
#################################################################################
#################################################################################
"""
# create and save a plot of generated images (reversed grayscale)
def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()
"""
#################################################################################
#################################################################################
#################################################################################

# evaluate the discriminator, plot generated images, save generator model
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
###################################	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

#################################################################################
#################################################################################
#################################################################################

# train the generator and discriminator
def train(dataX, g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(len(dataset) / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(dataX, g_model,  half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' % (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)


#################################################################################
#################################################################################
#################################################################################

# size of the latent space
latent_dim = 100

# create the discriminator
d_model = define_discriminator()

# create the generator
g_model = define_generator()

# create the gan
gan_model = define_gan(g_model, d_model)


# train model
train(dataX, g_model, d_model, gan_model, raw_text, latent_dim)














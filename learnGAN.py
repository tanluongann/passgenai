from __future__ import print_function, division

import keras
import numpy as np
import tensorflow as tf

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt

import sys
import os

import numpy as np

class GAN():

    def __init__(self, folder='data/leaked10000', min_password_size=8):

        self.folder = folder
        self.data = []
        self.min_password_size = min_password_size

        self.word_rows = 1 # A word is just 2 dimensions
        self.word_cols = 100 # Passwords with more than 100 characters are quite rare (Warning: Unicode chars can count as 4!!!)
        self.channels = 1 # Coming from image processing, I guess, not relevant here? (Could be language?)
        self.word_shape = (self.word_rows, self.word_cols, self.channels)

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated words
        z = Input(shape=(100,))
        word = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.discriminator(word)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        # 9527

    def load_data(self):
        """
        Load the learning data from the data.txt file.
        """
        full_filename = os.path.join(self.folder, 'data.txt')
        with open(full_filename, 'r') as f:
            d = f.read()
            res = []
            for e in d.split('\n'):
                e = e.strip()
                if len(e) > self.min_password_size:
                    w = np.zeros((1, self.word_cols), dtype=int)
                    e = e.encode('utf-8')                    
                    for i in range(len(e)):
                        w[0][i] = e[i]
                    res.append(w)
            self.data = np.zeros((len(res), 1, self.word_cols))
            for i in range(len(res)):
                self.data[i] = res[i]

    def build_generator(self):
        noise_shape = (100,)
        model = Sequential()
        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.word_shape), activation='tanh'))
        model.add(Reshape(self.word_shape))
        model.summary()

        noise = Input(shape=noise_shape)
        word = model(noise)
        return Model(noise, word)

    def build_discriminator(self):
        word_shape = (self.word_rows, self.word_cols, self.channels)
        model = Sequential()
        model.add(Flatten(input_shape=word_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        word = Input(shape=word_shape)
        validity = model(word)
        return Model(word, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        self.load_data()
        # (X_train, _), (_, _) = self.data
        X_train = self.data

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            words = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))

            # Generate a half batch of new images
            gen_words = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(words, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_words, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, 100))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)

            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_words(epoch)

    def save_words(self, epoch):
        pass
        # r, c = 5, 5
        # noise = np.random.normal(0, 1, (r * c, 100))
        # gen_words = self.generator.predict(noise)

        # # Rescale images 0 - 1
        # gen_words = 0.5 * gen_words + 0.5

        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_words[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("gan/images/mnist_%d.png" % epoch)
        # plt.close()


if __name__ == '__main__':
    gan = GAN()
    gan.load_data()

    gan.train(epochs=30000, batch_size=32, save_interval=200)

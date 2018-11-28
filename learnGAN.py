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
import string

import numpy as np

class GAN():

    ALLOWED_CHARS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%&*+=-_|:?'

    def __init__(self, folder='data/leaked10000', min_password_size=8):

        self.folder = folder
        self.data = []
        self.min_password_size = min_password_size

        self.generated = []

        self.word_rows = 1 # A word is just 2 dimensions
        self.word_cols = 32 # Passwords with more than 32 characters are quite rare (Warning: Unicode chars can count as 4!!!)
        self.channels = 1 # Coming from image processing, I guess, not relevant here? (Could be language?)
        self.word_shape = (self.word_rows, self.word_cols, self.channels)

        self.noise_width = 16

        optimizer = Adam(0.0002, 0.5)


        self.char_range = len(self.ALLOWED_CHARS) + 1

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        # Build and compile the generator
        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        # The generator takes noise as input and generated words
        z = Input(shape=(self.noise_width,))
        word = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The valid takes generated words as input and determines validity
        valid = self.discriminator(word)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates words => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.train_infos = []

        # 9527

    def string_to_intarray(self, str_val):
        # Initialize the list of int representing the word
        w = np.zeros((1, self.word_cols), dtype=int)
        for i in range(len(str_val)):
            # Get the letter at the index
            c = str_val[i]
            c = self.ALLOWED_CHARS.find(c)
            # Normalize c, so the value starts at 0
            c += 1
            w[0][i] = c
        return w
    def intarray_to_string(self, int_arr):
        res = []
        for u in int_arr[0]:
            u = u - 1
            if u >= 0:
                try:
                    res.append(self.ALLOWED_CHARS[u])
                except:
                    print('u out of bound: %s' % u)
                    res.append(' ')
            else:
                res.append(' ')
        return ''.join(res).strip()

    def intarray_to_normalized(self, int_arr):
        shift = self.char_range / 2
        res = (int_arr.astype(np.float32) - shift) / shift
        return res
    def normalized_to_intarray(self, nmz_arr):
        shift = self.char_range / 2
        res = (nmz_arr * shift + shift).astype(int)
        return res

    def load_data(self):
        """
        Load the learning data from the data.txt file.
        """
        full_filename = os.path.join(self.folder, 'data.txt')
        with open(full_filename, 'r') as f:
            d = f.read()
            res = []
            for e in d.split('\n'):
                e = e[:32]
                e = e.strip()
                if len(e) > self.min_password_size:
                    w = self.string_to_intarray(e)
                    res.append(w)

            # Now we know the size of the list of word, we initalize the data size with 0s
            self.data = np.zeros((len(res), 1, self.word_cols))
            # And we copy our data
            for i in range(len(res)):
                self.data[i] = res[i]

                # d = []
                # for e in res[i][0]:
                #     d.append('%2d' % e)
                # print(' '.join(d))

    def build_generator(self):
        noise_shape = (self.noise_width,)
        model = Sequential()

        model.add(Dense(32, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(32))
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
        model.add(Dense(128))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(16))
        model.add(LeakyReLU(alpha=0.2))

        model.add(Dense(1, activation='sigmoid'))
        # model.add(Dense(1, activation='sigmoid'))
        model.summary()

        word = Input(shape=word_shape)
        validity = model(word)
        return Model(word, validity)

    def train(self, epochs, batch_size=128, save_interval=50):

        # Load the dataset
        self.load_data()

        X_train = self.intarray_to_normalized(self.data)
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)

        self.train_infos = []
        self.generated = []

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of real words
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_words = X_train[idx]

            # Generate a half batch of new fake words
            noise = np.random.normal(0, 1, (half_batch, self.noise_width))
            fake_words = self.generator.predict(noise)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(real_words, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(fake_words, np.zeros((half_batch, 1)))

            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Generate a half batch of '1' as target
            valid_y = np.array([1] * batch_size)
            noise = np.random.normal(0, 1, (batch_size, self.noise_width))
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # ---------------------
            #  Data mgt.
            # ---------------------

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                # self.train_infos.append((epoch, d_loss[0], 100*d_loss[1], g_loss))
                print('')
                print('d_loss_real', d_loss_real)
                print('d_loss_fake', d_loss_fake)
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))
                self.create_words()

        # Once all completed
        full_filename = os.path.join(self.folder, 'stats.csv')
        with open(full_filename, 'w') as f:
            f.write('epoch, d_loss, acc, g_loss\n')
            for a in self.train_infos:
                f.write('%d, %f, %.2f%%, %f\n' % a)

        self.save_results()

    def create_words(self, sample_size=10):
        noise = np.random.normal(0, 1, (sample_size, self.noise_width))
        fake_word = self.generator.predict(noise)
        for t in range(sample_size):
            v = fake_word[t, :, :, 0]
            w = self.normalized_to_intarray(v)
            r = self.intarray_to_string(w)
            self.generated.append(r)
            print('New word:', r)


    def save_results(self):
        """Save the generated words in the data folder as list.txt.
        """
        full_filename = os.path.join(self.folder, 'list_gan.txt')
        with open(full_filename, 'w') as f:
            for e in self.generated:
                f.write(e + '\n')


        # # Rescale words 0 - 1
        # gen_words = 0.5 * gen_words + 0.5

        # for i in range(self.)
        # fig, axs = plt.subplots(r, c)
        # cnt = 0
        # for i in range(r):
        #     for j in range(c):
        #         axs[i,j].imshow(gen_words[cnt, :,:,0], cmap='gray')
        #         axs[i,j].axis('off')
        #         cnt += 1
        # fig.savefig("gan/words/mnist_%d.png" % epoch)
        # plt.close()


if __name__ == '__main__':

    gan = GAN()
    gan.load_data()
    gan.train(epochs=1000000, batch_size=1024, save_interval=100)

    # print(gan.generated)

from __future__ import print_function, division
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import numpy as np
from argparse import ArgumentParser
import os

class DCGAN():
    def __init__(self,
        dataset,
        saved_model='saved_model',
        output_images='images'):
        # Args
        self.dataset = dataset
        self.saved_model = saved_model
        self.output_images = output_images

        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = tf.keras.optimizers.Adam(0.0002, 0.5)

        # Build and compile the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator
        self.generator = self.build_generator()

        # The generator takes noise as input and generates imgs
        z = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = tf.keras.models.Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(tf.keras.layers.Reshape((7, 7, 128)))
        model.add(tf.keras.layers.UpSampling2D())
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, padding="same"))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.UpSampling2D())
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, padding="same"))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.Activation("relu"))
        model.add(tf.keras.layers.Conv2D(self.channels, kernel_size=3, padding="same"))
        model.add(tf.keras.layers.Activation("tanh"))

        model.summary()

        noise = tf.keras.layers.Input(shape=(self.latent_dim,))
        img = model(noise)

        return tf.keras.models.Model(noise, img)

    def build_discriminator(self):

        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Conv2D(32, kernel_size=3, strides=2, input_shape=self.img_shape, padding="same"))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(tf.keras.layers.ZeroPadding2D(padding=((0,1),(0,1))))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(tf.keras.layers.BatchNormalization(momentum=0.8))
        model.add(tf.keras.layers.LeakyReLU(alpha=0.2))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

        model.summary()

        img = tf.keras.layers.Input(shape=self.img_shape)
        validity = model(img)

        return tf.keras.models.Model(img, validity)

    def train(self, epochs, batch_size=128, save_interval=50):
        # Dataset
        gen = tf.keras.preprocessing.image.ImageDataGenerator(
            preprocessing_function=lambda x: x/127.5 - 1.
        )
        files = Path(self.dataset).rglob('*.jpg')
        files = map(os.path.abspath, files)
        df = pd.DataFrame({'filename': files})
        if not len(df): exit('No images found.')
        ds = gen.flow_from_dataframe(df,
            class_mode=None,
            batch_size=batch_size,
            target_size=(self.img_rows, self.img_cols))

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Get a random batch
            imgs = next(ds)

            # Sample noise and generate a batch of new images
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator (real classified as ones and generated as zeros)
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  Train Generator
            # ---------------------

            # Train the generator (wants discriminator to mistake images as real)
            g_loss = self.combined.train_on_batch(noise, valid)

            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % save_interval == 0:
                self.save_imgs(epoch)
                self.combined.save(f'{self.saved_model}/combined.h5')
                self.generator.save(f'{self.saved_model}/generator.h5')
                self.discriminator.save(f'{self.saved_model}/discriminator.h5')

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig(f'{self.output_images}/epoch_{epoch}.png')
        plt.close()


if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--dataset', type=str, required=True)
    arg_parser.add_argument('--output-images', type=str, required=True)
    arg_parser.add_argument('--saved-model', type=str, required=True)
    dcgan = DCGAN(**vars(arg_parser.parse_args()))
    dcgan.train(epochs=4000, batch_size=32, save_interval=50)
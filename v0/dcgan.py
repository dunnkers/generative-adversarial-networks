import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
import os
if os.path.dirname(__file__) != '':
    os.chdir(os.path.dirname(__file__))
import tensorflow as tf
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
print(tf.__version__)

dataset_folder = "C:\\Users\\thijs\\Documents\\Studie\\IIa\\Deep Learning\\deep_learning_2_dataset\\dataset\\"
results_dir = "results"
#latent_dims = 2048
latent_dims = 32
latent_dims_half = latent_dims//2
output_channels = 3
image_dims = (212, 212)
batch_size = 16
test_examples = (2,2)
test_input = np.random.normal(0, 1, (test_examples[0]*test_examples[1], latent_dims))

def make_encoder_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), input_shape=(image_dims[0], image_dims[1], output_channels, ), activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(latent_dims, activation='tanh'))
    model.add(layers.BatchNormalization(momentum=0.8))

    return model

def make_generator_model():
    model = tf.keras.Sequential()
    
    model.add(layers.Dense(7*7*128, input_shape=(latent_dims,), activation='tanh'))
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Reshape((7, 7, 128)))
    model.add(layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(256, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(512, (3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(layers.Conv2DTranspose(output_channels, (3, 3), strides=(2, 2), activation='sigmoid', padding='same'))
    #model.add(layers.Conv2D(output_channels, (5, 5), activation='sigmoid', padding='valid'))
    model.add(layers.Cropping2D(cropping=((6,6), (6,6))))
    print(model.output_shape)
    assert model.output_shape == (None, image_dims[0], image_dims[1], output_channels)

    return model


'''
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*128, use_bias=False, input_shape=(latent_dims,), activation='tanh'))
    model.add(layers.BatchNormalization(momentum=0.8))

    model.add(layers.Reshape((7, 7, 128)))
    assert model.output_shape == (None, 7, 7, 128) # Note: None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 7, 7, 128)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 14, 14, 128)

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 128)
    model.add(layers.BatchNormalization(momentum=0.8))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.LeakyReLU())
    assert model.output_shape == (None, 56, 56, 128)

    model.add(layers.Conv2DTranspose(output_channels, (5, 5), strides=(2, 2), use_bias=False, activation='sigmoid'))
    model.add(layers.Cropping2D(cropping=((7, 7), (7, 7))))
    print(model.output_shape)
    assert model.output_shape == (None, image_dims[0], image_dims[1], output_channels)

    return model
'''

'''
def make_generator_model():
    
    first_layer = (7, 7, 256)

    model = tf.keras.Sequential()

    model.add(layers.Dense(first_layer[0] * first_layer[1] * first_layer[2], activation="relu", input_dim=latent_dims))
    model.add(layers.Reshape(first_layer))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size=3)) # padding="same"
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size=3)) # padding="same"
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size=3)) # padding="same"
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size=3)) # padding="same"
    #model.add(UpSampling2D())
    #model.add(Conv2D(128, kernel_size=3)) # padding="same"
    model.add(layers.UpSampling2D())
    model.add(layers.Conv2D(256, kernel_size=5)) # padding="same"
    model.add(layers.BatchNormalization(momentum=0.8))
    model.add(layers.Activation("relu"))
    model.add(layers.Conv2D(output_channels, kernel_size=5)) # padding="same"
    model.add(layers.Activation("sigmoid"))
    model.add(layers.Cropping2D(cropping=((3, 3), (3, 3))))

    assert model.output_shape == (None, image_dims[0], image_dims[1], output_channels)
    return model

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[image_dims[0], image_dims[1], output_channels]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model
'''

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[image_dims[0], image_dims[1], output_channels]))
    model.add(layers.LeakyReLU())
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    #model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2D(512, (3, 3), strides=(2, 2), activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same', activation='relu'))
    model.add(layers.Conv2D(512, (3, 3), activation='relu'))
    #model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

def image_batch_generator():
    img_list = [os.path.join(dataset_folder, img_name) for img_name in os.listdir(dataset_folder)]
    n_samples = len(img_list)
    indices = np.arange(0, n_samples)
    indices = np.concatenate([indices, np.random.randint(0, n_samples, (batch_size - n_samples) % batch_size)])
    n_batches = indices.shape[0] // batch_size
    indices = np.reshape(indices, (n_batches, batch_size))

    imgs = np.zeros( (batch_size, image_dims[0], image_dims[1], output_channels) )
    for n in range(n_batches):
        idx = indices[n,:]
        for k, i in enumerate(idx):
            imgs[k,:,:,:] = cv2.imread(img_list[i])/ 255.
        batches_left = n_batches - 1 - n
        yield imgs, batches_left

def generate_and_save_images(model, epoch, real_latent_vec):
  global test_examples, test_input
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  test_input = np.random.normal(0, 1, (test_examples[0]*test_examples[1], latent_dims))
  test_input[0,:] = real_latent_vec[0]
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(10,10))

  for i in range(predictions.shape[0]):
      plt.subplot(test_examples[0], test_examples[1], i+1)
      plt.imshow( cv2.cvtColor(predictions[i].numpy(), cv2.COLOR_BGR2RGB) )
      plt.axis('off')

  plt.savefig( os.path.join(results_dir, '{:04d}.png'.format(epoch+1)) )
  plt.close()


if __name__ == '__main__':
    os.makedirs(results_dir, exist_ok=True)
    encoder = make_encoder_model()
    print("\nENCODER:")
    encoder.summary()
    generator = make_generator_model()
    print("\nGENERATOR:")
    generator.summary()
    discriminator = make_discriminator_model()
    print("\nDISCRIMINATOR:")
    discriminator.summary()
    img = layers.Input(shape=(image_dims[0], image_dims[1], output_channels))
    autoencoder = tf.keras.Model(img, generator(encoder(img)))
    autoencoder.compile(optimizer=tf.keras.optimizers.Nadam(1e-4), loss='mse')
    print("\nAUTOENCODER:")
    autoencoder.summary()

    generator_optimizer = tf.keras.optimizers.Nadam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Nadam(1e-4)
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def generator_loss(fake_output):
        return cross_entropy(tf.ones_like(fake_output), fake_output)
    
    def discriminator_loss(real_output, fake_output):
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss
    
    def autoencoder_loss(input_image, reconstructed_image):
        return tf.keras.losses.MSE(input_image, reconstructed_image)

    @tf.function
    def train_step(images):
        noise = tf.random.normal([batch_size, latent_dims])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as autoenc_tape:

            generated_images = generator(noise, training=True)

            real_output = discriminator(images, training=True)
            fake_output = discriminator(generated_images, training=True)

            gen_loss = generator_loss(fake_output)
            disc_loss = discriminator_loss(real_output, fake_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
            discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
            return gen_loss, disc_loss
        

    def train(data_generator, epochs, autoencoder_pretrain_epochs=0):
        for epoch in range(autoencoder_pretrain_epochs):
            start = time.time()
            for image_batch, batches_left in data_generator():
                autoenc_loss = autoencoder.train_on_batch(image_batch, image_batch)
            print("%-4d Autoencoder loss: %8.4e -- Time: %6.3fs" % (epoch+1, autoenc_loss, time.time()-start))
        
        for epoch in range(epochs):
            start = time.time()
            for image_batch, batches_left in data_generator():
                autoenc_loss = autoencoder.train_on_batch(image_batch, image_batch)
                gen_loss, disc_loss = train_step(image_batch)
                if batches_left == 0:
                    real_latent_vec = encoder(image_batch[:1])
            print("%-4d Generator loss: %8.4e -- Discriminator loss: %8.4e -- Autoencoder loss: %8.4e -- Time: %6.3fs" % (epoch+1, gen_loss, disc_loss, autoenc_loss, time.time()-start))
            generate_and_save_images(generator, epoch, real_latent_vec)

    train(image_batch_generator, epochs = 2000, autoencoder_pretrain_epochs=0)

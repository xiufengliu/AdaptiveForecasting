import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class GANModel:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.generator = self.define_generator()
        self.discriminator = self.define_discriminator()
        self.gan_model = self.define_gan()

    def define_generator(self):
        # Define the generator neural network
        model = tf.keras.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=self.input_shape))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='linear'))
        return model

    def define_discriminator(self):
        # Define the discriminator neural network
        model = tf.keras.Sequential()
        model.add(layers.Dense(128, activation='relu', input_shape=self.input_shape))
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        return model

    def define_gan(self):
        # Define the GAN model
        self.discriminator.trainable = False
        model = tf.keras.Sequential()
        model.add(self.generator)
        model.add(self.discriminator)
        return model

    def train_gan_model(self, data, num_epochs, batch_size):
        # Train the GAN model on the preprocessed dataset
        for epoch in range(num_epochs):
            # Train the discriminator
            idx = np.random.randint(0, data.shape[0], batch_size)
            real_data = data[idx]
            fake_data = self.generator.predict(np.random.normal(0, 1, (batch_size, self.input_shape[0])))
            x = np.concatenate([real_data, fake_data])
            y = np.zeros(2 * batch_size)
            y[:batch_size] = 0.9
            d_loss = self.discriminator.train_on_batch(x, y)

            # Train the generator
            z = np.random.normal(0, 1, (batch_size, self.input_shape[0]))
            y = np.ones(batch_size)
            g_loss = self.gan_model.train_on_batch(z, y)

            # Evaluate the GAN model
            print(f'Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}')

        return self.gan_model

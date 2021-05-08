import os
import numpy as np
import tensorflow as tf

from PIL import Image
from models import GAN
from dataset import IconGenerator


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)


class Trainer:
    def __init__(self, batch_size=64):
        self.gan = GAN()
        self.batch_size = batch_size
        # Load the dataset
        self.icon_generator = IconGenerator(self.batch_size)

    def train(self, epochs, sample_interval=50):
        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake = np.ones((self.batch_size, 1))

        for epoch in range(epochs):
            for i, inputs in enumerate(self.icon_generator):
                if i > len(self.icon_generator):
                    break
                s1, s2, s3, contour = inputs

                # Translate images to their opposite domain
                fake = self.gan.generator([s1, contour])
                fake_color = tf.concat([fake, s2], axis=1)
                real_color = tf.concat([s1, s2], axis=1)
                fake_shape = tf.concat([fake, contour], axis=1)
                real_shape = tf.concat([s3, contour], axis=1)
                # print(fake_color.shape, real_color.shape, fake_shape.shape, real_shape.shape)

                # Train the discriminators
                with tf.GradientTape() as tape:
                    d_out_color_real = self.gan.color_discriminator(real_color)
                    d_out_color_fake = self.gan.color_discriminator(fake_color)

                    d_color_loss = self.gan.color_discriminator_loss(d_out_color_real, d_out_color_fake)
                    d_color_grads = tape.gradient(d_color_loss, self.gan.color_discriminator.trainable_variables)
                    self.gan.d_optimizer.apply_gradients(
                        zip(d_color_grads, self.gan.color_discriminator.trainable_variables))

                with tf.GradientTape() as tape:
                    d_out_shape_real = self.gan.shape_discriminator(real_shape)
                    d_out_shape_fake = self.gan.shape_discriminator(fake_shape)

                    d_shape_loss = self.gan.shape_discriminator_loss(d_out_shape_real, d_out_shape_fake)
                    d_shape_grads = tape.gradient(d_shape_loss, self.gan.shape_discriminator.trainable_variables)
                    self.gan.d_optimizer.apply_gradients(
                        zip(d_shape_grads, self.gan.shape_discriminator.trainable_variables))

                # Clip discriminator weights
                # for d in [self.color_discriminator, self.shape_discriminator]:
                #     for l in d.layers:
                #         weights = l.get_weights()
                #         weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                #         l.set_weights(weights)

                # ------------------
                #  Train Generators
                # ------------------

                # Train the generators: contour, color_img, target_color_img, target_origin_img
                g_loss = self.gan.combined.train_on_batch([contour, s1, s2, s3], [valid, valid])

                # Plot the progress
                # print(d_color_loss, d_shape_loss, g_loss)
                print("%d [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                      % (epoch, d_color_loss, d_shape_loss, g_loss[0]))

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    # self.save_images(epoch, fake_color, fake_shape)
                    self.save_image(epoch, fake)
                    self.save_model(epoch)

    @classmethod
    def save_images(cls, epoch, fake_color, fake_shape):
        np.save(f'{epoch}-fake-color.png', fake_color)
        np.save(f'{epoch}-fake-shape.png', fake_shape)

    def save_image(self, epoch, arr):
        index = np.random.randint(self.batch_size, size=1)[0]
        arr = arr[index].numpy()
        arr = arr.reshape((64, 64, 3))
        arr = (arr + 1) / 2.0 + 255.0
        arr = arr.astype(np.uint8)
        image = Image.fromarray(arr)
        image.save(f'{epoch}-fake.png')

    def save_model(self, epoch):
        if not os.path.isdir('models'):
            os.mkdir('models')
        save_path = 'models/checkpoint_{epoch:04d}.ckpt'
        save_path = save_path.format(epoch=epoch)
        self.gan.generator.save_weights(save_path)


if __name__ == '__main__':
    trainer = Trainer()
    trainer.train(500)

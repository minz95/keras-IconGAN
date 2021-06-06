import datetime
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, AvgPool2D, UpSampling2D
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.optimizers import Adam

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


class itemGAN:
    def __init__(self):
        self.img_dim = 64
        self.img_ch = 3
        self.con_ch = 1

        self.img_shape = (self.img_dim, self.img_dim, self.img_ch)
        self.con_shape = (self.img_dim, self.img_dim, self.con_ch)

        self.d_optimizer = Adam(learning_rate=2e-4, beta_1=0, beta_2=0.999)
        self.g_optimizer = Adam(learning_rate=5e-5, beta_1=0, beta_2=0.999)

        self.shape_discriminator = self.build_discriminator(3, 1)
        self.color_discriminator = self.build_discriminator(3, 3)
        self.generator = self.build_generator()

        self.shape_discriminator.compile(optimizer=self.d_optimizer,
                                         loss=self.discriminator_loss,
                                         metrics=['accuracy'])
        self.color_discriminator.compile(optimizer=self.d_optimizer,
                                         loss=self.discriminator_loss,
                                         metrics=['accuracy'])
        self.shape_discriminator.trainable = False
        self.color_discriminator.trainable = False
        self.generator.compile(optimizer=self.g_optimizer,
                               loss=self.generator_loss)
        # self.combined.compile(optimizer=self.g_optimizer,
        #                       loss=[self.generator_loss, self.generator_loss],
        #                       loss_weights=[1, 1])

    @classmethod
    def discriminator_loss(cls, real, fake):
        loss_real = K.mean(K.relu(1.0 - real))
        loss_fake = K.mean(K.relu(1.0 + fake))
        return loss_real + loss_fake

    @classmethod
    def generator_loss(cls, _, fake):
        # d_out = D(fake)
        return -K.mean(fake)

    def train(self, epochs=1000, sample_interval=100):
        icon_generator = IconGenerator(64)
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = tf.summary.create_file_writer(f'./logs/{current_time}')
        n_minibatches = len(icon_generator)
        for epoch in range(epochs):
            for i, inputs in enumerate(icon_generator):
                step = n_minibatches * epoch + i
                if i > len(icon_generator):
                    break

                color_img, same_color_img, shape_img, contour = inputs
                fake = self.generator([color_img, contour])
                # fake_color = tf.concat([fake, same_color_img], axis=-1)
                # real_color = tf.concat([color_img, same_color_img], axis=-1)
                # fake_shape = tf.concat([fake, contour], axis=-1)
                # real_shape = tf.concat([shape_img, contour], axis=-1)

                d_color_loss = 0
                with tf.GradientTape() as tape:
                    d_out_color_real = self.color_discriminator([color_img, same_color_img])
                    d_out_color_fake = self.color_discriminator([fake, same_color_img])
                    d_color_loss = self.discriminator_loss(d_out_color_real, d_out_color_fake)
                    d_color_grads = tape.gradient(d_color_loss, self.color_discriminator.trainable_variables)
                    with summary_writer.as_default():
                        tf.summary.scalar('d-color-loss', d_color_loss, step=step)

                    self.d_optimizer.apply_gradients(
                        zip(d_color_grads, self.color_discriminator.trainable_variables))

                d_shape_loss = 0
                with tf.GradientTape() as tape:
                    d_out_shape_real = self.shape_discriminator([shape_img, contour])
                    d_out_shape_fake = self.shape_discriminator([fake, contour])
                    d_shape_loss = self.discriminator_loss(d_out_shape_real, d_out_shape_fake)
                    d_shape_grads = tape.gradient(d_shape_loss, self.shape_discriminator.trainable_variables)
                    with summary_writer.as_default():
                        tf.summary.scalar('d-shape-loss', d_shape_loss, step=step)

                    self.d_optimizer.apply_gradients(
                        zip(d_shape_grads, self.shape_discriminator.trainable_variables))

                g_loss = 0
                with tf.GradientTape() as tape:
                    fake = self.generator([color_img, contour])
                    # fake_color = tf.concat([fake, same_color_img], axis=-1)
                    # fake_shape = tf.concat([fake, contour], axis=-1)

                    d_out_color_fake = self.color_discriminator([fake, same_color_img])
                    d_out_shape_fake = self.shape_discriminator([fake, contour])
                    g_color_loss = self.generator_loss(self.color_discriminator, d_out_color_fake)
                    g_shape_loss = self.generator_loss(self.shape_discriminator, d_out_shape_fake)
                    g_loss = g_color_loss + g_shape_loss
                    g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
                    with summary_writer.as_default():
                        tf.summary.scalar(f'g-shape-loss', g_shape_loss, step=step)
                        tf.summary.scalar(f'g-color-loss', g_color_loss, step=step)
                        tf.summary.scalar(f'g_loss', g_loss, step=step)
                    self.g_optimizer.apply_gradients(
                        zip(g_grads, self.generator.trainable_variables))

                print(f'[D-COLOR-LOSS] {d_color_loss} :: [D-SHAPE-LOSS] {d_shape_loss} :: [G-LOSS] {g_loss}')

                if step % sample_interval == 0:
                    print(f'Save samples at step: {step}')
                    self.save_image(step, 'input-img', color_img)
                    self.save_image(step, 'same-color-img', same_color_img)
                    self.save_image(step, 'origin-of-contour', shape_img)
                    self.save_image(step, 'contour', contour)
                    self.save_image(step, 'fake', fake)
                    self.generator.save(f'models', save_format='tf')
                    with summary_writer.as_default():
                        tf.summary.image(f'{step}-color', color_img, max_outputs=64, step=step)
                        tf.summary.image(f'{step}-same-color', same_color_img,  max_outputs=64, step=step)
                        tf.summary.image(f'{step}-fake', fake, max_outputs=64, step=step)

    @classmethod
    def save_image(cls, step, name, arr):
        arr = np.clip((arr + 1) / 2.0 * 255.0, 0.0, 255.0)
        arr = arr.astype(np.uint8)
        arr = arr[0]
        if arr.shape[2] == 1:
            arr = arr.squeeze()
        image = Image.fromarray(arr)
        image.save(f'samples/{step}-{name}.png')

    def build_discriminator(self, ch_fake, ch_real):
        input1 = Input(shape=(self.img_dim, self.img_dim, ch_fake))
        input2 = Input(shape=(self.img_dim, self.img_dim, ch_real))
        x = tf.concat([input1, input2], axis=-1)
        x = SpectralNormalization(Conv2D(self.img_dim * 1, kernel_size=3, strides=2, padding='same'))(x)
        x = LeakyReLU(0.2)(x)

        x = SpectralNormalization(Conv2D(self.img_dim * 2, kernel_size=3, strides=2, padding='same'))(x)
        x = LeakyReLU(0.2)(x)

        x = SpectralNormalization(Conv2D(self.img_dim * 4, kernel_size=3, strides=2, padding='same'))(x)
        x = LeakyReLU(0.2)(x)

        x = SpectralNormalization(Conv2D(self.img_dim * 8, kernel_size=3, strides=1, padding='same'))(x)
        x = LeakyReLU(0.2)(x)

        x = SpectralNormalization(Conv2D(1, kernel_size=3, strides=1, padding='same'))(x)
        return Model([input1, input2], x)

    @classmethod
    def residual_block(cls, in_ch, out_ch, x, sample=False):
        sampling = UpSampling2D(interpolation='nearest')
        leaky_relu = LeakyReLU(0.2)

        h = BatchNormalization()(x)
        h = leaky_relu(h)

        if sample:
            h = sampling(h)
            x = sampling(x)

        h = SpectralNormalization(Conv2D(out_ch, 3, 1, padding='same'))(h)
        h = BatchNormalization()(h)
        h = leaky_relu(h)
        h = SpectralNormalization(Conv2D(out_ch, 3, 1, padding='same'))(h)

        if in_ch != out_ch:
            x = SpectralNormalization(Conv2D(out_ch, 1, 1, padding='valid'))(x)

        return x + h

    def build_generator(self):
        ch_output = 3
        color_input = Input(shape=(self.img_dim, self.img_dim, self.img_ch))
        shape_input = Input(shape=(self.img_dim, self.img_dim, self.con_ch))

        c = SpectralNormalization(Conv2D(self.img_dim * 1, 3, 1, padding='same'))(color_input)
        c = BatchNormalization()(c)
        c = LeakyReLU(0.2)(c)
        c = AvgPool2D(2)(c)
        c = SpectralNormalization(Conv2D(self.img_dim * 2, 3, 1, padding='same'))(c)
        c = BatchNormalization()(c)
        c = LeakyReLU(0.2)(c)
        c = AvgPool2D(2)(c)
        c = SpectralNormalization(Conv2D(self.img_dim * 4, 3, 1, padding='same'))(c)
        c = BatchNormalization()(c)
        c = LeakyReLU(0.2)(c)
        c = AvgPool2D(2)(c)

        s = SpectralNormalization(Conv2D(self.img_dim * 1, 3, 1, padding='same'))(shape_input)
        s = BatchNormalization()(s)
        s = LeakyReLU(0.2)(s)
        s = AvgPool2D(2)(s)
        s = SpectralNormalization(Conv2D(self.img_dim * 2, 3, 1, padding='same'))(s)
        s = BatchNormalization()(s)
        s = LeakyReLU(0.2)(s)
        s = AvgPool2D(2)(s)
        s = SpectralNormalization(Conv2D(self.img_dim * 4, 3, 1, padding='same'))(s)
        s = BatchNormalization()(s)
        s = LeakyReLU(0.2)(s)
        s = AvgPool2D(2)(s)

        h = tf.concat([c, s], axis=-1)
        h = SpectralNormalization(Conv2D(self.img_dim * 4, 3, 1, padding='same'))(h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 4, h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 4, h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 4, h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 4, h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 2, h, sample=True)
        h = self.residual_block(self.img_dim * 2, self.img_dim * 1, h, sample=True)
        h = self.residual_block(self.img_dim * 1, self.img_dim * 1, h, sample=True)
        h = BatchNormalization()(h)
        h = LeakyReLU(0.2)(h)
        h = SpectralNormalization(Conv2D(ch_output, 3, 1, padding='same'))(h)
        h = tf.nn.tanh(h)

        return Model([color_input, shape_input], h)


if __name__ == '__main__':
    gan = itemGAN()
    gan.train()

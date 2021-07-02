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


class ItemGAN:
    def __init__(self):
        self.batch_size = 64
        self.img_dim = 64
        self.img_ch = 3
        self.con_ch = 1

        self.img_shape = (self.img_dim, self.img_dim, self.img_ch)
        self.con_shape = (self.img_dim, self.img_dim, self.con_ch)

        self.sd_optimizer = Adam(learning_rate=2e-4, beta_1=0, beta_2=0.999)
        self.cd_optimizer = Adam(learning_rate=2e-4, beta_1=0, beta_2=0.999)
        self.g_optimizer = Adam(learning_rate=5e-5, beta_1=0, beta_2=0.999)

        self.shape_discriminator = self.build_discriminator(3, 1)
        self.color_discriminator = self.build_discriminator(3, 3)
        self.generator = self.build_generator()

        self.shape_discriminator.compile(optimizer=self.sd_optimizer,
                                         loss=self.disc_loss,
                                         metrics=['accuracy'])
        self.color_discriminator.compile(optimizer=self.cd_optimizer,
                                         loss=self.disc_loss,
                                         metrics=['accuracy'])
        self.shape_discriminator.trainable = False
        self.color_discriminator.trainable = False

        # -------------------------
        # Construct Computational
        #   Graph of Generator
        # -------------------------

        # Input images and their conditioning images
        img_input = Input(shape=self.img_shape)
        con_input = Input(shape=self.con_shape)
        color_input = Input(shape=self.img_shape)

        # By conditioning on B generate a fake version of A
        # self.generator = tf.keras.models.load_model('models')
        fake = self.generator([img_input, con_input])

        valid_shape = self.shape_discriminator([fake, con_input])
        valid_color = self.color_discriminator([fake, color_input])

        self.combined = Model(inputs=[img_input, con_input, color_input], outputs=[valid_shape, valid_color])
        self.combined.compile(loss=[self.generator_loss, self.generator_loss],
                              loss_weights=[1, 1],
                              optimizer=self.g_optimizer)

    @classmethod
    def discriminator_loss(cls, real, fake):
        loss_real = K.mean(K.relu(1.0 - real))
        loss_fake = K.mean(K.relu(1.0 + fake))
        return loss_real + loss_fake

    @classmethod
    def disc_loss(cls, labels, preds):
        # sign = (2 * K.mean(tf.reshape(labels, (-1, 64)), axis=1)) - 1
        # loss = K.mean(K.relu(1.0 - tf.math.multiply(sign, tf.reshape(preds, (-1, 64)))))
        loss_real = K.mean(K.relu(1.0 - preds[:64]))
        loss_fake = K.mean(K.relu(1.0 + preds[64:]))
        return loss_real + loss_fake

    @classmethod
    def generator_loss(cls, _, fake):
        # d_out = D(fake)
        return -K.mean(fake)

    def train(self, epochs=2000, sample_interval=100):
        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((self.batch_size, 8, 8, 1))
        fake = np.zeros((self.batch_size, 8, 8, 1))

        icon_generator = IconGenerator(64)
        for anchor, _, _, contour in icon_generator:
            fixed_img = tf.identity(anchor)
            fixed_contour = tf.identity(contour)
            break

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        summary_writer = tf.summary.create_file_writer(f'./logs/{current_time}')
        n_minibatches = len(icon_generator)
        for epoch in range(epochs):
            for i, inputs in enumerate(icon_generator):
                step = n_minibatches * epoch + i
                if i >= len(icon_generator):
                    break

                color_img, same_color_img, shape_img, contour = inputs
                fake_img = self.generator.predict([color_img, contour])

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Train the discriminators (original images = real / generated = Fake)
                # d_color_loss_real = self.color_discriminator.train_on_batch([color_img, same_color_img], valid)
                # d_color_loss_fake = self.color_discriminator.train_on_batch([fake_img, same_color_img], fake)
                # d_color_loss = 0.5 * np.add(d_color_loss_real, d_color_loss_fake)
                d_color_loss = self.color_discriminator.train_on_batch([
                    tf.concat((color_img, fake_img), axis=0),
                    tf.concat((same_color_img, same_color_img), axis=0)
                ], tf.concat((valid, fake), axis=0))

                # d_shape_loss_real = self.shape_discriminator.train_on_batch([shape_img, contour], valid)
                # d_shape_loss_fake = self.shape_discriminator.train_on_batch([fake_img, contour], fake)
                # d_shape_loss = 0.5 * np.add(d_shape_loss_real, d_shape_loss_fake)
                d_shape_loss = self.shape_discriminator.train_on_batch([
                    tf.concat((shape_img, fake_img), axis=0),
                    tf.concat((contour, contour), axis=0)
                ], tf.concat((valid, fake), axis=0))
                d_loss = 0.5 * np.add(d_color_loss, d_shape_loss)

                # -----------------
                #  Train Generator
                # -----------------

                # Train the generators
                g_loss = self.combined.train_on_batch([color_img, contour, same_color_img], [valid, valid])


                elapsed_time = datetime.datetime.now() - start_time
                # Plot the progress
                print("[Epoch %d/%d] [Step %d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                                                  step,
                                                                                                  d_loss[0],
                                                                                                  100 * d_loss[1],
                                                                                                  g_loss[0],
                                                                                                  elapsed_time))

                if step % sample_interval == 0:
                    print(f'Save samples at step: {step}')
                    fixed_fake = self.generator.predict([fixed_img, fixed_contour])
                    grid_img = self.image_grid(fixed_fake)[0]

                    grid_img = grid_img * 0.5 + 0.5
                    grid_img = np.clip(grid_img * 255.0, 0.0, 255.0)
                    grid_img = grid_img.astype(np.uint8)
                    image = Image.fromarray(grid_img)
                    image.save(f'samples2/{step}-grid-fake.png')

                    # self.save_image(step, 'input-img', color_img)
                    # self.save_image(step, 'same-color-img', same_color_img)
                    # self.save_image(step, 'origin-of-contour', shape_img)
                    # self.save_image(step, 'contour', contour)
                    # self.save_image(step, 'fake', fake_img)
                    self.generator.save(f'models3', save_format='tf')
                    with summary_writer.as_default():
                        tf.summary.image(f'{step}-color', color_img, max_outputs=64, step=step)
                        tf.summary.image(f'{step}-same-color', same_color_img,  max_outputs=64, step=step)
                        tf.summary.image(f'{step}-fake', fake_img, max_outputs=64, step=step)
                        tf.summary.scalar('d-loss', d_loss[0], step=step)
                        tf.summary.scalar('g_loss', g_loss[0], step=step)
                        tf.summary.scalar('d-acc', d_loss[1] * 100, step=step)

    @classmethod
    def image_grid(cls, x, size=8):
        t = tf.unstack(x[:size * size], num=size * size, axis=0)
        rows = [tf.concat(t[i * size:(i + 1) * size], axis=0)
                for i in range(size)]
        image = tf.concat(rows, axis=1)
        return image[None]

    @classmethod
    def save_image(cls, step, name, arr):
        arr = arr * 0.5 + 0.5
        arr = np.clip(arr * 255.0, 0.0, 255.0)
        arr = arr.astype(np.uint8)
        arr = arr[0]
        if arr.shape[2] == 1:
            arr = arr.squeeze()
        image = Image.fromarray(arr)
        image.save(f'samples2/{step}-{name}.png')

    def build_discriminator(self, ch_fake, ch_real):

        def d_layer(layer_input, filters, strides=2, relu=True):
            """Discriminator layer"""
            d = SpectralNormalization(Conv2D(filters, kernel_size=3, strides=strides, padding='same'))(layer_input)
            if relu:
                d = LeakyReLU(alpha=0.2)(d)

            return d

        input1 = Input(shape=(self.img_dim, self.img_dim, ch_fake))
        input2 = Input(shape=(self.img_dim, self.img_dim, ch_real))
        combined_img = tf.concat([input1, input2], axis=-1)
        d1 = d_layer(combined_img, self.img_dim * 1)
        d2 = d_layer(d1, self.img_dim * 2)
        d3 = d_layer(d2, self.img_dim * 4)
        d4 = d_layer(d3, self.img_dim * 8, 1)
        validity = d_layer(d4, 1, 1, relu=False)

        return Model([input1, input2], validity)

    @classmethod
    def residual_block(cls, in_ch, out_ch, x, sample=False):
        sampling = UpSampling2D(interpolation='nearest')
        leaky_relu = LeakyReLU(0.2)

        h = BatchNormalization(epsilon=1e-05, momentum=0.1)(x)
        h = leaky_relu(h)

        if sample:
            h = sampling(h)
            x = sampling(x)

        h = SpectralNormalization(Conv2D(out_ch, 3, 1, padding='same'))(h)
        h = BatchNormalization(epsilon=1e-05, momentum=0.1)(h)
        h = leaky_relu(h)
        h = SpectralNormalization(Conv2D(out_ch, 3, 1, padding='same'))(h)

        if in_ch != out_ch:
            x = SpectralNormalization(Conv2D(out_ch, 1, 1, padding='valid'))(x)

        return x + h

    def build_generator(self):

        def encoder(layer_input, filters):
            """Layers used during downsampling"""
            en = SpectralNormalization(Conv2D(filters, kernel_size=3, strides=1, padding='same'))(layer_input)
            en = BatchNormalization(epsilon=1e-05, momentum=0.1)(en)
            en = LeakyReLU(0.2)(en)
            en = AvgPool2D(2)(en)

            return en

        # Image input
        color_input = Input(shape=(self.img_dim, self.img_dim, self.img_ch))
        shape_input = Input(shape=(self.img_dim, self.img_dim, self.con_ch))

        # Color Encoder
        ce1 = encoder(color_input, self.img_dim * 1)
        ce2 = encoder(ce1, self.img_dim * 2)
        ce3 = encoder(ce2, self.img_dim * 4)

        # Shape Encoder
        se1 = encoder(shape_input, self.img_dim * 1)
        se2 = encoder(se1, self.img_dim * 2)
        se3 = encoder(se2, self.img_dim * 4)

        h = tf.concat([ce3, se3], axis=-1)

        # Decoder
        h = SpectralNormalization(Conv2D(self.img_dim * 4, kernel_size=3, strides=1, padding='same'))(h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 4, h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 4, h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 4, h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 4, h)
        h = self.residual_block(self.img_dim * 4, self.img_dim * 2, h, sample=True)
        h = self.residual_block(self.img_dim * 2, self.img_dim * 1, h, sample=True)
        h = self.residual_block(self.img_dim * 1, self.img_dim * 1, h, sample=True)
        h = BatchNormalization(epsilon=1e-05, momentum=0.1)(h)
        h = LeakyReLU(0.2)(h)
        h = SpectralNormalization(Conv2D(3, 3, 1, padding='same'))(h)
        output_img = tf.nn.tanh(h)

        return Model([color_input, shape_input], output_img)


if __name__ == '__main__':
    gan = ItemGAN()
    gan.train()

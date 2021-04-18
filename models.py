import numpy as np
import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, AvgPool2D, Conv2DTranspose
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.optimizers import Adam
from tensorflow.keras import backend as K

from dataset import IconGenerator


class GAN:
    def __init__(self):
        self.generator = Generator()
        self.shape_discriminator = Discriminator(3 + 1)
        self.color_discriminator = Discriminator(3 + 3)
        self.d_optimizer = Adam(learning_rate=2e-4, beta_1=0, beta_2=0.999)

        contour = Input(shape=(1, 64, 64))
        color_img = Input(shape=(3, 64, 64))
        target_color_img = Input(shape=(3, 64, 64))
        target_origin_img = Input(shape=(3, 64, 64))
        fake_img = self.generator([color_img, contour])
        valid_color = self.color_discriminator(tf.concat([fake_img, target_color_img], dim=1))
        valid_shape = self.shape_discriminator(tf.concat([fake_img, target_origin_img], dim=1))
        self.combined = Model(
            inputs=[contour, color_img, target_color_img, target_origin_img],
            outputs=[valid_color, valid_shape]
        )

        self.compile()

    def compile(self, g_lr=5e-5, d_lr=2e-4):
        self.shape_discriminator.compile(optimizer=self.d_optimizer,
                                         loss=self.shape_discriminator_loss,
                                         metrics=['accuracy'])
        self.color_discriminator.compile(optimizer=self.d_optimizer,
                                         loss=self.color_discriminator_loss,
                                         metrics=['accuracy'])
        g_optimizer = Adam(learning_rate=g_lr, beta_1=0, beta_2=0.999)
        self.combined.compile(optimizer=g_optimizer,
                              loss=[self.generator_loss, self.generator_loss],
                              loss_weights=[1, 1],
                              metrics=['accuracy'])

    def train(self, epochs, batch_size=64, sample_interval=50):
        # Load the dataset
        icon_generator = IconGenerator(batch_size)

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):
            for i, inputs in enumerate(icon_generator):
                if i > len(icon_generator):
                    break
                s1, s2, s3, contour = inputs
                s1 = (s1.astype(np.float32) - 127.5) / 127.5
                s2 = (s2.astype(np.float32) - 127.5) / 127.5
                s3 = (s3.astype(np.float32) - 127.5) / 127.5
                contour = (contour.astype(np.float32) - 127.5) / 127.5

                # Translate images to their opposite domain
                fake = self.generator([s1, s2])
                fake_color = tf.concat([fake, s2], axis=1)
                real_color = tf.concat([s1, s2], axis=1)
                fake_shape = tf.concat([fake, contour], axis=1)
                real_shape = tf.concat([s3, contour], axis=1)

                # Train the discriminators
                d_out_color_real = self.shape_discriminator(real_color)
                d_out_color_fake = self.shape_discriminator(fake_color)
                d_out_shape_real = self.shape_discriminator(real_shape)
                d_out_shape_fake = self.shape_discriminator(fake_shape)
                d_color_loss, d_color_grads = self.color_discriminator_loss(d_out_color_real, d_out_color_fake)
                d_shape_loss, d_shape_grads = self.shape_discriminator_loss(d_out_shape_real, d_out_shape_fake)
                self.d_optimizer.apply_gradients(zip(d_color_grads, self.color_discriminator.trainable_variables))
                self.d_optimizer.apply_gradients(zip(d_shape_grads, self.shape_discriminator.trainable_variables))

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
                g_loss = self.combined.train_on_batch([contour, s1, s2, s3], [valid, valid])


                # Plot the progress
                print("%d [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                      % (epoch, d_color_loss[0], d_shape_loss[0], g_loss[0]))

                # If at save interval => save generated image samples
                if epoch % sample_interval == 0:
                    self.save_images(epoch, fake_color, fake_shape)

    @classmethod
    def generator_loss(cls, _, prediction):
        return -prediction.mean()

    def shape_discriminator_loss(self, real, fake):
        with tf.GradientTape() as tape:
            loss_real = K.relu(1.0 - real).mean()
            loss_fake = K.relu(1.0 + fake).mean()
            loss_value = loss_real + loss_fake
        return loss_value, tape.gradient(loss_value, self.shape_discriminator.trainable_variables)

    def color_discriminator_loss(self, real, fake):
        with tf.GradientTape() as tape:
            loss_real = K.relu(1.0 - real).mean()
            loss_fake = K.relu(1.0 + fake).mean()
            loss_value = loss_real + loss_fake
        return loss_value, tape.gradient(loss_value, self.color_discriminator.trainable_variables)


class Generator(tf.keras.Model):
    def __init__(self, ch_color=3, ch_shape=1, img_dim=64, num_spectral_layer=3):
        super(Generator, self).__init__(name='')
        self.ch_color = ch_color
        self.ch_shape = ch_shape
        self.ch_output = 3
        self.dim = img_dim
        self.num_spectral_layer = num_spectral_layer

        self.color_input_shape = (self.ch_color, self.dim, self.dim)
        self.shape_input_shape = (self.ch_shape, self.dim, self.dim)
        self.decoder_input_shape = (self.ch_color, self.dim * 4, self.dim * 4)

        self.color_encoder = self._build_color_encoder()
        self.shape_encoder = self._build_shape_encoder()
        self.decoder = self._build_decoder()

        self._color_input = None
        self._shape_input = None
        self.model = self._build_model()

    def summary(self, **kwargs):
        self.style_encoder.summary()
        self.content_encoder.summary()
        self.decoder.summary()
        self.model.summary()

    def generator_loss(self, fake):
        d_out = self.discriminator(fake)
        return -d_out.mean()

    def call(self, input_tensor, **kwargs):
        color_tensor = input_tensor[0]
        shape_tensor = input_tensor[1]
        color_h = self.color_encoder(color_tensor)
        shape_h = self.shape_encoder(shape_tensor)
        h = tf.concat([color_h, shape_h], axis=1)
        return self.decoder(h)

    def _build(self):
        self._build_style_encoder()
        self._build_content_encoder()
        self._build_decoder()

    def _build_color_encoder(self):
        color_encoder_input = Input(shape=self.color_input_shape,
                                    name="color_encoder_input")
        self._color_input = color_encoder_input
        spectral_out = self._add_spectral_layers(color_encoder_input)
        return Model(color_encoder_input, spectral_out)

    def _build_shape_encoder(self):
        content_encoder_input = Input(shape=self.shape_input_shape,
                                      name="content_encoder_input")
        spectral_layers = self._add_spectral_layers(content_encoder_input)
        return spectral_layers

    def _build_decoder(self):
        decoder_input = Input(shape=self.decoder_input_shape,
                              name="decoder_input")
        conv2a = Conv2D(self.dim * 4, 3, 1,
                        padding='same',
                        data_format='channels_first')(decoder_input)
        sn2a = SpectralNormalization(conv2a)

        rb1 = ResidualBlock(self.dim * 4, self.dim * 4)(sn2a)
        rb2 = ResidualBlock(self.dim * 4, self.dim * 4)(rb1)
        rb3 = ResidualBlock(self.dim * 4, self.dim * 4)(rb2)
        rb4 = ResidualBlock(self.dim * 4, self.dim * 4)(rb3)
        rb5 = ResidualBlock(self.dim * 4, self.dim * 2, sample='up')(rb4)
        rb6 = ResidualBlock(self.dim * 2, self.dim * 1, sample='up')(rb5)
        rb7 = ResidualBlock(self.dim * 1, self.dim * 1, sample='up')(rb6)

        bn = BatchNormalization(rb7)
        leaky_relu = LeakyReLU(0.2)(bn)
        conv2b = Conv2D(self.ch_output, 3, 1,
                        padding='same',
                        data_format='channels_first')(leaky_relu)
        sn2b = SpectralNormalization()(conv2b)
        th = tf.nn.tanh()(sn2b)

        return Model(decoder_input, th)

    def _build_model(self):
        model_input = [self._color_input, self._shape_input]
        color_encoder_output = self.color_encoder(model_input[0])
        shape_encoder_output = self.shape_encoder(model_input[1])
        decoder_input = tf.concat([color_encoder_output, shape_encoder_output], axis=1)
        model_output = self.decoder(decoder_input)

        return Model(model_input, model_output, name="generator")

    def _add_spectral_layers(self, encoder_input):
        x = encoder_input
        for layer_index in range(self.num_spectral_layer):
            x = self._add_spectral_layer(layer_index, x)
        return x

    def _add_spectral_layer(self, layer_index, x):
        """SpectralNorm + BatchNorm2D +  LeakyReLu + AvgPool2D"""
        layer_number = layer_index + 1
        conv2d = Conv2D(
            filters=self.ch_output,
            kernel_size=3,
            strides=1,
            padding="same",
            data_format='channels_first',
            name=f"encoder_conv2d_{layer_number}"
        )(x)
        spectral_layer = SpectralNormalization(
            name=f"encoder_spectral_norm_{layer_number}"
        )(conv2d)
        x = spectral_layer(x)
        x = BatchNormalization(name=f"encoder_bn_{layer_number}")(x)
        x = LeakyReLU(0.2, name=f"encoder_leaky_relu_{layer_number}")(x)
        x = AvgPool2D(name=f"encoder_avg_pool_{layer_number}")(x)
        return x


class Discriminator(tf.keras.Model):
    def __init__(self, ch_input):
        super(Discriminator, self).__init__(name='')
        self.base_dim = 64
        self.conv2a = Conv2D(ch_input, kernel_size=3, strides=2,
                             padding='same', data_format='channels_first')
        self.sn2a = SpectralNormalization()
        self.leaky_relu1 = LeakyReLU(0.2)

        self.conv2b = Conv2D(self.base_dim * 2, kernel_size=3, strides=2,
                             padding='same', data_format='channels_first')
        self.sn2b = SpectralNormalization()
        self.leaky_relu2 = LeakyReLU(0.2)

        self.conv2c = Conv2D(self.base_dim * 4, kernel_size=3, strides=2,
                             padding='same', data_format='channels_first')
        self.sn2c = SpectralNormalization()
        self.leaky_relu3 = LeakyReLU(0.2)

        self.conv2d = Conv2D(self.base_dim * 8, kernel_size=3, strides=1,
                             padding='same', data_format='channels_first')
        self.sn2d = SpectralNormalization()
        self.leaky_relu4 = LeakyReLU(0.2)

        self.conv2e = Conv2D(1, kernel_size=3, strides=1,
                             padding='same', data_format='channels_first')
        self.sn2e = SpectralNormalization()

    def call(self, input_tensor, **kwargs):
        conv2a = self.conv2a(input_tensor)
        sn2a = self.sn2a(conv2a)
        leaky_relu1 = self.leaky_relu1(sn2a)

        conv2b = self.conv2b(leaky_relu1)
        sn2b = self.sn2b(conv2b)
        leaky_relu2 = self.leaky_relu2(sn2b)

        conv2c = self.conv2c(leaky_relu2)
        sn2c = self.sn2c(conv2c)
        leaky_relu3 = self.leaky_relu3(sn2c)

        conv2d = self.conv2d(leaky_relu3)
        sn2d = self.sn2d(conv2d)
        leaky_relu4 = self.leaky_relu4(sn2d)

        conv2e = self.conv2e(leaky_relu4)
        sn2e = self.sn2e(conv2e)

        return sn2e


class ResidualBlock(tf.keras.Model):
    def __init__(self, ch_in, ch_out, sample='none', training=False):
        super(ResidualBlock, self).__init__(name='')
        self.training = training

        self.conv2a = Conv2D(ch_out, kernel_size=3, strides=1,
                             padding='same', data_format='channels_first')
        self.sn2a = SpectralNormalization()

        self.conv2b = Conv2D(ch_out, kernel_size=3, strides=1,
                             padding='same', data_format='channels_first')
        self.sn2b = SpectralNormalization()

        self.conv2c = Conv2D(ch_out, kernel_size=1, strides=1,
                             padding='valid', data_format='channels_first') if ch_in != ch_out else False
        self.sn2c = SpectralNormalization()

        if sample == 'up':
            self.sample = Conv2DTranspose(1, kernel_size=1, strides=2,
                                          padding='valid', data_format='channels_first')
        else:
            self.sample = None

        self.bn1 = BatchNormalization()
        self.bn2 = BatchNormalization()

        self.leaky_relu = LeakyReLU(0.2)

    def call(self, input_tensor, **kwargs):
        h = self.leaky_relu(self.bn1(input_tensor))
        x = input_tensor

        if self.sample:
            h = self.sample(h)
            x = self.sample(x)

        h = self.conv2a(h)
        h = self.sn2a(h)
        h = self.leaky_relu(self.bn2(h))

        h = self.conv2b(h)
        h = self.sn2b(h)

        if self.conv2c:
            h = self.conv2c(h)
            h = self.sn2c(h)

        return x + h

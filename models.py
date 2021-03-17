import tensorflow as tf

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, AvgPool2D, Conv2DTranspose
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.optimizers import Adam
from tensorflow.keras import backend as K


class GAN:
    def __init__(self):
        self.generator = Generator()
        self.discriminator = Discriminator()

    def train(self, input_tensor):
        style1, style2, style3, contour = input_tensor
        fake = self.generator.predict(style1, contour)
        style_fake = tf.concat([fake, style2], dim=1)
        style_real = tf.concat([style1, style2], dim=1)
        content_fake = tf.concat([fake, contour], dim=1)
        content_real = tf.concat([style3, contour], dim=1)



    def generator_loss(self, fake):
        critic = self.discriminator(fake)
        return -critic.mean()

    def discriminator_loss(self, real, fake):
        critic_real = self.discriminator(real)
        critic_fake = self.discriminator(fake)
        loss_real = K.relu(1.0 - critic_real).mean()
        loss_fake = K.relu(1.0 + critic_fake).mean()
        return loss_real + loss_fake


class Generator:
    def __init__(self, ch_style=3, ch_content=1, img_dim=64, num_spectral_layer=3):
        self.ch_style = ch_style
        self.ch_content = ch_content
        self.ch_output = 3
        self.dim = img_dim
        self.num_spectral_layer = num_spectral_layer

        self.style_input_shape = (self.ch_style, self.dim, self.dim)
        self.content_input_shape = (self.ch_content, self.dim, self.dim)
        self.decoder_input_shape = (self.ch_style, self.dim * 4, self.dim * 4)

        self.style_encoder = None
        self.content_encoder = None
        self.decoder = None
        self.discriminator = None
        self.model = None

        self._model_input = None

        self._build()

    def summary(self):
        self.style_encoder.summary()
        self.content_encoder.summary()
        self.model.summary()

    def compile(self, lr=5e-5):
        optimizer = Adam(learning_rate=lr, beta_1=0, beta_2=0.999)
        self.model.compile(optimizer=optimizer, loss=self.generator_loss)

    def generator_loss(self, fake):
        d_out = self.discriminator(fake)
        return -d_out.mean()

    def train(self):
        pass

    def predict(self, style_tensor, content_tensor):
        style_h = self.style_encoder(style_tensor)
        content_h = self.content_encoder(content_tensor)
        h = tf.concat([style_h, content_h], axis=1)
        return self.decoder(h)

    def _build(self):
        self._build_style_encoder()
        self._build_content_encoder()
        self._build_decoder()

    def _build_style_encoder(self):
        style_encoder_input = Input(shape=self.style_input_shape,
                                    name="style_encoder_input")
        spectral_layers = self._add_spectral_layers(style_encoder_input)
        return spectral_layers

    def _build_content_encoder(self):
        content_encoder_input = Input(shape=self.content_input_shape,
                                      name="content_encoder_input")
        spectral_layers = self._add_spectral_layers(content_encoder_input)
        return spectral_layers

    def _build_decoder(self):
        decoder_input = Input(shape=self.decoder_input_shape,
                              name="decoder_input")
        conv2a = Conv2D(self.dim * 4, 3, 1, padding='same')(decoder_input)
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
        conv2b = Conv2D(self.ch_output, 3, 1, padding='same')(leaky_relu)
        sn2b = SpectralNormalization()(conv2b)
        th = tf.nn.tanh()(sn2b)

        return th

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
    def __init__(self):
        super(Discriminator, self).__init__(name='')
        self.base_dim = 64
        self.conv2a = Conv2D(self.base_dim * 1, kernel_size=3, strides=2, padding='same')
        self.sn2a = SpectralNormalization()
        self.leaky_relu1 = LeakyReLU(0.2)

        self.conv2b = Conv2D(self.base_dim * 2, kernel_size=3, strides=2, padding='same')
        self.sn2b = SpectralNormalization()
        self.leaky_relu2 = LeakyReLU(0.2)

        self.conv2c = Conv2D(self.base_dim * 4, kernel_size=3, strides=2, padding='same')
        self.sn2c = SpectralNormalization()
        self.leaky_relu3 = LeakyReLU(0.2)

        self.conv2d = Conv2D(self.base_dim * 8, kernel_size=3, strides=1, padding='same')
        self.sn2d = SpectralNormalization()
        self.leaky_relu4 = LeakyReLU(0.2)

        self.conv2e = Conv2D(1, kernel_size=3, strides=1, padding='same')
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

        self.conv2a = Conv2D(ch_out, kernel_size=3, strides=1, padding='same')
        self.sn2a = SpectralNormalization()

        self.conv2b = Conv2D(ch_out, kernel_size=3, strides=1, padding='same')
        self.sn2b = SpectralNormalization()

        self.conv2c = Conv2D(ch_out, kernel_size=1, strides=1, padding='valid') if ch_in != ch_out else False
        self.sn2c = SpectralNormalization()

        if sample == 'up':
            self.sample = Conv2DTranspose(1, kernel_size=1, strides=2, padding='valid')
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

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization, AvgPool2D, Conv2DTranspose
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.optimizers import Adam
from tensorflow.keras import backend as K


class GAN:
    def __init__(self):
        self.generator = Generator()
        self.shape_discriminator = Discriminator(3 + 1)
        self.color_discriminator = Discriminator(3 + 3)

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
        d_optimizer = Adam(learning_rate=d_lr, beta_1=0, beta_2=0.999)
        self.shape_discriminator.compile(optimizer=d_optimizer,
                                         loss=self.discriminator_loss,
                                         metrics=['accuracy'])
        self.color_discriminator.compile(optimizer=d_optimizer,
                                         loss=self.discriminator_loss,
                                         metrics=['accuracy'])
        g_optimizer = Adam(learning_rate=g_lr, beta_1=0, beta_2=0.999)
        self.combined.compile(optimizer=g_optimizer,
                              loss=[self.generator_loss, self.generator_loss],
                              loss_weights=[1, 1],
                              metrics=['accuracy'])

    def train(self, epochs, batch_size=64, sample_interval=50):
        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5

        # Domain A and B (rotated)
        X_A = X_train[:int(X_train.shape[0] / 2)]
        X_B = scipy.ndimage.interpolation.rotate(X_train[int(X_train.shape[0] / 2):], 90, axes=(1, 2))

        X_A = X_A.reshape(X_A.shape[0], self.img_dim)
        X_B = X_B.reshape(X_B.shape[0], self.img_dim)

        clip_value = 0.01
        n_critic = 4

        # Adversarial ground truths
        valid = -np.ones((batch_size, 1))
        fake = np.ones((batch_size, 1))

        for epoch in range(epochs):

            # Train the discriminator for n_critic iterations
            for _ in range(n_critic):

                # ----------------------
                #  Train Discriminators
                # ----------------------

                # Sample generator inputs
                imgs_A = self.sample_generator_input(X_A, batch_size)
                imgs_B = self.sample_generator_input(X_B, batch_size)

                # Translate images to their opposite domain
                fake_B = self.G_AB.predict(imgs_A)
                fake_A = self.G_BA.predict(imgs_B)

                # Train the discriminators
                D_A_loss_real = self.D_A.train_on_batch(imgs_A, valid)
                D_A_loss_fake = self.D_A.train_on_batch(fake_A, fake)

                D_B_loss_real = self.D_B.train_on_batch(imgs_B, valid)
                D_B_loss_fake = self.D_B.train_on_batch(fake_B, fake)

                D_A_loss = 0.5 * np.add(D_A_loss_real, D_A_loss_fake)
                D_B_loss = 0.5 * np.add(D_B_loss_real, D_B_loss_fake)

                # Clip discriminator weights
                for d in [self.D_A, self.D_B]:
                    for l in d.layers:
                        weights = l.get_weights()
                        weights = [np.clip(w, -clip_value, clip_value) for w in weights]
                        l.set_weights(weights)

            # ------------------
            #  Train Generators
            # ------------------

            # Train the generators
            g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, valid, imgs_A, imgs_B])

            # Plot the progress
            print("%d [D1 loss: %f] [D2 loss: %f] [G loss: %f]" \
                  % (epoch, D_A_loss[0], D_B_loss[0], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.save_imgs(epoch, X_A, X_B)

    @classmethod
    def generator_loss(cls, _, prediction):
        return -prediction.mean()

    @classmethod
    def discriminator_loss(cls, real, prediction):
        loss_real = K.relu(1.0 - real).mean()
        loss_fake = K.relu(1.0 + prediction).mean()
        return loss_real + loss_fake


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

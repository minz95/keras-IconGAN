import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, \
    BatchNormalization, AvgPool2D, Conv2DTranspose, ZeroPadding2D, UpSampling2D
from tensorflow_addons.layers import SpectralNormalization
from tensorflow.keras.optimizers import Adam

class itemGAN:
    def __init__(self):
        self.img_dim = 64
        self.img_ch = 3
        self.con_ch = 1

        self.img_shape = (self.img_dim, self.img_dim, self.img_ch)
        self.con_shape = (self.img_dim, self.img_dim, self.con_ch)

        self.d_optimizer = Adam(learning_rate=2e-4, beta_1=0, beta_2=0.999)
        self.g_optimizer = Adam(learning_rate=5e-5, beta_1=0, beta_2=0.999)

        self.shape_discriminator = self.build_discriminator()
        self.color_discriminator = self.build_discriminator()
        self.generator = self.build_generator(color_ch=3, shape_ch=1)

        self.shape_discriminator.compile(optimizer=self.d_optimizer,
                                         loss=self.discriminator_loss,
                                         metrics=['accuracy'])
        self.color_discriminator.compile(optimizer=self.d_optimizer,
                                         loss=self.discriminator_loss,
                                         metrics=['accuracy'])
        self.shape_discriminator.trainable = False
        self.color_discriminator.trainable = False
        # self.generator.compile(optimizer=self.g_optimizer,
        #                        loss=self.generator_loss)
        self.combined.compile(optimizer=self.g_optimizer,
                              loss=[self.generator_loss, self.generator_loss],
                              loss_weights=[1, 1])

    def discriminator_loss(self, D, real, fake):
        d_out_real = D(real)
        d_out_fake = D(fake)
        loss_real = K.mean(K.relu(1.0 - d_out_real))
        loss_fake = K.mean(K.relu(1.0 + d_out_fake))
        return loss_real + loss_fake

    def generator_loss(self, D, fake):
        d_out = D(fake)
        return -K.mean(d_out)

    def train(self):
        pass

    def build_discriminator(self, ch_fake, ch_real):
        input1 = Input()
        conv2a = Conv2D(self.img_dim * 1, kernel_size=3, strides=2, padding='same')
        self.sn2a = SpectralNormalization(self.conv2a)
        self.leaky_relu1 = LeakyReLU(0.2)
        input_tensor = ZeroPadding2D(padding=1)(input_tensor)
        sn2a = self.sn2a(input_tensor)
        leaky_relu1 = self.leaky_relu1(sn2a)

        leaky_relu1 = ZeroPadding2D(padding=1)(leaky_relu1)
        sn2b = self.sn2b(leaky_relu1)
        leaky_relu2 = self.leaky_relu2(sn2b)

        leaky_relu2 = ZeroPadding2D(padding=1)(leaky_relu2)
        sn2c = self.sn2c(leaky_relu2)
        leaky_relu3 = self.leaky_relu3(sn2c)

        leaky_relu3 = ZeroPadding2D(padding=1)(leaky_relu3)
        sn2d = self.sn2d(leaky_relu3)
        leaky_relu4 = self.leaky_relu4(sn2d)

        leaky_relu4 = ZeroPadding2D(padding=1)(leaky_relu4)
        sn2e = self.sn2e(leaky_relu4)

        return sn2e

    def build_generator(self, color_ch=3, shape_ch=1):
        pass



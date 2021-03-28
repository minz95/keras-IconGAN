import tensorflow as tf

from models import GAN


def train(self, input_tensor):
    color1, color2, color3, contour = input_tensor
    fake = self.generator.predict(color1, contour)
    style_fake = tf.concat([fake, color2], dim=1)
    style_real = tf.concat([color1, color2], dim=1)
    content_fake = tf.concat([fake, contour], dim=1)
    content_real = tf.concat([color3, contour], dim=1)


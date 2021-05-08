import cv2
import numpy as np
import tensorflow as tf

from PIL import Image

from models import Generator
from dataset import IconGenerator


if __name__ == '__main__':
    model_path = 'models/checkpoint_450.ckpt'
    model = Generator()
    model.load_weights(model_path)
    preprocessor = IconGenerator(batch_size=1)

    color_path = './preprocessed_data/img/000004.png'
    draw_path = 'testdata/bird.png'

    img = cv2.imread(draw_path)
    con_img = np.invert(img)
    con_img = Image.fromarray(con_img)
    con_img.save('edge_result.png', 'png')
    con_img.show()

    s1 = Image.open(color_path).convert('RGB')

    h, w = 64, 64
    s1 = tf.image.resize(s1, size=(h, w), method=tf.image.ResizeMethod.BICUBIC)
    con_img = tf.image.resize(con_img, size=(h, w), method=tf.image.ResizeMethod.BICUBIC)

    s1 = (s1.astype(np.float32) - 127.5) / 127.5
    con_img = (con_img.astype(np.float32) - 127.5) / 127.5
    s1 = np.moveaxis(s1, -1, 0)
    con_img = np.moveaxis(con_img, -1, 0)[:1, :, :]

    s1 = np.expand_dims(s1, axis=0)
    con_img = np.expand_dims(con_img, axis=0)
    fake = model([s1, con_img]).numpy()
    fake = fake.reshape((64, 64, 3))
    fake = (fake + 1) / 2.0 + 255.0
    fake = fake.astype(np.uint8)
    fake_img = Image.fromarray(fake)
    fake_img.show()

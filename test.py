import cv2
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, default='models',
                        help='model path')
    parser.add_argument('color_input', type=str, default='test_data/000004.png',
                        help='color image')
    parser.add_argument('shape_input', type=str, default='test_data/bird.png',
                        help='shape image')
    parser.add_argument('output_path', type=str, default='test_data/output.png',
                        help='test output image path')

    args = parser.parse_args()
    model_path = args.model_path
    model = tf.keras.models.load_model(model_path)

    color_path = args.color_input
    draw_path = args.shape_input

    img = cv2.imread(draw_path)
    con_img = np.invert(img)
    con_img = Image.fromarray(con_img)
    # con_img.save('edge_result.png', 'png')
    # con_img.show()

    s1 = Image.open(color_path).convert('RGB')

    h, w = 64, 64
    s1 = tf.image.resize(s1, size=(h, w), method=tf.image.ResizeMethod.BICUBIC)
    con_img = tf.image.resize(con_img, size=(h, w), method=tf.image.ResizeMethod.BICUBIC)

    color_image = (s1.astype(np.float32) - 127.5) / 127.5
    con_img = (con_img.astype(np.float32) - 127.5) / 127.5
    con_img = con_img[:, :, :1]

    color_image = np.expand_dims(color_image, axis=0)
    con_img = np.expand_dims(con_img, axis=0)
    fake = model([color_image, con_img]).numpy()
    fake = fake.reshape((64, 64, 3))
    fake = (fake + 1) / 2.0 + 255.0
    fake = fake.astype(np.uint8)
    fake_img = Image.fromarray(fake)

    output_path = args.output_path
    fake_img.save(output_path)

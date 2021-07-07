import cv2
import argparse
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.keras.preprocessing.image import img_to_array


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models', required=False,
                        help='model path')
    parser.add_argument('--color_input', type=str, default='test_data/000004.png',
                        required=False, help='color image')
    parser.add_argument('--shape_input', type=str, default='test_data/bird.png',
                        required=False, help='shape image')
    parser.add_argument('--output_path', type=str, default='test_data/output.png',
                        required=False, help='test output image path')

    args = parser.parse_args()
    model_path = args.model_path
    model = tf.keras.models.load_model(model_path)

    color_path = args.color_input
    draw_path = args.shape_input

    img = cv2.imread(draw_path)
    canny = cv2.Canny(img, 100, 255)
    contour_img = Image.fromarray(canny)
    contour_img.save('test_data/contour.png')
    # contour_img.save('test_data/edge_result.png', 'png')
    # contour_img.show()

    color_img = Image.open(color_path).convert('RGB')

    color_img = img_to_array(color_img) / 255.0
    contour_img = img_to_array(contour_img) / 255.0

    color_img = (color_img - 0.5) / 0.5
    contour_img = (contour_img - 0.5) / 0.5

    h, w = 64, 64
    color_img = tf.image.resize(color_img, size=(h, w), method=tf.image.ResizeMethod.BICUBIC)
    contour_img = tf.image.resize(contour_img, size=(h, w), method=tf.image.ResizeMethod.BICUBIC)
    contour_img = contour_img[:, :, :1]

    color_img = np.expand_dims(color_img, axis=0)
    contour_img = np.expand_dims(contour_img, axis=0)
    color_img = np.array(color_img)
    contour_img = np.array(contour_img)

    fake = model([color_img, contour_img]).numpy()
    fake = fake * 0.5 + 0.5
    fake = np.clip(fake * 255.0, 0.0, 255.0)
    fake = fake.astype(np.uint8)
    fake = fake[0]
    fake = Image.fromarray(fake)

    output_path = args.output_path
    fake.save(output_path)

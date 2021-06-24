import os
import math
import pickle
import random
import numpy as np
import tensorflow as tf

from PIL import Image

from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.image import img_to_array


class IconGenerator(Sequence):
    def __init__(self, batch_size, dim=64, pad_ratio=8, data_path='./preprocessed_data', shuffle=True):
        self.img_size = (64, 64)
        self.data_path = data_path
        self.contour_dir = os.path.join(self.data_path, 'contour')
        self.img_dir = os.path.join(self.data_path, 'img')

        self.pad_ratio = pad_ratio
        self.min_crop_area = ((pad_ratio + 1) / (pad_ratio + 2)) ** 2
        self.crop_size = int(64 * self.min_crop_area)
        print(self.crop_size)

        print(os.walk(self.contour_dir))
        _, _, self.contour_list = next(os.walk(self.contour_dir))
        _, _, self.image_list = next(os.walk(self.img_dir))

        with open(os.path.join(self.data_path, 'labels.pickle'), 'rb') as fp:
            labels = pickle.load(fp)
        if labels:
            self.labels = labels['labels']
            self.groups = labels['groups']
        else:
            self.labels = []
            self.groups = []

        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle

        self.indexes = np.arange(len(self.contour_list))
        self.size = len(self.indexes)
        self.icon_paths = [os.path.join(self.img_dir, '{:06d}.png'.format(i)) for i in self.indexes]
        self.contour_paths = [os.path.join(self.contour_dir, '{:06d}.png'.format(i)) for i in self.indexes]

        self.scale = (0.84, 0.9)
        self.ratio = (1.0, 1.0)

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.contour_list) / self.batch_size))

    def style_img_processing(self, img):
        prev_size = img.shape
        crop_size = [int(self.min_crop_area * prev_size[0]), prev_size[1], 3]
        img = tf.image.random_crop(img, size=crop_size)
        img = tf.image.resize(img, size=self.img_size, method=tf.image.ResizeMethod.BICUBIC)

        p = random.random()
        if p < 0.5:
            img = tf.image.random_flip_left_right(img, seed=None)

        p = random.random()
        if p < 0.5:
            img = tf.image.random_flip_up_down(img, seed=None)
        return img

    def paired_img_processing(self, img1, img2):
        prev_size1 = img1.shape
        crop_size = [int(self.min_crop_area * prev_size1[0]), prev_size1[1], 3]
        seed = random.randint(1, 1000)

        img1 = tf.image.stateless_random_crop(img1, size=crop_size, seed=[seed, seed])
        img2 = tf.image.stateless_random_crop(img2, size=crop_size, seed=[seed, seed])
        img1 = tf.image.resize(img1, self.img_size, method=tf.image.ResizeMethod.BICUBIC)
        img2 = tf.image.resize(img2, self.img_size, method=tf.image.ResizeMethod.BICUBIC)
        p = random.random()
        if p < 0.5:
            img1 = tf.image.flip_left_right(img1)
            img2 = tf.image.flip_left_right(img2)

        p = random.random()
        if p < 0.5:
            img1 = tf.image.flip_up_down(img1)
            img2 = tf.image.flip_up_down(img2)
        return img1, img2

    @classmethod
    def get_random_size(cls, img, scale, ratio):
        area = img.shape[0] * img.shape[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.shape[0] and h <= img.shape[1]:
                i = random.randint(0, img.shape[1] - h)
                j = random.randint(0, img.shape[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.shape[0] / img.shape[1]
        if in_ratio < min(ratio):
            w = img.shape[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.shape[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.shape[0]
            h = img.shape[1]
        i = (img.shape[1] - h) // 2
        j = (img.shape[0] - w) // 2
        return i, j, h, w

    def __getitem__(self, index):
        s1_arr = []
        s2_arr = []
        s3_arr = []
        contour_arr = []
        for i in range(self.batch_size):

            label = self.labels[index * self.batch_size + i]
            group = self.groups[label]

            # pick the icon in the same color cluster
            idx2 = random.choice(group)
            idx3 = random.choice(self.indexes)

            s1 = Image.open(self.icon_paths[index * self.batch_size + i]).convert('RGB')
            s2 = Image.open(self.icon_paths[idx2]).convert('RGB')
            s3 = Image.open(self.icon_paths[idx3]).convert('RGB')
            contour = Image.open(self.contour_paths[idx3]).convert('RGB')

            s1 = img_to_array(s1) / 255.0
            s2 = img_to_array(s2) / 255.0
            s3 = img_to_array(s3) / 255.0
            contour = img_to_array(contour) / 255.0

            s1 = (s1 - 0.5) / 0.5
            s2 = (s2 - 0.5) / 0.5
            s3 = (s3 - 0.5) / 0.5
            contour = (contour - 0.5) / 0.5

            # s1 = (s1.astype(np.float32) - 127.5) / 127.5
            # s2 = (s2.astype(np.float32) - 127.5) / 127.5
            # s3 = (s3.astype(np.float32) - 127.5) / 127.5
            # contour = (contour.astype(np.float32) - 127.5) / 127.5

            s1 = self.style_img_processing(s1)
            s2 = self.style_img_processing(s2)
            s3, contour = self.paired_img_processing(s3, contour)
            # s1 = np.moveaxis(s1, -1, 0)
            # s2 = np.moveaxis(s2, -1, 0)
            # s3 = np.moveaxis(s3, -1, 0)
            # contour = np.moveaxis(contour, -1, 0)

            s1_arr.append(s1)
            s2_arr.append(s2)
            s3_arr.append(s3)
            contour_arr.append(contour[:, :, :1])

        return np.array(s1_arr), np.array(s2_arr), np.array(s3_arr), np.array(contour_arr)

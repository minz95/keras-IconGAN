import os
import math
import pickle
import random
import numpy as np
import tensorflow as tf

from PIL import Image

from tensorflow.keras.utils import Sequence


class IconGenerator(Sequence):
    def __init__(self, batch_size, dim=128, pad_ratio=8, data_path='./datasets', shuffle=True):
        self.data_path = data_path
        self.contour_dir = os.path.join(self.data_path, 'contour')
        self.img_dir = os.path.join(self.data_path, 'img')

        self.pad_ratio = pad_ratio
        _, _, self.contour_list = next(os.walk(self.contour_dir))
        _, _, self.image_list = next(os.walk(self.img_dir))

        with open(os.path.join(self.data_path, 'labels'), 'rb') as fp:
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

        self.on_epoch_end()

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.floor(len(self.contour_list) / self.batch_size))

    @classmethod
    def img_normalization(cls, img):
        return (img - 0.5) / 0.5

    def style_img_processing(self, img, scale=(0.84, 0.9), ratio=(1.0, 1.0)):
        _, _, h, w = self.get_random_size(img, scale, ratio)
        img = tf.image.resize_with_crop_or_pad(img, h, w, method='bicubic')
        img = tf.image.random_flip_left_right(img, seed=None)
        img = tf.image.random_flip_up_down(img, seed=None)
        return img

    def paired_img_processing(self, img1, img2):
        _, _, h, w = self.get_params(img1, self.scale, self.ratio)

        img1 = tf.image.resize_with_crop_or_pad(img1, h, w, method='bicubic')
        img2 = tf.image.resize_with_crop_or_pad(img2, h, w, method='bicubic')

        p = random.random()
        if p < 0.5:
            img1 = tf.image.flip_left_right(img1, seed=None)
            img2 = tf.image.flip_left_right(img2, seed=None)

        p = random.random()
        if p < 0.5:
            img1 = tf.image.flip_up_down(img1)
            img2 = tf.image.flip_up_down(img2)
        return img1, img2

    @classmethod
    def get_random_size(cls, img, scale, ratio):
        area = img.size[0] * img.size[1]

        for attempt in range(10):
            target_area = random.uniform(*scale) * area
            log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
            aspect_ratio = math.exp(random.uniform(*log_ratio))

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= img.size[0] and h <= img.size[1]:
                i = random.randint(0, img.size[1] - h)
                j = random.randint(0, img.size[0] - w)
                return i, j, h, w

        # Fallback to central crop
        in_ratio = img.size[0] / img.size[1]
        if in_ratio < min(ratio):
            w = img.size[0]
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = img.size[1]
            w = int(round(h * max(ratio)))
        else:  # whole image
            w = img.size[0]
            h = img.size[1]
        i = (img.size[1] - h) // 2
        j = (img.size[0] - w) // 2
        return i, j, h, w

    def __getitem__(self, index):
        label = self.labels[index]
        group = self.groups[label]

        # pick the icon in the same color cluster
        idx2 = random.choice(group)
        idx3 = random.choice(self.indexes)

        s1 = Image.open(self.icon_paths[index]).convert('RGB')
        s2 = Image.open(self.icon_paths[idx2]).convert('RGB')
        s3 = Image.open(self.icon_paths[idx3]).convert('RGB')
        contour = Image.open(self.contour_paths[idx3]).convert('RGB')

        s1 = self.style_img_processing(s1)
        s2 = self.style_img_processing(s2)
        s3, contour = self.paired_img_processing(s3, contour)

        s1 = self.img_normalization(s1)
        s2 = self.img_normalization(s2)
        s3 = self.img_normalization(s3)
        contour = self.img_normalization(contour)

        return s1, s2, s3, contour[:1, :, :]

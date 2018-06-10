from __future__ import division
import cv2
import numpy as np
from numpy import random
import math
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
__all__ = ['Compose','ResizeImg',"Normalize","RandomResizedCrop","RandomHflip", 'ExpandBorder','deNormalize']

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *args):
        for t in self.transforms:
            args = t(*args)
        return args
def fixed_crop(src, x0, y0, w, h, size=None):
    out = src[y0:y0 + h, x0:x0 + w]
    if size is not None and (w, h) != size:
        out = cv2.resize(out, (size[0], size[1]), interpolation=cv2.INTER_CUBIC)
    return out
def scale_down(src_size, size):
    w, h = size
    sw, sh = src_size
    if sh < h:
        w, h = float(w * sh) / h, sh
    if sw < w:
        w, h = sw, float(h * sw) / w
    return int(w), int(h)

def center_crop(src, size):
    h, w = src.shape[0:2]
    new_w, new_h = scale_down((w, h), size)

    x0 = int((w - new_w) / 2)
    y0 = int((h - new_h) / 2)

    out = fixed_crop(src, x0, y0, new_w, new_h, size)
    return out

class ResizeImg(object):
    def __init__(self, size,inter=cv2.INTER_LINEAR):
        self.size = size
        self.inter = inter

    def __call__(self, image, mask):
        return cv2.resize(image, (self.size[1], self.size[0]), interpolation=self.inter), \
               cv2.resize(mask, (self.size[1], self.size[0]), interpolation=self.inter)

class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.25, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self,img, mask):
        h, w, _ = img.shape
        area = h * w
        for attempt in range(10):
            target_area = random.uniform(self.scale[0], self.scale[1]) * area
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            new_w = int(round(math.sqrt(target_area * aspect_ratio)))
            new_h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                new_h, new_w = new_w, new_h

            if new_w < w and new_h < h:
                x0 = random.randint(0, w - new_w)
                y0 = random.randint(0, h - new_h)
                out_img = fixed_crop(img, x0, y0, new_w, new_h, self.size)
                out_mask = fixed_crop(mask, x0, y0, new_w, new_h, self.size)

                return out_img, out_mask
        # Fallback
        return center_crop(img, self.size), center_crop(mask, self.size)
class RandomHflip(object):
    def __call__(self, image, mask):
        if random.randint(2):
            return cv2.flip(image, 1), cv2.flip(mask, 1),
        else:
            return image, mask


class ExpandBorder(object):
    def __init__(self, mode='constant', value=255, size=(336,336), resize=False):
        self.mode = mode
        self.value = value
        self.resize = resize
        self.size = size

    def __call__(self, image, mask):

        h, w, _ = image.shape
        if h > w:
            pad1 = (h-w)//2
            pad2 = h - w - pad1
            if self.mode == 'constant':
                image = np.pad(image, ((0, 0), (pad1, pad2), (0, 0)),
                               self.mode, constant_values=self.value)
                mask = np.pad(mask, ((0, 0), (pad1, pad2)),
                               self.mode, constant_values=0)
            else:
                image = np.pad(image,((0,0), (pad1, pad2),(0,0)), self.mode)
                mask = np.pad(mask, ((0, 0), (pad1, pad2)),
                               self.mode, constant_values=0)
        elif h < w:
            pad1 = (w-h)//2
            pad2 = w-h - pad1
            if self.mode == 'constant':
                image = np.pad(image, ((pad1, pad2),(0, 0), (0, 0)),
                               self.mode,constant_values=self.value)
                mask = np.pad(mask, ((pad1, pad2),(0, 0)),
                               self.mode,constant_values=self.value)
            else:
                image = np.pad(image, ((pad1, pad2), (0, 0), (0, 0)),self.mode)
                mask = np.pad(mask, ((pad1, pad2),(0, 0)),
                               self.mode,constant_values=self.value)

        if self.resize:
            image = cv2.resize(image, (self.size[0], self.size[0]),interpolation=cv2.INTER_LINEAR)
            mask = cv2.resize(mask, (self.size[0], self.size[0]), interpolation=cv2.INTER_LINEAR)

        return image, mask

class Normalize(object):
    def __init__(self,mean, std):
        '''
        :param mean: RGB order
        :param std:  RGB order
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        '''
        self.mean = np.array(mean).reshape(1,1,3)
        self.std = np.array(std).reshape(1,1,3)
    def __call__(self, img, lm):
        '''
        :param image:  (H,W,3)  RGB
        :return:
        '''
        return (img/ 255. - self.mean) / self.std, lm


class deNormalize(object):
    def __init__(self,mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        '''
        :param mean: RGB order
        :param std:  RGB order
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        '''
        self.mean = np.array(mean).reshape(1,1,3)
        self.std = np.array(std).reshape(1,1,3)
    def __call__(self, img):
        '''
        :param image:  (H,W,3)  RGB
        :return:
        '''
        # return (img/ 255. - self.mean) / self.std
        return (255*(img*self.std + self.mean)).clip(0,255).astype(np.uint8)
from __future__ import division
import cv2
import numpy as np
from numpy import random
import math
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
# __all__ = ['Compose','ResizeImg',"Normalize","RandomResizedCrop","RandomHflip", 'ExpandBorder','deNormalize']

def rotate_bound(image, angle, borderValue=0, borderMode=None):
    # grab the dimensions of the image and then determine the
    # center
    h, w = image.shape[:2]

    (cX, cY) = (w // 2, h // 2)

    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    if borderMode is None:
        rotated = cv2.warpAffine(image, M, (nW, nH), borderValue=borderValue, borderMode=0)
    else:
        rotated = cv2.warpAffine(image, M, (nW, nH),borderValue=borderValue, borderMode=borderMode)

    return rotated


def rotate_nobound(image, angle,borderValue=0, borderMode=None):
    (h, w) = image.shape[:2]


    # if the center is None, initialize it as the center of
    # the image
    center = (w // 2, h // 2)

    # perform the rotation
    M = cv2.getRotationMatrix2D(center, angle, 1.)
    if borderMode is None:
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=borderValue, borderMode=0)
    else:
        rotated = cv2.warpAffine(image, M, (w, h), borderValue=borderValue, borderMode=borderMode)

    return rotated


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

class RandomSelect(object):
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, image, mask):
        idx = random.randint(len(self.transforms))
        # print('  %s %s' % (self.transforms[idx].__class__.__name__, self.transforms[idx].__dict__))
        return self.transforms[idx](image, mask)


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



class RandomRotate(object):
    def __init__(self, angles, bound, borderMode='REFLECT', borderValue=None):
        self.angles = angles
        self.bound = bound
        if borderMode=='REFLECT':
            self.borderMode = cv2.BORDER_REFLECT
        else:
            self.borderMode = borderMode
        self.borderValue = borderValue

    def __call__(self,img, mask):
        angle = np.random.uniform(self.angles[0], self.angles[1])
        if isinstance(self.bound, str) and self.bound.lower() == 'random':
            bound = random.randint(2)
        else:
            bound = self.bound

        if bound:
            img = rotate_bound(img, angle, borderMode=self.borderMode, borderValue=self.borderValue)
            mask = rotate_bound(mask, angle, borderValue=0)
        else:
            img = rotate_nobound(img, angle, borderMode=self.borderMode, borderValue=self.borderValue)
            mask = rotate_nobound(mask, angle,borderValue=0)
        return img, mask


class RandomBrightness(object):
    def __init__(self, delta=10):
        assert delta >= 0
        assert delta <= 255
        self.delta = delta

    def __call__(self, image, mask):
        delta = random.uniform(-self.delta, self.delta)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        image[:,:,2] += delta
        image = image.clip(0, 255).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

        return image, mask


class RandomSmall(object):
    def __init__(self, ratio=0.1):
        self.ratio = ratio

    def __call__(self, image, mask):
        h,w = image.shape[0:2]

        ratio = random.uniform(0, self.ratio)
        dw = max(1,int(w*ratio))
        dh = max(1,int(h*ratio))


        w_shift = random.randint(-dw, dw)
        h_shift = random.randint(-dh, dh)

        pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        pts2 = np.float32([[dw, dh],
                         [w-dw, dh],
                         [dw, h-dh],
                         [w-dw, h-dh]])
        pts2[:,0] += w_shift
        pts2[:,1] += h_shift


        M = cv2.getPerspectiveTransform(pts1, pts2)
        image = cv2.warpPerspective(image, M, (w,h), borderMode=cv2.BORDER_REFLECT)
        mask = cv2.warpPerspective(mask, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return image, mask



class Normalize(object):
    def __init__(self,mean=None, std=None):
        '''
        :param mean: RGB order
        :param std:  RGB order
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        '''
        if mean is not None:
            self.mean = np.array(mean).reshape(1,1,3)
        else:
            self.mean = None

        if std is not None:
            self.std = np.array(std).reshape(1,1,3)
        else:
            self.std = None
    def __call__(self, img, mask):
        '''
        :param image:  (H,W,3)  RGB
        :return:
        '''
        if self.mean is None and self.std is None:
            return  img / 255., mask
        elif self.mean is not None and self.std is not None:
            return  (img / 255. - self.mean) / self.std, mask
        elif self.mean is None:
            return img / 255. / self.std, mask
        else:
            return (img / 255. - self.mean) , mask




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

if __name__ == '__main__':
    from matplotlib import pyplot as plt
    import os
    from glob import glob
    import pandas as pd
    from Sdataset import Sdata
    import logging


    class valAug(object):
        def __init__(self, size=(224, 224)):
            self.augment = Compose([
                RandomSelect([
                    RandomSmall(ratio=0.1),
                    RandomRotate(angles=(-20, 20), bound='Random'),
                    RandomResizedCrop(size=size),
                ]),
                RandomBrightness(delta=30),
                # ExpandBorder(value=0),
                ResizeImg(size=size),
                Normalize(mean=None, std=None)
            ])

        def __call__(self, *args):
            return self.augment(*args)

    # prepare data
    eval_data_root = '/media/hszc/data1/seg_data/diy_seg'
    img_paths = sorted(glob(os.path.join(eval_data_root, 'seg_img/*.jpg')))
    mask_paths = ['/'.join(x.split('/')[:-2]) + '/seg_mask/' + x.split('/')[-1].replace('.jpg', '.png') for x in
                  img_paths]
    anno = pd.DataFrame({'image_paths': img_paths, 'mask_paths': mask_paths})
    print anno.info()
    data_set = {}
    data_set['val'] = Sdata(anno_pd=anno, transforms=valAug(size=(448, 448)))

    img, mask = data_set['val'][114]
    img = img.astype(np.uint8)
    print img.shape, mask.shape, img.mean()

    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask[:,:,0])
    plt.show()

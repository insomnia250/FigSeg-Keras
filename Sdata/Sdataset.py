# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt

class Sdata(data.Dataset):
    def __init__(self, anno_pd, transforms, dis=False):
        anno_pd.index = range(anno_pd.shape[0])
        self.image_paths = anno_pd['image_paths'].tolist()
        self.mask_paths = anno_pd['mask_paths'].tolist()
        self.mask_teacher_paths = anno_pd['mask_teacher_paths'].tolist()
        self.transforms = transforms
        self.dis = dis
        # deal with label

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self.image_paths[item]), cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
        mask = cv2.imread(self.mask_paths[item], cv2.IMREAD_GRAYSCALE)
        h, w = mask.shape
        if self.dis:
            mask_teacher = np.load(self.mask_teacher_paths[item])
            mask_teacher = cv2.resize(mask_teacher, (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            mask_teacher = None

        mask[mask==2] = 1

        if self.transforms:
            img, mask, mask_teacher = self.transforms(img, mask, mask_teacher)

        if not self.dis:
            return img.astype(np.float32), mask[:, :, np.newaxis].astype(np.float32)
        else:
            return img.astype(np.float32), mask[:,:,np.newaxis].astype(np.float32),\
                    mask_teacher[:,:,np.newaxis].astype(np.float32)


def collate_fn(batch):
    imgs = []
    masks = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])

    return np.stack(imgs, 0), \
           np.stack(masks, 0)

def collate_fn2(batch):
    imgs = []
    masks = []
    masks_teacher = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])
        masks_teacher.append(sample[2])

    return np.stack(imgs, 0), \
           np.stack(masks, 0), \
           np.stack(masks_teacher, 0)


# if __name__ == '__main__':

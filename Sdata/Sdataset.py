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
    def __init__(self, anno_pd, transforms):
        anno_pd.index = range(anno_pd.shape[0])
        self.img_path = anno_pd['image_paths']
        self.mask_path = anno_pd['mask_paths']
        self.transforms = transforms
        # deal with label

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self.img_path[item]),  cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
        mask = cv2.imread(self.mask_path[item],cv2.IMREAD_GRAYSCALE)
        mask[mask==2] = 1

        if self.transforms:
            img, mask = self.transforms(img, mask)
        return img.astype(np.float32), mask[:,:,np.newaxis].astype(np.float32)

def collate_fn(batch):
    imgs = []
    masks = []

    for sample in batch:
        imgs.append(sample[0])
        masks.append(sample[1])

    return np.stack(imgs, 0), \
           np.stack(masks, 0)


if __name__ == '__main__':
    from utils.preprocessing import gen_dataloader
    img_root = '/media/hszc/data1/seg_data/'

    data_set, data_loader = gen_dataloader(img_root, validation_split=0.1, train_bs=8, val_bs=4)
    print len(data_set['train']), len(data_set['val'])

    img, mask = data_set['train'][13]
    img = img.astype(np.uint8)
    print img.shape
    print mask.max()
    print mask.shape

    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.figure()
    plt.imshow(mask[:,:,0])
    plt.show()
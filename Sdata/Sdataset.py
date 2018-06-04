# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
cv2.setNumThreads(0)

class LMdata(data.Dataset):
    def __init__(self, anno_pd, transforms):
        anno_pd.index = range(anno_pd.shape[0])
        self.img_path = anno_pd['image_id']
        self.mask_path = anno_pd['image_mask']
        self.transforms = transforms
        # deal with label

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, item):
        img = cv2.cvtColor(cv2.imread(self.img_path[item]),  cv2.COLOR_BGR2RGB)  # [h,w,3]  RGB
        mask = cv2.imread(self.mask_path[item],cv2.IMREAD_GRAYSCALE)

        if self.transforms:
            img, mask = self.transforms(img, mask)
        return img, mask[:,:,np.newaxis]

if __name__ == '__main__':
    from utils.preprocessing import gen_dataloader
    img_root = '/media/gserver/data/seg_data/'

    data_set, data_loader = gen_dataloader(img_root, validation_split=0.1, train_bs=8, val_bs=4)
    print len(data_set['train']), len(data_set['val'])

    img, mask = data_set['train'][0]
    img = img.astype(np.uint8)
    print img.shape
    print img
    print mask.max()

    from matplotlib import pyplot as plt
    plt.imshow(img)
    plt.figure()
    plt.imshow(mask)
    plt.show()
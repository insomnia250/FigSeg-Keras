# coding=utf8
from __future__ import division
import os
import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import cv2
from collections import OrderedDict
from itertools import chain
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

attr2idx_map = {'coat_length_labels':0,
            'collar_design_labels':1,
            'lapel_design_labels':2,
            'neck_design_labels':3,
            'neckline_design_labels':4,
            'pant_length_labels':5,
            'skirt_length_labels':6,
            'sleeve_length_labels':7}

idx2attr_map = OrderedDict({0:'coat_length_labels',
                            1:'collar_design_labels',
                            2:'lapel_design_labels',
                            3:'neck_design_labels',
                            4:'neckline_design_labels',
                            5:'pant_length_labels',
                            6:'skirt_length_labels',
                            7:'sleeve_length_labels'})

attr2length_map = { 0: 8,
                    1:5,
                    2:5,
                    3:5,
                    4: 10,
                    5:6,
                    6:6,
                    7:9}

attr2catidx_map = { 0: [0,8],
                    1:[8,13],
                    2:[13,18],
                    3:[18,23],
                    4: [23,33],
                    5:[33,39],
                    6:[39,45],
                    7:[45,54]}

label_map = {'n':0., 'm':0.5, 'y':1.}


class FSdata(data.Dataset):
    def __init__(self, root_path, anno_pd, transforms=None, label_map=label_map, select=range(0,8)):
        self.root_path = root_path
        self.paths = anno_pd['ImageName'].tolist()
        self.attrs = anno_pd['AttrKey'].tolist()
        self.labels = anno_pd['AttrValues'].tolist()
        self.transforms = transforms
        self.label_map = label_map
        self.select = select
        self.catlen = 0
        self.catidx_map = {}
        for attr_idx in select:
            self.catidx_map[attr_idx] = [self.catlen, self.catlen+attr2length_map[attr_idx]]
            self.catlen += attr2length_map[attr_idx]

        self.num_classes = [attr2length_map[x] for x in select]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        img_path = os.path.join(self.root_path, self.paths[item])

        img = cv2.cvtColor(cv2.imread(img_path),cv2.COLOR_BGR2RGB)   # [h,w,3]  RGB

        attr = self.attrs[item]
        attr_idx = attr2idx_map[attr]

        label_str = [s for s in self.labels[item]]
        label = np.array([self.label_map[s] for s in self.labels[item]])

        label = label / label.sum()

        attr_mask = np.zeros(self.catlen, dtype=np.int32)
        attr_mask[self.catidx_map[attr_idx][0]:self.catidx_map[attr_idx][1]] = 1

        cat_label = np.zeros(self.catlen, dtype=np.float32)
        cat_label[self.catidx_map[attr_idx][0]:self.catidx_map[attr_idx][1]] = label

        cat_label_str = ['n']*self.catlen
        cat_label_str[self.catidx_map[attr_idx][0]:self.catidx_map[attr_idx][1]] = label_str

        if self.transforms is not None:
            img, attr_idx = self.transforms(img, attr_idx)

        return img, attr_idx, attr_mask, cat_label, cat_label_str

def collate_fn(batch):
    imgs = []
    attr_idx = []
    attr_mask = []
    cat_label = []
    cat_label_str = []

    for sample in batch:
        imgs.append(sample[0])
        attr_idx.append(sample[1])
        attr_mask.append(sample[2])
        cat_label.append(sample[3])
        cat_label_str.append(sample[4])

    return np.stack(imgs, 0), \
           attr_idx, \
           np.stack(attr_mask, 0), \
           np.stack(cat_label, 0), \
           cat_label_str

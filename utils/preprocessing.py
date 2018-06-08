from copy import copy
import torch
import numpy as np
from torchvision import datasets, transforms
from Sdata.Saug import *
import torch.utils.data as torchdata
from sklearn.model_selection import train_test_split
import logging
from Sdata.Sdataset import *
import glob

class trainAug(object):
    def __init__(self, size=(224,224)):
        self.augment = Compose([
            RandomResizedCrop(size=size),
            # ResizeImg(size=(224, 224)),
            # ExpandBorder(mode='constant', value=255, size=size, resize=False),
            RandomHflip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args):
        return self.augment(*args)

class valAug(object):
    def __init__(self,size=(224,224)):
        self.augment = Compose([
            # ExpandBorder(mode='constant', value=255, size=size, resize=False),
            ResizeImg(size=size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args):
        return self.augment(*args)


def gen_dataloader(img_root,size=(224,224), validation_split=0.1,train_bs =8,val_bs=4):
    annotation = pd.DataFrame(columns=["image_paths","mask_paths"])
    imglist = sorted(list(glob.glob(os.path.join(img_root, "seg_img","*.jpg"))))

    annotation['image_paths']=imglist
    annotation["mask_paths"]=annotation['image_paths'].apply(lambda x :os.path.join(img_root, "seg_mask",x.split("/")[-1].split(".")[0]+".png"))
    train_pd, val_pd = train_test_split(annotation, test_size=validation_split, random_state=42)
    train_pd.index = range(train_pd.shape[0])
    val_pd.index = range(val_pd.shape[0])
    data_set = {}
    data_set['train'] = Sdata(train_pd, trainAug(size=size))
    data_set['val'] = Sdata(val_pd, valAug(size=size))
    data_loader = {}
    data_loader['train'] = torchdata.DataLoader(data_set['train'], train_bs, num_workers=8,
                                                shuffle=True, pin_memory=True,collate_fn=collate_fn)
    data_loader['val'] = torchdata.DataLoader(data_set['val'], val_bs, num_workers=4,
                                              shuffle=False, pin_memory=True,collate_fn=collate_fn)
    return data_set,data_loader
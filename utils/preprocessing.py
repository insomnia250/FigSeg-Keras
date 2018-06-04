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
    def __init__(self):
        self.augment = Compose([
            RandomResizedCrop(size=(385, 385)),
            # ResizeImg(size=(224, 224)),
            RandomHflip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args):
        return self.augment(*args)

class valAug(object):
    def __init__(self):
        self.augment = Compose([
            ResizeImg(size=(385, 385)),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args):
        return self.augment(*args)


def gen_dataloader(img_root,validation_split=0.1,train_bs =8,val_bs=4):
    annotation = pd.DataFrame(columns=["image_id","image_mask"])
    imglist = glob.glob(os.path.join(img_root, "seg_img","*.jpg"))

    annotation['image_id']=pd.Series(imglist)
    annotation["image_mask"]=annotation['image_id'].apply(lambda x :os.path.join(img_root, "seg_mask",x.split("/")[-1].split(".")[0]+".png"))
    train_pd, val_pd = train_test_split(annotation, test_size=validation_split, random_state=42)
    data_set = {}
    data_set['train'] = LMdata(train_pd, trainAug())
    data_set['val'] = LMdata(val_pd, trainAug())
    data_loader = {}
    data_loader['train'] = torchdata.DataLoader(data_set['train'], train_bs, num_workers=8,
                                                shuffle=True, pin_memory=False, collate_fn=collate_fn)
    data_loader['val'] = torchdata.DataLoader(data_set['val'], val_bs, num_workers=4,
                                              shuffle=False, pin_memory=False, collate_fn=collate_fn)
    return data_set,data_loader
import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import keras
from glob import glob
from Sdata.Sdataset import Sdata, collate_fn
from Sdata.Saug import *
import pandas as pd
import torch.utils.data as torchdata
from utils.predicting import predict,predict_vis
import cv2
from matplotlib import pyplot as plt
from keras import callbacks, optimizers
from keras.utils.generic_utils import CustomObjectScope
import logging
from keras import losses
from nets.MobileUnet import MobileUNet
from nets.MobileUnet_small import MobileUNet as MobileUNet_s
from loss import dice_coef_loss, dice_coef, recall, precision
from utils.logs import *
from utils.train import train
from nets.MobileUnet import custom_objects
from nets.FigSeg import FigSeg
from keras.utils import multi_gpu_model



class trainAug(object):
    def __init__(self, size=(224,224)):
        self.augment = Compose([
            RandomResizedCrop(size=size),
            # ResizeImg(size=(224, 224)),
            RandomHflip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args):
        return self.augment(*args)

class valAug(object):
    def __init__(self,size=(224,224)):
        self.augment = Compose([
            ResizeImg(size=size),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, *args):
        return self.augment(*args)



eval_data_root = '/media/hszc/data1/seg_data/diy_seg'
resume = '/home/hszc/zhangchi/FigSeg-Keras/saved_models/MobileUnet_s-448_reg_adam/weights-[153-94500]-[0.9123].h5'
bs = 4


# prepare data
img_paths = sorted(glob(os.path.join(eval_data_root, 'seg_img/*.jpg')))
mask_paths = ['/'.join(x.split('/')[:-2])+'/seg_mask/'+x.split('/')[-1].replace('.jpg', '.png') for x in img_paths]
anno = pd.DataFrame({'image_paths':img_paths, 'mask_paths':mask_paths})
print anno.info()
data_set = {}
data_set['val'] = Sdata(anno_pd=anno, transforms=valAug(size=(448,448)))

data_loader = {}
data_loader['val'] = torchdata.DataLoader(data_set['val'], bs, num_workers=8,
                                                shuffle=False, pin_memory=True,collate_fn=collate_fn)




# prepare model
model = MobileUNet_s(input_shape=(448, 448, 3),
                   alpha=1,
                   alpha_up=0.25)

parallel_model = multi_gpu_model(model,gpus=2)

if resume:
    # with CustomObjectScope(custom_objects()):
    #     model = keras.models.load_model(resume)
    parallel_model.load_weights(resume)
    # model.set_weights(parallel_model.get_weights())
    logging.info('resumed model from %s'%resume)

model.summary()

#

mIOU = predict(model, data_set['val'], data_loader['val'], verbose=True)
print mIOU
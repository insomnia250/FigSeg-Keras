import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from loss import dice_coef_loss, dice_coef, recall, precision
from learning_rate import create_lr_schedule
from utils.logs import *
from utils.train import train
from nets.MobileUnet import custom_objects
from nets.FigSeg import FigSeg
from nets.DeeplabV3 import Deeplabv3




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



eval_data_root = '/media/gserver/data/seg_data/diy_data'
resume = './saved_models/deeplab_moblie_adam/weights-21000-[0.9235].h5'
bs = 32


# prepare data
img_paths = sorted(glob(os.path.join(eval_data_root, 'seg_img/*.jpg')))
mask_paths = ['/'.join(x.split('/')[:-2])+'/seg_mask/'+x.split('/')[-1].replace('.jpg', '.png') for x in img_paths]
anno = pd.DataFrame({'image_paths':img_paths, 'mask_paths':mask_paths})

data_set = {}
data_set['val'] = Sdata(anno_pd=anno, transforms=valAug(size=(224,224)))

data_loader = {}
data_loader['val'] = torchdata.DataLoader(data_set['val'], bs, num_workers=8,
                                                shuffle=False, pin_memory=True,collate_fn=collate_fn)




# prepare model
model = Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(224, 224, 3), classes=1, backbone='mobilenetv2', OS=16, alpha=1.)

model.compile(
    # optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
    optimizer=optimizers.Adam(lr=1e-4),
    # optimizer=optimizers.RMSprop(),
    loss={'proba': losses.binary_crossentropy},
    metrics={'proba': [
        recall,
        precision,
        'acc'
    ]},
)

out_layers = ['proba']
model.metrics_tensors += [layer.output for layer in model.layers if layer.name in out_layers]

if resume:
    # with CustomObjectScope(custom_objects()):
    #     model = keras.models.load_model(resume)
    model.load_weights(resume)
    logging.info('resumed model from %s'%resume)

model.summary()

#

mIOU = predict_vis(model, data_set['val'], data_loader['val'], verbose=True)
print mIOU
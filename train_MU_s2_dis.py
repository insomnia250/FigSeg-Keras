import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import keras
from keras.utils import multi_gpu_model
from keras import callbacks, optimizers
from keras.utils.generic_utils import CustomObjectScope
import logging
from keras import losses
from nets.MobileUnet import MobileUNet
from nets.MobileUnet_small import MobileUNet as MobileUNet_s
from nets_dis.MobileUnet_small2_dis import MobileUNet as MobileUNet_s2_dis
from loss import dice_coef_loss, dice_coef, recall, precision
from utils.logs import *
from utils.train_dis import train
from utils.preprocessing import get_train_val, gen_dataloader
from Sdata.Saug import *
from keras import backend as K
from functools import partial

class trainAug(object):
    def __init__(self, size=(256, 256)):
        self.augment = Compose([
            RandomSelect([
                RandomSmall(ratio=0.1),
                RandomRotate(angles=(-20, 20), bound='Random'),
                RandomResizedCrop(size=size),
            ]),
            RandomBrightness(delta=30),
            ResizeImg(size=size),
            RandomHflip(),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)


class valAug(object):
    def __init__(self,size=(256,256)):
        self.augment = Compose([
            ResizeImg(size=size),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)

train_root = '/media/hszc/data1/seg_data'
val_root = '/media/hszc/data1/seg_data/diy_seg'
img_shape = (256,256)
save_dir = './saved_models/MUs2(4-2 0p5)_256-dis/'
alpha = 0.95
T = 20
bs = 8
do_para = False
resume = './saved_models/MUs2(4-2 0p5)_256-dis/weights-[0-0]-[0.4009].h5'




if not os.path.exists(save_dir):
    os.makedirs(save_dir)

logfile = '%s/trainlog.log' % save_dir
trainlog(logfile)

# prepare data
train_pd, _ = get_train_val(train_root, test_size=0.0, dis='pred_mask(T20)')
_, val_pd = get_train_val(val_root, test_size=1.0)

print train_pd.info()
print val_pd.info()


data_set,data_loader = gen_dataloader(train_pd, val_pd, trainAug(), valAug(), train_bs = bs, val_bs=2,
                                      dis=True)

# logging info of dataset
logging.info(train_pd.shape)
logging.info(val_pd.shape)
logging.info('train augment:')
for item in data_set['train'].transforms.augment.transforms:
    logging.info('  %s %s' % (item.__class__.__name__, item.__dict__))

logging.info('val augment:')
for item in data_set['val'].transforms.augment.transforms:
    logging.info('  %s %s' % (item.__class__.__name__, item.__dict__))

img_height = img_shape[0]
img_width = img_shape[1]

# x = np.random.rand(10,448,448,3)

def CrossEntropyKD_with_T(y_true, y_pred, T, from_logits, weights):
    '''
    :param y_true:  proba (already smoothed by temperature )
    :param y_pred:  logits (will be smoothed by temperature and then compute BCE loss with y_true)
    :return: loss
    '''
    return weights * K.mean(K.binary_crossentropy(y_true, y_pred/T, from_logits=from_logits), axis=-1)

CrossEntropyKD1 = partial(CrossEntropyKD_with_T, T=10, from_logits=True, weights=alpha*T*T)  # alpha*T*T
CrossEntropyKD2 = partial(CrossEntropyKD_with_T, T=1, from_logits=False, weights=1-alpha)   # 1-alpha

model = MobileUNet_s2_dis(input_shape=(img_height, img_width, 3),
                   alpha=0.5,
                   alpha_up=0.25)

model.summary(print_fn=logging.info)

print model.output[0].name
print model.output[1].name

if do_para:
    parallel_model = multi_gpu_model(model,gpus=2)
    parallel_model.compile(
        # optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
        optimizer=optimizers.Adam(lr=1e-4),
        # optimizer=optimizers.RMSprop(),
        loss={'logits': CrossEntropyKD1,
              'proba': CrossEntropyKD2},
        metrics={'proba': [
            recall,
            precision,
            'acc'
        ]},
    )

    out_layers = ['proba']
    parallel_model.metrics_tensors += [layer.output for layer in model.layers if layer.name in out_layers]
else:
    parallel_model = None
    model.compile(
        # optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
        optimizer=optimizers.Adam(lr=1e-4),
        # optimizer=optimizers.RMSprop(),
        loss={'logits': CrossEntropyKD1,
              'proba': CrossEntropyKD2},
        metrics={'proba': [
            recall,
            precision,
            'acc'
        ]},
    )
    out_layers = ['proba']
    model.metrics_tensors += [layer.output for layer in model.layers if layer.name in out_layers]
print [x.name for x in model.metrics_tensors]
print model.metrics_names
# [logits_loss, proba_loss, recall, precision, acc]


if resume:
    model.load_weights(resume)
    logging.info('resumed model from %s'%resume)


def lr_scheduler(epoch,base_lr=1e-4):
    if epoch < 100:
        return 1*base_lr
    elif epoch < 170:
        return 0.5*base_lr
    else:
        return 0.01*base_lr


train(model,
      parallel_model,
      epoch_num=250,
      start_epoch=0,
      lr_scheduler=lr_scheduler,
      data_set=data_set,
      data_loader=data_loader,
      save_dir='%s'%save_dir,
      print_inter=200,
      val_inter=3500
      )

import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

import keras

from keras import callbacks, optimizers
from keras.utils.generic_utils import CustomObjectScope
from utils.preprocessing import *
import logging
from keras import losses
from nets.MobileUnet import MobileUNet
from loss import dice_coef_loss, dice_coef, recall, precision
from learning_rate import create_lr_schedule
from utils.logs import *
from utils.train import train
from nets.MobileUnet import custom_objects

img_root = '/media/hszc/data1/seg_data'
img_shape = (224,224)
save_dir = './saved_models/test/'
bs = 24

resume = '/home/gserver/zhangchi/FigSeg-Keras/saved_models/test/model-80000-[0.8121].pth'
resume = '/home/hszc/zhangchi/FigSeg-Keras/saved_models/test/weights-500-[0.2007].h5'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#
data_set, data_loader = gen_dataloader(img_root=img_root, size=(224,224),train_bs=bs, val_bs=4)

print len(data_set['train'])
print len(data_set['val'])
img_height = img_shape[0]
img_width = img_shape[1]
lr_base = 0.01 * (float(8) / 16)


x = np.random.rand(10,224,224,3)

logfile = '%s/trainlog.log' % save_dir
trainlog(logfile)



model = MobileUNet(input_shape=(img_height, img_width, 3),
                   alpha=1,
                   alpha_up=0.25)
model.compile(
    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
    # optimizer=Adam(lr=0.001),
    # optimizer=optimizers.RMSprop(),
    loss={'proba': losses.binary_crossentropy},
    metrics={'proba': [
        recall,
        precision,
    ]},
)

out_layers = ['proba']
model.metrics_tensors += [layer.output for layer in model.layers if layer.name in out_layers]

if resume:
    # with CustomObjectScope(custom_objects()):
    #     model = keras.models.load_model(resume)
    model.load_weights(resume)
    logging.info('resumed model from %s'%resume)


# learning scheduler
step1_bs_rate = 25. / 24.
step2_bs_rate = 50. / 24.
step3_bs_rate = 75. / 24.
steps_bs_rate = 100. / 24.
step1 = int(bs * step1_bs_rate)
step2 = int(bs * step2_bs_rate)
step3 = int(bs * step3_bs_rate)
steps = int(bs * steps_bs_rate)
logging.info('lr steps1: %d' % step1)
logging.info('lr steps2: %d' % step2)
logging.info('lr steps3: %d' % step3)
logging.info('total steps: %d' % steps)


def lr_scheduler(epoch,base_lr=1e-4):
    if epoch < step1:
        return 1*base_lr
    elif epoch < step2:
        return 0.1*base_lr
    elif epoch < step3:
        return 0.05*base_lr
    else:
        return 0.01*base_lr

train(model,
      epoch_num=100,
      start_epoch=0,
      lr_scheduler=lr_scheduler,
      data_set=data_set,
      data_loader=data_loader,
      save_dir='%s'%save_dir,
      print_inter=200,
      val_inter=3500
      )

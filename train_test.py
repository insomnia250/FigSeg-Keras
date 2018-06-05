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

img_root = '/media/gserver/data/seg_data'
img_shape = (224,224)
save_dir = './saved_models/test/'

resume = '/home/gserver/zhangchi/FigSeg-Keras/saved_models/test/model-80000-[0.8121].pth'
# resume=''
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
#
data_set, data_loader = gen_dataloader(img_root=img_root, size=(224,224),train_bs=8, val_bs=4)

print len(data_set['train'])
print len(data_set['val'])
img_height = img_shape[0]
img_width = img_shape[1]
lr_base = 0.01 * (float(8) / 16)


x = np.random.rand(10,224,224,3)

logfile = '%s/trainlog.log' % save_dir
trainlog(logfile)


if resume:
    with CustomObjectScope(custom_objects()):
        model = keras.models.load_model(resume)
    logging.info('resumed model from %s'%resume)
else:

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
# print model.metrics_tensors
#
# # model.summary()
#
# mask = np.random.randint(0,2,size=(10,224,224,1))
# out = model.train_on_batch(x,  mask)
# print out
# print model.metrics_names
#
# output = model.predict_on_batch(x)
# print output.shape
# print len(output)

train(model,
      epoch_num=2000,
      start_epoch=10,
      data_set=data_set,
      data_loader=data_loader,
      save_dir='%s'%save_dir,
      print_inter=200,
      val_inter=4000
      )

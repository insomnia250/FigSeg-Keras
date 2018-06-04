import os
import time,random
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import callbacks, optimizers
from utils.logs import trainlog
from utils.preprocessing import *
from FSdata.FSaug import *
import logging
from keras import losses
from nets.MobileUnet import MobileUNet
from loss import dice_coef_loss, dice_coef, recall, precision
from learning_rate import create_lr_schedule
from utils.logs import *
from utils.train import train

img_root = '/media/gserver/data/seg_data'
img_shape = (224,224)
#
data_set, data_loader = gen_dataloader(img_root=img_root, train_bs=8, val_bs=4)

print type(data_loader['train'])
img_height = img_shape[0]
img_width = img_shape[1]
lr_base = 0.01 * (float(8) / 16)


x = np.random.rand(10,224,224,3)

model = MobileUNet(input_shape=(img_height, img_width, 3),
                   alpha=1,
                   alpha_up=0.25)

logits, y = model.predict(x)
print logits.min(), logits.max()
print '=='*20
print y.shape


#
model.compile(
    optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
    # optimizer=Adam(lr=0.001),
    # optimizer=optimizers.RMSprop(),
    loss={'logits':None, 'proba':dice_coef_loss},
    metrics={'proba':[
        dice_coef,
        recall,
        precision,
        'binary_crossentropy',
    ]},
)
model.summary()
mask = np.random.randint(0,2,size=(10,224,224,1))
out = model.train_on_batch(x,  mask)
print out
print model.metrics_names
#
#
#
# # # callbacks
# # scheduler = callbacks.LearningRateScheduler(
# #     create_lr_schedule(epochs, lr_base=lr_base, mode='progressive_drops'))
# # tensorboard = callbacks.TensorBoard(log_dir='./logs')
# # csv_logger = callbacks.CSVLogger('logs/training.csv')
# # checkpoint = callbacks.ModelCheckpoint(filepath=checkpoint_path,
# #                                        save_weights_only=True,
# #                                        save_best_only=True)
# #
# train(model,
#       epoch_num=20,
#       start_epoch=0,
#       data_set=data_set,
#       data_loader=data_loader,
#       save_dir='./temp_modles/fuckyou',
#       print_inter=200,
#       val_inter=3500
#       )

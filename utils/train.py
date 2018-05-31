from __future__ import division
import torch
import os,time,datetime
from torch.autograd import Variable
import logging
import torch.nn.functional as F
import numpy as np
from math import ceil
import copy
from metrics import cal_mAP
from utils.predicting import predict

from logs import *

def train(model,
          epoch_num,
          start_epoch,
          optimizer,
          criterion,
          exp_lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          augloss=False,
          print_inter=200,
          val_inter=3500
          ):

    best_acc = 0.0
    best_mAP = 0.0
    best_model = copy.deepcopy(model)

    step = 0
    for epoch in range(start_epoch, epoch_num):
        for batch_cnt, data in enumerate(data_loader['train']):
            model.train(True)
            # print data
            inputs, attr, attr_mask, labels, labels_str = data

            loss,outputs = model.train_on_batch(inputs, labels)

            # batch loss
            if step % print_inter == 0:
                logging.info('%s [%d-%d] | batch-loss: %.3f'
                             % (dt(), epoch, batch_cnt, loss.data[0]))


            if step % val_inter == 0:
                logging.info('--' * 30)
                val_mAP, val_acc = predict(model, data_set['val'], data_loader['val'])

                if val_mAP > best_mAP:
                    best_mAP = val_mAP
                    best_acc = val_acc
                    best_model = copy.deepcopy(model)

                # save model
                save_path = os.path.join(save_dir,
                                         'weights-%d-%d-[%.4f]-[%.4f].pth' % (epoch, batch_cnt, val_acc, val_mAP))
                model.save_weights(save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)

            # save best model
            save_path = os.path.join(save_dir,
                                     'bestweights-[%.4f]-[%.4f].pth' % (best_acc, best_mAP))
            best_model.save_weights(save_path)
            logging.info('saved model to %s' % (save_path))
from __future__ import division
import os
from utils.predicting import predict_dis
from metrics import cal_IOU
from logs import *
from keras import backend as K
import numpy as np
from matplotlib import pyplot as plt
def train(model,
          para_model,
          epoch_num,
          start_epoch,
          lr_scheduler,
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500
          ):

    best_mIOU = 0.0
    best_weights = model.get_weights()

    step = start_epoch * len(data_set['train']) // data_loader['train'].batch_size
    for epoch in range(start_epoch, epoch_num):
        for batch_cnt, data in enumerate(data_loader['train']):
            if para_model is None:
                K.set_value(model.optimizer.lr, lr_scheduler(epoch))
            else:
                K.set_value(para_model.optimizer.lr, lr_scheduler(epoch))

            if step % val_inter == 0:
                logging.info('--' * 30)
                if para_model is None:
                    logging.info('current lr:%s'%K.eval(model.optimizer.lr))
                    mIOU = predict_dis(model, data_set['val'], data_loader['val'])
                else:
                    logging.info('current lr:%s' % K.eval(para_model.optimizer.lr))
                    mIOU = predict_dis(para_model, data_set['val'], data_loader['val'])

                if mIOU > best_mIOU:
                    best_mIOU = mIOU
                    best_weights = model.get_weights()
                # save model

                save_path = os.path.join(save_dir, 'weights-[%d-%d]-[%.4f].h5' % (epoch,step, mIOU))
                model.save_weights(save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)

            # training
            inputs, masks, mask_teacher = data   # (H, W, 3)   (H , W , 1)

            # for i in range(inputs.shape[0]):
            #     plt.subplot(131)
            #     plt.imshow(inputs[i])
            #     plt.subplot(132)
            #     plt.imshow(masks[i,:, :, 0])
            #     plt.subplot(133)
            #     plt.imshow(mask_teacher[i,:, :, 0])
            #     plt.show()

            if para_model is None:
                outputs = model.train_on_batch(inputs, [mask_teacher, masks])
            else:
                outputs = para_model.train_on_batch(inputs, [mask_teacher, masks])

            loss, logits_loss, proba_loss, recall, prec, acc, proba = outputs
            bs_mIOU = cal_IOU(proba.round()[:,:,:,0], masks[:,:,:,0], 2)

            # batch loss
            if step % print_inter == 0:
                logging.info('%s [%d-%d]| loss: %.3f, %.3f+%.3f, recall: %.2f, prec: %.3f, acc:%.3f, iou: %.3f'
                             % (dt(), epoch, step, loss, logits_loss, proba_loss, recall, prec, acc, bs_mIOU.mean()))

            step += 1

    # save best model
    save_path = os.path.join(save_dir,'bestmodel-[%.4f].h5' % (best_mIOU))
    model.set_weights(best_weights)
    model.save(save_path)
    logging.info('saved best model to %s' % (save_path))
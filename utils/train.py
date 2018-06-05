from __future__ import division
import os
from utils.predicting import predict
from metrics import cal_IOU
from logs import *

def train(model,
          epoch_num,
          start_epoch,
          data_set,
          data_loader,
          save_dir,
          print_inter=200,
          val_inter=3500
          ):

    best_mIOU = 0.0
    best_weights = model.get_weights()

    step = 0
    for epoch in range(start_epoch, epoch_num):
        for batch_cnt, data in enumerate(data_loader['train']):
            if step % val_inter == 0:
                logging.info('--' * 30)
                mIOU = predict(model, data_set['val'], data_loader['val'])

                if mIOU > best_mIOU:
                    best_mIOU = mIOU
                    best_weights = model.get_weights()

                # save model

                save_path = os.path.join(save_dir, 'model-%d-[%.4f].h5' % (step, mIOU))
                model.save(save_path)
                logging.info('saved model to %s' % (save_path))
                logging.info('--' * 30)

            # training
            inputs, masks = data

            outputs = model.train_on_batch(inputs, masks)
            loss, recall, prec, proba = outputs

            bs_mIOU = cal_IOU(proba.round()[:,:,:,0], masks[:,:,:,0], 2)

            # batch loss
            if step % print_inter == 0:
                logging.info('%s [%d-%d]| loss: %.3f, recall: %.2f, prec: %.3f, iou: %.3f'
                             % (dt(), epoch, batch_cnt, loss, recall, prec, bs_mIOU.mean()))

            step += 1


    # save best model
    save_path = os.path.join(save_dir,'bestmodel-[%.4f].h5' % (best_mIOU))
    model.set_weights(best_weights)
    model.save(save_path)
    logging.info('saved best model to %s' % (save_path))
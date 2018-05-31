#coding=utf8
from __future__ import division
import time
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import pandas as pd
from metrics import cal_mAP
from FSdata.FSdataset import attr2catidx_map, attr2idx_map, idx2attr_map

def predict(
        model,
        data_set,
        data_loader,
):
    t0 = time.time()


    val_preds = np.zeros((len(data_set), data_set.catlen), dtype=np.float32)
    val_true = np.zeros((len(data_set), data_set.catlen), dtype=np.float32)
    val_attr = np.zeros(len(data_set), dtype=np.int)
    val_attr_mask = np.zeros((len(data_set), data_set.catlen), dtype=np.int)
    val_labels_str = np.empty((len(data_set), data_set.catlen), dtype='|S1')

    idx = 0
    for batch_cnt_val, data_val in enumerate(data_loader):
        print '%d/%d'%(batch_cnt_val, len(data_set)/data_loader.batch_size)
        inputs, attr, attr_mask, labels, labels_str = data_val

        # forward
        outputs = model.predict_on_batch(inputs)

        # statistics
        val_preds[idx:(idx + labels.size(0))] = outputs
        val_true[idx:(idx + labels.size(0))] = labels
        val_attr[idx:(idx + labels.size(0))] = attr
        val_attr_mask[idx:(idx + labels.size(0))] = attr_mask
        val_labels_str[idx:(idx + labels.size(0))] = labels_str

        idx += labels.size(0)

    preds_y = np.argmax(val_preds * val_attr_mask,1)
    labels_yonly = np.argmax(val_true,1)

    val_acc = (preds_y==labels_yonly).sum() * 1. /  labels_yonly.shape[0]
    val_mAP, APs, accs = cal_mAP(val_labels_str, val_preds, val_attr, data_set.catidx_map)

    t1 = time.time()
    since = t1 - t0
    print('val-acc@1: %.4f ||val-mAP: %.4f ||time: %d'
                 % (val_acc, val_mAP, since))
    for key in APs.keys():
        print('acc: %.4f, AP: %.4f %s' % (accs[key], APs[key], key))


    return val_preds, val_labels_str, val_attr, val_true

def gen_submission(test_pd, preds, catidx_map):
    test_pd = test_pd[['ImageName', 'AttrKey']].copy()
    test_pd['AttrValueProbs'] = list(preds)

    for Attr_idx in catidx_map.keys():
        Attr = idx2attr_map[Attr_idx]
        idx1 = catidx_map[Attr_idx][0]
        idx2 = catidx_map[Attr_idx][1]

        test_pd.loc[test_pd['AttrKey'] == Attr, 'AttrValueProbs'] = \
            test_pd.loc[test_pd['AttrKey']==Attr, 'AttrValueProbs'].apply(
            lambda probs: ';'.join(['%.10f'%x for x in list(probs[idx1:idx2])]))
    return test_pd

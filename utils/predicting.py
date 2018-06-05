from __future__ import division
import time
import numpy as np
from metrics import cal_IOU

def predict(
        model,
        data_set,
        data_loader,
):

    ious = np.zeros(len(data_set), dtype=float)

    idx = 0
    for batch_cnt_val, data_val in enumerate(data_loader):
        imgs, masks = data_val
        # forward
        proba = model.predict_on_batch(imgs)
        # statistics
        iou = cal_IOU(proba.round()[:,:,:,0], masks[:,:,:,0], 2)
        print iou
        ious[idx: idx + imgs.shape[0]] = iou
        idx += imgs.shape[0]

    return ious.mean()


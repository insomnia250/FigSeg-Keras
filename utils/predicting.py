from __future__ import division
import time
import numpy as np
from metrics import cal_IOU
from Sdata.Saug import deNormalize
from utils.plotting import AddHeatmap
from matplotlib import pyplot as plt


def predict(
        model,
        data_set,
        data_loader,
        verbose=False
):

    ious = np.zeros(len(data_set), dtype=float)

    idx = 0
    for batch_cnt_val, data_val in enumerate(data_loader):
        if verbose:
            print('%d / %d' % (batch_cnt_val, len(data_set)//data_loader.batch_size))
        imgs, masks = data_val
        # forward
        proba = model.predict_on_batch(imgs)
        # statistics
        iou = cal_IOU(proba.round()[:,:,:,0], masks[:,:,:,0], 2)
        ious[idx: idx + imgs.shape[0]] = iou
        idx += imgs.shape[0]

    return ious.mean()


def predict_vis(
        model,
        data_set,
        data_loader,
        verbose=False
):
    deNormalizer = deNormalize()
    ious = np.zeros(len(data_set), dtype=float)

    idx = 0
    for batch_cnt_val, data_val in enumerate(data_loader):
        if verbose:
            print('%d / %d' % (batch_cnt_val, len(data_set)//data_loader.batch_size))
        imgs, masks = data_val
        # forward
        proba = model.predict_on_batch(imgs)
        # statistics
        iou = cal_IOU(proba.round()[:,:,:,0], masks[:,:,:,0], 2)
        ious[idx: idx + imgs.shape[0]] = iou
        idx += imgs.shape[0]

        # visualize

        pred_hm = proba[0, :, :, 0]
        true_hm = masks[0, :, :, 0]


        ori_img = deNormalizer(imgs[0])
        print pred_hm.shape, true_hm.shape, ori_img.shape
        plt.figure()
        plt.subplot(221)
        plt.imshow(ori_img)
        plt.subplot(222)
        plt.imshow(true_hm)
        plt.subplot(224)
        plt.imshow(pred_hm)
        plt.show()


        pred_hm = AddHeatmap(ori_img, pred_hm)
        true_hm = AddHeatmap(ori_img, true_hm)

        plt.figure()
        plt.subplot(221)
        plt.imshow(ori_img)
        plt.subplot(222)
        plt.imshow(true_hm)
        plt.subplot(224)
        plt.imshow(pred_hm)
        plt.show()








    return ious.mean()


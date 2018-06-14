from __future__ import division
import time
import numpy as np
from metrics import cal_IOU
from Sdata.Saug import deNormalize
from utils.plotting import AddHeatmap
from matplotlib import pyplot as plt

import cv2

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


def predict_dis(
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
        _, proba = model.predict_on_batch(imgs)
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
    deNormalizer = deNormalize(mean=None,std=None)
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
        # print proba.round()[:,:,:,0]
        # print masks[:,:,:,0]
        ious[idx: idx + imgs.shape[0]] = iou
        idx += imgs.shape[0]

        # # visualize
        for i in xrange(imgs.shape[0]):
            pred_hm = proba[i, :, :, 0]
            true_hm = masks[i, :, :, 0]


            ori_img = deNormalizer(imgs[i])
            plt.figure()
            plt.subplot(221)
            plt.imshow(ori_img)
            plt.subplot(224)
            plt.imshow(true_hm)
            plt.subplot(222)
            plt.title('%.3f'%iou[i],fontsize=10)
            plt.imshow(pred_hm)
            plt.show()


            pred_hm = AddHeatmap(ori_img, pred_hm)
            true_hm = AddHeatmap(ori_img, true_hm)

            plt.figure()
            plt.subplot(221)
            plt.imshow(ori_img)
            plt.subplot(224)
            plt.imshow(true_hm)
            plt.subplot(222)
            plt.title('%.3f' % iou[i])
            plt.imshow(pred_hm)
            plt.show()

    return ious.mean()


def predict_save(
        model,
        data_set,
        data_loader,
        save_paths,
        verbose=False,
        visual=False
):

    idx = 0
    for batch_cnt_val, data_val in enumerate(data_loader):
        if verbose:
            print('%d / %d' % (batch_cnt_val, len(data_set)//data_loader.batch_size))
        imgs, masks = data_val
        # forward
        proba = model.predict_on_batch(imgs)
        for pred_mask in proba[:,:,:,0]:
            save_path = save_paths[idx]
            if visual:
                print('shape:', pred_mask.shape)
                print('to be saved: %s'%save_path)
                plt.imshow(pred_mask)
                plt.show()
            np.save(save_path,pred_mask)
            idx += 1


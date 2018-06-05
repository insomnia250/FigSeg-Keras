import numpy as np
from FSdata.FSdataset import attr2catidx_map,idx2attr_map,attr2idx_map,attr2length_map
from collections import OrderedDict
from sklearn.metrics import confusion_matrix


def cal_IOU(pred_b, gt_b, num_classes):
    '''
    :param pred_b: shape of (bs, H, W), per pixel class
    :param gt_b:  shape of (bs, H, W), per pixel class
    :param num_classes:
    :return:
    '''
    bs = pred_b.shape[0]
    ious = np.zeros((bs, num_classes),dtype=float)
    cnt = np.zeros(bs)
    for i in xrange(num_classes):
        inter = (((pred_b==i) & (gt_b==i))>0).sum(axis=1).sum(axis=1)
        union = (((pred_b==i) + (gt_b==i))>0).sum(axis=1).sum(axis=1)
        valid = (union>0).astype(int)
        ious[:,i] = 1. * inter  / (union+(1-valid))
        cnt += valid

    miou = ious.sum(axis=1) / cnt
    return miou


if __name__ == '__main__':

    y_true = np.array([
        ['y','n','n','m'],
                       ['n','n','y','m'],
                       ['n','y','n','m'],
                       ['n','n','n','y'],
                       ['n','n','n','y'],
        ['y', 'n', 'n', 'm'],
        ['n', 'n', 'y', 'm'],
        ['n', 'y', 'n', 'm'],
        ['n', 'n', 'n', 'y'],
        ['n', 'n', 'n', 'y'],
        ['y', 'n', 'n', 'm'],
        ['n', 'n', 'y', 'm'],
        ['n', 'y', 'n', 'm'],
        ['n', 'n', 'n', 'y'],
        ['n', 'n', 'n', 'y'],
                       ], dtype='|S1')


    y_pred = np.array([
                        [0.2,0.5,0.2,0.1],
                       [0.2,0.1,0.5,0.2],
                       [0.1,0.7,0.15,0.05],
                       [0.25,0.25,0.2,0.3],
                       [0.05,0.25,0.2,0.5],
        [0.2, 0.5, 0.2, 0.1],
        [0.2, 0.1, 0.5, 0.2],
        [0.1, 0.7, 0.15, 0.05],
        [0.25, 0.25, 0.2, 0.3],
        [0.05, 0.25, 0.2, 0.5],
        [0.2, 0.5, 0.2, 0.1],
        [0.2, 0.1, 0.5, 0.2],
        [0.1, 0.7, 0.15, 0.05],
        [0.25, 0.25, 0.2, 0.3],
        [0.05, 0.25, 0.2, 0.5],
                       ])

    print y_true
    print y_pred

    print cal_AP(y_true,y_pred)
import numpy as np
from FSdata.FSdataset import attr2catidx_map,idx2attr_map,attr2idx_map,attr2length_map
from collections import OrderedDict
from sklearn.metrics import confusion_matrix

def cal_AP(y_true, y_pred):
    '''
    :param y_true: shape of  (n, C), numpy.array, dtype of '|S1',
    :param y_pred: shape of  (n, C), numpy.array, dtype of float
    :return:
    '''

    MaxAttrProbs = np.max(y_pred,axis=1)
    MaxAttrValues = np.argmax(y_pred,axis=1)

    AllProbThresholds = set(list(MaxAttrProbs))
    P = np.zeros(len(AllProbThresholds),dtype=np.float32)
    basic_acc = 0

    for i,ProbThreshold in enumerate(sorted(AllProbThresholds)):
        PRED_CORRECT_COUNT = ((MaxAttrProbs >= ProbThreshold) & (np.choose(MaxAttrValues, y_true.T)=='y')).sum()
        PRED_WRONG_COUNT = ((MaxAttrProbs >= ProbThreshold) & (np.choose(MaxAttrValues, y_true.T)=='n')).sum()
        PRED_COUNT = PRED_CORRECT_COUNT + PRED_WRONG_COUNT

        if PRED_COUNT == 0:
            PRED_COUNT = 1
            print ProbThreshold
            print MaxAttrProbs

        P[i] = 1.0 * PRED_CORRECT_COUNT / PRED_COUNT
        if i == 0:
            basic_acc = P[i]

    return P.mean(), basic_acc

def cal_mAP(y_true, y_pred_str, attr, catidx_map):
    APs = []
    APs_dict = OrderedDict({})
    acc_dict = OrderedDict({})
    for attr_key in sorted(set(list(attr))):
        idx1 = catidx_map[attr_key][0]
        idx2 = catidx_map[attr_key][1]
        AP,acc = cal_AP(y_true[attr == attr_key, idx1:idx2], y_pred_str[attr == attr_key, idx1:idx2])
        APs.append(AP)
        APs_dict[idx2attr_map[attr_key]] = AP
        acc_dict[idx2attr_map[attr_key]] = acc
    return np.mean(APs), APs_dict, acc_dict

def cal_csv_mAP(merged_pd,val_pd, attr, NMS=False):
    APs = []
    APs_dict = OrderedDict({})
    acc_dict = OrderedDict({})

    for attr_key in attr:
        merged_pd2 =merged_pd[merged_pd["AttrKey"]==attr_key]
        val_pd2 = val_pd[val_pd["AttrKey"]==attr_key]
        cat_len= attr2length_map[attr2idx_map[attr_key]]
        val_labels_str = np.empty((len(val_pd2), cat_len), dtype='|S1')
        val_preds = np.zeros((len(val_pd2),cat_len), dtype=np.float32)
        for ind in range(len(val_pd2)):
            val_labels_str[ind,:]=np.array([x for x in val_pd2["AttrValues"].iloc[ind]])
            val_preds[ind,:]=np.array(merged_pd2["AttrValueProbs"].iloc[ind])
        if NMS:
            modified_row = (val_preds>0.995).sum(axis=1)
            modified = np.repeat(modified_row[:,np.newaxis],val_preds.shape[1],1).astype(bool)

            val_preds[val_preds>0.995]=1
            val_preds[modified & (val_preds<1)] = 0

        AP,acc = cal_AP(val_labels_str,val_preds)
        APs.append(AP)
        APs_dict[attr_key] = AP
        acc_dict[attr_key] = acc
    return np.mean(APs), APs_dict, acc_dict

def confusion_report(y_true, y_pred,  attr, img_paths, catidx_map):
    wrong_dicts = {}
    c_mats = {}
    for attr_key in sorted(set(list(attr))):
        attr_name = idx2attr_map[attr_key]
        idx1 = catidx_map[attr_key][0]
        idx2 = catidx_map[attr_key][1]

        y_true_part = y_true[attr == attr_key, idx1:idx2]
        y_pred_part = y_pred[attr == attr_key, idx1:idx2]

        img_paths_part = img_paths[attr == attr_key]

        y_true_label = np.argmax(y_true_part,axis=1)
        y_pred_label = np.argmax(y_pred_part,axis=1)

        wrong_dicts[attr_name] = {}
        for label1 in np.unique(y_true_label):
            for label2 in np.unique(y_pred_label):
                if label1 != label2:
                    wrong_img_paths = img_paths_part[(y_true_label==label1) & (y_pred_label==label2)]
                    wrong_dicts[attr_name]['%s_to_%s'%(label1,label2)] = wrong_img_paths

        c_mats[attr_name] = confusion_matrix(y_true_label, y_pred_label)

    return wrong_dicts, c_mats

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
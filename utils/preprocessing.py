import torch
import pandas as pd
from FSdata.FSdataset import attr2length_map, attr2idx_map, idx2attr_map
import warnings
import logging
import os

def addFakeLabel(test_pd):
    AttrValues = []
    AttrKey = []
    for attr in attr2idx_map.keys():
        attr_idx = attr2idx_map[attr]
        AttrKey.append(attr)
        AttrValues.append('n' * attr2length_map[attr_idx])

    fake_label = pd.DataFrame({'AttrKey': AttrKey, 'AttrValues': AttrValues})
    test_pd = test_pd[['ImageName', 'AttrKey']].merge(fake_label, how='left', on='AttrKey')
    return test_pd

def init_weight_from_model(tg_model, tg_name=None, src_name=None, **kwargs):
    '''
    use weights in src_model to initialize tg_model,
    src_model's weights of name in src_name will be given to tg_model's layer of name in tg_name

    :param tg_model:  model that use provided weights
    :param src_name:  layer name in src_model
    :param tg_name:  layer name in tg_name
    :param kwargs: src_weights or src_model
    :return: initialized tg_model
    '''
    if 'src_weights' in kwargs.keys():
        weights_src = kwargs['src_weights']
    elif 'src_model' in kwargs.keys():
        weights_src = kwargs['src_model'].state_dict()
    else:
        raise ValueError('source weights or src model should be given')

    weights_tg = tg_model.state_dict()
    for key in weights_tg.keys():
        if key in tg_name:
            idx = tg_name.index(key)
            if src_name[idx]:
                weights_tg[key] = weights_src[src_name[idx]]
        elif key in weights_src.keys():
            weights_tg[key] = weights_src[key]
        else:
            print('"%s" not in src model weights, using init value' % key)
    tg_model.load_state_dict(weights_tg)
    return tg_model


def join_path_to_df(df, *path_to_join ):
    path_to_join = os.path.join(*path_to_join)
    df['ImageName'] = df['ImageName'].apply(lambda x: os.path.join(path_to_join, x))
    return df


def select_attr_from_df(df, select_AttrIdx):
    select_AttrKey = [idx2attr_map[x] for x in select_AttrIdx]
    df = df[df['AttrKey'].apply(lambda x: True if x in select_AttrKey else False)]
    df.index = range(df.shape[0])
    return df


def do_SWA(resumes):
    res = torch.load(resumes[0])

    for resume in resumes[1:]:
        weights = torch.load(resume)
        for key, value in weights.items():
            print key
            res[key] += value

    for key, value in res.items():
        res[key] /= len(resumes)

    return res




if __name__ =='__main__':
    pass

import os
import time,random
import pandas as pd
from sklearn.model_selection import train_test_split
from FSdata.FSdataset import FSdata, collate_fn,attr2length_map, idx2attr_map
import torch
import torch.utils.data as torchdata
import torch.optim as optim
from torch.optim import lr_scheduler

from utils.logs import trainlog
from utils.preprocessing import *
from FSdata.FSaug import *
import logging
from keras import losses
from keras.applications.inception_v3 import InceptionV3

class FSAug(object):
    def __init__(self):
        self.augment = Compose([
            ExpandBorder(select=[0, 5, 6, 7], mode='constant', resize=True, size=(336, 336)),
            RandomUpperCrop(size=(368, 368), select=[1 ,2 ,3, 4]),
            Resize(size=(368, 368), select=[1, 2, 3, 4]),
            Resize(size=(336, 336), select=[0, 5, 6, 7]),
            RandomHflip(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            RandomErasing(select=[0,5,6,7]),
        ])

    def __call__(self, image,attr_idx):
        return self.augment(image,attr_idx)

class FSAugVal(object):
    def __init__(self):
        self.augment = Compose([
            ExpandBorder(select=[0,5, 6, 7], mode='constant',resize=True,size=(336,336)),
            UpperCrop(size=(368, 368), select=[1,2, 3, 4]),
            Resize(size=(368, 368), select=[1, 2, 3, 4]),
            Resize(size=(336, 336), select=[0, 5, 6, 7]),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, image,attr_idx):
        return self.augment(image,attr_idx)

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
rawdata_root = '/media/gserver/data/FashionAI'

round1_df = pd.read_csv(os.path.join(rawdata_root,'round1/base/Annotations/label.csv'),
                        header=None, names=['ImageName', 'AttrKey', 'AttrValues'])
round1_df = join_path_to_df(round1_df, rawdata_root, 'round1/base')


round2_df = pd.read_csv(os.path.join(rawdata_root,'round2/train/Annotations/label.csv'),
                        header=None, names=['ImageName', 'AttrKey', 'AttrValues'])
round2_df = join_path_to_df(round2_df, rawdata_root, 'round2/train')

extra_df = pd.read_csv(os.path.join(rawdata_root,'round2/round2_data_add_skirt_legth.txt'),
                        header=None, names=['ImageName', 'AttrKey', 'AttrValues'])
extra_df = join_path_to_df(extra_df, rawdata_root, 'round1/web')


round2_train_pd, val_pd = train_test_split(round2_df, test_size=0.1, random_state=37,
                                    stratify=round2_df['AttrKey'])

train_pd = pd.concat([round2_train_pd, round1_df, extra_df], axis=0, ignore_index=True)
train_pd.index = range(train_pd.shape[0])


select_AttrIdx = [0,5,6,7]
train_pd = select_attr_from_df(train_pd, select_AttrIdx)
val_pd = select_attr_from_df(val_pd, select_AttrIdx)

print(train_pd.shape)
print(val_pd.shape)


# saving dir
save_dir = '/media/gserver/extra/FashionAI/round2/res101_seed0'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
logfile = '%s/trainlog.log'%save_dir
trainlog(logfile)


data_set = {}
data_set['train'] = FSdata(anno_pd=train_pd,
                           transforms=FSAug(),
                           select=select_AttrIdx,
                           )

data_set['val'] = FSdata(anno_pd=val_pd,
                         transforms=FSAugVal(),
                         select=select_AttrIdx
                         )

data_loader = {}
data_loader['train'] = torchdata.DataLoader(data_set['train'], 8, num_workers=4,
                                            shuffle=True, pin_memory=True,collate_fn=collate_fn)
data_loader['val'] = torchdata.DataLoader(data_set['val'], batch_size=2, num_workers=4,
                                          shuffle=False, pin_memory=True,collate_fn=collate_fn)
#
# logging info of dataset
logging.info(train_pd.shape)
logging.info(val_pd.shape)
logging.info('train augment:')
for item in data_set['train'].transforms.augment.transforms:
    logging.info('  %s %s' % (item.__class__.__name__, item.__dict__))

logging.info('val augment:')
for item in data_set['val'].transforms.augment.transforms:
    logging.info('  %s %s' % (item.__class__.__name__, item.__dict__))
#
#
# model prepare
resume = None
model = resnet101_cat(pretrained=True, num_classes=[attr2length_map[x] for x in select_AttrIdx])

if resume:
    model.eval()
    try:
        model.load_state_dict(torch.load(resume))
        logging.info('resuming finetune from %s' % resume)
    except KeyError:
        model = torch.nn.DataParallel(model)
        model.load_state_dict(torch.load(resume))
        logging.info('resuming finetune from %s' % resume)

model.cuda()
logging.info(model)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
criterion = SoftmaxCrossEntropy()

lr_lambda = lambda x: 1 if x < 4 else (0.1 if x < 8 else 0.01)
exp_lr_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# training
best_acc,best_model_wts = train(  model,
                                  epoch_num=25,
                                  start_epoch=0,
                                  optimizer=optimizer,
                                  criterion=criterion,
                                  exp_lr_scheduler=exp_lr_scheduler,
                                  data_set=data_set,
                                  data_loader=data_loader,
                                  save_dir=save_dir,
                                  augloss=False,
                                  print_inter=100,
                                  val_inter=4000,
                                  )

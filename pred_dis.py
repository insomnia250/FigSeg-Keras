import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from keras.models import Model
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))
from Sdata.Saug import *

from utils.predicting import predict,predict_vis,predict_save

from utils.preprocessing import get_train_val, gen_dataloader
from nets.MobileUnet_small2 import MobileUNet as MobileUNet_s2
from keras.layers import Activation, Lambda

class valAug(object):
    def __init__(self,size=(256,256)):
        self.augment = Compose([
            ResizeImg(size=size),
            Normalize(mean=None, std=None)
        ])

    def __call__(self, *args):
        return self.augment(*args)

train_root = '/media/hszc/data1/seg_data'
val_root = '/media/hszc/data1/seg_data/diy_seg'

save_dir = '/media/hszc/data1/seg_data/diy_seg/pred_mask(T20)'
img_shape = (256,256)
bs = 8
do_para = False
resume = './saved_models/MUs2(4-2 0p5)_256/bestmodel-[0.8534].h5'
T = 20

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

# prepare data
train_pd, _ = get_train_val(train_root, test_size=0.0)
_, val_pd = get_train_val(val_root, test_size=1.0)

print train_pd.info()
print val_pd.info()


data_set, data_loader = gen_dataloader(train_pd, val_pd,  valAug(), valAug(), train_bs = bs, val_bs=2,
                                       train_shuffle=False, val_shuffle=False)


trained_model = MobileUNet_s2(input_shape=(256, 256, 3),
                              alpha=0.5,
                              alpha_up=0.25)

trained_model.load_weights(resume)
trained_model.summary()

# predict through dataset
logits = trained_model.get_layer('logits').output
logits = Lambda(lambda inputs: inputs/T)(logits)
proba_smooth = Activation('sigmoid', name='proba')(logits)
model = Model(inputs=trained_model.inputs, outputs=proba_smooth)
model.summary()

mask_paths = data_set['train'].mask_paths
save_paths = [os.path.join(save_dir, x.split('/')[-1]) for x in mask_paths]
save_paths = [x.replace('.png', '.npy') for x in save_paths]
predict_save(model, data_set['train'], data_loader['train'], save_paths=save_paths,verbose=True,
             visual=False)


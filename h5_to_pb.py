# coding=utf-8
import sys

from keras.models import Model,load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K
from keras.applications import mobilenet
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.layers import Input, GlobalAveragePooling2D, BatchNormalization, Dense,SeparableConv2D,Convolution2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Dropout, Flatten, merge, Reshape, Activation
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

from keras.utils import multi_gpu_model
from keras import callbacks, optimizers
from keras.utils.generic_utils import CustomObjectScope
from utils.preprocessing import *
import logging
from nets.DeeplabV3 import Deeplabv3
from keras import losses
# from nets.MobileUnet import MobileUNet
from nets.MobileUnet_small import MobileUNet as MobileUNet_s
from nets.MobileUnet_small2 import MobileUNet as MobileUNet_s2
from loss import dice_coef_loss, dice_coef, recall, precision

def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a prunned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    prunned so subgraphs that are not neccesary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


input_fld = 'artifacts/'
weight_file = 'artifacts/Ms2.h5'
output_graph_name = 'Ms2.pb'
multi_gpu = 0

output_fld = input_fld
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = weight_file

K.set_learning_phase(0)

# prepare model
model = MobileUNet_s2(input_shape=(256, 256, 3),
                   alpha=1,
                   alpha_up=0.25)
# model =  Deeplabv3(weights='pascal_voc', input_tensor=None, input_shape=(224, 224, 3), classes=1, backbone='mobilenetv2', OS=16, alpha=0.5)
if multi_gpu:
    parallel_model = multi_gpu_model(model,gpus=2)
    parallel_model.load_weights(weight_file)
    model.save_weights('artifacts/weights-[182-112000]-[0.8056].h5')
else:
    model.load_weights(weight_file)
logging.info('resumed model from %s'%weight_file)

model.summary()

# print("Start!")
# net_model=mobilenet.MobileNet(input_shape=(128, 128, 3),
#                               alpha=0.5,
#                               include_top=False,
#                               weights='/home/heils-server/User/gsj/liveness_pretr_mobilenetv1_keras/res/mobilenet_5_0_128_tf_no_top.h5')
# net_model = add_top_layers(net_model)
# net_model.load_weights(weight_file_path, by_name=True)
print("Load Over!")

print('input is :', model.input.name)
print ('output is:', model.output.name)

sess = K.get_session()

frozen_graph = freeze_session(K.get_session(), output_names=[model.output.op.name])

from tensorflow.python.framework import graph_io

graph_io.write_graph(frozen_graph, output_fld, output_graph_name, as_text=False)

print('saved the constant graph (ready for inference) at: ', osp.join(output_fld, output_graph_name))

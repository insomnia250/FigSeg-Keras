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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

from keras.utils import multi_gpu_model
from keras import callbacks, optimizers
from keras.utils.generic_utils import CustomObjectScope
from utils.preprocessing import *
import logging
from keras import losses
from nets.MobileUnet import MobileUNet
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
weight_file = 'artifacts/weights-122500-[0.9186].h5'
output_graph_name = 'MobileUnet9186.pb'

output_fld = input_fld
if not os.path.isdir(output_fld):
    os.mkdir(output_fld)
weight_file_path = weight_file

K.set_learning_phase(0)



model = MobileUNet(input_shape=(224, 224, 3),
                   alpha=1,
                   alpha_up=0.25)
model.compile(
    # optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),
    optimizer=optimizers.Adam(lr=1e-4),
    # optimizer=optimizers.RMSprop(),
    loss={'proba': losses.binary_crossentropy},
    metrics={'proba': [
        recall,
        precision,
        'acc'
    ]},
)

out_layers = ['proba']
model.metrics_tensors += [layer.output for layer in model.layers if layer.name in out_layers]

# with CustomObjectScope(custom_objects()):
#     model = keras.models.load_model(resume)
model.load_weights(weight_file_path)


# net_model = load_model(weight_file_path)

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

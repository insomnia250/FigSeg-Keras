import os
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
import keras
import keras.backend as K
from keras import Input
from keras.layers import BatchNormalization, Activation, Conv2D, \
    concatenate, Conv2DTranspose, Add, MaxPooling2D, GlobalAveragePooling2D
from keras.engine import Model
from layers.BilinearUpSampling import BilinearUpSampling2D
from keras import regularizers as regu
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

weight_decay = 1e-5
channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
def _GlobalConvModule(x, out_idm, kernel_size, name='GCM1'):
    x_l = Conv2D(filters=out_idm, kernel_size=(kernel_size[0], 1), padding='same', name='%s_conv_l1'%name,
                 kernel_regularizer=regu.l2(weight_decay),
                 bias_regularizer=regu.l2(weight_decay))(x)
    x_l = Conv2D(filters=out_idm, kernel_size=(1, kernel_size[1]), padding='same', name='%s_conv_l2'%name,
                 kernel_regularizer=regu.l2(weight_decay),
                 bias_regularizer=regu.l2(weight_decay))(x_l)
    x_r = Conv2D(filters=out_idm, kernel_size=(1, kernel_size[0]), padding='same', name='%s_conv_r1'%name,
                 kernel_regularizer=regu.l2(weight_decay),
                 bias_regularizer=regu.l2(weight_decay))(x)
    x_r = Conv2D(filters=out_idm, kernel_size=(kernel_size[1], 1), padding='same', name='%s_conv_r2'%name,
                 kernel_regularizer=regu.l2(weight_decay),
                 bias_regularizer=regu.l2(weight_decay))(x_r)
    return Add()([x_l, x_r])

def _BoundaryRefineModule(x, dim, name='BRM1'):
    residual = Conv2D(filters=dim, kernel_size=3, padding='same', name='%s_conv1'%name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay))(x)
    residual = Activation('relu', name='%s_relu'%name)(residual)
    residual = Conv2D(filters=dim, kernel_size=3, padding='same', name='%s_conv2' % name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay))(residual)
    return Add()([residual, x])

def _BoundaryRefineModule2(x, dim, name='BRM1'):
    residual = Conv2D(filters=dim, kernel_size=3, padding='same', name='%s_conv1'%name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay))(x)
    residual = Activation('relu', name='%s_relu'%name)(residual)
    residual = Conv2D(filters=dim, kernel_size=3, padding='same', name='%s_conv2' % name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay))(residual)
    x = Conv2D(filters=dim, kernel_size=3, padding='same', name='%s_shortcut_conv'%name,
               kernel_regularizer=regu.l2(weight_decay),
               bias_regularizer=regu.l2(weight_decay))(x)
    return Add()([residual, x])


def Bottleneck(x, filters, name , strides=(1, 1), downsample=False, squeeze=8):

    filters1 = filters // squeeze
    filters2 = filters // squeeze
    filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    residual = Conv2D(filters1, (1, 1),name='%s_conv1' % name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay))(x)
    residual = BatchNormalization(axis=bn_axis, name='%s_bn1'%name,
                                  gamma_regularizer=regu.l2(weight_decay),
                                  beta_regularizer=regu.l2(weight_decay))(residual)
    residual = Activation('relu',name='%s_relu1'%name)(residual)

    residual = Conv2D(filters2, 3, strides=strides , padding='same',
                      name='%s_conv2' % name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay)
                      )(residual)
    residual = BatchNormalization(axis=bn_axis, name='%s_bn2'%name,
                                  beta_regularizer=regu.l2(weight_decay),
                                  gamma_regularizer=regu.l2(weight_decay)
                                  )(residual)
    residual = Activation('relu',name='%s_relu2'%name)(residual)

    residual = Conv2D(filters3, (1, 1), name='%s_conv3'%name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay)
                      )(residual)
    residual = BatchNormalization(axis=bn_axis, name='%s_bn3'%name,
                                  gamma_regularizer=regu.l2(weight_decay),
                                  beta_regularizer=regu.l2(weight_decay)
                                  )(residual)


    if downsample:
        x = Conv2D(filters3, (1, 1), strides=strides,
                   name='%s_neckConv' % name)(x)
        x = BatchNormalization(axis=bn_axis, name='%s_neckBn'%name)(x)

    x = Add(name='%s_add'%name)([residual, x])
    x = Activation('relu', name='%s_relu'%name)(x)
    return x


def Bottleneck_up(x, filters, name,  strides=(1, 1), squeeze=8, in_dim=64, upsample=False):
    filters1 = in_dim // squeeze
    filters2 = in_dim // squeeze
    filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    residual = Conv2D(filters1, (1, 1),name='%s_conv1'%name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay)
                      )(x)
    residual = BatchNormalization(axis=bn_axis, name='%s_bn1'%name,
                                  gamma_regularizer=regu.l2(weight_decay),
                                  beta_regularizer=regu.l2(weight_decay)
                                  )(residual)
    residual = Activation('relu',name='%s_relu1'%name)(residual)

    residual = Conv2DTranspose(filters2, 3, padding='same', strides=strides,
               name='%s_deConv2'%name,
               kernel_regularizer=regu.l2(weight_decay),
               bias_regularizer=regu.l2(weight_decay)
               )(residual)
    residual = BatchNormalization(axis=bn_axis, name='%s_bn2'%name,
                                  beta_regularizer=regu.l2(weight_decay),
                                  gamma_regularizer=regu.l2(weight_decay)
                                  )(residual)
    residual = Activation('relu',name='%s_relu2'%name)(residual)

    residual = Conv2D(filters3, (1, 1), name='%s_conv3'%name,
                      kernel_regularizer=regu.l2(weight_decay),
                      bias_regularizer=regu.l2(weight_decay)
                      )(residual)
    residual = BatchNormalization(axis=bn_axis, name='%s_bn3'%name,
                                  gamma_regularizer=regu.l2(weight_decay),
                                  beta_regularizer=regu.l2(weight_decay)
                                  )(residual)

    if upsample:
        x = Conv2DTranspose(filters3, (1, 1), strides=strides, padding='same',
                          name='%s_neckConv'%name,
                            kernel_regularizer=regu.l2(weight_decay),
                            bias_regularizer=regu.l2(weight_decay)
                            )(x)
        x = BatchNormalization(axis=bn_axis, name='%s_neckbn'%name,
                               gamma_regularizer=regu.l2(weight_decay),
                               beta_regularizer=regu.l2(weight_decay)
                               )(x)

    x = Add(name='%s_add'%name)([residual, x])
    x = Activation('relu', name='%s_relu'%name)(x)
    return x

def _EncoderBlock(x, inplanes, planes, num_blocks, strides, name='Enc'):
    for i in xrange(num_blocks):
        if strides != 1 or inplanes != planes:
            downsample = True
        else:
            downsample = False

        x = Bottleneck(x, planes, strides=strides, name='%s_%d'%(name, i), downsample=downsample)
        strides = 1
        inplanes = planes

    return x

def _DecoderBlock(x, inplanes, planes, num_blocks, strides, name='Dec'):
    for i in xrange(num_blocks):
        if strides != 1 or inplanes != planes:
            upsample = True
        else:
            upsample = False

        x = Bottleneck_up(x, planes, strides=strides, name='%s_%d'%(name, i), upsample=upsample)
        strides = 1
        inplanes = planes

    return x


def FigSeg(input_shape=None,
        input_tensor=None):
    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    fm0 = Conv2D(64, kernel_size=7, strides=2, padding='same', use_bias=False,
                  kernel_regularizer=regu.l2(weight_decay),
                  bias_regularizer=regu.l2(weight_decay), name='enc0_conv')(img_input)
    fm0 = BatchNormalization(axis=channel_axis, name='enc0_bn',
                           beta_regularizer=regu.l2(weight_decay), gamma_regularizer=regu.l2(weight_decay))(fm0)
    fm0 = Activation('relu', name='enc0_relu')(fm0)

    fm1 = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(fm0)
    fm1 = _EncoderBlock(fm1, 64, 128, num_blocks=3, strides=1, name='Enc1')

    fm2 = _EncoderBlock(fm1, 128, 256, num_blocks=4, strides=2, name='Enc2')
    fm3 = _EncoderBlock(fm2, 256, 512, num_blocks=6, strides=2, name='Enc3')
    fm4 = _EncoderBlock(fm3, 512, 1024, num_blocks=3, strides=2, name='Enc4')

    gcfm1 = _GlobalConvModule(fm4, 2, (7,7), name='GCM1')
    gcfm1 = _BoundaryRefineModule(gcfm1, 2, name='BRM1')

    gcfm2 = _GlobalConvModule(fm3, 2, (7,7), name='GCM2')
    gcfm2 = _BoundaryRefineModule(gcfm2, 2, name='BRM2')

    gcfm3 = _GlobalConvModule(fm2, 2, (7,7), name='GCM3')
    gcfm3 = _BoundaryRefineModule(gcfm3, 2, name='BRM3')

    gcfm4 = _GlobalConvModule(fm1, 2, (7,7), name='GCM4')
    gcfm4 = _BoundaryRefineModule(gcfm4, 2, name='BRM4')

    fs1 = BilinearUpSampling2D(size=(2, 2),name='upsample1')(gcfm1)
    fs1 = concatenate([gcfm2,fs1],axis=3,name='cat1')
    fs1 = _BoundaryRefineModule2(fs1, 2, name='BRM5')

    fs2 = BilinearUpSampling2D(size=(2, 2),name='upsample2')(fs1)
    fs2 = concatenate([gcfm3,fs2],axis=3,name='cat2')
    fs2 = _BoundaryRefineModule2(fs2, 2, name='BRM6')

    fs3 = BilinearUpSampling2D(size=(2, 2),name='upsample3')(fs2)
    fs3 = concatenate([gcfm4,fs3],axis=3,name='cat3')
    fs3 = _BoundaryRefineModule2(fs3, 2, name='BRM7')

    fs4 = BilinearUpSampling2D(size=(2, 2),name='upsample4')(fs3)
    fs4 = concatenate([fm0,fs4],axis=3,name='cat4')
    fs4 = _BoundaryRefineModule2(fs4, 16, name='BRM8')

    fs4 = _DecoderBlock(fs4, 16, 2, 3, 2, name='Dec')
    logits = _BoundaryRefineModule2(fs4, 1, name='logits')
    proba = Activation('sigmoid', name='proba')(logits)

    model = Model(img_input, proba)

    return model


if __name__=='__main__':
    import numpy as np

    x = np.empty((10,224,224,3),dtype=np.float32)
    model = FigSeg(input_shape=(224, 224, 3))
    model.summary()
    y = model.predict(x)
    print '=='*20
    print y.shape
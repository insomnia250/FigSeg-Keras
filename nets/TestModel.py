import keras.backend as K
from keras import Input
from keras.applications import mobilenet
from keras.applications.mobilenet import DepthwiseConv2D, relu6
from keras.engine import Model
from keras.layers import BatchNormalization, Activation, Conv2D, concatenate, Conv2DTranspose
import numpy as np
import loss
from layers.BilinearUpSampling import BilinearUpSampling2D

from keras.layers import Input, Dense, Lambda
np.random.seed(13)

loss_f = lambda y_true,y_pred: y_pred

def mask_MSE(x):
    y_true, y_pred, mask = x
    return K.sum(mask*K.square(y_pred - y_true),axis=1)


input_shape=(32,)
mask_shape = (10,)

img = Input(shape=input_shape)
mask = Input(shape=mask_shape)
y = Input(shape=mask_shape)
b = Dense(10)(img)
b = Activation('relu')(b)
pred = Dense(10)(b)

loss = Lambda(mask_MSE)([y, pred, mask])
# loss = K.sum(mask *(K.square(pred - y)))

model = Model(inputs=[img,mask,y], outputs=[loss, pred])
print model.summary()
model.compile(loss={'lambda_1':loss_f}, optimizer='sgd')

x = np.arange(3*32).reshape(3,32)
mask = np.random.randint(2, size=(3,10))
y = np.arange(3*10).reshape(3,10)+1

print x
print mask
print y


loss, pred = model.predict([x, mask, y])
print (((pred - y)**2)*mask).sum(1)
print loss
# -*- coding: utf-8 -*-
# python 2

from __future__ import print_function

import numpy as np

from keras import backend as K
from keras.layers import Input, Activation
from keras.layers import concatenate, SpatialDropout2D
from keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.optimizers import Adam
from metrics import dice_coefficient_loss, dice_coefficient

K.set_image_data_format('channels_first')

def u_net2d_model(dropout_rate = 0.5,
                  optimizer = Adam,
                  initial_learning_rate = 5e-4,
                  loss_function = dice_coefficient_loss,
                  metrics_function = [dice_coefficient],
                  input_shape = (4, 240, 240),
                  convolution_kernel = (3, 3),
                  convolution_strides = (1, 1),
                  maxpooling_kernel = (2, 2),
                  maxpooling_strides = (2, 2),
                  deconvolution_kernel = (3, 3),
                  deconvolution_strides = (2, 2)):

    inputs = Input(input_shape) # (4, 240, 240)

    conv_block1 = convolution_block(input_layer = inputs, filters = [16, 32], dropout_rate = dropout_rate, kernel = convolution_kernel, strides = convolution_strides, activation = 'relu')
    # conv1 : (16, 240, 240)
    # conv2 : (32, 240, 240)

    maxpool1 = maxpooling2D(input_layer = conv_block1, pool_size = maxpooling_kernel, strides = maxpooling_strides, padding = 'valid')
    # maxpool1 : (32, 120, 120)

    conv_block2 = convolution_block(input_layer = maxpool1, filters = [64, 128], dropout_rate = dropout_rate, kernel = convolution_kernel, strides = convolution_strides, activation = 'relu')
    # conv3 : (64, 120, 120)
    # conv4 : (128, 120, 120)

    maxpool2 = maxpooling2D(input_layer = conv_block2, pool_size = maxpooling_kernel, strides = maxpooling_strides, padding = 'valid')
    # (128, 60, 60)

    conv_block3 = convolution_block(input_layer = maxpool2, filters = [256, 512, 128], dropout_rate = dropout_rate, kernel = convolution_kernel, strides = convolution_strides, activation = 'relu')
    # conv5 : (256, 60, 60)
    # conv6 : (512, 60, 60)
    # conv7 : (128, 60, 60)

    deconv1 = deconvolution2D(input_layer = conv_block3, output_filters = 128, kernel = deconvolution_kernel, strides = deconvolution_strides, padding = 'same')
    # deconv1 : (128, 120, 120)
    merge1 = residual_connection(deconv1, conv_block2)
    # merge1 : (128 + 128, 120, 120)

    conv_block4 = convolution_block(input_layer = merge1, filters = [64, 32], dropout_rate = dropout_rate, kernel = convolution_kernel, strides = convolution_strides, activation = 'relu')
    # conv8 : (64, 120, 120)
    # conv9 : (32, 120, 120)

    deconv2 = deconvolution2D(input_layer = conv_block4, output_filters = 32, kernel = deconvolution_kernel, strides = deconvolution_strides, padding = 'same')
    # deconv2 : (32, 240, 240)
    merge2 = residual_connection(deconv2, conv_block1)
    # merge2 : (32 + 32, 240, 240)

    conv_block5 = convolution_block(input_layer = merge2, filters = [16, 8], dropout_rate = dropout_rate, kernel = convolution_kernel, strides = convolution_strides, activation = 'relu')
    # conv10 : (16, 240, 240)
    # conv11 : (8, 240, 240)

    outputs = last_convolution2D(input_layer = conv_block5)

    model = Model(inputs = inputs, outputs = outputs )

    model.compile(optimizer = optimizer(lr = initial_learning_rate), loss = loss_function, metrics = metrics_function)

    return model

def convolution_block(input_layer, filters, dropout_rate, kernel, strides, activation):
    if len(filters) == 2:
        layer1 = convolution2D(input_layer, output_filters = filters[0], kernel = kernel, strides = strides, activation = activation)
        dropout = dropout2D(layer1, dropout_rate = dropout_rate)
        layer2 = convolution2D(dropout, output_filters = filters[1], kernel = kernel, strides = strides, activation = activation)
        return layer2
    elif len(filters) == 3:
        layer1 = convolution2D(input_layer, output_filters = filters[0], kernel = kernel, strides = strides, activation = activation)
        dropout1 = dropout2D(layer1, dropout_rate = dropout_rate)
        layer2 = convolution2D(dropout1, output_filters = filters[1], kernel = kernel, strides = strides, activation = activation)
        dropout2 = dropout2D(layer2, dropout_rate = dropout_rate)
        layer3 = convolution2D(dropout2, output_filters = filters[2], kernel = kernel, strides = strides, activation = activation)
        return layer3

def last_convolution2D(input_layer, activation = 'sigmoid', output_filter = 4, kernel = (1, 1), strides = (1, 1), padding = 'same'):
    layer = Conv2D(output_filter, kernel_size = kernel, padding = padding,
                   strides = strides, data_format = 'channels_first')(input_layer)
    layer = BatchNormalization()(layer)
    layer = Activation(activation)(layer)
    return layer

def dropout2D(input_layer, dropout_rate):
    layer = SpatialDropout2D(rate = dropout_rate, data_format = 'channels_first')(input_layer)
    return layer

def convolution2D(input_layer, output_filters, kernel, strides, activation):
    layer = Conv2D(output_filters, kernel_size = kernel, padding = 'same',
                   strides = strides, data_format = 'channels_first')(input_layer)
    layer = BatchNormalization()(layer)
    layer = Activation(activation)(layer)
    return layer

def maxpooling2D(input_layer, pool_size, strides, padding):
    layer = MaxPooling2D(pool_size = pool_size, strides = strides, padding = padding, data_format = 'channels_first')(input_layer)
    return layer

def deconvolution2D(input_layer, output_filters, kernel, strides, padding):
    layer = Conv2DTranspose(filters = output_filters, kernel_size = kernel, strides = strides,
                            padding = padding, data_format = 'channels_first')(input_layer)
    return layer

def residual_connection(deconv_layer, conv_layer):
    residual = concatenate([deconv_layer, conv_layer], axis = 1)
    return residual

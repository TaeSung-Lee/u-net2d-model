# -*- coding: utf-8 -*-
# python 2

from __future__ import print_function

import os
import time
import numpy as np

from keras.utils.np_utils import to_categorical
from keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, TensorBoard
from keras.optimizers import Adam

from metrics import dice_coefficient, dice_coefficient_loss
from u_net2d_model import u_net2d_model

output_path = input('INPUT OUTPUT PATH : ')
model_path = output_path + 'model/'
try:
    os.makedirs(model_path)
except OSError:
    pass

train_patch_name = output_path + 'train.pat'
label_patch_name = output_path + 'label_4_channels.pat'

check_point = model_path + 'model_check_point-{epoch:03d}-{val_dice_coefficient:.2f}.hdf5'
log_file = model_path + 'training.log'
model_file = model_path + 'u_net2d_model.h5'
tensorboard_log_file_path = model_path + 'tensorboard_log/'
try:
    os.makedirs(tensorboard_log_file_path)
except OSError:
    pass

dropout_rate = 0.5
validation_rate = 0.2
optimizer = Adam
initial_learning_rate = 5e-4
loss_function = dice_coefficient_loss
metrics_function = [dice_coefficient]
learning_rate_drop = 0.5
learning_rate_patience = 20
batch_size = 10
epochs = 100
overwrite = False

def get_callback(model_file, log_file, learning_rate_drop, learning_rate_patience, tensorboard_log_file_path):
    callback = []
    callback.append(ModelCheckpoint(model_file, monitor = 'val_dice_coefficient', verbose = 1, save_best_only = True,  mode = 'max'))
    callback.append(CSVLogger(log_file, append = False))
    callback.append(ReduceLROnPlateau(factor = learning_rate_drop,
                                      patience = learning_rate_patience,
                                      verbose = 1))
    callback.append(TensorBoard(log_dir = tensorboard_log_file_path))
    return callback

def train(overwrite = overwrite,
          batch_size = batch_size,
          epochs = epochs,
          validation_rate = validation_rate,
          optimizer = optimizer,
          initial_learning_rate = initial_learning_rate,
          loss_function = loss_function,
          metrics_function = metrics_function,
          learning_rate_drop = learning_rate_drop,
          learning_rate_patience = learning_rate_patience):

    global train_patch_name, label_patch_name, check_point, log_file, model_file
    global tensorboard_log_file_path, model_path

    model = u_net2d_model(input_shape = (4, 240, 240),
                          dropout_rate = dropout_rate,
                          optimizer = optimizer,
                          initial_learning_rate = initial_learning_rate,
                          loss_function = loss_function,
                          metrics_function = metrics_function)
    if overwrite:
        for directory in [model_path, tensorboard_log_file_path]:
            files = [f for f in os.listdir(directory) if os.path.isfile(f)]
            for f in files:
                os.remove(model_path + files)
    else:
        check_points = [f for f in os.listdir(model_path) if f.endswith('.hdf5')]
        if trained_model in os.listdir(model_path):
            model.load_weight(model_path + trained_model)
        else:
            best_check_point = check_point[-1]
            model.load_weights(model_path + best_check_point)

    train_patch = np.memmap(train_patch_name, mode = 'r', dtype = np.float32)
    train_patch = train_patch.reshape(-1, 4, 240, 240)
    label_patch = np.memmap(label_patch_name, mode = 'r', dtype = np.int8,
                            shape = (train_patch.shape[0], 4, 240, 240))

    model.fit(x = train_patch[:],
              y = label_patch[:],
              validation_split = validation_rate,
              batch_size = batch_size,
              epochs = epochs,
              callbacks = get_callback(model_file = check_point,
                                       log_file = log_file,
                                       learning_rate_drop = learning_rate_drop,
                                       learning_rate_patience = learning_rate_patience,
                                       tensorboard_log_file_path = tensorboard_log_file_path))

    model.save(model_file)

if __name__ == '__main__':
    train()

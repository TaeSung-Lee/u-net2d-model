from __future__ import print_function

import os
import pickle

import numpy as np
import SimpleITK as sitk
from keras.models import load_model

from source.pre_process import get_data_list, get_time
from source.pre_process import get_image, irs_train_image
from source.metrics import (dice_coefficient, dice_coefficient_loss)

DATATYPE = {'HGG':0, 'LGG':1, 'TEST':2}
SHAPE = (240, 240, 155)
PATCH = (33, 33, 33)
MODS = 4
SEGMENTATION = 5

test_samples = 110
batch_size = 120
total_pixels = 240 * 240 * 155

output_path = input('Input output path : ')
data_path = input('Input data path : ')

model_dir = output_path + 'model/'
model_path = model_dir + 'u_net3d_model.h5'
predict_dir = output_path + 'prediction/'

try:
    os.mkdir(predict_dir)
except OSError:
    pass

model = load_model(model_path, custom_objects = {'dice_coefficient' : dice_coefficient,
                                                 'dice_coefficient_loss' : dice_coefficient_loss})
class BatchSizeError(Exception):
    pass

def batch_sample_check(num_of_samples, batch_size):
    if num_of_samples in [batch_size, SHAPE[2] % batch_size]:
        pass
    else:
        raise BatchSizeError

def test_predict():
    data_list = get_data_list(data_path)[DATATYPE['TEST']]
    test_size = len(data_list) // MODS
    test_data = get_data(data_list, lower_bound = 0, upper_bound = test_size)
    for sample in range(test_size):
        name = data_list[sample, 0]
        VSD_id = name.split('.')[-2]
        new_name = 'VSD.test_predict_{:03}.{}.mha'.format(sample + 1, VSD_id)
        prediction_sample = test_data[sample]
        steps = SHAPE[2] / batch_size
        remainder = SHAPE[2] % batch_size
        prediction_mri = np.zeros(shape = SHAPE, dtype = np.int16)
        for step in range(steps + 1):
            if step < range(steps + 1)[-1]:
                batch_sample = prediction_sample[:, :, :, (step * batch_size) : ((step + 1) * batch_size)]
                progress_presentage = float((step + 1) * batch_size) / float(SHAPE[2]) * 100
            else:
                batch_sample = prediction_sample[:, :, :, ((step - 1) * batch_size) : (step * batch_size + remainder)]
                progress_presentage = float((step * batch_size + remainder)) / float(SHAPE[2]) * 100
            batch_sample_size = batch_sample.shape[0]
            try:
                batch_sample_check(batch_sample_size, batch_size)
            except BatchSizeError:
                print('Reset the batch_size in [1 - 155]')
            temp_prediction = model.predict(x = batch_sample, batch_size = batch_sample_size, verbose = 0)
            prediction = np.zeros(shape = (batch_sample_size, 240, 240), dtype = np.int16)
            for batch in range(batch_sample_size):
                for channel in range(4):
                    mask = np.in1d(temp_prediction[batch, channel], 1).reshape(240, 240)
                    prediction[batch][mask] = channel + 1
            prediction_mri[:, :, (step * batch_size) : ((step + 1) * batch_size)] = prediction
            print('\r', get_time() + ' : creating {}, {:.1f}%, [{} / {}]'.format(new_name, progress_presentage, sample+1, test_size), end = '')
        sitk_img = sitk.GetImageFromArray(prediction_mri.transpose())
        sitk.WriteImage(sitk_img, predict_dir + new_name)

def manual_predict(data_class, lower_bound, upper_bound = None):
    if upper_bound == None:
        upper_bound = lower_bound + 1
    total = upper_bound - lower_bound
    data_list = get_data_list(data_path)[DATATYPE[data_class]]
    test_data = get_data(data_list, lower_bound, upper_bound)
    for sample in range(total):
        flair_file_name = data_list[sample + lower_bound, 0]
        VSD_id = flair_file_name.split['.'][-2]
        new_name = 'VSD.{}_validation_{:03}.{}.mha'.format(data_class.lower(), sample + 1, VSD_id)
        prediction_sample = test_data[sample]
        steps = SHAPE[2] / batch_size
        remainder = SHAPE[2] % batch_size
        prediction_mri = np.zeros(shape = SHAPE, dtype = np.int16)
        for stpe in range(steps + 1):
            if step < range(steps + 1)[-1]:
                batch_sample = prediction_sample[:, :, :, (step * batch_size) : ((step + 1) * batch_size)]
                progress_presentage = float((step + 1) * batch_size) / float(SHAPE[2]) * 100
            else:
                batch_sample = prediction_sample[:, :, :, (step * batch_size) : ((step + 1) * batch_size)]
                progress_presentage = float((step + 1) * batch_size) / float(SHAPE[2]) * 100
            batch_sample_size = batch_sample.shape[0]
            try:
                batch_sample_check(batch_sample_size, batch_size)
            except BatchSizeError:
                print('Reset the batch_size in [1 - 155]')
            temp_prediction = model.predict(x = batch_sample, batch_size = batch_sample_size, verbose = 0)
            prediction = np.zeros(shape = (batch_sample_size, 240, 240), dtype = np.int16)
            for batch in range(batch_sample_size):
                for channel in range(4):
                    mask = np.in1d(temp_prediction[batch, channel], 1).reshape(240, 240)
                    prediction[batch][mask] = channel + 1
            prediction_mri[:, :, (step * batch_size) : ((step + 1) * batch_size)] = prediction
            print('\r', get_time() + ' : creating {}, {:.1f}%, [{} / {}]'.format(new_name, progress_presentage, sample+1, total), end = '')
        sitk_img = sitk.GetImageFromArray(prediction_mri.transpose())
        sitk.WriteImage(sitk_img, prediction_dir + new_name)

def get_data(data_list, lower_bound, upper_bound):
    total = upper_bound - lower_bound
    if os.path.exists(predict_dir + 'predict.dat'):
        overwrite = input('The predict file exist. Do you want to overwrite ? [y or n]')
        if overwrite == 'n':
            fp = np.memmap(predict_dir + 'predict.dat', mode = 'r', dtype = np.float32,
                           shape = (total, MODS, SHAPE[0], SHAPE[1], SHAPE[2]))
            return fp
    fp = np.memmap(predict_dir + 'predict.dat', mode = 'w+', dtype = np.float32,
                   shape = (total, MODS, SHAPE[0], SHAPE[1], SHAPE[2]))
    with open(output_path + 'Flair_irs.pkl', 'r') as f1:
        flair_irs = pickle.load(f1)
    with open(output_path + 'T1_irs.pkl', 'r') as f2:
        t1_irs = pickle.load(f2)
    with open(output_path + 'T1c_irs.pkl', 'r') as f3:
        t1c_irs = pickle.load(f3)
    with open(output_path + 'T2_irs.pkl', 'r') as f4:
        t2_irs = pickle.load(f4)
    irs_list = [flair_irs, t1_irs, t1c_irs, t2_irs]
    for sample in range(total):
        print('\r', get_time() + ' : getting prediction data {}%'.format(float(sample + 1) / float(total) * 100), end = '')
        for mod in range(MODS):
            img = get_image(data_list[(sample + lower_bound), mod], data_path)
            irs_img = irs_train_image(img, irs_list[mod])
            minpix = np.min(irs_img)
            if minpix < 0:
                irs_img[irs_img != 0] -= minpix
            if mod == MOD['MR_T1']:
                img_e = histogram_equalizing(irs_img.astype(np.uint16))
                fp[sample, mod] = img_e.astype(np.float32)
            else:
                fp[sample, mod] = irs_img
    means, stds = np.load(output_path + 'MeanAndStd.npy')
    for mod in range(MODS):
        print('\r', get_time() + ' : prediction data normalizing {}%'.format(float(mod + 1) / float(MODS) * 100), end = '')
        fp[:, mod, :, :, :] -= means[mod]
        fp[:, mod, :, :, :] /= stds[mod]
    return fp

if __name__ == '__main__':
    prediction_mod = int(input('Prediction MOD (0 : manual_prediction, 1 : test_prediction) : '))
    if prediction_mod:
        test_predict()
    else:
        sample = input('Type (HGG, LGG, TEST)? ')
        data_range = input('''
Input data range
    - HGG : 1 - 220
    - LGG : 1 - 54
    - TEST : 1 - 110
    e.g.)
        input : [98]     - [98] One sample will be predicted
        input : [126, 130] - [126, 127, 128, 129, 130] samples will be predicted
''')
        lower = data_range[0] - 1
        upper = data_range[1]
        manual_predict(sample, lower_bound = lower, upper_bound = upper)

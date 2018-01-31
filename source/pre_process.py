# -*- coding: utf-8 -*-
# python 2

from __future__ import print_function

import os
import pickle
from time import localtime, strftime

import numpy as np
import SimpleITK as sitk
from medpy.filter import IntensityRangeStandardization
from medpy.io import save

from multi_channels_label import get_4_channels_label

MOD = {'MR_Flair':0, 'MR_T1':1, 'MR_T1c':2, 'MR_T2':3, 'OT':4}
MODS = {'HGG':5, 'LGG':5, 'TEST':4}
DATATYPE = {'HGG':0, 'LGG':1, 'TEST':2}
SHAPE = [240, 240, 155]
PATCH = [240, 240]
ROTATE = 4
FLIP = 4

def get_time():
    return strftime('%Y-%m-%d  %H:%M:%S', localtime())

def get_data_list(path):
    name_list = os.listdir(path)
    HGG = {}
    LGG = {}
    TEST = {}
    for filename in name_list:
        temp = filename.split('.')
        typ = temp[0]
        cnt = int(temp[1])
        mod = int(MOD[temp[-3]])
        if typ == "HGG":
            HGG[cnt, mod] = filename
        elif typ == 'LGG':
            LGG[cnt, mod] = filename
        else: #TEST(HGG_LGG)
            TEST[cnt, mod] = filename
    return np.array([HGG, LGG, TEST])

def get_image(img_name, data_path):
    path = data_path
    sitk_img = sitk.ReadImage(path + img_name)
    img = sitk.GetArrayFromImage(sitk_img).transpose()
    minpix = np.min(img)
    if minpix < 0:
        img[img != 0] -= minpix
    return img

def get_orig_data(data_list, dataClass, output_path, data_path):
    mods = MODS[dataClass]
    total = len(data_list[DATATYPE[dataClass]]) // mods
    if mods == 5:
        mods = 4
    fp = np.memmap(output_path + dataClass + '_orig.dat', dtype = np.float32, mode = 'w+',
                   shape = (total, mods, SHAPE[0], SHAPE[1], SHAPE[2]))
    print(get_time() + ': %s get_orig_data STARTED' %(dataClass))
    for sample in range(total):
        for mod in range(mods):
            img = get_image(data_list[DATATYPE[dataClass]][sample, mod], data_path)
            fp[sample, mod] = img
    print(get_time() + ': %s get_orig_data ENDED' %(dataClass))

'''
parameters : cutoff : (float, float)
               Lower and upper cut-off percentiles to exclude outliers.
             landmarks : sequence of floats
               List of percentiles serving as model landmarks, must lie between the cutoffp values.
reference : http://loli.github.io/medpy/generated/medpy.filter.IntensityRangeStandardization.IntensityRangeStandardization.html
'''
def get_trained_irs(data_list, output_path, cutoffp = (1, 20), landmarkp = [2,3,4,5,6, 8,10,12,14, 15,16,17,18,19]): # Default : cutoffp = (1, 99), landmarkp = [10, 20, 30, 40, 50, 60, 70, 90]
    flair_irs = IntensityRangeStandardization(cutoffp = cutoffp, landmarkp = landmarkp)
    t1_irs    = IntensityRangeStandardization(cutoffp = cutoffp, landmarkp = landmarkp)
    t1c_irs   = IntensityRangeStandardization(cutoffp = cutoffp, landmarkp = landmarkp)
    t2_irs    = IntensityRangeStandardization(cutoffp = cutoffp, landmarkp = landmarkp)
    irs_list = [flair_irs, t1_irs, t1c_irs, t2_irs]
    for dataClass in ['HGG', 'LGG', 'TEST']:
        mods = MODS[dataClass]
        total = len(data_list[DATATYPE[dataClass]]) // mods
        if mods == 5:
            mods = 4
        fp = np.memmap(output_path + dataClass + '_orig.dat', dtype = np.float32, mode = 'r',
                       shape = (total, mods, SHAPE[0], SHAPE[1], SHAPE[2]))
        print(get_time() + ': %s StandardIntensityModel training STARTED' %(dataClass))
        for mod in range(mods):
            images = fp[:, mod, :, :, :]
            irs_list[mod] = irs_train_image(images, irs_list[mod], train_mode = True)
        print(get_time() + ': %s StandardIntensityModel training ENDED' %(dataClass))
    with open(output_path + 'Flair_irs.pkl', 'wb') as f1:
        pickle.dump(irs_list[0], f1)
    with open(output_path + 'T1_irs.pkl', 'wb') as f2:
        pickle.dump(irs_list[1], f2)
    with open(output_path + 'T1c_irs.pkl', 'wb') as f3:
        pickle.dump(irs_list[2], f3)
    with open(output_path + 'T2_irs.pkl', 'wb') as f4:
        pickle.dump(irs_list[3], f4)

def get_label_data(dataClass, dataList, output_path, data_path):
    mods = MODS[dataClass]
    total = len(dataList[DATATYPE[dataClass]]) // mods
    fp = np.memmap(output_path + dataClass + '_label.dat', mode = 'w+', dtype = np.int8,
                   shape = (total, SHAPE[0], SHAPE[1], SHAPE[2]))
    print(get_time() + ': %s get_label_data STARTED' %(dataClass))
    for sample in range(total):
        label = get_image(dataList[DATATYPE[dataClass]][sample, mods - 1], data_path)
        fp[sample] = label.astype(np.uint8)
    print(get_time() + ': %s get_label_data ENDED' %(dataClass))

def irs_train_image(image, irs, train_mode = False):
    if train_mode:
        irs = irs.train([image[image > 0]])
        return irs
    else:
        image[image > 0] = irs.transform(image[image > 0], surpress_mapping_check = True)
        return image

def get_data(dataClass, dataList, output_path, data_path):
    mods = MODS[dataClass]
    total = len(dataList[DATATYPE[dataClass]]) // mods
    if mods == 5:
        mods = 4
    fp = np.memmap(output_path + dataClass + '.dat', mode = 'w+', dtype = np.float32,
                   shape = (total, mods, SHAPE[0], SHAPE[1], SHAPE[2]))
    with open(output_path + 'Flair_irs.pkl', 'rb') as f1:
        flair_irs = pickle.load(f1)
    with open(output_path + 'T1_irs.pkl', 'rb') as f2:
        t1_irs = pickle.load(f2)
    with open(output_path + 'T1c_irs.pkl', 'rb') as f3:
        t1c_irs = pickle.load(f3)
    with open(output_path + 'T2_irs.pkl', 'rb') as f4:
        t2_irs = pickle.load(f4)
    irs_list = [flair_irs, t1_irs, t1c_irs, t2_irs]
    print(get_time() + ': %s get_data STARTED' %(dataClass))
    for sample in range(total):
        for mod in range(mods):
            img = get_image(dataList[DATATYPE[dataClass]][sample, mod], data_path)
            irs_img = irs_train_image(img, irs_list[mod])
            irs_img[irs_img == 0] = 0
            minpix = np.min(irs_img)
            if minpix < 0:
                irs_img[irs_img != 0] -= minpix
            if mod == MOD['MR_T1']:
                img_e = histogram_equalizing(irs_img.astype(np.uint16))
                fp[sample, mod] = img_e.astype(np.float32)
            else:
                fp[sample, mod] = irs_img
    print(get_time() + ': %s get data ENDED' %(dataClass))

'''
reference : http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_histograms/py_histogram_equalization/py_histogram_equalization.html#histogram-equalization
'''
def histogram_equalizing(img):
    maxpix = np.max(img)
    hist, bins = np.histogram(img.flatten(), maxpix+1, [0, maxpix+1])
    cdf = hist.cumsum()
    # cdf_normalized = cdf * hist.max() / cdf.max()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min()) * maxpix / (cdf_m.max() - cdf_m.min())
    cdf_e = np.ma.filled(cdf_m, 0).astype(np.uint16)
    img_e = cdf_e[img]
    # hist_e, bins_e = np.histogram(img_e.flatten(), maxpix+1, [0,maxpix+1])
    # cdf_normalized = cdf_e * hist_e.max() / cdf_e.max()
    return img_e

def image_rotate(img, number_of_rotate):
    if np.ndim(img) == 3:
        img = np.rot90(img, number_of_rotate, axes = (1, 2))
    elif np.ndim(img) == 2:
        img = np.rot90(img, number_of_rotate, axes = (0, 1))
    return img

def image_flip(train, label, number_of_flip):
    mods = 4
    fliped_train = np.zeros(shape = (mods, SHAPE[0], SHAPE[1]), dtype = np.float32)
    fliped_lable = np.zeros(shape = (SHAPE[0], SHAPE[1]), dtype = np.int8)
    if number_of_flip == 1:
        for mod in range(mods):
            fliped_train[mod] = np.fliplr(train[mod])
        fliped_label = np.fliplr(label)
    elif number_of_flip == 2:
        for mod in range(mods):
            fliped_train[mod] = np.flipud(train[mod])
        fliped_label = np.flipud(label)
    elif number_of_flip == 3:
        for mod in range(mods):
            fliped_train[mod] = np.fliplr(np.flipud(train[mod]))
        fliped_label = np.fliplr(np.flipud(label))
    return fliped_train, fliped_label

def get_patch(dataClasses, dataList, output_path):
    hgg_mods = MODS[dataClasses[0]]
    hgg_total = len(dataList[DATATYPE[dataClasses[0]]]) // hgg_mods
    lgg_mods = MODS[dataClasses[1]]
    lgg_total = len(dataList[DATATYPE[dataClasses[1]]]) // lgg_mods

    if hgg_mods == 5 and lgg_mods == 5:
        mods = 4
    total = {'HGG' : hgg_total, 'LGG' : lgg_total}
    height = SHAPE[2]
    train_size = np.sum(total.values()) * height * ROTATE * FLIP
    train_patch = np.memmap(output_path + 'train.pat', mode = 'w+', dtype = np.float32,
                         shape = (train_size, mods, SHAPE[0], SHAPE[1]))
    label_patch = np.memmap(output_path + 'label.pat', mode = 'w+', dtype = np.int8,
                         shape = (train_patch.shape[0], SHAPE[0], SHAPE[1]))
    cnt = 0
    for dataClass in dataClasses:
        fp = np.memmap(output_path + dataClass + '.dat', mode = 'r', dtype = np.float32,
                       shape = (total[dataClass], mods, SHAPE[0], SHAPE[1], SHAPE[2]))
        fp_label = np.memmap(output_path + dataClass + '_label.dat', mode = 'r', dtype = np.int8,
                       shape = (total[dataClass], SHAPE[0], SHAPE[1], SHAPE[2]))
        print(get_time() + ': %s get_patch STARTED' %(dataClass))
        for sample in range(total[dataClass]):
            for h in range(height):
                plain_img = fp[sample, :, :, :, h]
                plain_label = fp_label[sample, :, :, h]
                for flip in range(FLIP):
                    if flip == 0:
                        for rot in range(ROTATE):
                            train_patch[cnt] = image_rotate(plain_img, rot)
                            label_patch[cnt] = image_rotate(plain_label, rot)
                            cnt += 1
                    else:
                        fliped_train, fliped_label = image_flip(plain_img, plain_label, flip)
                        for rot in range(ROTATE):
                            train_patch[cnt] = image_rotate(fliped_train, rot)
                            label_patch[cnt] = image_rotate(fliped_label, rot)
                            cnt += 1
    print(get_time() + ': %s get_patch ENDED' %(dataClasses))
    state = np.random.get_state()
    np.random.shuffle(train_patch)
    print(get_time() + ': %s train data shuffle ENDED' %(dataClasses))
    np.random.set_state(state)
    np.random.shuffle(label_patch)
    print(get_time() + ': %s label data shuffle ENDED' %(dataClasses))

def get_mean_and_std(output_path):
    mods = 4
    fp = np.memmap(output_path + 'train.pat', mode = 'r', dtype = np.float32)
    fp = fp.reshape(-1, mods, SHAPE[0], SHAPE[1])
    stds = np.zeros(mods, dtype = np.float32)
    print(get_time() + ': get_mean_and_std STARTED')
    for mod in range(mods):
        stds[mod] = np.std(fp[:100, mod, :, :])
    means = np.mean(fp, axis=(0,2,3))
    mean_and_std = np.array([means, stds]).astype(np.float32)
    np.save(output_path + 'MeanAndStd.npy', mean_and_std)
    print(get_time() + ': get_mean_and_std ENDED')

def gauss_norm(output_path):
    mods = 4
    fp = np.memmap(output_path + 'train.pat', mode = 'r+', dtype = np.float32)
    fp = fp.reshape(-1, mods, PATCH[0], PATCH[1])
    means, stds = np.load(output_path + 'MeanAndStd.npy')
    print(get_time() + ': gauss_normalization STARTED')
    for mod in range(mods):
        fp[:, mod, :, :] -= means[mod]
        fp[:, mod, :, :] /= stds[mod]
    print(get_time() + ': gauss_normalization ENDED')

def main_process():
    data_path = input('input data path : ')
    output_path = input('input output path : ')
    data_list = get_data_list(data_path)
    get_orig_data(data_list, 'HGG', output_path, data_path)
    get_orig_data(data_list, 'LGG', output_path, data_path)
    get_orig_data(data_list, 'TEST', output_path, data_path)
    get_trained_irs(data_list, output_path)
    get_label_data('HGG', data_list, output_path, data_path)
    get_label_data('LGG', data_list, output_path, data_path)
    get_data('HGG', data_list, output_path, data_path)
    get_data('LGG', data_list, output_path, data_path)
    get_patch(['HGG','LGG'], data_list, output_path)
    get_mean_and_std(output_path)
    gauss_norm(output_path)
    get_4_channels_label(output_path)
    print('-'*20, 'all programs done', '-'*20)

if __name__ == '__main__':
    main_process()

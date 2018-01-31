from __future__ import print_function

import numpy as np

def get_4_channels_label(output_path):
    label_1_channel = np.memmap(output_path + 'label.pat', dtype = np.int8, mode = 'r')
    label_1_channel = label_1_channel.reshape(-1, 1, 240, 240)

    totoal = label_1_channel.shape[0]
    num_of_channels = 4

    label_4_channels = np.memmap(output_path + 'label_4_channels.pat', dtype = np.int8, mode = 'w+',
                                 mode = (total, num_of_channels, 240, 240))
    for step in range(total):
        for channel in range(num_of_channels):
            temp_img = np.zeros(shape = (240, 240), dtype = np.int8)
            mask = np.in1d(label_1_channel[step, 0], channel + 1).reshape(240, 240)
            temp_img[mask] = 1
            label_4_channels[step, channel] = temp_img
        print('\r{} / {}'.format(step + 1, total), end = '')

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def load_weights_and_biases(numpy_file_name):
    try:
        variable_scope_to_name_to_value_dict = np.load(numpy_file_name).item()
        print(variable_scope_to_name_to_value_dict)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))


if __name__ == "__main__":
    alexnet_caffe_to_tf_numpy_file_path = '/home/gao/Downloads/Faster-RCNN_TF_gby/data/pretrain_model/bvlc_alexnet.npy'
    alexnet_faster_r_cnn_numpy_file_path = '/home/gao/Downloads/Faster-RCNN_TF_gby/data/pretrain_model/bvlc_alexnet_faster_r_cnn.npy'
    load_weights_and_biases(alexnet_faster_r_cnn_numpy_file_path)
    load_weights_and_biases(alexnet_caffe_to_tf_numpy_file_path)

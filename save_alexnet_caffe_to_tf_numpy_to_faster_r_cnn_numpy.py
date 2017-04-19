from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


def save_weights_and_biases_to_numpy_array_from_checkpoint_file(in_numpy_file_name, out_numpy_file_name) :
    try:
        variable_scope_to_name_to_value_dict = {}
        caffe_to_tf_numpy_data_dict = np.load(in_numpy_file_name).item()
        all_variable_name_list = caffe_to_tf_numpy_data_dict.keys()
        for uKey in sorted(all_variable_name_list):
            key = uKey.encode('ascii', 'ignore')
            variable_name_scope_prefix = key
            variable_weights_name = 'weights'
            variable_biases_name = 'biases'
            if len(caffe_to_tf_numpy_data_dict[key][0].shape) > 1:
                weights_index = 0
                biases_index = 1
            else:
                weights_index = 1
                biases_index = 0
            variable_weights_value = caffe_to_tf_numpy_data_dict[key][weights_index]
            variable_biases_value = caffe_to_tf_numpy_data_dict[key][biases_index]
            variable_scope_to_name_to_value_dict[variable_name_scope_prefix] = {variable_weights_name: variable_weights_value,
                                                                                variable_biases_name: variable_biases_value}
            print("tensor_name: ", key)
        # np.savez(numpy_file_name, variable_scope_to_name_to_value_dict)
        np.save(out_numpy_file_name, variable_scope_to_name_to_value_dict)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))


if __name__ == "__main__":
    alexnet_caffe_to_tf_numpy_file_path = '/home/gao/Downloads/Faster-RCNN_TF_gby/data/pretrain_model/bvlc_alexnet.npy'
    alexnet_faster_r_cnn_numpy_file_path = '/home/gao/Downloads/Faster-RCNN_TF_gby/data/pretrain_model/bvlc_alexnet_faster_r_cnn.npy'
    save_weights_and_biases_to_numpy_array_from_checkpoint_file(alexnet_caffe_to_tf_numpy_file_path, alexnet_faster_r_cnn_numpy_file_path)

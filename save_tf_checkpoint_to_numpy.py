# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""A simple script for inspect checkpoint files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os

import numpy as np

import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
from tensorflow.python.platform import app
from tensorflow.python.platform import flags

FLAGS = None


def save_weights_and_biases_to_numpy_array_from_checkpoint_file(checkpoint_file_name, numpy_file_name,
                                                                variable_name_filter_func):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        all_variable_name_list = var_to_shape_map.keys()
        to_save_variable_name_list = filter(variable_name_filter_func, all_variable_name_list)
        variable_scope_to_name_to_value_dict = {}
        for key in sorted(to_save_variable_name_list):
            name_scopes = key.split('/')
            variable_name_scope_prefix = str.join('/', name_scopes[:-1])
            variable_name = name_scopes[-1]
            variable_value = reader.get_tensor(key)
            variable_scope_to_name_to_value_dict[variable_name_scope_prefix] = {variable_name: variable_value}
            print("tensor_name: ", key)
            print(variable_value)
        # np.savez(numpy_file_name, variable_scope_to_name_to_value_dict)
        np.save(numpy_file_name, variable_scope_to_name_to_value_dict)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in checkpoint_file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(checkpoint_file_name.split(".")[0:-1])
            v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))


def print_tensors_in_checkpoint_file(file_name, tensor_name, all_tensors):
    """Prints tensors in a checkpoint file.

  If no `tensor_name` is provided, prints the tensor names and shapes
  in the checkpoint file.

  If `tensor_name` is provided, prints the content of the tensor.

  Args:
    file_name: Name of the checkpoint file.
    tensor_name: Name of the tensor in the checkpoint file to print.
    all_tensors: Boolean indicating whether to print all tensors.
  """
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        if all_tensors:
            var_to_shape_map = reader.get_variable_to_shape_map()
            for key in sorted(var_to_shape_map):
                print("tensor_name: ", key)
                print(reader.get_tensor(key))
        elif not tensor_name:
            print(reader.debug_string().decode("utf-8"))
        else:
            print("tensor_name: ", tensor_name)
            print(reader.get_tensor(tensor_name))
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))
        if "corrupted compressed block contents" in str(e):
            print("It's likely that your checkpoint file has been compressed "
                  "with SNAPPY.")
        if ("Data loss" in str(e) and
                (any([e in file_name for e in [".index", ".meta", ".data"]]))):
            proposed_file = ".".join(file_name.split(".")[0:-1])
            v2_file_error_template = """
It's likely that this is a V2 checkpoint and you need to provide the filename
*prefix*.  Try removing the '.' and extension.  Try:
inspect checkpoint --file_name = {}"""
            print(v2_file_error_template.format(proposed_file))


def main(unused_argv):
    if not FLAGS.file_name:
        print(
            "Usage: save_tf_checkpoint_to_numpy.py --file_name=checkpoint_file_name --out_numpy_file_path=output_numpy_data_file_path")
        sys.exit(1)
    else:
        if os.path.isdir(FLAGS.file_name):
            checkpoint_path = tf.train.latest_checkpoint(FLAGS.file_name)
            if checkpoint_path is not None:
                save_weights_and_biases_to_numpy_array_from_checkpoint_file(checkpoint_path, FLAGS.out_numpy_file_path,
                                                                            lambda s: s.endswith(
                                                                                'weights') or s.endswith('biases'))
            else:
                raise NameError("%s does not contain any checkpoints!" % FLAGS.file_name)
        else:
            save_weights_and_biases_to_numpy_array_from_checkpoint_file(FLAGS.file_name, FLAGS.out_numpy_file_path,
                                                                        lambda s: s.endswith('weights') or s.endswith(
                                                                            'biases'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--file_name", type=str, default="", help="Checkpoint filename. "
                                                  "Note, if using Checkpoint V2 format, file_name is the "
                                                  "shared prefix between all files in the checkpoint.")
    parser.add_argument(
        "--out_numpy_file_path",
        type=str,
        default="",
        help="Output numpy obj file path")
    parser.add_argument(
        "--tensor_name",
        type=str,
        default="",
        help="Name of the tensor to inspect")
    parser.add_argument(
        "--all_tensors",
        nargs="?",
        const=True,
        type="bool",
        default=False,
        help="If True, print the values of all the tensors.")
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

import numpy as np

from tensorflow.python.platform import app

FLAGS = None


def load_weights_and_biases_to_numpy_array_from_checkpoint_file(numpy_file_name):
    try:
        variable_scope_to_name_to_value_dict = np.load(numpy_file_name).item()
        print(variable_scope_to_name_to_value_dict)
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))


def main(unused_argv):
    if not FLAGS.file_name:
        print(
            "Usage: save_tf_checkpoint_to_numpy.py --file_name=numpy_data_file_path")
        sys.exit(1)
    else:
        load_weights_and_biases_to_numpy_array_from_checkpoint_file(FLAGS.file_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--file_name", type=str, default="", help="Checkpoint filename. "
                                                  "Note, if using Checkpoint V2 format, file_name is the "
                                                  "shared prefix between all files in the checkpoint.")
    FLAGS, unparsed = parser.parse_known_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)

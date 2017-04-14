# import _init_paths
import tensorflow as tf
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import os, sys, cv2
import argparse
from networks.factory import get_network


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='VGGnet_test')
    parser.add_argument('--model', dest='model', help='Model path',
                        default=' ')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))

    with tf.Graph().as_default():
        # init session
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            net = get_network(args.demo_net)
            saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
            saver.restore(sess, args.model)
            print ('\n\nLoaded network {:s}'.format(args.model))

            saver_gby = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
            saver_gby.save(sess, '/home/gao/Data/faster_r_cnn/tf-models/VGGnet_fast_rcnn_iter_70000.ckpt')
            writer = tf.summary.FileWriter('/home/gao/Data/faster_r_cnn/tf-models', sess.graph)
            writer.close()


# import _init_paths
from __future__ import print_function
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
from caffe_classes import class_names as imagenet_class_names


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Faster R-CNN demo')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        default='Alexnet_Alexnet-test-debug')
    parser.add_argument('--model', dest='model', help='Model path',
                        # default='/home/gao/Downloads/Faster-RCNN_TF_gby/output/faster_rcnn_end2end/voc_2007_trainval/VGGnet_fast_rcnn_iter_70000.ckpt')
                        default='/home/gao/Downloads/Faster-RCNN_TF_gby/data/pretrain_model/bvlc_alexnet_faster_r_cnn.npy')

    args = parser.parse_args()

    return args
if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals

    args = parse_args()

    if args.model == ' ':
        raise IOError(('Error: Model not found.\n'))
        
    # init session
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    # load network
    net = get_network(args.demo_net)
    #sess.run(tf.initialize_all_variables())
    # load model
    saver = tf.train.Saver(write_version=tf.train.SaverDef.V1)
    net.load(args.model, sess, saver)

    print ('\n\nLoaded network {:s}'.format(args.model))

    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']

    im_path_list = tf.gfile.Glob(cfg.DATA_DIR+"/demo/*.*g")
    im_names = map(lambda p: os.path.basename(p), im_path_list)

    for im_name in im_names:
        print ('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print ('Demo for data/demo/{}'.format(im_name))
        im_file = os.path.join(cfg.DATA_DIR, 'demo', im_name)
        im_orig_int = cv2.imread(im_file)
        im_orig_float = im_orig_int.astype(np.float32, copy=True)
        im_orig_sub_mean = im_orig_float - cfg.PIXEL_MEANS
        im = cv2.resize(im_orig_sub_mean, (227, 227), interpolation=cv2.INTER_LINEAR)
        im_tensor = im.reshape((1, 227, 227, 3))
        feed_dict={net.data: im_tensor}
        class_prob = sess.run(net.get_output('cls_prob'), feed_dict=feed_dict)
        class_name = imagenet_class_names[np.argmax(class_prob)]
        im_display = im_orig_int[:, :, (2, 1, 0)]
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.imshow(im_display, aspect='equal')
        print("Class: " + class_name + ", probability: %.4f" %class_prob[0, np.argmax(class_prob)])
        plt.show()
        # demo(sess, net, im_name)




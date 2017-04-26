import tensorflow as tf
from networks.network import Network

n_classes = 1000
_feat_stride = [16,]
anchor_scales = [8, 16, 32] 

class Alexnet_test_debug(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info})
        self.trainable = trainable
        self.setup()

    def setup(self):
        (self.feed('data')
         .conv(11, 11, 96, 4, 4, name='conv1', trainable=True, padding='VALID')
         .lrn(radius=2, alpha=2e-05, beta=0.75, name='norm1', bias=1.0)
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool1')
         .conv(5, 5, 256, 1, 1, name='conv2', trainable=True, padding='SAME', group=2)
         .lrn(radius=2, alpha=2e-05, beta=0.75, name='norm2', bias=1.0)
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
         .conv(3, 3, 384, 1, 1, name='conv3', trainable=True, padding='SAME')
         .conv(3, 3, 384, 1, 1, name='conv4', trainable=True, padding='SAME', group=2)
         .conv(3, 3, 256, 1, 1, name='conv5', trainable=True, padding='SAME', group=2)
         .max_pool(3, 3, 2, 2, padding='VALID', name='pool5')
         .fc(4096, name='fc6', is_feed_in_transpose=False)
         .fc(4096, name='fc7', is_feed_in_transpose=False)
         .fc(n_classes, relu=False, name='fc8', is_feed_in_transpose=False)
         .softmax(name='cls_prob'))



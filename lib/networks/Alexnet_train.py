import tensorflow as tf
from networks.network import Network


#define

n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class Alexnet_train(Network):
    def __init__(self, trainable=True):
        self.inputs = []
        self.data = tf.placeholder(tf.float32, shape=[None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape=[None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape=[None, 5])
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data':self.data, 'im_info':self.im_info, 'gt_boxes':self.gt_boxes})
        self.trainable = trainable
        self.setup()

        # create ops and placeholders for bbox normalization process
        with tf.variable_scope('bbox_pred', reuse=True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape=weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape=biases.get_shape())

            self.bbox_weights_assign = weights.assign(self.bbox_weights)
            self.bbox_bias_assign = biases.assign(self.bbox_biases)

    def setup(self):
        (self.feed('data')
             .conv(11, 11, 64, 4, 4, name='alexnet_v2/conv1', trainable=True, padding='VALID')
             .max_pool(3, 3, 2, 2, padding='VALID', name='alexnet_v2/pool1')
             .conv(5, 5, 192, 1, 1, name='alexnet_v2/conv2', trainable=True)
             .max_pool(3, 3, 2, 2, padding='VALID', name='alexnet_v2/pool2')
             .conv(3, 3, 384, 1, 1, name='alexnet_v2/conv3')
             .conv(3, 3, 384, 1, 1, name='alexnet_v2/conv4')
             .conv(3, 3, 256, 1, 1, name='alexnet_v2/conv5')
             .max_pool(3, 3, 2, 2, padding='VALID', name='alexnet_v2/pool5'))

        #========= RPN ============
        (self.feed('alexnet_v2/pool5')
             .conv(3,3,512,1,1,name='rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*2 ,1 , 1, padding='VALID', relu = False, name='rpn_cls_score'))

        (self.feed('rpn_cls_score','gt_boxes','im_info','data')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data' ))

        # Loss of rpn_cls & rpn_boxes

        (self.feed('rpn_conv/3x3')
             .conv(1,1,len(anchor_scales)*3*4, 1, 1, padding='VALID', relu = False, name='rpn_bbox_pred'))

        #========= RoI Proposal ============
        (self.feed('rpn_cls_score')
             .reshape_layer(2,name = 'rpn_cls_score_reshape')
             .softmax(name='rpn_cls_prob'))

        (self.feed('rpn_cls_prob')
             .reshape_layer(len(anchor_scales)*3*2,name = 'rpn_cls_prob_reshape'))

        (self.feed('rpn_cls_prob_reshape','rpn_bbox_pred','im_info')
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN',name = 'rpn_rois'))

        (self.feed('rpn_rois','gt_boxes')
             .proposal_target_layer(n_classes,name = 'roi-data'))


        #========= RCNN ============
        (self.feed('alexnet_v2/pool5', 'roi-data')
             .roi_pool(7, 7, 1.0/16, name='pool_5')
             .fc(4096, name='alexnet_v2/fc6')
             .dropout(0.5, name='drop6')
             .fc(4096, name='alexnet_v2/fc7')
             .dropout(0.5, name='drop7')
             .fc(n_classes, relu=False, name='cls_score')
             .softmax(name='cls_prob'))

        (self.feed('drop7')
             .fc(n_classes*4, relu=False, name='bbox_pred'))


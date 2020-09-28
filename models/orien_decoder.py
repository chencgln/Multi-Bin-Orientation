import tensorflow as tf

from models import common
from config.config import cfg
from models.backbones import backbone_provider

class OrienDecoder():
    def __init__(self, trainable, backbone="darknet53"):
        self.trainable        = trainable
        self.BINS             = cfg.MODEL.ORIEN_BINS
        self.backbone         = backbone_provider.backbone_fn(backbone)
        self.orien_conf_output = None
        self.orien_offset_output = None

    def forward(self, input_data):
        input_data = input_data/255.
        feat_32, _, _ = self.backbone(input_data, self.trainable)
        with tf.variable_scope("orien_head", reuse=tf.AUTO_REUSE): 
            net = common.flatten_layer(feat_32)

            orien_conf = common.dot_product_layer(net, neurons=256, name='fc_conf_0')
            orien_conf = tf.nn.dropout(orien_conf, 0.5, name='do_conf_0')
            orien_conf = common.dot_product_layer(orien_conf, neurons=self.BINS, name='fc_conf_1', activation="identity")
            self.orien_conf_output = orien_conf
            pred_conf = tf.nn.softmax(orien_conf)
    
            orien_offset = common.dot_product_layer(net, neurons=256, name='fc_offset_0')
            orien_offset = tf.nn.dropout(orien_offset, 0.5, name='do_offset_0')
            orien_offset = common.dot_product_layer(orien_offset, neurons=self.BINS*2, name='fc_offset_1', activation="identity")
            orien_offset = tf.reshape(orien_offset, [-1, self.BINS, 2])
            orien_offset = tf.nn.l2_normalize(orien_offset, dim=2)
            self.orien_offset_output = orien_offset
            pred_offset = orien_offset

        return pred_conf, pred_offset

    def cal_loss(self, label_conf, label_offset):
        assert((self.orien_conf_output is not None) and (self.orien_offset_output is not None))
        loss_conf = self.confidence_loss(label_conf, self.orien_conf_output)
        loss_offset = self.offset_loss(label_offset, self.orien_offset_output)

        total_loss = 8.*loss_offset + loss_conf
        return total_loss, loss_conf, loss_offset

    def confidence_loss(self, label_conf, conf_output):
        loss_c = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label_conf, logits=conf_output))
        return loss_c

    def offset_loss(self, label_offset, offset_output):
        # Find number of anchors
        anchors = tf.reduce_sum(tf.square(label_offset), axis=2)
        anchors = tf.greater(anchors, tf.constant(0.5))
        anchors = tf.reduce_sum(tf.cast(anchors, tf.float32), 1)

        # Define the loss
        loss = (label_offset[:,:,0]*offset_output[:,:,0] + label_offset[:,:,1]*offset_output[:,:,1])
        loss = tf.reduce_sum((2 - 2 * tf.reduce_mean(loss, axis=0))) / anchors

        return tf.reduce_mean(loss) 
import os
import cv2
import time
import shutil
import argparse
import numpy as np
import os.path as osp
import tensorflow as tf

from utils import utils
from models import common
from config.config import cfg
from models.orien_decoder import OrienDecoder
from utils.dataset.kitti_loader import KittiLoader

class Configs():
    def __init__(self):
        self.checkpoint_path    = cfg.EVAL.CHECKPOINT
        self.moving_ave_decay   = cfg.MODEL.MOVING_AVE_DECAY
        self.output_dir           = cfg.EVAL.OUTPUT_PATH
        self.eval_file_list           = cfg.DATASET.VAL_LIST
        self.eval_root           = cfg.DATASET.ROOT
        self.BINS               = cfg.MODEL.ORIEN_BINS
        
def evaluate(config):
    if osp.exists(config.output_dir) and len(os.listdir(config.output_dir)):
        print('=> clear %s dir' % config.output_dir)
        os.system('rm -rf %s/*' % config.output_dir)

    with tf.name_scope('input'):
        input_data = tf.placeholder(dtype=tf.float32, shape=[None, 224,224,3], name='input_data')
        trainable  = tf.placeholder(dtype=tf.bool,    name='trainable')

    
    with tf.name_scope('model'):
        model = OrienDecoder(trainable, "darknet53")
        tf_pred_conf, tf_pred_off = model.forward(input_data)
    # """
    # :param input_checkpoint:
    # :return:
    # """
    # saver = tf.train.import_meta_graph(config.checkpoint_path + '.meta', clear_devices=True)
    # graph = tf.get_default_graph()
    # input_graph_def = graph.as_graph_def()
    # with tf.Session() as sess:
    #     saver.restore(sess, config.checkpoint_path)
    #     output_graph_def = tf.graph_util.convert_variables_to_constants(
    #         sess=sess,
    #         input_graph_def=input_graph_def,
    #         output_node_names=[var.name[:-2] for var in tf.global_variables()])
    #     for var in tf.global_variables():
    #         print(var.name[:-2])
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        saver = tf.train.Saver()#ema_obj.variables_to_restore())
        saver.restore(sess, config.checkpoint_path)

        image_list, label_list = utils.load_dataset_list(config.eval_file_list)
        for idx, image_path in enumerate(image_list):
            img = cv2.imread(image_path)
            bbox = np.array(label_list[idx][:4], dtype=np.int16)
            label_confs = label_list[idx][4:4+config.BINS]
            label_offs = label_list[idx][4+config.BINS:]

            roi = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            roi = cv2.resize(roi[..., [2,1,0]], (224,224))
            roi = roi[np.newaxis, ...]
            print(label_confs, label_offs)
            pred_conf, pred_off = sess.run([tf_pred_conf, tf_pred_off], feed_dict={input_data: roi, trainable: False})
            print(pred_conf, pred_off)
                    
# def do_inference(sess, eval_list):
#     with open(config.eval_file_list, 'r') as lf:
#         for num, line in enumerate(lf):
#     # for _ in range(1):
#     # data_dir = "/data/cgl/tf_yolo_gitlab/tensorflow_yolov3/yyang_data"
#     # for root, dirs, files in os.walk(data_dir):
#     #     for ff in files:
#     #         if ff[-4:]!=".jpg":
#     #             continue
#             # annotation = [os.path.join(root, ff)]
#             annotation = line.strip().split()
#             if len(annotation)==2:

if __name__ == '__main__': 
    evaluate(Configs())
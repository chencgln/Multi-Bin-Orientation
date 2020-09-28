import random
import numpy as np
import tensorflow as tf

from utils import utils
from config.config import cfg

class KittiLoader():
    def __init__(self, list_file, batch_size, shuffle=True):
        # self.parser = DatasetParser()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.iters_per_epoch = -1

        self.build_dataset(list_file)

    def build_dataset(self, list_file):
        image_list, label_list = utils.load_dataset_list(list_file)
        self.iters_per_epoch  = int(np.ceil(len(image_list) / self.batch_size))
        dataset = tf.data.Dataset.from_tensor_slices((image_list, label_list))
        dataset = dataset.map(self.parse_image_label, num_parallel_calls=4)
        if self.shuffle:
            dataset = dataset.shuffle(4)
        dataset = dataset.batch(self.batch_size, drop_remainder=True).repeat(cfg.TRAIN.TRAINING_EPOCHS)
        dataset = dataset.prefetch(buffer_size=4)
        self.iterator = dataset.make_one_shot_iterator()

    def parse_image_label(self, image_file, label_data):
        BINS = cfg.MODEL.ORIEN_BINS
        image_data = tf.image.decode_jpeg(tf.read_file(image_file), channels=3)
        image = tf.cast(image_data, tf.float32)
        image_hw = tf.cast(tf.shape(image)[:2], tf.float32)
        label_bbox = tf.concat([label_data[1:2]/image_hw[0], label_data[0:1]/image_hw[1], label_data[3:4]/image_hw[0], label_data[2:3]/image_hw[1]], axis=0)
        batch_ids = tf.constant([0])
        target_size = (cfg.MODEL.INPUT_SIZE_HW[0], cfg.MODEL.INPUT_SIZE_HW[1])
        image = tf.image.crop_and_resize(image[tf.newaxis, ...], label_bbox[tf.newaxis, ...], batch_ids, crop_size=target_size)[0]
        
        label_orien_conf   = label_data[4:4+BINS]
        label_orien_offset = tf.reshape(label_data[4+BINS:], (BINS, 2))
        
        return image, label_orien_conf, label_orien_offset

    def get_next(self):
        return self.iterator.get_next()
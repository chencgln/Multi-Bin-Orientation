import cv2
import numpy as np
import tensorflow as tf

from config.config import cfg
from utils import utils
from utils.dataset import kitti_loader

def test_data_loader():
    data_loader = kitti_loader.KittiLoader(cfg.DATASET.VAL_LIST, 2)
    batch_image, batch_conf, batch_offset = data_loader.get_next()
    with tf.Session() as sess:
        try:
            while True:
                b_img, b_conf, b_offset = sess.run([batch_image, batch_conf, batch_offset])
                print(b_img.shape, b_conf.shape, b_offset.shape)
                for idx, img in enumerate(b_img):
                    img = np.asarray(img, dtype=np.uint8)
                    h, w, _ = img.shape

                    confs = b_conf[idx]
                    offsets = b_offset[idx]
                    alpha = utils.parse_multi_bin(confs, offsets)
                    alpha = alpha/np.pi*180
                 
                    cv2.putText(img, "%.1f"%alpha, (10, int(h/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                    cv2.imshow("image", img)
                    cv2.waitKey()
        except tf.errors.OutOfRangeError:
            pass

if __name__=="__main__":
    test_data_loader()
from easydict import EasyDict as edict

__C                             = edict()
cfg                             = __C

__C.MODEL                       = edict()
__C.MODEL.BACKBONE              = "darknet53"
__C.MODEL.INPUT_SIZE_HW         = [224, 224]
__C.MODEL.MOVING_AVE_DECAY      = 0.9995
__C.MODEL.ORIEN_BINS            = 2
__C.MODEL.ORIEN_BIN_OVERLAP     = 0.1

__C.DATASET                     = edict()
__C.DATASET.CLASSES              = 'config/names/kitti.names'
__C.DATASET.ROOT                = None
__C.DATASET.TRAIN_LIST          = "/data/cgl/tf_yolo_gitlab/multi_tasks/tensorflow_yolov3_orien_dir/train_val_sets/train_kitti_sense.txt"
__C.DATASET.VAL_LIST            = "/data/cgl/datasets/driving_stereo/train_val_append_sets/train_kitti.txt"

# __C.DATASET.TRANS_AUG           = False
__C.DATASET.COLOR_AUG           = True

__C.TRAIN                       = edict()
__C.TRAIN.OPTIMIZER             = "Adam"
__C.TRAIN.SAVING_STEP           = 3000
__C.TRAIN.BATCH_SIZE            = 10
__C.TRAIN.LEARN_RATE_INIT       = 1e-5
__C.TRAIN.LEARN_RATE_END        = 1e-8
__C.TRAIN.TRAINING_EPOCHS       = 80
__C.TRAIN.CHECKPOINT            = None#"./checkpoint/kitti_sense_1/yolov3_1.1644.ckpt-78"
__C.TRAIN.SAVER_DIR             = "./checkpoint/kitti_sense_2"
__C.TRAIN.LOG_STEPS             = 10

__C.EVAL                        = edict()
__C.EVAL.BATCH_SIZE             = 12
__C.EVAL.OUTPUT_PATH            = "./output_kitti_sense_0917"
__C.EVAL.CHECKPOINT             = "./checkpoint/kitti_sense_2/modesl_16.7375.ckpt-3"
__C.EVAL.SCORE_THRESHOLD        = 0.4
__C.EVAL.IOU_THRESHOLD          = 0.5
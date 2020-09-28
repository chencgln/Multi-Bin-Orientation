import os
import time
import shutil
import argparse
import numpy as np
import tensorflow as tf

from models import common
from config.config import cfg
from models.orien_decoder import OrienDecoder
from utils.dataset.kitti_loader import KittiLoader

class Configs():
    def __init__(self):
        self.batch_size         = cfg.TRAIN.BATCH_SIZE
        self.learn_rate_init    = cfg.TRAIN.LEARN_RATE_INIT
        self.learn_rate_end     = cfg.TRAIN.LEARN_RATE_END
        self.training_epochs    = cfg.TRAIN.TRAINING_EPOCHS
        self.checkpoint_path    = cfg.TRAIN.CHECKPOINT
        self.moving_ave_decay   = cfg.MODEL.MOVING_AVE_DECAY
        self.optimizer_func     = common.tf_optimizer_func(cfg.TRAIN.OPTIMIZER)

        self.saving_step        = cfg.TRAIN.SAVING_STEP
        self.log_step           = cfg.TRAIN.LOG_STEPS
        self.save_dir           = cfg.TRAIN.SAVER_DIR

        self.train_list         = cfg.DATASET.TRAIN_LIST
        self.val_list           = cfg.DATASET.VAL_LIST
        
def train(config):
    
    global_step = tf.Variable(1.0, dtype=tf.float64, trainable=False, name='global_step')
    trainable   = tf.placeholder(dtype=tf.bool, name='training')

    ### dataloader
    print("Loading dataset...")
    trainset = KittiLoader(config.train_list, config.batch_size)
    valset  = KittiLoader(config.val_list, config.batch_size)
    batch_image, batch_conf, batch_offset = tf.cond(trainable, lambda: trainset.get_next(), lambda: valset.get_next())
    print("Dataset loaded.")
 
    with tf.name_scope('learning_rate'):
        train_steps = tf.constant( config.training_epochs * trainset.iters_per_epoch,
                                    dtype=tf.float64, name='train_steps')
        learning_rate = config.learn_rate_end + 0.5 * (config.learn_rate_init - config.learn_rate_end) * (1 + tf.cos((global_step) / (train_steps) * np.pi))
    
    with tf.name_scope("model"):
        model = OrienDecoder(trainable, backbone="darknet53")
        pred_conf, pred_offset = model.forward(batch_image)
        total_loss, conf_loss, offset_loss = model.cal_loss(batch_conf, batch_offset)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            optimizer = config.optimizer_func(learning_rate).minimize(total_loss, global_step=global_step)
    
    with tf.name_scope('savers'):
        loader = tf.train.Saver(tf.global_variables())
        # self.loader = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(exclude=["learning_rate"]))
        # self.loader = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(exclude=["learning_rate", "yolov3/featuremap*", "train_op"]))
        # self.loader = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(exclude=["learning_rate",
        #                                                                         "yolov3/.*Adam.*", "yolov3/.*ExponentialMovingAverage.*"]))
        # loader = tf.train.Saver(var_list=tf.contrib.framework.get_variables_to_restore(exclude=["learning_rate", "darknet/.*Adam.*", 
        #                                                                                             "darknet/.*ExponentialMovingAverage.*"]))
        saver  = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    sess_config = tf.ConfigProto(allow_soft_placement=True)
    sess_config.gpu_options.per_process_gpu_memory_fraction = 0.6
    with tf.Session(config=sess_config) as sess:
        ## load weights
        sess.run(tf.global_variables_initializer())
        if config.checkpoint_path is None:
            print('=> Train from scratch ...')
        else:
            print('=> Restoring weights: %s ... ' % config.checkpoint_path)
            loader.restore(sess, config.checkpoint_path)

        write_op, train_summary_writer, val_summary_writer = build_log_summary([total_loss, conf_loss, offset_loss], 
                                                                            ["total_loss", "conf_loss", "offset_loss"], sess)
        try:
            for epoch in range(config.training_epochs):
                train_epoch_losses = []
                # val_epoch_loss = []
                for iter in range(1, trainset.iters_per_epoch + 1):
                    _, summary, g_step, t_loss, c_loss, o_loss = sess.run(
                        [optimizer, write_op, global_step, total_loss, conf_loss, offset_loss], feed_dict={trainable: True})
                    train_epoch_losses.append(t_loss)
                    train_summary_writer.add_summary(summary, g_step)

                    if g_step % config.log_step==0:
                        log_time = time.strftime('%Y-%m-%d-%H:%M:%S', time.localtime(time.time()))
                        losses_mess = "total loss: %.2f, conf loss: %.2f, offset loss: %.2f."%(t_loss, c_loss, o_loss)
                        print("=>%d %s Epoch: %2d Step: %2d/%d => Train loss: %.2f. "
                                            %(g_step, log_time, epoch, iter, trainset.iters_per_epoch, t_loss)+losses_mess)

                    if g_step % cfg.TRAIN.SAVING_STEP==0:
                        print("Evaluating...")
                        val_loss, val_summary = sess.run([total_loss, write_op], feed_dict={trainable: False})
                        # val_epoch_loss.append(val_loss)
                        val_summary_writer.add_summary(val_summary, g_step)
                        avg_training_loss, val_epoch_loss = np.mean(train_epoch_losses), np.mean(val_loss)
                        ckpt_file = os.path.join(config.save_dir, "modesl_%.4f.ckpt" % val_loss)
                        log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
                        print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                                        %(epoch, log_time, avg_training_loss, val_epoch_loss, ckpt_file))
                        saver.save(sess, ckpt_file, global_step=epoch)
                                            
        except tf.errors.OutOfRangeError:
            print("end of sequence.")

def build_log_summary(var_list, name_list, sess):
    logdir = os.path.join(cfg.TRAIN.SAVER_DIR, "logs")
    if os.path.exists(logdir): shutil.rmtree(logdir)
    os.makedirs(logdir)

    with tf.name_scope('summary'):
        for idx, var in enumerate(var_list):
            tf.summary.scalar(name_list[idx], var)            
    
    write_op = tf.summary.merge_all()
    train_summary_writer  = tf.summary.FileWriter(os.path.join(logdir, "train"), graph=sess.graph)
    val_summary_writer  = tf.summary.FileWriter(os.path.join(logdir, "test"), graph=sess.graph)

    return write_op, train_summary_writer, val_summary_writer

if __name__ == '__main__': 
    train(Configs())
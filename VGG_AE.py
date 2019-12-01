import numpy as np
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets.vgg import vgg_16
import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np
import glob, random, os


class VGG_Encoder(object):
    def __init__(self,
                 im_height=224,
                 im_width=244,
                 channels=3,
                 save_path='VGG_AE/VGG_AE.ckpt'):

        self.save_path = save_path
        self.im_height = im_height
        self.im_width = im_width
        self.im_channels = channels

        with tf.variable_scope("VGG"):
            self.inputs = tf.placeholder(tf.float32, shape=[None, self.im_height, self.im_width, channels])
            with slim.arg_scope(vgg.vgg_arg_scope()):
                net, end_points = vgg_16(self.inputs, is_training=False)

    def set_session(self, session):
        self.session = session

    def decode(self, images):
        return self.session.run(self.predict_op, feed_dict={self.inputs: images})

    def update(self, states, actions, targets):
        c, _ = self.session.run(
            [self.cost, self.train_op],
            feed_dict={
                self.inputs: states,
                self.goals: targets,
                self.actions: actions
            }
        )
        return c

    def load(self):
        self.saver = tf.train.Saver(tf.global_variables())
        load_was_success = True
        try:
            save_dir = '/'.join(self.save_path.split('/')[:-1])
            ckpt = tf.train.get_checkpoint_state(save_dir)
            load_path = ckpt.model_checkpoint_path
            self.saver.restore(self.session, load_path)
        except:
            print("no saved model to load. starting new session")
            load_was_success = False
        else:
            print("loaded model: {}".format(load_path))
            saver = tf.train.Saver(tf.global_variables())
            episode_number = int(load_path.split('-')[-1])

    def save(self, n):
        self.saver.save(self.session, self.save_path, global_step=n)
        print("SAVED MODEL #{}".format(n))

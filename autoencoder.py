from tensorflow.contrib.slim.nets import vgg
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import os


# Create graph
inputs = tf.placeholder(tf.float32, shape=[None, 300, 300, 3])
resized_input = tf.image.resize_images(inputs, [224, 224])

with slim.arg_scope(vgg.vgg_arg_scope()):
    net, end_points = vgg.vgg_16(resized_input, is_training=True)

fc6_variables = tf.contrib.framework.get_variables_to_restore(exclude=["vgg/fc7"])
fc6_init = tf.contrib.framework.assign_from_checkpoint_fn(
    os.path.join("VGG_weights/vgg_16.ckpt", '{}.ckpt'.format("vgg_16")), fc6_variables)
fc7_variables = tf.contrib.framework.get_variables()
fc8_init = tf.variables_initializer(fc7_variables)
fc7_activation = tf.nn.relu(end_points["vgg_16/fc7"])
fc7_reshape = tf.reshape(fc7_activation, [-1, 8, 8, 512])
conv_9 = tf.layers.conv2d_transpose(fc7_reshape, filters=46, kernel_size=2, strides=2,padding="same", name="conv_9")
conv_10 = tf.layers.conv2d_transpose(conv_9, filters=28, kernel_size=3, strides=2,padding="same", name="conv_10")
conv_11 = tf.layers.conv2d_transpose(conv_10, filters=14, kernel_size=3, strides=2,padding="same", name="conv_11")
conv_12 = tf.layers.conv2d_transpose(conv_11, filters=7, kernel_size=3, strides=2,padding="same", name="conv_12")
conv_13 = tf.layers.conv2d_transpose(conv_12, filters=3, kernel_size=3, strides=2,padding="same", name="conv_13")
#conv_14 = tf.layers.conv2d_transpose(conv_13, filters=3, kernel_size=1, strides=2,padding="same", name="conv_14")
#conv_15 = tf.layers.conv2d_transpose(conv_14, filters=3, kernel_size=3, strides=2,padding="same", name="conv_14")

print(conv_9)




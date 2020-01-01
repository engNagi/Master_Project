from DQnetwork_bitflipping import DQN
import tensorflow as tf
#from bitflipping import BitFlip
import time
from Environment_dataset_generation import Environment
from image_poses import Pose_CNN
import numpy as np
import matplotlib.pyplot as plt

env = Environment(top_view_cam=False)
env.make()
encoder = Pose_CNN(samples_size=0)
x_corr  = []
z_corr  = []
x_corr_ = []
z_corr_ = []
with tf.Session() as sess:
    encoder.set_session(sess)
    sess.run(tf.global_variables_initializer())
    encoder.load()
    observation, _, poses, _, _, _ = env.reset("FloorPlan220")  # reset environment
    for n in range(1000):
            action = env.action_sampler()
            observation, agent_position, _, _, _ = env.take_action(action)
            observation = np.expand_dims(observation, axis=0)
            estimated_position = sess.run([encoder.estimated_poses], feed_dict={encoder.image: observation})
            print("1pose", agent_position,"\n2pose", estimated_position)
            x =estimated_position[0]
            estimated_position = x[0]
            x_corr.append(agent_position[0])
            z_corr.append(agent_position[2])
            x_corr_.append(estimated_position[0])
            z_corr_.append(estimated_position[2])
    plt.plot(np.absolute(np.subtract(x_corr, x_corr_)), "b-", label="ground truth")
    plt.plot(np.absolute(np.subtract(z_corr, z_corr_)), "r-", label="estimated")
    plt.legend(loc="best")
    plt.show()

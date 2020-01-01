import tensorflow as tf
import numpy as np
import glob, random, os
from Environment_dataset_generation import Environment
from collections import deque

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

training = 60000
validation_iter = 20000
validation_samples = deque(maxlen=20000)
batch_size = 120


class Pose_CNN(object):
    def __init__(self,
                 poses_dims=3,
                 samples_size=20000):

        self.sample_size = samples_size
        self.pose_dims = poses_dims
        self.save_path = 'CNN_images_poses/Pose_CNN.ckpt'
        self.sample_buffer = deque(maxlen=self.sample_size)
        #   input
        self.image = tf.placeholder(tf.float32, [None, 300, 300, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [256, 256])
        self.normalized_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.resized_image)
        #   poses
        self.positions = tf.placeholder(tf.float32, [None, poses_dims])

        #tf.summary.image('input_image', self.normalized_image, 20)

        self.estimated_poses = self.encoder(self.normalized_image)
        tf.summary.histogram("poses", self.positions)
        tf.summary.histogram("estimated_poses", self.estimated_poses)
        self.loss, self.error = self.compute_loss()         # self.poses_flat, self.estimated_poses_flat, self.error, self.batch_shape

        optimizer = tf.train.AdamOptimizer(1e-5)
        gradients = optimizer.compute_gradients(self.loss)
        clipped_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        self.train_op = optimizer.apply_gradients(clipped_grad)

        tf.summary.scalar("loss", self.loss)
        tf.summary.histogram("element_wise_error", self.error)

        self.merged = tf.summary.merge_all()

    def encoder(self, x):
        conv_1 = tf.layers.conv2d(x, filters=4, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal(), name="conv_1_down_sampling")

        conv_1_1 = tf.layers.conv2d(conv_1, filters=4, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal(), name="conv_1_feature_extractor")

        conv_2 = tf.layers.conv2d(conv_1_1, filters=8, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal(), name="conv2_down_sampling")
        conv_2_1 = tf.layers.conv2d(conv_2, filters=8, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal(), name="conv2_feature_extractor")

        conv_3 = tf.layers.conv2d(conv_2_1, filters=16, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal(), name="conv3_down_sampling")
        conv_3_1 = tf.layers.conv2d(conv_3, filters=16, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal(), name="conv3_feature_extractor")

        conv_4 = tf.layers.conv2d(conv_3_1, filters=32, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal(), name="conv4_down_sampling")
        conv_4_1 = tf.layers.conv2d(conv_4, filters=32, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal(), name="conv4_feature_extractor")

        conv_5 = tf.layers.conv2d(conv_4_1, filters=64, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal(), name="conv5_down_sampling")
        conv_5_1 = tf.layers.conv2d(conv_5, filters=64, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal(), name="conv5_feature_extractor")

        conv_6 = tf.layers.conv2d(conv_5_1, filters=128, kernel_size=3, strides=1, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal(), name="conv6_down_sampling")
        conv_6_1 = tf.layers.conv2d(conv_6, filters=128, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal(), name="conv_6_feature_extractor")

        conv_flatten = tf.layers.flatten(conv_6_1, name="conv_flatten")

        flatten_dense_1 = tf.layers.dense(conv_flatten, units=1024,
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name='Dense_1')
        flatten_dense_2 = tf.layers.dense(flatten_dense_1, units=215,
                                        activation=tf.nn.leaky_relu,
                                        kernel_initializer=tf.initializers.he_normal(),
                                        name='Dense_2')

        estimated_poses = tf.layers.dense(flatten_dense_2, units=self.pose_dims,
                                          name='pose_estimation')
        return estimated_poses

    def compute_loss(self):
        batch_shape = tf.shape(self.positions)[0]
        poses_flat = tf.reshape(self.positions, [batch_shape, -1])
        estimated_poses_flat = tf.reshape(self.estimated_poses, [batch_shape, -1])
        error = tf.subtract(poses_flat, estimated_poses_flat, name="error")
        loss = tf.reduce_sum(tf.square(error), axis=1)
        mean_loss = tf.reduce_mean(loss)

        return mean_loss, error #poses_flat, estimated_poses_flat , batch_shape

    def batch_sampler(self, batch_size):
        if len(self.sample_buffer) < batch_size:  # if there's no enough transitions, do nothing
            return 0
        # sample batches from samples_buffer
        else:
            samples = random.sample(self.sample_buffer, batch_size)
            observations, agent_poses = map(np.array, zip(*samples))

        return observations, agent_poses

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

    def set_session(self, session):
        self.session = session

def generate_scene_name():
    scene_name_list = []
    for k in range(430):
        if k in range(1, 31):
            scene_name_list.append("FloorPlan" + str(k))
        elif k in range(201, 231):
            scene_name_list.append("FloorPlan" + str(k))
        elif k in range(301, 331):
            scene_name_list.append("FloorPlan" + str(k))
        elif k in range(400, 431):
            scene_name_list.append("FloorPlan" + str(k + 1))

    return scene_name_list


def train_vae():
    scene_name_list = generate_scene_name()
    env = Environment(top_view_cam=False)
    env.make()
    encoder = Pose_CNN(samples_size=20000)

    for scene in scene_name_list:
        scene_number = scene[9:]
        observation, agent_position, agent_pose, _, _, _ = env.reset(scene=scene, random_init=True)
        validation_samples.append((observation, agent_position))
        encoder.sample_buffer.append((observation, agent_position))

        for batches_ in range(batch_size):
            action = env.action_sampler()
            observation_, agent_position_, agent_pose, _, _ = env.take_action(action)
            if batches_ % 5 == 0:
                validation_samples.append((observation_, agent_position_))
            else:
                encoder.sample_buffer.append((observation_, agent_position_))

    with tf.Session() as sess:
        encoder.set_session(session=sess)
        sess.run(tf.global_variables_initializer())
        encoder.load()

        loss_writer = tf.summary.FileWriter('logdir_CNN_pose/train', sess.graph)
        validation_writer = tf.summary.FileWriter('logdir_CNN_pose/validation')

        for i in range(training):
            observations, agent_poses = encoder.batch_sampler(8)
            training_loss, _, loss_summary = sess.run([encoder.loss, encoder.train_op, encoder.merged],
                                                      feed_dict={encoder.image: observations,
                                                                 encoder.positions: agent_poses})
            print('step {}: training loss {:.6f}'.format(i, training_loss))
            loss_writer.add_summary(loss_summary)

            if i % 50 == 0:
                encoder.save(i)

            # print("input", encoder.estimated_poses.eval(feed_dict={encoder.image: observations}))
            # print("input_shape", np.shape(encoder.estimated_poses.eval(feed_dict={encoder.image: observations})))
            # print("poses", encoder.positions.eval(feed_dict={encoder.positions: agent_poses}))
            # print("poses_shape", np.shape(encoder.positions.eval(feed_dict={encoder.positions: agent_poses})))
            # print("error", encoder.loss.eval(feed_dict={encoder.image: observations,
            #                                             encoder.positions: agent_poses}))
            # print("error_shape", np.shape(encoder.loss.eval(feed_dict={encoder.image: observations,
            #                                                            encoder.positions: agent_poses})))
            # print("pose_flat", np.shape(encoder.poses_flat.eval(feed_dict={encoder.positions: agent_poses})))
            # print("estimated_pose_flat", np.shape(encoder.estimated_poses_flat.eval(feed_dict={encoder.image: observations,
            #                                                                                    encoder.positions: agent_poses})))
            # print("normal_loss", encoder.error.eval(feed_dict={encoder.image: observations,
            #                                                             encoder.positions: agent_poses}))
            # print("normal_loss_shape", np.shape(encoder.error.eval(feed_dict={encoder.image: observations,
            #                                                             encoder.positions: agent_poses})))
            # print("batch_shape", np.shape(encoder.batch_shape.eval(feed_dict={encoder.positions: agent_poses})))

            if i % 5000 == 0 and i > 0:  # validation
                for i in range(validation_iter):
                    validation_observations, validation_agent_poses = encoder.batch_sampler(8)
                    validation_loss, validation_summary = sess.run([encoder.loss, encoder.merged],
                                                                   feed_dict={encoder.image: validation_observations,
                                                                              encoder.positions: agent_poses})
                    validation_writer.add_summary(validation_summary)
                    print('step {}: validation loss {:.6f}'.format(i, validation_loss))




def load_vae():
    graph = tf.Graph()
    with graph.as_default():
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config, graph=graph)

        network = Network()
        init = tf.global_variables_initializer()
        sess.run(init)

        saver = tf.train.Saver(max_to_keep=1)
        training_data = data_iterator(batch_size=32)

        try:
            saver.restore(sess, tf.train.latest_checkpoint(model_path))
        except:
            raise ImportError("Could not restore saved model")

        return sess, network


if __name__ == "__main__":
    train_vae()

import tensorflow as tf
# from tensorflow.python import debug as tf_debug
import numpy as np
import glob, random

model_path = "saved_models/"
model_name = model_path + 'model'
training_dataset_path = "/home/WIN-UNI-DUE/sjmonagi/Desktop/dataset/Train"
validation_dataset_path = "/home/WIN-UNI-DUE/sjmonagi/Desktop/dataset/Validation"
training = 50000
validation_iter = 5000
total_t = 0
validation_period = 5000


class Network(object):
    # Create model
    def __init__(self):
        self.save_path = '/Users/mohamednagi/Desktop/Master_Project/Autoencoder/CNN_AE.ckpt'
        self.image = tf.placeholder(tf.float32, [None, 300, 300, 3], name='image')
        self.resized_image = tf.image.resize_images(self.image, [256, 256])
        self.normalized_image = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), self.resized_image)

        # self.normalized_image = tf.image.per_image_standardization([self.resized_image[1],
        #                                                             self.resized_image[2],
        #                                                             self.resized_image[3]])
        tf.summary.image('resized', self.normalized_image, 20)

        self.features = self.encoder(self.normalized_image)
        self.reconstructions = self.decoder(self.features)
        tf.summary.image('reconstructed_normalized_image', self.reconstructions, 20)
        tf.summary.histogram("reconstructed", self.reconstructions)

        self.loss = self.compute_loss()
        optimizer = tf.train.AdamOptimizer(1e-3)
        gradients = optimizer.compute_gradients(self.loss)
        clipped_grad = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients]
        self.train_op = optimizer.apply_gradients(clipped_grad)

        tf.summary.scalar('loss', self.loss)

        self.merged = tf.summary.merge_all()

    def encoder(self, x):
        conv_1 = tf.layers.conv2d(x, filters=4, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal())

        conv_1_1 = tf.layers.conv2d(conv_1, filters=4, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal())

        conv_2 = tf.layers.conv2d(conv_1_1, filters=8, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal())
        conv_2_1 = tf.layers.conv2d(conv_2, filters=8, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal())

        conv_3 = tf.layers.conv2d(conv_2_1, filters=16, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal())
        conv_3_1 = tf.layers.conv2d(conv_3, filters=16, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal())

        conv_4 = tf.layers.conv2d(conv_3_1, filters=32, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal())
        conv_4_1 = tf.layers.conv2d(conv_4, filters=32, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal())

        conv_5 = tf.layers.conv2d(conv_4_1, filters=64, kernel_size=3, strides=2, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal())
        conv_5_1 = tf.layers.conv2d(conv_5, filters=64, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal())

        conv_6 = tf.layers.conv2d(conv_5_1, filters=128, kernel_size=3, strides=1, padding='same',
                                  activation=tf.nn.leaky_relu,
                                  kernel_initializer=tf.initializers.he_normal())
        conv_6_1 = tf.layers.conv2d(conv_6, filters=128, kernel_size=3, strides=1, padding='same',
                                    activation=tf.nn.leaky_relu,
                                    kernel_initializer=tf.initializers.he_normal())

        x = tf.layers.flatten(conv_6_1)
        z = tf.layers.dense(x, units=512, name='z_mu')
        return z

    def decoder(self, z):
        x = tf.layers.dense(z, 1024, activation=tf.nn.leaky_relu)  # 65536
        x = tf.reshape(x, [-1, 8, 8, 16])
        conv_8 = tf.layers.conv2d_transpose(x, filters=256, kernel_size=3, strides=2, padding='same',
                                            activation=tf.nn.leaky_relu,
                                            kernel_initializer=tf.initializers.he_normal())
        conv_8_1 = tf.layers.conv2d_transpose(conv_8, filters=256, kernel_size=3, strides=1, padding='same',
                                              activation=tf.nn.leaky_relu,
                                              kernel_initializer=tf.initializers.he_normal())

        conv_9 = tf.layers.conv2d_transpose(conv_8_1, filters=128, kernel_size=3, strides=2, padding='same',
                                            activation=tf.nn.leaky_relu,
                                            kernel_initializer=tf.initializers.he_normal())
        conv_9_1 = tf.layers.conv2d_transpose(conv_9, filters=128, kernel_size=3, strides=1, padding='same',
                                              activation=tf.nn.leaky_relu,
                                              kernel_initializer=tf.initializers.he_normal())

        conv_10 = tf.layers.conv2d_transpose(conv_9_1, filters=64, kernel_size=3, strides=2, padding='same',
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.initializers.he_normal())

        conv_10_1 = tf.layers.conv2d_transpose(conv_10, filters=64, kernel_size=3, strides=1, padding='same',
                                               activation=tf.nn.leaky_relu,
                                               kernel_initializer=tf.initializers.he_normal())

        conv_11 = tf.layers.conv2d_transpose(conv_10_1, filters=32, kernel_size=3, strides=2, padding='same',
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.initializers.he_normal())
        conv_11_1 = tf.layers.conv2d_transpose(conv_11, filters=32, kernel_size=3, strides=1, padding='same',
                                               activation=tf.nn.leaky_relu,
                                               kernel_initializer=tf.initializers.he_normal())

        conv_12 = tf.layers.conv2d_transpose(conv_11_1, filters=16, kernel_size=3, strides=2, padding='same',
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.initializers.he_normal())
        conv_12_1 = tf.layers.conv2d_transpose(conv_12, filters=16, kernel_size=3, strides=1, padding='same',
                                               activation=tf.nn.leaky_relu,
                                               kernel_initializer=tf.initializers.he_normal())
        conv_13 = tf.layers.conv2d_transpose(conv_12_1, filters=3, kernel_size=3, strides=1, padding='same',
                                             activation=tf.nn.leaky_relu,
                                             kernel_initializer=tf.initializers.he_normal())
        conv_13_1 = tf.layers.conv2d_transpose(conv_13, filters=3, kernel_size=3, strides=1, padding='same',
                                               activation=None,
                                               kernel_initializer=tf.initializers.he_normal())
        return conv_13_1

    def compute_loss(self):
        batch_shape = tf.shape(self.normalized_image)[0]
        logits_flat = tf.reshape(self.reconstructions, [batch_shape, -1])
        labels_flat = tf.reshape(self.normalized_image, [batch_shape, -1])
        reconstruction_loss = tf.reduce_sum(tf.square(logits_flat - labels_flat), axis=1)
        vae_loss = tf.reduce_mean(reconstruction_loss)

        return vae_loss

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


def data_iterator(batch_size, path):
    data_files = glob.glob(path + '/**/VAE_FloorPlan*', recursive=True)
    while True:
        data = np.load(random.sample(data_files, 1)[0])
        np.random.shuffle(data)
        np.random.shuffle(data)
        N = data.shape[0]
        start = np.random.randint(0, N - batch_size)
        yield data[start:start + batch_size]


def train_vae():
    network = Network()
    with tf.Session() as sess:
        network.set_session(session=sess)
        sess.run(tf.global_variables_initializer())
        network.load()

        loss_writer = tf.summary.FileWriter('logdir_AE/train', sess.graph)
        validation_writer = tf.summary.FileWriter('logdir_AE/validation')

        training_data = data_iterator(batch_size=32, path=training_dataset_path)
        validation_data = data_iterator(batch_size=16, path=validation_dataset_path)

        for i in range(training):
            training_images = next(training_data)
            training_loss, _, loss_summary = sess.run([network.loss, network.train_op, network.merged],
                                                      feed_dict={network.image: training_images})
            print('step {}: training loss {:.6f}'.format(i, training_loss))
            loss_writer.add_summary(loss_summary)

            # print("reconstructed",network.reconstructions.eval(feed_dict={network.image: training_images}))
            # print("input resized",network.normalized_image.eval(feed_dict={network.image: training_images}))
            # print("input",network.image.eval(feed_dict={network.image: training_images}))

            if i % 10000 == 0 and i > 0:  # validation
                print("validation")
                for i in range(validation_iter):
                    validation_images = next(validation_data)
                    validation_loss, validation_summary = sess.run([network.loss, network.merged],
                                                                   feed_dict={network.image: validation_images})
                    validation_writer.add_summary(validation_summary)
                    print('step {}: validation loss {:.6f}'.format(i, validation_loss))

            if i % 50 == 0:
                network.save(i)


def load_autoencoder():
    graph = tf.Graph()
    with graph.as_default():
        sess = tf.Session(graph=graph)
        network = Network()
        network.set_session(sess)
        init = tf.global_variables_initializer()
        sess.run(init)
        network.load()
        return sess, network


if __name__ == "__main__":
    train_vae()
